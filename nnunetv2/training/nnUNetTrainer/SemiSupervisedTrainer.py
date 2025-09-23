import copy
import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List
from torch import autocast
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset, nnUNetDatasetNumpy
from batchgenerators.utilities.file_and_folder_operations import load_pickle, write_pickle
import os
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.paths import nnUNet_preprocessed


class UnlabeledDataset(nnUNetBaseDataset):
    """专门用于无标签数据的数据集类"""
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.identifiers)
    
    def load_case(self, identifier):
        """加载无标签数据案例"""
        data_file = join(self.source_folder, identifier + '.npz')
        data_dict = np.load(data_file)
        data = data_dict['data']
        
        # 无标签数据没有分割标签，返回None
        seg = None
        seg_prev = None
        
        # 加载属性文件
        properties_file = join(self.source_folder, identifier + '.pkl')
        if os.path.exists(properties_file):
            properties = load_pickle(properties_file)
        else:
            # 如果没有属性文件，创建基本属性
            properties = {'spacing': [1.0, 1.0, 1.0], 'origin': [0.0, 0.0, 0.0]}
        
        # 确保无标签数据的properties包含必要的字段
        if 'class_locations' not in properties:
            # 无标签数据没有类别位置信息，设置为空字典
            properties['class_locations'] = {}
        
        return data, seg, seg_prev, properties
    
    @staticmethod
    def save_case(data: np.ndarray, seg: np.ndarray, properties: dict, output_filename_truncated: str):
        """保存数据案例（无标签数据通常不需要保存）"""
        np.savez_compressed(output_filename_truncated + '.npz', data=data)
        if properties is not None:
            write_pickle(properties, output_filename_truncated + '.pkl')
    
    @staticmethod
    def get_identifiers(folder: str) -> List[str]:
        """获取文件夹中所有数据的标识符"""
        if not os.path.exists(folder):
            print(f"警告：无标签数据文件夹不存在: {folder}")
            return []
        
        files = os.listdir(folder)
        npz_files = [i for i in files if i.endswith('.npz')]
        case_identifiers = [i[:-4] for i in npz_files]
        
        print(f"在文件夹 {folder} 中找到 {len(case_identifiers)} 个无标签数据文件")
        if len(case_identifiers) == 0:
            print(f"文件夹中的所有文件: {files[:10]}...")  # 只显示前10个文件
        
        return case_identifiers


class SemiSupervisedTrainer(nnUNetTrainer):
    """
    半监督训练器，实现教师-学生框架
    
    核心特性：
    1. 教师模型：使用EMA权重，不参与梯度更新
    2. 学生模型：实际训练的模型
    3. 一致性损失：在无标签数据上保持预测一致性
    4. 动态权重：随训练进程调整一致性损失权重
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # 保存配置参数
        self.configuration = configuration
        
        # 半监督学习参数
        self.ema_decay = 0.99  # EMA衰减率
        self.consistency_weight = 1.0  # 一致性损失权重
        self.consistency_rampup_starts = 0  # 一致性损失开始epoch
        self.consistency_rampup_ends = 200  # 一致性损失达到最大权重的epoch
        self.unlabeled_batch_size = 2  # 无标签数据批次大小
        self.unlabeled_data_path = None  # 无标签数据路径
        
        # 设置保存间隔（从配置文件读取，默认为1以支持每个epoch保存）
        self.save_every = 1  # 每个epoch保存一次
        
        # 教师模型（将在initialize中创建）
        self.teacher_model = None
        
        # 无标签数据加载器
        self.unlabeled_dataloader = None
        
    def initialize(self):
        """初始化训练器，包括创建教师模型"""
        super().initialize()
        
        # 创建教师模型（学生模型的深拷贝）
        self.teacher_model = copy.deepcopy(self.network)
        self.teacher_model.eval()
        
        # 冻结教师模型参数
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        # 初始化无标签数据加载器
        self._setup_unlabeled_dataloader()
        
    def _setup_unlabeled_dataloader(self):
        """设置无标签数据加载器"""
        try:
            # 获取无标签数据路径
            if self.unlabeled_data_path is not None:
                unlabeled_folder = self.unlabeled_data_path
            else:
                unlabeled_folder = join(nnUNet_preprocessed, self.dataset_json['name'], 
                                      self.configuration)
            
            print(f"使用无标签数据路径: {unlabeled_folder}")
            
            # 创建无标签数据集
            unlabeled_dataset = UnlabeledDataset(
                folder=unlabeled_folder,
                identifiers=None,  # 自动获取所有标识符
                folder_with_segs_from_previous_stage=None
            )
            
            # 获取patch_size和transforms
            patch_size = self.configuration_manager.patch_size
            
            # 获取训练时的transforms
            deep_supervision_scales = self._get_deep_supervision_scales()
            (
                rotation_for_DA,
                do_dummy_2d_data_aug,
                initial_patch_size,
                mirror_axes,
            ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
            
            tr_transforms = self.get_training_transforms(
                patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
                use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
                is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
                regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
                ignore_label=self.label_manager.ignore_label)
            
            # 创建无标签数据加载器
            self.unlabeled_dataloader = nnUNetDataLoader(
                unlabeled_dataset,
                self.unlabeled_batch_size,
                initial_patch_size,
                patch_size,
                self.label_manager,
                oversample_foreground_percent=0.0,  # 无标签数据不需要前景过采样
                sampling_probabilities=None,
                pad_sides=None,
                probabilistic_oversampling=False,
                transforms=tr_transforms  # 使用训练时的数据增强
            )
            
            print(f"成功设置无标签数据加载器，数据量: {len(unlabeled_dataset)}")
            
        except Exception as e:
            print(f"警告：无法设置无标签数据加载器: {e}")
            print("将仅使用有标签数据进行训练")
            self.unlabeled_dataloader = None
    
    def update_teacher_model(self):
        """使用EMA更新教师模型权重"""
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher_model.parameters(), 
                                                   self.network.parameters()):
                teacher_param.data = (self.ema_decay * teacher_param.data + 
                                    (1 - self.ema_decay) * student_param.data)
    
    def get_consistency_weight(self, epoch: int) -> float:
        """计算当前epoch的一致性损失权重"""
        if epoch < self.consistency_rampup_starts:
            return 0.0
        elif epoch >= self.consistency_rampup_ends:
            return self.consistency_weight
        else:
            # 线性增长
            rampup_length = self.consistency_rampup_ends - self.consistency_rampup_starts
            current_rampup = epoch - self.consistency_rampup_starts
            return self.consistency_weight * (current_rampup / rampup_length)
    
    def compute_consistency_loss(self, student_output, teacher_output):
        """计算一致性损失"""
        # 使用MSE损失
        return F.mse_loss(student_output, teacher_output)
    
    def train_step(self, batch: dict) -> dict:
        """重写训练步骤，加入半监督学习逻辑"""
        # 标准的监督学习步骤
        supervised_loss_dict = super().train_step(batch)
        
        # 如果没有无标签数据，返回监督损失
        if self.unlabeled_dataloader is None:
            return supervised_loss_dict
        
        # 获取一致性损失权重
        consistency_weight = self.get_consistency_weight(self.current_epoch)
        
        if consistency_weight == 0.0:
            return supervised_loss_dict
        
        try:
            # 获取无标签数据
            unlabeled_batch = next(self.unlabeled_dataloader)
            unlabeled_data = unlabeled_batch['data'].to(self.device, non_blocking=True)
            
            # 教师模型预测（弱增强）
            with torch.no_grad():
                with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                    teacher_output = self.teacher_model(unlabeled_data)
            
            # 学生模型预测（强增强，这里简化为同样的数据）
            with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                student_output = self.network(unlabeled_data)
            
            # 计算一致性损失
            if self.enable_deep_supervision:
                # 如果启用深度监督，只使用最高分辨率的输出
                consistency_loss = self.compute_consistency_loss(student_output[0], teacher_output[0])
            else:
                consistency_loss = self.compute_consistency_loss(student_output, teacher_output)
            
            # 加权一致性损失
            weighted_consistency_loss = consistency_weight * consistency_loss
            
            # 反向传播一致性损失
            if self.grad_scaler is not None:
                self.grad_scaler.scale(weighted_consistency_loss).backward()
            else:
                weighted_consistency_loss.backward()
            
            # 更新返回的损失字典
            total_loss = supervised_loss_dict['loss'] + weighted_consistency_loss.detach().cpu().numpy()
            
            return {
                'loss': total_loss,
                'supervised_loss': supervised_loss_dict['loss'],
                'consistency_loss': consistency_loss.detach().cpu().numpy(),
                'consistency_weight': consistency_weight
            }
            
        except Exception as e:
            print(f"一致性损失计算出错: {e}")
            return supervised_loss_dict
    
    def on_train_epoch_end(self, train_outputs: List[dict]):
        """训练epoch结束时的处理"""
        super().on_train_epoch_end(train_outputs)
        
        # 更新教师模型
        self.update_teacher_model()
        
        # 记录半监督学习相关指标（不使用logger.log，因为它只支持预定义的键）
        if len(train_outputs) > 0 and 'consistency_loss' in train_outputs[0]:
            consistency_losses = [o.get('consistency_loss', 0) for o in train_outputs]
            consistency_weights = [o.get('consistency_weight', 0) for o in train_outputs]
            
            avg_consistency_loss = np.mean(consistency_losses)
            avg_consistency_weight = np.mean(consistency_weights)
            
            # 直接打印到日志文件而不是使用logger.log
            self.print_to_log_file(f"Epoch {self.current_epoch}: 一致性损失 = {avg_consistency_loss:.4f}, "
                                 f"一致性权重 = {avg_consistency_weight:.4f}")
            
            print(f"Epoch {self.current_epoch}: 一致性损失 = {avg_consistency_loss:.4f}, "
                  f"一致性权重 = {avg_consistency_weight:.4f}")
    
    def save_checkpoint(self, filename: str) -> None:
        """保存检查点，包括教师模型"""
        super().save_checkpoint(filename)
        
        # 保存教师模型
        if self.teacher_model is not None:
            teacher_filename = filename.replace('.pth', '_teacher.pth')
            torch.save({
                'teacher_model_state_dict': self.teacher_model.state_dict(),
                'ema_decay': self.ema_decay,
            }, teacher_filename)
            print(f"教师模型已保存到: {teacher_filename}")
    
    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        """加载检查点，包括教师模型"""
        super().load_checkpoint(filename_or_checkpoint)
        
        # 加载教师模型
        if isinstance(filename_or_checkpoint, str):
            # 首先尝试加载专门的教师模型文件
            teacher_filename = filename_or_checkpoint.replace('.pth', '_teacher.pth')
            teacher_loaded = False
            
            try:
                teacher_checkpoint = torch.load(teacher_filename, map_location=self.device, weights_only=False)
                if self.teacher_model is not None:
                    self.teacher_model.load_state_dict(teacher_checkpoint['teacher_model_state_dict'])
                    self.ema_decay = teacher_checkpoint.get('ema_decay', self.ema_decay)
                    print(f"教师模型已从 {teacher_filename} 加载")
                    teacher_loaded = True
            except FileNotFoundError:
                print(f"未找到教师模型文件: {teacher_filename}")
            except Exception as e:
                print(f"加载教师模型时出错: {e}")
            
            # 如果没有专门的教师模型文件，使用主模型文件初始化教师模型
            if not teacher_loaded:
                try:
                    print(f"使用主模型文件初始化教师模型: {filename_or_checkpoint}")
                    if self.teacher_model is not None:
                        self.teacher_model.load_state_dict(self.network.state_dict())
                        print("教师模型已用学生模型权重初始化")
                except Exception as e:
                    print(f"初始化教师模型时出错: {e}")