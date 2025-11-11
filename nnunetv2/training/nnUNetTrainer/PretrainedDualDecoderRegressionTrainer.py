import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List
from torch import autocast
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.architectures.pretrained_dual_decoder_regression_unet import PretrainedDualDecoderRegressionUNet
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
import os
from batchgenerators.utilities.file_and_folder_operations import join


class PretrainedDualDecoderRegressionTrainer(nnUNetTrainer):
    """
    基于预训练权重的双解码器回归训练器
    
    特点：
    1. 加载半监督训练的分割模型权重
    2. 冻结分割解码器，只训练回归解码器
    3. 支持联合损失（分割损失 + 回归损失）
    4. 支持单向注意力机制
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # 回归相关参数
        self.regression_dim = 1  # 回归输出维度
        self.regression_loss_weight = 1.0  # 回归损失权重
        self.segmentation_loss_weight = 0.1  # 分割损失权重（较小，因为已经预训练）
        self.freeze_segmentation = True  # 是否冻结分割解码器
        self.enable_cross_attention = True  # 是否启用交叉注意力
        
        # 预训练权重路径
        self.pretrained_checkpoint_path = None
        
        # 损失函数
        self.regression_loss = nn.MSELoss()
        
    def set_pretrained_checkpoint_path(self, checkpoint_path: str):
        """设置预训练检查点路径"""
        self.pretrained_checkpoint_path = checkpoint_path
        print(f"设置预训练检查点路径: {checkpoint_path}")
    
    def set_regression_parameters(self, regression_dim: int = 1, 
                                regression_loss_weight: float = 1.0,
                                segmentation_loss_weight: float = 0.1):
        """设置回归相关参数"""
        self.regression_dim = regression_dim
        self.regression_loss_weight = regression_loss_weight
        self.segmentation_loss_weight = segmentation_loss_weight
        print(f"回归参数设置: dim={regression_dim}, reg_weight={regression_loss_weight}, seg_weight={segmentation_loss_weight}")
    
    def build_network_architecture(self, architecture_class_name: str,
                                 arch_init_kwargs: dict,
                                 arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                 num_input_channels: int,
                                 num_output_channels: int,
                                 enable_deep_supervision: bool = True) -> nn.Module:
        """
        构建双解码器回归网络架构
        """
        # 使用我们自定义的双解码器网络
        network = PretrainedDualDecoderRegressionUNet(
            input_channels=num_input_channels,
            n_stages=len(arch_init_kwargs['features_per_stage']),
            features_per_stage=arch_init_kwargs['features_per_stage'],
            conv_op=arch_init_kwargs['conv_op'],
            kernel_sizes=arch_init_kwargs['kernel_sizes'],
            strides=arch_init_kwargs['strides'],
            n_conv_per_stage=arch_init_kwargs['n_conv_per_stage'],
            num_classes=num_output_channels,
            n_conv_per_stage_decoder=arch_init_kwargs['n_conv_per_stage_decoder'],
            conv_bias=arch_init_kwargs['conv_bias'],
            norm_op=arch_init_kwargs['norm_op'],
            norm_op_kwargs=arch_init_kwargs['norm_op_kwargs'],
            dropout_op=arch_init_kwargs['dropout_op'],
            dropout_op_kwargs=arch_init_kwargs['dropout_op_kwargs'],
            nonlin=arch_init_kwargs['nonlin'],
            nonlin_kwargs=arch_init_kwargs['nonlin_kwargs'],
            deep_supervision=enable_deep_supervision,
            nonlin_first=arch_init_kwargs['nonlin_first'],
            # 回归相关参数
            regression_dim=self.regression_dim,
            enable_cross_attention=self.enable_cross_attention,
            # 预训练权重相关
            pretrained_checkpoint_path=self.pretrained_checkpoint_path,
            freeze_segmentation_decoder=self.freeze_segmentation
        )
        
        return network
    
    def _get_loss(self):
        """
        配置损失函数
        """
        # 分割损失（使用原有的损失函数）
        if self.label_manager.has_regions:
            segmentation_loss = DC_and_BCE_loss({}, {}, self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
        else:
            segmentation_loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                              'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, 
                                             self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
        
        # 如果启用深度监督，包装分割损失
        if self.enable_deep_supervision:
            segmentation_loss = DeepSupervisionWrapper(segmentation_loss, None)
        
        return segmentation_loss
    
    def train_step(self, batch: dict) -> dict:
        """
        训练步骤，支持双解码器训练
        """
        data = batch['data']
        target_seg = batch['target']  # 分割标签
        target_reg = batch.get('regression_target', None)  # 回归标签
        
        data = data.to(self.device, non_blocking=True)
        if isinstance(target_seg, list):
            target_seg = [i.to(self.device, non_blocking=True) for i in target_seg]
        else:
            target_seg = target_seg.to(self.device, non_blocking=True)
        
        if target_reg is not None:
            target_reg = target_reg.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad()
        
        # 前向传播
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            seg_output, reg_output = self.network(data)
            
            # 计算分割损失
            seg_loss = self.loss(seg_output, target_seg)
            
            # 计算回归损失
            reg_loss = torch.tensor(0.0, device=self.device)
            if target_reg is not None:
                reg_loss = self.regression_loss(reg_output, target_reg)
            
            # 总损失
            total_loss = (self.segmentation_loss_weight * seg_loss + 
                         self.regression_loss_weight * reg_loss)
        
        # 反向传播
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        
        return {
            'loss': total_loss.detach().cpu().numpy(),
            'seg_loss': seg_loss.detach().cpu().numpy(),
            'reg_loss': reg_loss.detach().cpu().numpy() if target_reg is not None else 0.0
        }
    
    def validation_step(self, batch: dict) -> dict:
        """
        验证步骤
        """
        data = batch['data']
        target_seg = batch['target']
        target_reg = batch.get('regression_target', None)
        
        data = data.to(self.device, non_blocking=True)
        if isinstance(target_seg, list):
            target_seg = [i.to(self.device, non_blocking=True) for i in target_seg]
        else:
            target_seg = target_seg.to(self.device, non_blocking=True)
        
        if target_reg is not None:
            target_reg = target_reg.to(self.device, non_blocking=True)
        
        self.network.eval()
        
        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                seg_output, reg_output = self.network(data)
                
                # 计算分割损失
                seg_loss = self.loss(seg_output, target_seg)
                
                # 计算回归损失
                reg_loss = torch.tensor(0.0, device=self.device)
                if target_reg is not None:
                    reg_loss = self.regression_loss(reg_output, target_reg)
                
                # 总损失
                total_loss = (self.segmentation_loss_weight * seg_loss + 
                             self.regression_loss_weight * reg_loss)
        
        return {
            'loss': total_loss.detach().cpu().numpy(),
            'seg_loss': seg_loss.detach().cpu().numpy(),
            'reg_loss': reg_loss.detach().cpu().numpy() if target_reg is not None else 0.0,
            'seg_output': seg_output,
            'reg_output': reg_output
        }
    
    def on_epoch_end(self):
        """
        Epoch结束时的处理
        """
        super().on_epoch_end()
        
        # 打印训练信息
        if hasattr(self, 'logger') and self.logger is not None:
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.log('learning_rate', current_lr, self.current_epoch)
            
            # 记录损失权重
            self.logger.log('segmentation_loss_weight', self.segmentation_loss_weight, self.current_epoch)
            self.logger.log('regression_loss_weight', self.regression_loss_weight, self.current_epoch)
    
    def save_checkpoint(self, filename: str) -> None:
        """
        保存检查点，包含额外的回归训练信息
        """
        checkpoint = {
            'network_weights': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
            'logging': self.logger.get_checkpoint() if self.logger is not None else None,
            'epoch': self.current_epoch + 1,
            'init': self.get_init_dict(),
            'trainer_name': self.__class__.__name__,
            'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
            # 回归训练特定信息
            'regression_dim': self.regression_dim,
            'regression_loss_weight': self.regression_loss_weight,
            'segmentation_loss_weight': self.segmentation_loss_weight,
            'freeze_segmentation': self.freeze_segmentation,
            'enable_cross_attention': self.enable_cross_attention,
            'pretrained_checkpoint_path': self.pretrained_checkpoint_path
        }
        
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        """
        加载检查点，恢复回归训练信息
        """
        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        else:
            checkpoint = filename_or_checkpoint
        
        # 加载基本信息
        super().load_checkpoint(checkpoint)
        
        # 加载回归训练特定信息
        if 'regression_dim' in checkpoint:
            self.regression_dim = checkpoint['regression_dim']
        if 'regression_loss_weight' in checkpoint:
            self.regression_loss_weight = checkpoint['regression_loss_weight']
        if 'segmentation_loss_weight' in checkpoint:
            self.segmentation_loss_weight = checkpoint['segmentation_loss_weight']
        if 'freeze_segmentation' in checkpoint:
            self.freeze_segmentation = checkpoint['freeze_segmentation']
        if 'enable_cross_attention' in checkpoint:
            self.enable_cross_attention = checkpoint['enable_cross_attention']
        if 'pretrained_checkpoint_path' in checkpoint:
            self.pretrained_checkpoint_path = checkpoint['pretrained_checkpoint_path']
        
        print(f"已加载回归训练检查点，回归维度: {self.regression_dim}")
    
    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        """
        重写日志打印，添加回归训练信息
        """
        super().print_to_log_file(*args, also_print_to_console=also_print_to_console, add_timestamp=add_timestamp)
        
        if self.current_epoch == 0:  # 只在第一个epoch打印配置信息
            self.print_to_log_file("=" * 50)
            self.print_to_log_file("双解码器回归训练配置:")
            self.print_to_log_file(f"  回归输出维度: {self.regression_dim}")
            self.print_to_log_file(f"  回归损失权重: {self.regression_loss_weight}")
            self.print_to_log_file(f"  分割损失权重: {self.segmentation_loss_weight}")
            self.print_to_log_file(f"  冻结分割解码器: {self.freeze_segmentation}")
            self.print_to_log_file(f"  启用交叉注意力: {self.enable_cross_attention}")
            self.print_to_log_file(f"  预训练权重路径: {self.pretrained_checkpoint_path}")
            self.print_to_log_file("=" * 50)


# 导入必要的损失函数
try:
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
except ImportError:
    from nnunetv2.training.loss.dice import SoftDiceLoss as MemoryEfficientSoftDiceLoss