import os
import numpy as np
import torch
from typing import Union, List, Tuple
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.utilities.file_and_folder_operations import join, load_json, subfiles
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed


class UnlabeledDataset(nnUNetBaseDataset):
    """
    无标签数据集类，专门用于加载无标签的图像数据
    """
    
    def __init__(self, folder: str, case_identifiers: List[str] = None,
                 num_images_properties_loading_threshold: int = 1000):
        """
        初始化无标签数据集
        
        Args:
            folder: 数据文件夹路径
            case_identifiers: 病例标识符列表，如果为None则自动发现
            num_images_properties_loading_threshold: 加载属性的阈值
        """
        super().__init__(folder, case_identifiers, num_images_properties_loading_threshold)
        
    def load_case(self, key: str):
        """
        加载单个病例，返回图像数据和虚拟标签
        
        Args:
            key: 病例标识符
            
        Returns:
            tuple: (data, dummy_seg, dummy_seg_prev, properties)
        """
        # 加载图像数据和属性
        data, seg, seg_prev, properties = super().load_case(key)
        
        # 为无标签数据创建虚拟分割标签（全零）
        dummy_seg = np.zeros_like(seg) if seg is not None else np.zeros(
            (1, *data.shape[1:]), dtype=np.uint8)
        
        return data, dummy_seg, seg_prev, properties


class SemiSupervisedDataLoader:
    """
    半监督数据加载器，同时管理有标签和无标签数据
    """
    
    def __init__(self, 
                 labeled_dataloader: nnUNetDataLoader,
                 unlabeled_folder: str,
                 unlabeled_batch_size: int = 2,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 label_manager: LabelManager = None,
                 transforms=None):
        """
        初始化半监督数据加载器
        
        Args:
            labeled_dataloader: 有标签数据加载器
            unlabeled_folder: 无标签数据文件夹路径
            unlabeled_batch_size: 无标签数据批次大小
            patch_size: 补丁大小
            label_manager: 标签管理器
            transforms: 数据变换
        """
        self.labeled_dataloader = labeled_dataloader
        self.unlabeled_batch_size = unlabeled_batch_size
        
        # 设置无标签数据加载器
        self.unlabeled_dataloader = self._setup_unlabeled_dataloader(
            unlabeled_folder, patch_size, label_manager, transforms
        )
        
        # 创建迭代器
        self.labeled_iter = iter(self.labeled_dataloader)
        self.unlabeled_iter = iter(self.unlabeled_dataloader) if self.unlabeled_dataloader else None
        
    def _setup_unlabeled_dataloader(self, unlabeled_folder: str, 
                                   patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                                   label_manager: LabelManager,
                                   transforms) -> nnUNetDataLoader:
        """
        设置无标签数据加载器
        
        Args:
            unlabeled_folder: 无标签数据文件夹
            patch_size: 补丁大小
            label_manager: 标签管理器
            transforms: 数据变换
            
        Returns:
            nnUNetDataLoader: 无标签数据加载器
        """
        try:
            # 检查无标签数据文件夹是否存在
            if not os.path.exists(unlabeled_folder):
                print(f"警告：无标签数据文件夹不存在: {unlabeled_folder}")
                return None
            
            # 创建无标签数据集
            unlabeled_dataset = UnlabeledDataset(
                unlabeled_folder,
                case_identifiers=None,  # 自动发现所有病例
                num_images_properties_loading_threshold=0
            )
            
            if len(unlabeled_dataset) == 0:
                print(f"警告：无标签数据文件夹为空: {unlabeled_folder}")
                return None
            
            # 创建无标签数据加载器
            unlabeled_dataloader = nnUNetDataLoader(
                unlabeled_dataset,
                self.unlabeled_batch_size,
                patch_size,
                patch_size,  # final_patch_size与patch_size相同
                label_manager,
                oversample_foreground_percent=0.0,  # 无标签数据不需要前景过采样
                sampling_probabilities=None,
                pad_sides=None,
                probabilistic_oversampling=False,
                transforms=transforms
            )
            
            print(f"成功创建无标签数据加载器，包含 {len(unlabeled_dataset)} 个病例")
            return unlabeled_dataloader
            
        except Exception as e:
            print(f"创建无标签数据加载器失败: {e}")
            return None
    
    def get_labeled_batch(self):
        """
        获取有标签数据批次
        
        Returns:
            dict: 有标签数据批次
        """
        try:
            return next(self.labeled_iter)
        except StopIteration:
            # 重新开始迭代
            self.labeled_iter = iter(self.labeled_dataloader)
            return next(self.labeled_iter)
    
    def get_unlabeled_batch(self):
        """
        获取无标签数据批次
        
        Returns:
            dict: 无标签数据批次，如果没有无标签数据则返回None
        """
        if self.unlabeled_iter is None:
            return None
            
        try:
            return next(self.unlabeled_iter)
        except StopIteration:
            # 重新开始迭代
            self.unlabeled_iter = iter(self.unlabeled_dataloader)
            return next(self.unlabeled_iter)
    
    def __iter__(self):
        """迭代器接口"""
        return self
    
    def __next__(self):
        """
        获取下一个批次（有标签数据）
        
        Returns:
            dict: 有标签数据批次
        """
        return self.get_labeled_batch()
    
    def __len__(self):
        """返回有标签数据加载器的长度"""
        return len(self.labeled_dataloader)


class UnlabeledDataDiscovery:
    """
    无标签数据发现工具，用于从原始数据文件夹中发现无标签图像
    """
    
    @staticmethod
    def discover_unlabeled_images(raw_data_folder: str, 
                                labeled_case_ids: List[str] = None) -> List[str]:
        """
        从原始数据文件夹中发现无标签图像
        
        Args:
            raw_data_folder: 原始数据文件夹路径
            labeled_case_ids: 已标注的病例ID列表
            
        Returns:
            List[str]: 无标签图像文件路径列表
        """
        if not os.path.exists(raw_data_folder):
            print(f"原始数据文件夹不存在: {raw_data_folder}")
            return []
        
        # 获取所有图像文件
        image_files = subfiles(raw_data_folder, suffix='.nii.gz', join=True)
        
        if labeled_case_ids is None:
            labeled_case_ids = []
        
        # 过滤出无标签图像
        unlabeled_images = []
        for img_file in image_files:
            # 提取病例ID
            case_id = os.path.basename(img_file).split('_')[0]
            
            # 如果不在已标注列表中，则为无标签数据
            if case_id not in labeled_case_ids:
                unlabeled_images.append(img_file)
        
        print(f"发现 {len(unlabeled_images)} 个无标签图像文件")
        return unlabeled_images
    
    @staticmethod
    def create_unlabeled_dataset_structure(unlabeled_images: List[str], 
                                         output_folder: str,
                                         dataset_name: str = "UnlabeledDataset"):
        """
        为无标签数据创建nnUNet格式的数据集结构
        
        Args:
            unlabeled_images: 无标签图像文件路径列表
            output_folder: 输出文件夹路径
            dataset_name: 数据集名称
        """
        import shutil
        from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
        
        # 创建输出文件夹
        images_folder = join(output_folder, 'imagesTr')
        maybe_mkdir_p(images_folder)
        
        # 复制无标签图像
        for i, img_path in enumerate(unlabeled_images):
            case_id = f"unlabeled_{i:03d}_0000.nii.gz"
            dst_path = join(images_folder, case_id)
            shutil.copy2(img_path, dst_path)
            print(f"复制: {img_path} -> {dst_path}")
        
        # 创建dataset.json文件
        dataset_json = {
            "name": dataset_name,
            "description": "Unlabeled dataset for semi-supervised learning",
            "tensorImageSize": "4D",
            "reference": "Semi-supervised learning",
            "licence": "Unknown",
            "release": "1.0",
            "modality": {
                "0": "CT"
            },
            "labels": {
                "background": 0
            },
            "numTraining": len(unlabeled_images),
            "numTest": 0,
            "training": [
                {
                    "image": f"./imagesTr/unlabeled_{i:03d}_0000.nii.gz",
                    "label": None  # 无标签
                }
                for i in range(len(unlabeled_images))
            ],
            "test": []
        }
        
        # 保存dataset.json
        import json
        with open(join(output_folder, 'dataset.json'), 'w') as f:
            json.dump(dataset_json, f, indent=2)
        
        print(f"无标签数据集结构已创建: {output_folder}")
        print(f"包含 {len(unlabeled_images)} 个无标签图像")


def create_semi_supervised_dataloader(labeled_dataloader: nnUNetDataLoader,
                                     unlabeled_data_path: str,
                                     unlabeled_batch_size: int = 2) -> SemiSupervisedDataLoader:
    """
    创建半监督数据加载器的便捷函数
    
    Args:
        labeled_dataloader: 有标签数据加载器
        unlabeled_data_path: 无标签数据路径
        unlabeled_batch_size: 无标签数据批次大小
        
    Returns:
        SemiSupervisedDataLoader: 半监督数据加载器
    """
    return SemiSupervisedDataLoader(
        labeled_dataloader=labeled_dataloader,
        unlabeled_folder=unlabeled_data_path,
        unlabeled_batch_size=unlabeled_batch_size,
        patch_size=labeled_dataloader.patch_size,
        label_manager=labeled_dataloader.label_manager,
        transforms=labeled_dataloader.transforms
    )