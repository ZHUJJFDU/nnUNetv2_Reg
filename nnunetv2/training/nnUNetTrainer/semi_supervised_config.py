#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
半监督训练配置文件
定义半监督学习的默认参数和配置选项
"""

import os
from typing import Dict, Any, Optional


class SemiSupervisedConfig:
    """
    半监督训练配置类
    
    包含所有半监督学习相关的参数配置
    """
    
    def __init__(self):
        # 基础训练参数
        self.num_epochs = 1000
        self.batch_size = 2
        self.learning_rate = 3e-4
        self.weight_decay = 3e-5
        
        # 半监督参数
        self.unlabeled_batch_size = 2
        self.consistency_weight = 1.0
        self.consistency_ramp_up_epochs = 100
        self.consistency_ramp_up_type = 'linear'  # 'linear', 'exp', 'cosine'
        
        # EMA参数
        self.ema_decay = 0.99
        self.ema_warmup_steps = 0
        
        # 一致性损失参数
        self.consistency_loss_type = 'mse'  # 'mse', 'kl', 'ce'
        self.consistency_temperature = 1.0
        self.use_confidence_mask = True
        self.confidence_threshold = 0.95
        
        # 伪标签参数
        self.use_pseudo_labels = True
        self.pseudo_label_threshold = 0.95
        self.use_entropy_filter = True
        self.entropy_threshold = 0.5
        
        # 数据增强参数
        self.strong_augmentation = True
        self.weak_augmentation = True
        self.augmentation_strength = 1.0
        
        # 训练策略参数
        self.save_interval = 50
        self.validation_interval = 25
        self.log_interval = 10
        
        # 设备和性能参数
        self.device = 'cuda'
        self.num_workers = 8
        self.pin_memory = True
        self.mixed_precision = True
        
        # 路径参数
        self.output_folder = None
        self.unlabeled_data_path = None
        self.pretrained_weights_path = None
        
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """
        从字典更新配置
        
        Args:
            config_dict: 配置字典
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"警告: 未知配置参数 {key}")
                
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            配置字典
        """
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_')}
                
    def save_to_file(self, file_path: str):
        """
        保存配置到文件
        
        Args:
            file_path: 文件路径
        """
        import json
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            
    @classmethod
    def load_from_file(cls, file_path: str) -> 'SemiSupervisedConfig':
        """
        从文件加载配置
        
        Args:
            file_path: 文件路径
            
        Returns:
            配置实例
        """
        import json
        config = cls()
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
                config.update_from_dict(config_dict)
        return config
        
    def validate(self) -> bool:
        """
        验证配置参数的有效性
        
        Returns:
            配置是否有效
        """
        errors = []
        
        # 检查基础参数
        if self.num_epochs <= 0:
            errors.append("num_epochs必须大于0")
            
        if self.batch_size <= 0:
            errors.append("batch_size必须大于0")
            
        if self.unlabeled_batch_size <= 0:
            errors.append("unlabeled_batch_size必须大于0")
            
        if self.learning_rate <= 0:
            errors.append("learning_rate必须大于0")
            
        # 检查半监督参数
        if not 0 <= self.ema_decay <= 1:
            errors.append("ema_decay必须在[0, 1]范围内")
            
        if not 0 <= self.confidence_threshold <= 1:
            errors.append("confidence_threshold必须在[0, 1]范围内")
            
        if self.consistency_loss_type not in ['mse', 'kl', 'ce']:
            errors.append("consistency_loss_type必须是'mse', 'kl'或'ce'之一")
            
        if self.consistency_ramp_up_type not in ['linear', 'exp', 'cosine']:
            errors.append("consistency_ramp_up_type必须是'linear', 'exp'或'cosine'之一")
            
        # 检查路径参数
        if self.unlabeled_data_path and not os.path.exists(self.unlabeled_data_path):
            errors.append(f"无标签数据路径不存在: {self.unlabeled_data_path}")
            
        if self.pretrained_weights_path and not os.path.exists(self.pretrained_weights_path):
            errors.append(f"预训练权重路径不存在: {self.pretrained_weights_path}")
            
        if errors:
            print("配置验证失败:")
            for error in errors:
                print(f"  - {error}")
            return False
            
        return True
        
    def print_config(self):
        """
        打印配置信息
        """
        print("半监督训练配置:")
        print("=" * 50)
        
        print("基础训练参数:")
        print(f"  训练轮数: {self.num_epochs}")
        print(f"  批次大小: {self.batch_size}")
        print(f"  学习率: {self.learning_rate}")
        print(f"  权重衰减: {self.weight_decay}")
        print()
        
        print("半监督参数:")
        print(f"  无标签批次大小: {self.unlabeled_batch_size}")
        print(f"  一致性损失权重: {self.consistency_weight}")
        print(f"  一致性权重上升轮数: {self.consistency_ramp_up_epochs}")
        print(f"  一致性权重上升类型: {self.consistency_ramp_up_type}")
        print()
        
        print("EMA参数:")
        print(f"  EMA衰减率: {self.ema_decay}")
        print(f"  EMA预热步数: {self.ema_warmup_steps}")
        print()
        
        print("一致性损失参数:")
        print(f"  损失类型: {self.consistency_loss_type}")
        print(f"  温度参数: {self.consistency_temperature}")
        print(f"  使用置信度掩码: {self.use_confidence_mask}")
        print(f"  置信度阈值: {self.confidence_threshold}")
        print()
        
        print("伪标签参数:")
        print(f"  使用伪标签: {self.use_pseudo_labels}")
        print(f"  伪标签阈值: {self.pseudo_label_threshold}")
        print(f"  使用熵过滤: {self.use_entropy_filter}")
        print(f"  熵阈值: {self.entropy_threshold}")
        print()
        
        print("数据增强参数:")
        print(f"  强增强: {self.strong_augmentation}")
        print(f"  弱增强: {self.weak_augmentation}")
        print(f"  增强强度: {self.augmentation_strength}")
        print()
        
        print("训练策略参数:")
        print(f"  保存间隔: {self.save_interval}")
        print(f"  验证间隔: {self.validation_interval}")
        print(f"  日志间隔: {self.log_interval}")
        print()
        
        print("设备和性能参数:")
        print(f"  设备: {self.device}")
        print(f"  工作进程数: {self.num_workers}")
        print(f"  固定内存: {self.pin_memory}")
        print(f"  混合精度: {self.mixed_precision}")
        print()
        
        print("路径参数:")
        print(f"  输出文件夹: {self.output_folder}")
        print(f"  无标签数据路径: {self.unlabeled_data_path}")
        print(f"  预训练权重路径: {self.pretrained_weights_path}")
        print("=" * 50)


# 预定义配置
class PresetConfigs:
    """
    预定义的配置模板
    """
    
    @staticmethod
    def get_default_config() -> SemiSupervisedConfig:
        """
        获取默认配置
        
        Returns:
            默认配置实例
        """
        return SemiSupervisedConfig()
        
    @staticmethod
    def get_lung_bullae_config() -> SemiSupervisedConfig:
        """
        获取肺大泡分割专用配置
        
        Returns:
            肺大泡分割配置实例
        """
        config = SemiSupervisedConfig()
        
        # 针对肺大泡分割的优化参数
        config.num_epochs = 500
        config.consistency_weight = 0.5
        config.consistency_ramp_up_epochs = 50
        config.ema_decay = 0.999
        config.confidence_threshold = 0.9
        config.pseudo_label_threshold = 0.9
        config.consistency_loss_type = 'mse'
        config.use_entropy_filter = True
        config.entropy_threshold = 0.4
        
        return config
        
    @staticmethod
    def get_fast_training_config() -> SemiSupervisedConfig:
        """
        获取快速训练配置（用于测试）
        
        Returns:
            快速训练配置实例
        """
        config = SemiSupervisedConfig()
        
        # 快速训练参数
        config.num_epochs = 100
        config.consistency_ramp_up_epochs = 20
        config.save_interval = 10
        config.validation_interval = 5
        config.log_interval = 1
        
        return config
        
    @staticmethod
    def get_high_quality_config() -> SemiSupervisedConfig:
        """
        获取高质量训练配置
        
        Returns:
            高质量训练配置实例
        """
        config = SemiSupervisedConfig()
        
        # 高质量训练参数
        config.num_epochs = 2000
        config.consistency_weight = 1.5
        config.consistency_ramp_up_epochs = 200
        config.ema_decay = 0.9999
        config.confidence_threshold = 0.98
        config.pseudo_label_threshold = 0.98
        config.use_entropy_filter = True
        config.entropy_threshold = 0.3
        
        return config


def create_config_from_args(args) -> SemiSupervisedConfig:
    """
    从命令行参数创建配置
    
    Args:
        args: 命令行参数
        
    Returns:
        配置实例
    """
    config = SemiSupervisedConfig()
    
    # 更新配置
    if hasattr(args, 'num_epochs'):
        config.num_epochs = args.num_epochs
    if hasattr(args, 'unlabeled_batch_size'):
        config.unlabeled_batch_size = args.unlabeled_batch_size
    if hasattr(args, 'consistency_weight'):
        config.consistency_weight = args.consistency_weight
    if hasattr(args, 'ema_decay'):
        config.ema_decay = args.ema_decay
    if hasattr(args, 'consistency_ramp_up_epochs'):
        config.consistency_ramp_up_epochs = args.consistency_ramp_up_epochs
    if hasattr(args, 'device'):
        config.device = args.device
    if hasattr(args, 'unlabeled_data_path'):
        config.unlabeled_data_path = args.unlabeled_data_path
    if hasattr(args, 'pretrained_weights'):
        config.pretrained_weights_path = args.pretrained_weights
        
    return config


if __name__ == '__main__':
    # 测试配置
    print("测试默认配置:")
    config = PresetConfigs.get_default_config()
    config.print_config()
    
    print("\n测试肺大泡配置:")
    lung_config = PresetConfigs.get_lung_bullae_config()
    lung_config.print_config()
    
    # 验证配置
    print("\n配置验证结果:", config.validate())