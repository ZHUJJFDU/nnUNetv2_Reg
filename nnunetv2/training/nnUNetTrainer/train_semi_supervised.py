#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
半监督训练脚本
基于nnUNet v2框架实现教师-学生半监督学习

使用方法:
python train_semi_supervised.py -d 102 -c 3d_fullres -f 0 --unlabeled_data_path /path/to/unlabeled/data
"""

import argparse
import os
import sys
import torch
from typing import Union, List

# 添加nnUNet路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from nnunetv2.training.nnUNetTrainer.SemiSupervisedTrainer import SemiSupervisedTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p, load_json


def get_trainer_from_args(dataset_name_or_id: Union[int, str],
                         configuration: str,
                         fold: Union[int, str],
                         trainer_name: str = 'SemiSupervisedTrainer',
                         plans_identifier: str = 'nnUNetPlans',
                         use_compressed: bool = False,
                         device: str = 'cuda'):
    """
    根据参数创建训练器实例
    
    Args:
        dataset_name_or_id: 数据集名称或ID
        configuration: 配置名称 (如 '3d_fullres')
        fold: 折数
        trainer_name: 训练器名称
        plans_identifier: 计划标识符
        use_compressed: 是否使用压缩数据
        device: 设备类型
        
    Returns:
        SemiSupervisedTrainer: 训练器实例
    """
    # 转换数据集名称
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    
    # 构建路径
    preprocessed_dataset_folder = join(nnUNet_preprocessed, dataset_name)
    
    if not isdir(preprocessed_dataset_folder):
        raise RuntimeError(f"预处理数据集文件夹不存在: {preprocessed_dataset_folder}")
    
    # 加载计划
    plans_manager = PlansManager(join(preprocessed_dataset_folder, plans_identifier + '.json'))
    
    # 加载dataset.json
    dataset_json = load_json(join(preprocessed_dataset_folder, 'dataset.json'))
    
    # 获取配置
    configuration_manager = plans_manager.get_configuration(configuration)
    
    # 创建训练器
    trainer = SemiSupervisedTrainer(
        plans=plans_manager.plans,
        configuration=configuration,
        fold=fold,
        dataset_json=dataset_json,
        device=torch.device(device)
    )
    
    return trainer


def run_training(dataset_name_or_id: Union[int, str],
                configuration: str,
                fold: Union[int, str],
                trainer_name: str = 'SemiSupervisedTrainer',
                plans_identifier: str = 'nnUNetPlans',
                pretrained_weights: str = None,
                num_epochs: int = 1000,
                use_compressed: bool = False,
                device: str = 'cuda',
                unlabeled_data_path: str = None,
                unlabeled_batch_size: int = 2,
                consistency_weight: float = 1.0,
                ema_decay: float = 0.99,
                consistency_ramp_up_epochs: int = 100,
                save_interval: int = 1,
                continue_training: bool = False,
                only_run_validation: bool = False,
                disable_checkpointing: bool = False,
                c: str = None,
                val_disable_overwrite: bool = False):
    """
    运行半监督训练
    
    Args:
        dataset_name_or_id: 数据集名称或ID
        configuration: 配置名称
        fold: 折数
        trainer_name: 训练器名称
        plans_identifier: 计划标识符
        pretrained_weights: 预训练权重路径
        num_epochs: 训练轮数
        use_compressed: 是否使用压缩数据
        device: 设备类型
        unlabeled_data_path: 无标签数据路径
        unlabeled_batch_size: 无标签数据批次大小
        consistency_weight: 一致性损失权重
        ema_decay: EMA衰减率
        consistency_ramp_up_epochs: 一致性损失权重上升轮数
        continue_training: 是否继续训练
        only_run_validation: 是否只运行验证
        disable_checkpointing: 是否禁用检查点
        c: 继续训练的检查点
        val_disable_overwrite: 验证时是否禁用覆写
    """
    # 创建训练器
    trainer = get_trainer_from_args(
        dataset_name_or_id=dataset_name_or_id,
        configuration=configuration,
        fold=fold,
        trainer_name=trainer_name,
        plans_identifier=plans_identifier,
        use_compressed=use_compressed,
        device=device
    )
    
    # 设置半监督参数
    if unlabeled_data_path:
        trainer.unlabeled_data_path = unlabeled_data_path
    trainer.unlabeled_batch_size = unlabeled_batch_size
    trainer.consistency_weight = consistency_weight
    trainer.ema_decay = ema_decay
    trainer.consistency_ramp_up_epochs = consistency_ramp_up_epochs
    trainer.save_every = save_interval  # 设置保存间隔
    
    # 初始化训练器
    trainer.initialize()
    
    # 加载预训练权重
    if pretrained_weights is not None:
        trainer.load_checkpoint(pretrained_weights)
        print(f"已加载预训练权重: {pretrained_weights}")
    
    # 继续训练
    if continue_training:
        if c is not None:
            trainer.load_checkpoint(c)
        else:
            trainer.load_checkpoint(join(trainer.output_folder, 'checkpoint_latest.pth'))
    
    # 只运行验证
    if only_run_validation:
        trainer.perform_actual_validation(trainer.val_gen)
        return
    
    # 运行训练
    trainer.run_training()
    
    print("半监督训练完成！")


def main():
    """
    主函数，解析命令行参数并运行训练
    """
    parser = argparse.ArgumentParser(description='nnUNet半监督训练脚本')
    
    # 必需参数
    parser.add_argument('-d', '--dataset', type=str, required=True,
                       help='数据集名称或ID (例如: Dataset102_quan 或 102)')
    parser.add_argument('-c', '--configuration', type=str, required=True,
                       help='配置名称 (例如: 3d_fullres, 3d_lowres, 2d)')
    parser.add_argument('-f', '--fold', type=str, required=True,
                       help='折数 (例如: 0, 1, 2, 3, 4 或 all)')
    
    # 半监督相关参数
    parser.add_argument('--unlabeled_data_path', type=str, default=None,
                       help='无标签数据路径')
    parser.add_argument('--unlabeled_batch_size', type=int, default=2,
                       help='无标签数据批次大小 (默认: 2)')
    parser.add_argument('--consistency_weight', type=float, default=1.0,
                       help='一致性损失权重 (默认: 1.0)')
    parser.add_argument('--ema_decay', type=float, default=0.99,
                       help='EMA衰减率 (默认: 0.99)')
    parser.add_argument('--consistency_ramp_up_epochs', type=int, default=100,
                       help='一致性损失权重上升轮数 (默认: 100)')
    
    # 训练参数
    parser.add_argument('-tr', '--trainer', type=str, default='SemiSupervisedTrainer',
                       help='训练器名称 (默认: SemiSupervisedTrainer)')
    parser.add_argument('-p', '--plans', type=str, default='nnUNetPlans',
                       help='计划标识符 (默认: nnUNetPlans)')
    parser.add_argument('--pretrained_weights', type=str, default=None,
                       help='预训练权重路径')
    parser.add_argument('--num_epochs', type=int, default=1000,
                       help='训练轮数 (默认: 1000)')
    
    # 设备和数据参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备类型 (默认: cuda)')
    parser.add_argument('--use_compressed', action='store_true',
                       help='使用压缩数据')
    
    # 训练控制参数
    parser.add_argument('--continue_training', action='store_true',
                       help='继续训练')
    parser.add_argument('--only_run_validation', action='store_true',
                       help='只运行验证')
    parser.add_argument('--disable_checkpointing', action='store_true',
                       help='禁用检查点保存')
    parser.add_argument('--val_disable_overwrite', action='store_true',
                       help='验证时禁用覆写')
    parser.add_argument('--c', type=str, default=None,
                       help='继续训练的检查点路径')
    
    args = parser.parse_args()
    
    # 打印参数
    print("半监督训练参数:")
    print(f"  数据集: {args.dataset}")
    print(f"  配置: {args.configuration}")
    print(f"  折数: {args.fold}")
    print(f"  训练器: {args.trainer}")
    print(f"  无标签数据路径: {args.unlabeled_data_path}")
    print(f"  无标签批次大小: {args.unlabeled_batch_size}")
    print(f"  一致性损失权重: {args.consistency_weight}")
    print(f"  EMA衰减率: {args.ema_decay}")
    print(f"  一致性权重上升轮数: {args.consistency_ramp_up_epochs}")
    print(f"  预训练权重: {args.pretrained_weights}")
    print(f"  训练轮数: {args.num_epochs}")
    print(f"  设备: {args.device}")
    print()
    
    # 运行训练
    try:
        run_training(
            dataset_name_or_id=args.dataset,
            configuration=args.configuration,
            fold=args.fold,
            trainer_name=args.trainer,
            plans_identifier=args.plans,
            pretrained_weights=args.pretrained_weights,
            num_epochs=args.num_epochs,
            use_compressed=args.use_compressed,
            device=args.device,
            unlabeled_data_path=args.unlabeled_data_path,
            unlabeled_batch_size=args.unlabeled_batch_size,
            consistency_weight=args.consistency_weight,
            ema_decay=args.ema_decay,
            consistency_ramp_up_epochs=args.consistency_ramp_up_epochs,
            continue_training=args.continue_training,
            only_run_validation=args.only_run_validation,
            disable_checkpointing=args.disable_checkpointing,
            c=args.c,
            val_disable_overwrite=args.val_disable_overwrite
        )
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    # Windows下multiprocessing的保护
    import multiprocessing
    multiprocessing.freeze_support()
    main()