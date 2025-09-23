#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
半监督训练启动脚本

使用示例:
1. 基本用法:
   python run_semi_supervised_training.py
   
2. 自定义参数:
   python run_semi_supervised_training.py --config custom_config.json
   
3. 命令行覆盖参数:
   python run_semi_supervised_training.py --ema_decay 0.995 --consistency_weight 2.0
"""

import os
import sys
import json
import argparse
from pathlib import Path

# 添加nnUNet路径到系统路径
nnunet_root = Path(__file__).parent
sys.path.insert(0, str(nnunet_root))

from nnunetv2.training.nnUNetTrainer.train_semi_supervised import run_training


def load_config(config_path: str) -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"已加载配置文件: {config_path}")
    return config


def merge_config_with_args(config: dict, args: argparse.Namespace) -> dict:
    """
    将命令行参数与配置文件合并
    
    Args:
        config: 配置字典
        args: 命令行参数
        
    Returns:
        dict: 合并后的配置
    """
    # 命令行参数优先级更高
    if args.dataset is not None:
        config['dataset']['name_or_id'] = args.dataset
    if args.configuration is not None:
        config['dataset']['configuration'] = args.configuration
    if args.fold is not None:
        config['dataset']['fold'] = args.fold
    if args.teacher_checkpoint is not None:
        config['semi_supervised']['teacher_checkpoint_path'] = args.teacher_checkpoint
    if args.unlabeled_data is not None:
        config['semi_supervised']['unlabeled_data_path'] = args.unlabeled_data
    if args.ema_decay is not None:
        config['semi_supervised']['ema_decay'] = args.ema_decay
    if args.consistency_weight is not None:
        config['semi_supervised']['consistency_weight'] = args.consistency_weight
    if args.consistency_ramp_up is not None:
        config['semi_supervised']['consistency_ramp_up_epochs'] = args.consistency_ramp_up
    if args.epochs is not None:
        config['trainer']['num_epochs'] = args.epochs
    if args.device is not None:
        config['trainer']['device'] = args.device
    if args.output_folder is not None:
        config['output']['base_folder'] = args.output_folder
    
    # 布尔参数
    if args.continue_training:
        config['trainer']['continue_training'] = True
    if args.validation_only:
        config['trainer']['validation_only'] = True
    if args.disable_checkpointing:
        config['trainer']['disable_checkpointing'] = True
    
    return config


def print_config_summary(config: dict):
    """
    打印配置摘要
    
    Args:
        config: 配置字典
    """
    print("\n" + "=" * 60)
    print("半监督训练配置摘要")
    print("=" * 60)
    
    print(f"数据集: {config['dataset']['name_or_id']}")
    print(f"配置: {config['dataset']['configuration']}")
    print(f"折数: {config['dataset']['fold']}")
    print(f"训练器: {config['trainer']['class_name']}")
    print(f"设备: {config['trainer']['device']}")
    print(f"训练轮数: {config['trainer']['num_epochs']}")
    
    print("\n半监督参数:")
    print(f"  教师模型检查点: {config['semi_supervised']['teacher_checkpoint_path']}")
    print(f"  无标签数据路径: {config['semi_supervised']['unlabeled_data_path']}")
    print(f"  EMA衰减率: {config['semi_supervised']['ema_decay']}")
    print(f"  一致性损失权重: {config['semi_supervised']['consistency_weight']}")
    print(f"  一致性权重上升周期: {config['semi_supervised']['consistency_ramp_up_epochs']}")
    
    print("\n伪标签参数:")
    print(f"  置信度阈值: {config['pseudo_labeling']['confidence_threshold']}")
    print(f"  使用熵过滤: {config['pseudo_labeling']['use_entropy_filtering']}")
    print(f"  熵阈值: {config['pseudo_labeling']['entropy_threshold']}")
    
    print("=" * 60)


def validate_config(config: dict) -> bool:
    """
    验证配置的有效性
    
    Args:
        config: 配置字典
        
    Returns:
        bool: 配置是否有效
    """
    errors = []
    
    # 检查必需的配置项
    required_keys = [
        ('dataset', 'name_or_id'),
        ('dataset', 'configuration'),
        ('dataset', 'fold'),
        ('trainer', 'class_name'),
        ('semi_supervised', 'teacher_checkpoint_path')
    ]
    
    for section, key in required_keys:
        if section not in config or key not in config[section]:
            errors.append(f"缺少必需配置: {section}.{key}")
        elif config[section][key] is None:
            errors.append(f"配置项不能为空: {section}.{key}")
    
    # 检查教师模型检查点是否存在
    teacher_checkpoint = config.get('semi_supervised', {}).get('teacher_checkpoint_path')
    if teacher_checkpoint and not os.path.exists(teacher_checkpoint):
        errors.append(f"教师模型检查点不存在: {teacher_checkpoint}")
    
    # 检查无标签数据路径（如果指定）
    unlabeled_path = config.get('semi_supervised', {}).get('unlabeled_data_path')
    if unlabeled_path and not os.path.exists(unlabeled_path):
        errors.append(f"无标签数据路径不存在: {unlabeled_path}")
    
    # 检查数值范围
    ema_decay = config.get('semi_supervised', {}).get('ema_decay', 0.99)
    if not 0 < ema_decay < 1:
        errors.append(f"EMA衰减率必须在(0,1)范围内: {ema_decay}")
    
    consistency_weight = config.get('semi_supervised', {}).get('consistency_weight', 1.0)
    if consistency_weight < 0:
        errors.append(f"一致性损失权重不能为负数: {consistency_weight}")
    
    if errors:
        print("\n配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("配置验证通过")
    return True


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(
        description="nnUNet半监督训练启动脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 配置文件
    parser.add_argument(
        "--config", 
        type=str, 
        default="nnunetv2/training/nnUNetTrainer/semi_supervised_config.json",
        help="配置文件路径"
    )
    
    # 基本参数（可覆盖配置文件）
    parser.add_argument("-d", "--dataset", type=str, help="数据集名称或ID")
    parser.add_argument("-c", "--configuration", type=str, help="配置名称")
    parser.add_argument("-f", "--fold", type=int, help="交叉验证折数")
    
    # 半监督参数
    parser.add_argument("-t", "--teacher_checkpoint", type=str, help="教师模型检查点路径")
    parser.add_argument("-u", "--unlabeled_data", type=str, help="无标签数据路径")
    parser.add_argument("--ema_decay", type=float, help="EMA衰减率")
    parser.add_argument("--consistency_weight", type=float, help="一致性损失权重")
    parser.add_argument("--consistency_ramp_up", type=int, help="一致性权重上升周期")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--device", type=str, help="训练设备")
    parser.add_argument("-o", "--output_folder", type=str, help="输出文件夹")
    
    # 训练选项
    parser.add_argument("--continue_training", action="store_true", help="继续训练")
    parser.add_argument("--validation_only", action="store_true", help="仅进行验证")
    parser.add_argument("--disable_checkpointing", action="store_true", help="禁用检查点保存")
    
    # 其他选项
    parser.add_argument("--dry_run", action="store_true", help="仅验证配置，不开始训练")
    parser.add_argument("--save_config", type=str, help="保存合并后的配置到指定文件")
    
    args = parser.parse_args()
    
    try:
        # 加载配置文件
        config = load_config(args.config)
        
        # 合并命令行参数
        config = merge_config_with_args(config, args)
        
        # 验证配置
        if not validate_config(config):
            sys.exit(1)
        
        # 打印配置摘要
        print_config_summary(config)
        
        # 保存合并后的配置（如果指定）
        if args.save_config:
            with open(args.save_config, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"\n配置已保存到: {args.save_config}")
        
        # 如果是dry run，则退出
        if args.dry_run:
            print("\n配置验证完成（dry run模式）")
            return
        
        # 开始训练
        print("\n开始半监督训练...")
        
        # 调用训练函数
        run_training(
            dataset_name_or_id=config['dataset']['name_or_id'],
            configuration=config['dataset']['configuration'],
            fold=config['dataset']['fold'],
            trainer_name=config['trainer']['class_name'],
            plans_identifier=config['dataset']['plans_identifier'],
            pretrained_weights=config['semi_supervised']['teacher_checkpoint_path'],
            num_epochs=config['trainer']['num_epochs'],
            device=config['trainer']['device'],
            unlabeled_data_path=config['semi_supervised']['unlabeled_data_path'],
            unlabeled_batch_size=config['semi_supervised']['unlabeled_batch_size'],
            consistency_weight=config['semi_supervised']['consistency_weight'],
            ema_decay=config['semi_supervised']['ema_decay'],
            consistency_ramp_up_epochs=config['semi_supervised']['consistency_ramp_up_epochs'],
            save_interval=config['trainer'].get('save_interval', 1),
            continue_training=config['trainer']['continue_training'],
            only_run_validation=config['trainer']['validation_only'],
            disable_checkpointing=config['trainer']['disable_checkpointing']
        )
        
        print("\n训练完成！")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()