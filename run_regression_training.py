#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
import argparse
from pathlib import Path

nnunet_root = Path(__file__).parent
sys.path.insert(0, str(nnunet_root))

from batchgenerators.utilities.file_and_folder_operations import join, isdir, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.nnUNetTrainer.RegnnUNetTrainer import RegnnUNetTrainer
import torch


def get_reg_trainer(dataset_name_or_id: str, configuration: str, fold: int, device: str = 'cuda'):
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    preprocessed_dataset_folder = join(nnunet_root, 'DATASET', 'nnUNet_preprocessed', dataset_name)
    if not isdir(preprocessed_dataset_folder):
        raise RuntimeError(f"预处理数据集文件夹不存在: {preprocessed_dataset_folder}")
    plans_file = join(preprocessed_dataset_folder, 'nnUNetPlans.json')
    plans_manager = PlansManager(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder, 'dataset.json'))
    trainer = RegnnUNetTrainer(
        plans=plans_manager.plans,
        configuration=configuration,
        fold=fold,
        dataset_json=dataset_json,
        device=torch.device(device)
    )
    return trainer


def main():
    parser = argparse.ArgumentParser(description='回归微调训练启动脚本')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='数据集名称或ID (例如: Dataset102_quan 或 102)')
    parser.add_argument('-c', '--configuration', type=str, required=True, help='配置名称 (例如: 3d_fullres)')
    parser.add_argument('-f', '--fold', type=int, required=True, help='折数 (例如: 0, 1, 2, 3, 4 或 all)')
    parser.add_argument('--device', type=str, default='cuda', help='设备类型 (默认: cuda)')
    parser.add_argument('--pretrained_weights', type=str, default=None, help='半监督阶段的检查点路径')
    parser.add_argument('--reg_weight', type=float, default=1.0, help='回归损失权重')
    parser.add_argument('--reg_key', type=str, default='bulla_thickness', help='回归值键名')
    args = parser.parse_args()

    trainer = get_reg_trainer(args.dataset, args.configuration, args.fold, args.device)

    trainer.initialize()
    trainer.set_regression_parameters(reg_weight=args.reg_weight, reg_key=args.reg_key, debug=True)

    if args.pretrained_weights:
        trainer.load_checkpoint(args.pretrained_weights)
        print(f"已加载预训练权重: {args.pretrained_weights}")

    trainer.run_training()
    print('回归微调训练完成！')


if __name__ == '__main__':
    main()

