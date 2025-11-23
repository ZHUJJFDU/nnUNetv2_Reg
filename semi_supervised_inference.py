#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
半监督训练模型推理脚本
使用半监督框架训练好的权重进行医学图像分割推理
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from typing import Union, List, Optional

# 添加nnUNet路径
nnunet_root = Path(__file__).parent
if str(nnunet_root) not in sys.path:
    sys.path.insert(0, str(nnunet_root))

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


class SemiSupervisedPredictor(nnUNetPredictor):
    """
    半监督训练模型预测器
    
    继承自 nnUNetPredictor，并增加了加载教师模型的功能。
    """
    
    def __init__(self, 
                 use_teacher_model: bool = False,
                 **kwargs):
        """
        初始化半监督预测器
        
        Args:
            use_teacher_model: 是否使用教师模型进行推理
            **kwargs: 传递给父类 nnUNetPredictor 的参数
        """
        # 调用父类的构造函数
        super().__init__(**kwargs)
        self.use_teacher_model = use_teacher_model
        if self.verbose:
            print(f"半监督预测器初始化完成。使用教师模型: {self.use_teacher_model}")

    def initialize_from_trained_model_folder(self, 
                                             model_training_output_dir: str, 
                                             use_folds: Optional[Union[int, str, List[int], List[str]]], 
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        重写此方法以支持加载教师模型。
        首先加载标准学生模型，然后如果需要，用教师模型权重覆盖。
        """
        # 1. 调用父类方法，加载标准的学生模型权重到 self.list_of_parameters
        super().initialize_from_trained_model_folder(model_training_output_dir, use_folds, checkpoint_name)
        
        # 2. 如果指定使用教师模型，则加载并替换权重
        if self.use_teacher_model:
            if self.verbose:
                print("正在加载教师模型权重...")
            
            # self.list_of_parameters 是一个列表，每个元素是一个模型的 state_dict
            # self.used_folds 是一个列表，记录了实际使用的折数
            
            new_params = []
            for i, fold in enumerate(self.used_folds):
                fold_dir = join(model_training_output_dir, f'fold_{fold}')
                # 教师模型的检查点通常带有 _teacher 后缀
                teacher_checkpoint_path = join(fold_dir, checkpoint_name.replace('.pth', '_teacher.pth'))

                if os.path.exists(teacher_checkpoint_path):
                    if self.verbose:
                        print(f"为 fold {fold} 加载教师模型权重: {teacher_checkpoint_path}")
                    
                    # 加载教师模型权重
                    teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location=torch.device('cpu'))
                    
                    # 教师权重在 'teacher_model_state_dict' 键中
                    if 'teacher_model_state_dict' in teacher_checkpoint:
                        new_params.append(teacher_checkpoint['teacher_model_state_dict'])
                        if self.verbose:
                            print(f"成功为 fold {fold} 加载教师模型权重。")
                    else:
                        print(f"警告: 在 {teacher_checkpoint_path} 中未找到 'teacher_model_state_dict'。将使用学生模型。")
                        new_params.append(self.list_of_parameters[i]) # 回退到学生模型
                else:
                    print(f"警告: 未找到 fold {fold} 的教师模型权重文件: {teacher_checkpoint_path}。将使用学生模型。")
                    new_params.append(self.list_of_parameters[i]) # 回退到学生模型
            
            # 替换权重列表
            self.list_of_parameters = new_params


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='半监督训练模型推理脚本 (继承自 nnUNetPredictor)')
    
    # 必需参数
    parser.add_argument('-i', '--input', type=str, required=True,
                       help='输入图像路径或文件夹')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='输出路径或文件夹')
    parser.add_argument('-d', '--dataset_name_or_id', type=str, required=True,
                        help='需要使用的训练模型的 Dataset name or id。')
    parser.add_argument('-tr', '--trainer_name', type=str, required=False, default='nnUNetTrainer_500epochs',
                        help='训练器的名称 (例如: nnUNetTrainer)。默认: nnUNetTrainer_500epochs')
    parser.add_argument('-p', '--plans_identifier', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. 默认: nnUNetPlans')
    parser.add_argument('-c', '--configuration', type=str, required=False, default='3d_fullres',
                        help='配置名称 (例如: 3d_fullres)。默认: 3d_fullres')
    
    # 可选参数
    parser.add_argument('-f', '--folds', type=int, nargs='+', default=None,
                       help='使用的折数 (默认: 使用所有可用的折)')
    parser.add_argument('-chk', '--checkpoint', type=str, default='checkpoint_best.pth',
                       help='检查点文件名 (默认: checkpoint_best.pth)')
    parser.add_argument('--use_teacher', action='store_true',
                       help='使用教师模型进行推理')
    parser.add_argument('-step_size', type=float, default=0.5, help='滑窗步长 (0-1之间, 默认 0.5)')
    parser.add_argument('--disable_tta', action='store_true', help='禁用测试时增强 (镜像)')
    parser.add_argument('--device', type=str, default='cuda', help='设备类型 (cuda 或 cpu, 默认: cuda)')
    parser.add_argument('--save_probabilities', action='store_true', help='保存概率图')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的文件')
    parser.add_argument('--num_processes_preprocessing', type=int, default=8, help='预处理进程数 (默认: 8)')
    parser.add_argument('--num_processes_segmentation_export', type=int, default=8, help='分割导出进程数 (默认: 8)')
    parser.add_argument('--quiet', action='store_true', help='静默模式')
    
    args = parser.parse_args()
    
    # 打印参数
    if not args.quiet:
        print("半监督模型推理参数:")
        for arg, value in sorted(vars(args).items()):
            print(f"  {arg}: {value}")
        print()
    
    try:
        # 准备传递给预测器的参数
        predictor_kwargs = {
            'tile_step_size': args.step_size,
            'use_gaussian': True, # nnUNet 默认推荐
            'use_mirroring': not args.disable_tta,
            'perform_everything_on_device': True if args.device == 'cuda' else False,
            'device': torch.device(args.device),
            'verbose': not args.quiet,
            'verbose_preprocessing': False, # 通常不需要预处理的详细日志
            'allow_tqdm': not args.quiet,
            'use_teacher_model': args.use_teacher
        }

        # 自动构建模型路径
        dataset_name = maybe_convert_to_dataset_name(args.dataset_name_or_id)
        model_folder = join(nnunet_root, "DATASET", "nnUNet_trained_models", dataset_name, 
                            f"{args.trainer_name}__{args.plans_identifier}__{args.configuration}")
        if not isdir(model_folder):
            raise FileNotFoundError(f"模型文件夹未找到: {model_folder}")

        # 创建预测器实例
        predictor = SemiSupervisedPredictor(**predictor_kwargs)
        
        # 初始化模型（这将调用我们重写的加载教师模型的方法）
        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=args.folds,
            checkpoint_name=args.checkpoint
        )
        
        # 检查输入是单个文件还是文件夹
        if os.path.isfile(args.input):
            # 单个文件预测
            # 注意：predict_single_nifti_image 是从父类继承的
            output_dir = os.path.dirname(args.output)
            if output_dir:
                maybe_mkdir_p(output_dir)
            
            predictor.predict_single_nifti_image(
                input_image=args.input,
                output_filename_truncated=args.output,
                save_probabilities=args.save_probabilities,
                overwrite=args.overwrite
            )
        else:
            # 批量预测
            # 注意：predict_from_files 是从父类继承的
            maybe_mkdir_p(args.output)
            predictor.predict_from_files(
                list_of_lists_or_source_folder=args.input,
                output_folder_or_list_of_truncated_output_files=args.output,
                save_probabilities=args.save_probabilities,
                overwrite=args.overwrite,
                num_processes_preprocessing=args.num_processes_preprocessing,
                num_processes_segmentation_export=args.num_processes_segmentation_export
            )
        
        if not args.quiet:
            print("\n推理完成！")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
