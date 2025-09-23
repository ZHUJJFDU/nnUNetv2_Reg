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
import numpy as np
from pathlib import Path
from typing import Union, List, Optional
import json

# 添加nnUNet路径
nnunet_root = Path(__file__).parent
sys.path.insert(0, str(nnunet_root))

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.nnUNetTrainer.SemiSupervisedTrainer import SemiSupervisedTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p, load_json


class SemiSupervisedPredictor:
    """
    半监督训练模型预测器
    
    支持加载半监督框架训练的模型权重进行推理
    """
    
    def __init__(self, 
                 model_folder: str,
                 use_folds: Union[List[int], int] = None,
                 checkpoint_name: str = 'checkpoint_best.pth',
                 use_teacher_model: bool = False,
                 device: str = 'cuda',
                 allow_tqdm: bool = True,
                 verbose: bool = True):
        """
        初始化半监督预测器
        
        Args:
            model_folder: 模型文件夹路径
            use_folds: 使用的折数，None表示使用所有可用的折
            checkpoint_name: 检查点文件名
            use_teacher_model: 是否使用教师模型进行推理
            device: 设备类型
            allow_tqdm: 是否显示进度条
            verbose: 是否显示详细信息
        """
        self.model_folder = model_folder
        self.use_folds = use_folds
        self.checkpoint_name = checkpoint_name
        self.use_teacher_model = use_teacher_model
        self.device = device
        self.allow_tqdm = allow_tqdm
        self.verbose = verbose
        
        # 初始化预测器
        self.predictor = None
        self._initialize_predictor()
    
    def _initialize_predictor(self):
        """初始化nnUNet预测器"""
        if self.verbose:
            print(f"初始化半监督模型预测器...")
            print(f"模型文件夹: {self.model_folder}")
            print(f"检查点文件: {self.checkpoint_name}")
            print(f"使用教师模型: {self.use_teacher_model}")
            print(f"设备: {self.device}")
        
        # 创建nnUNet预测器
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device(self.device),
            verbose=self.verbose,
            verbose_preprocessing=self.verbose,
            allow_tqdm=self.allow_tqdm
        )
        
        # 初始化预测器
        self.predictor.initialize_from_trained_model_folder(
            self.model_folder,
            use_folds=self.use_folds,
            checkpoint_name=self.checkpoint_name
        )
        
        # 如果使用教师模型，加载教师模型权重
        if self.use_teacher_model:
            self._load_teacher_weights()
    
    def _load_teacher_weights(self):
        """加载教师模型权重"""
        if self.verbose:
            print("加载教师模型权重...")
        
        # 查找教师模型检查点
        if self.use_folds is None:
            # 自动检测可用的折
            folds = []
            for fold in range(5):  # 通常最多5折
                fold_dir = join(self.model_folder, f'fold_{fold}')
                if isdir(fold_dir):
                    folds.append(fold)
        else:
            folds = [self.use_folds] if isinstance(self.use_folds, int) else self.use_folds
        
        # 为每个折加载教师模型权重
        for fold in folds:
            fold_dir = join(self.model_folder, f'fold_{fold}')
            teacher_checkpoint_path = join(fold_dir, self.checkpoint_name.replace('.pth', '_teacher.pth'))
            
            if os.path.exists(teacher_checkpoint_path):
                if self.verbose:
                    print(f"加载教师模型权重: {teacher_checkpoint_path}")
                
                # 加载教师模型权重
                teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location=self.device)
                
                # 获取对应的网络
                if hasattr(self.predictor, 'list_of_parameters'):
                    # 找到对应折的网络
                    fold_idx = folds.index(fold) if fold in folds else 0
                    if fold_idx < len(self.predictor.list_of_parameters):
                        network = self.predictor.list_of_parameters[fold_idx]
                        if 'teacher_model_state_dict' in teacher_checkpoint:
                            network.load_state_dict(teacher_checkpoint['teacher_model_state_dict'])
                            if self.verbose:
                                print(f"成功加载fold {fold}的教师模型权重")
            else:
                if self.verbose:
                    print(f"警告: 未找到教师模型权重文件: {teacher_checkpoint_path}")
                    print("将使用学生模型权重进行推理")
    
    def predict_single_nifti_image(self, 
                                   input_image: str, 
                                   output_filename: str,
                                   save_probabilities: bool = False,
                                   overwrite: bool = True):
        """
        对单个NIfTI图像进行预测
        
        Args:
            input_image: 输入图像路径
            output_filename: 输出文件路径
            save_probabilities: 是否保存概率图
            overwrite: 是否覆盖已存在的文件
        """
        if self.verbose:
            print(f"预测图像: {input_image}")
            print(f"输出路径: {output_filename}")
        
        self.predictor.predict_single_nifti_image(
            input_image=input_image,
            output_filename_truncated=output_filename,
            save_probabilities=save_probabilities,
            overwrite=overwrite
        )
        
        if self.verbose:
            print(f"预测完成: {output_filename}")
    
    def predict_from_files(self,
                          list_of_lists_or_source_folder: Union[str, List[List[str]]],
                          output_folder_or_list_of_truncated_output_files: Union[str, List[str]],
                          save_probabilities: bool = False,
                          overwrite: bool = True,
                          num_processes_preprocessing: int = 8,
                          num_processes_segmentation_export: int = 8,
                          folder_with_segs_from_prev_stage: str = None,
                          num_parts: int = 1,
                          part_id: int = 0):
        """
        批量预测文件
        
        Args:
            list_of_lists_or_source_folder: 输入文件列表或源文件夹
            output_folder_or_list_of_truncated_output_files: 输出文件夹或输出文件列表
            save_probabilities: 是否保存概率图
            overwrite: 是否覆盖已存在的文件
            num_processes_preprocessing: 预处理进程数
            num_processes_segmentation_export: 分割导出进程数
            folder_with_segs_from_prev_stage: 前一阶段分割结果文件夹
            num_parts: 总分片数
            part_id: 当前分片ID
        """
        if self.verbose:
            print(f"批量预测...")
            print(f"输入: {list_of_lists_or_source_folder}")
            print(f"输出: {output_folder_or_list_of_truncated_output_files}")
        
        self.predictor.predict_from_files(
            list_of_lists_or_source_folder=list_of_lists_or_source_folder,
            output_folder_or_list_of_truncated_output_files=output_folder_or_list_of_truncated_output_files,
            save_probabilities=save_probabilities,
            overwrite=overwrite,
            num_processes_preprocessing=num_processes_preprocessing,
            num_processes_segmentation_export=num_processes_segmentation_export,
            folder_with_segs_from_prev_stage=folder_with_segs_from_prev_stage,
            num_parts=num_parts,
            part_id=part_id
        )
        
        if self.verbose:
            print("批量预测完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='半监督训练模型推理脚本')
    
    # 必需参数
    parser.add_argument('-i', '--input', type=str, required=True,
                       help='输入图像路径或文件夹')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='输出路径或文件夹')
    parser.add_argument('-m', '--model', type=str, required=True,
                       help='模型文件夹路径')
    
    # 可选参数
    parser.add_argument('-f', '--folds', type=int, nargs='+', default=None,
                       help='使用的折数 (默认: 使用所有可用的折)')
    parser.add_argument('-chk', '--checkpoint', type=str, default='checkpoint_best.pth',
                       help='检查点文件名 (默认: checkpoint_best.pth)')
    parser.add_argument('--use_teacher', action='store_true',
                       help='使用教师模型进行推理')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备类型 (默认: cuda)')
    parser.add_argument('--save_probabilities', action='store_true',
                       help='保存概率图')
    parser.add_argument('--overwrite', action='store_true',
                       help='覆盖已存在的文件')
    parser.add_argument('--num_processes', type=int, default=8,
                       help='处理进程数 (默认: 8)')
    parser.add_argument('--quiet', action='store_true',
                       help='静默模式')
    
    args = parser.parse_args()
    
    # 打印参数
    if not args.quiet:
        print("半监督模型推理参数:")
        print(f"  输入: {args.input}")
        print(f"  输出: {args.output}")
        print(f"  模型: {args.model}")
        print(f"  折数: {args.folds}")
        print(f"  检查点: {args.checkpoint}")
        print(f"  使用教师模型: {args.use_teacher}")
        print(f"  设备: {args.device}")
        print(f"  保存概率图: {args.save_probabilities}")
        print(f"  覆盖文件: {args.overwrite}")
        print()
    
    try:
        # 创建预测器
        predictor = SemiSupervisedPredictor(
            model_folder=args.model,
            use_folds=args.folds,
            checkpoint_name=args.checkpoint,
            use_teacher_model=args.use_teacher,
            device=args.device,
            verbose=not args.quiet
        )
        
        # 检查输入是单个文件还是文件夹
        if os.path.isfile(args.input):
            # 单个文件预测
            predictor.predict_single_nifti_image(
                input_image=args.input,
                output_filename=args.output,
                save_probabilities=args.save_probabilities,
                overwrite=args.overwrite
            )
        else:
            # 批量预测
            predictor.predict_from_files(
                list_of_lists_or_source_folder=args.input,
                output_folder_or_list_of_truncated_output_files=args.output,
                save_probabilities=args.save_probabilities,
                overwrite=args.overwrite,
                num_processes_preprocessing=args.num_processes,
                num_processes_segmentation_export=args.num_processes
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