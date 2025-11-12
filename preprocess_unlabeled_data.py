#!/usr/bin/env python3
"""
使用nnUNet官方预处理流程处理无标签数据
确保与nnUNetv2_plan_and_preprocess命令执行结果完全一致
"""

import os
import numpy as np
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name

def preprocess_unlabeled_data_official(dataset_id: int = 102, configuration: str = '3d_fullres'):
    """
    使用nnUNet官方预处理流程处理无标签数据
    
    Args:
        dataset_id: 数据集ID，默认为102
        configuration: 配置名称，默认为'3d_fullres'
    """
    # 转换数据集名称
    dataset_name = convert_id_to_dataset_name(dataset_id)
    
    # 路径设置
    raw_dataset_folder = join(nnUNet_raw, dataset_name)
    preprocessed_dataset_folder = join(nnUNet_preprocessed, dataset_name)
    unlabeled_images_folder = r"c:\Users\960\Desktop\nnUNet_Reg\DATASET\nnUNet_raw\Dataset102_quan\image"
    
    print(f"处理数据集: {dataset_name}")
    print(f"原始图像路径: {unlabeled_images_folder}")
    print(f"预处理输出路径: {preprocessed_dataset_folder}")
    
    # 检查路径是否存在
    if not os.path.exists(unlabeled_images_folder):
        raise ValueError(f"无标签图像文件夹不存在: {unlabeled_images_folder}")
    
    if not os.path.exists(preprocessed_dataset_folder):
        raise ValueError(f"预处理文件夹不存在: {preprocessed_dataset_folder}")
    
    # 加载数据集配置
    dataset_json_file = join(preprocessed_dataset_folder, 'dataset.json')
    if not os.path.exists(dataset_json_file):
        raise ValueError(f"数据集配置文件不存在: {dataset_json_file}")
    
    dataset_json = load_json(dataset_json_file)
    
    # 加载plans文件
    plans_file = join(preprocessed_dataset_folder, 'nnUNetPlans.json')
    if not os.path.exists(plans_file):
        raise ValueError(f"Plans文件不存在: {plans_file}")
    
    plans = load_json(plans_file)
    plans_manager = PlansManager(plans)
    
    # 获取配置
    if configuration not in plans_manager.available_configurations:
        raise ValueError(f"配置 {configuration} 不在可用配置中: {plans_manager.available_configurations}")
    
    configuration_manager = plans_manager.get_configuration(configuration)
    
    # 创建预处理器
    preprocessor = configuration_manager.preprocessor_class(verbose=True)
    
    # 获取无标签图像文件列表
    image_files = []
    for ext in ['*.nii.gz', '*.nii', '*.mha', '*.mhd']:
        image_files.extend(Path(unlabeled_images_folder).glob(ext))
    
    print(f"找到 {len(image_files)} 个无标签图像文件")
    
    if len(image_files) == 0:
        print("警告: 未找到任何无标签图像文件")
        return
    
    # 使用第一个图像文件来确定读写器
    from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_file_ending
    rw = determine_reader_writer_from_file_ending(
        dataset_json['file_ending'],
        str(image_files[0]),
        allow_nonmatching_filename=True,
        verbose=True
    )
    
    # 创建输出目录
    output_folder = join(preprocessed_dataset_folder, 'SeminnUNet')
    maybe_mkdir_p(output_folder)
    
    # 预处理每个图像
    processed_count = 0
    for i, image_file in enumerate(image_files):
        print(f"处理 {i+1}/{len(image_files)}: {image_file.name}")
        
        # 获取case identifier
        case_identifier = image_file.stem.replace('.nii', '')
        
        try:
            # 使用nnUNet的图像读取器读取图像
            image_path = str(image_file)
            rw_instance = rw()
            image, properties = rw_instance.read_images([image_path])
            
            # 使用官方预处理器处理图像
            preprocessed_data, _, preprocessed_properties = preprocessor.run_case_npy(
                image,
                None,  # 无分割标签
                properties,
                plans_manager,
                configuration_manager,
                dataset_json
            )
            
            # 保存预处理后的图像（使用nnUNet标准格式）
            output_file = join(output_folder, f"{case_identifier}.npz")
            np.savez_compressed(
                output_file,
                data=preprocessed_data
            )
            
            # 保存属性文件
            properties_file = join(output_folder, f"{case_identifier}.pkl")
            import pickle
            with open(properties_file, 'wb') as f:
                pickle.dump(preprocessed_properties, f)
            
            processed_count += 1
            print(f"已保存: {output_file}")
            
        except Exception as e:
            print(f"处理 {image_file.name} 时出错: {e}")
            continue
    
    print(f"\n无标签数据预处理完成！成功处理 {processed_count} 个文件")
    print(f"输出目录: {output_folder}")
    print("\n注意: 此预处理使用了与nnUNetv2_plan_and_preprocess完全相同的流程，包括:")
    print("- 官方图像读取器")
    print("- 标准预处理步骤（裁剪、重采样、标准化）")
    print("- 相同的数据格式和属性保存")

def preprocess_unlabeled_data_simple():
    """
    简化版预处理（之前的实现）
    """
    import SimpleITK as sitk
    import pickle
    
    # 输入和输出路径
    input_folder = r"c:\Users\Administrator\Desktop\nnUNet_master\DATASET\nnUNet_raw\Dataset102_quan\image"
    output_folder = r"c:\Users\Administrator\Desktop\nnUNet_master\DATASET\nnUNet_preprocessed\Dataset102_quan\SeminnUNet"
    
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有图像文件
    image_files = []
    for ext in ['*.nii.gz', '*.nii', '*.mha', '*.mhd']:
        image_files.extend(Path(input_folder).glob(ext))
    
    if not image_files:
        print(f"在 {input_folder} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    processed_count = 0
    for image_file in image_files:
        print(f"处理: {image_file.name}")
        
        try:
            # 使用SimpleITK加载
            sitk_image = sitk.ReadImage(str(image_file))
            image_array = sitk.GetArrayFromImage(sitk_image)
            
            # 提取图像属性
            spacing = sitk_image.GetSpacing()
            origin = sitk_image.GetOrigin()
            direction = sitk_image.GetDirection()
            
            properties = {
                'spacing': list(spacing),
                'origin': list(origin),
                'direction': list(direction),
                'size_after_resampling': image_array.shape,
                'original_spacing': list(spacing),
                'original_size': image_array.shape
            }
            
            # 简单的Z-score标准化
            mean = np.mean(image_array)
            std = np.std(image_array)
            if std > 0:
                normalized = (image_array - mean) / std
            else:
                normalized = image_array - mean
            normalized = normalized.astype(np.float32)
            
            # 添加通道维度 (C, H, W, D)
            if len(normalized.shape) == 3:
                normalized = normalized[None]  # 添加通道维度
            
            # 生成输出文件名
            case_id = image_file.stem.replace('.nii', '')
            output_npz = os.path.join(output_folder, f"{case_id}.npz")
            output_pkl = os.path.join(output_folder, f"{case_id}.pkl")
            
            # 保存预处理后的数据
            np.savez_compressed(output_npz, data=normalized)
            
            # 保存属性
            with open(output_pkl, 'wb') as f:
                pickle.dump(properties, f)
            
            processed_count += 1
            print(f"已保存: {case_id}")
            
        except Exception as e:
            print(f"保存失败 {case_id}: {e}")
    
    print(f"\n预处理完成！成功处理 {processed_count} 个文件")
    print(f"输出目录: {output_folder}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='预处理无标签数据')
    parser.add_argument('--method', choices=['official', 'simple'], default='official',
                       help='预处理方法: official使用nnUNet官方流程, simple使用简化流程')
    parser.add_argument('--dataset_id', type=int, default=102, help='数据集ID')
    parser.add_argument('--configuration', type=str, default='3d_fullres', help='配置名称')
    
    args = parser.parse_args()
    
    try:
        if args.method == 'official':
            print("使用nnUNet官方预处理流程...")
            preprocess_unlabeled_data_official(args.dataset_id, args.configuration)
        else:
            print("使用简化预处理流程...")
            preprocess_unlabeled_data_simple()
    except Exception as e:
        print(f"错误: {e}")
        exit(1)