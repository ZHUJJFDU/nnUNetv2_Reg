# 半监督nnUNet模型推理指南

## 概述

本指南介绍如何使用半监督nnUNet框架训练好的模型进行推理预测。

## 文件说明

- `semi_supervised_inference.py`: 主要推理脚本
- `inference_config.json`: 推理配置文件
- 训练好的模型权重: `checkpoint_best.pth` 或 `checkpoint_latest.pth`

## 快速开始

### 1. 单张图像推理

```bash
python semi_supervised_inference.py \
    --input path/to/image.nii.gz \
    --output path/to/prediction.nii.gz \
    --model_folder "c:/Users/Administrator/Desktop/nnUNet_master/DATASET/nnUNet_trained_models/Dataset102_quan/SemiSupervisedTrainer__nnUNetPlans__3d_fullres" \
    --checkpoint checkpoint_best.pth
```

### 2. 批量推理

```bash
python semi_supervised_inference.py \
    --input_folder path/to/input/folder \
    --output_folder path/to/output/folder \
    --model_folder "c:/Users/Administrator/Desktop/nnUNet_master/DATASET/nnUNet_trained_models/Dataset102_quan/SemiSupervisedTrainer__nnUNetPlans__3d_fullres" \
    --checkpoint checkpoint_best.pth
```

### 3. 使用配置文件

```bash
python semi_supervised_inference.py --config inference_config.json
```

## 配置文件说明

### 模型配置 (model)

- `model_folder`: 训练好的模型文件夹路径
- `checkpoint_name`: 检查点文件名 (checkpoint_best.pth 或 checkpoint_latest.pth)
- `use_folds`: 使用的fold，null表示使用所有可用fold
- `use_teacher_model`: 是否使用教师模型权重 (通常设为false)

### 推理配置 (inference)

- `device`: 计算设备 ("cuda" 或 "cpu")
- `tile_step_size`: 滑动窗口步长 (0.5表示50%重叠)
- `use_gaussian`: 是否使用高斯权重
- `use_mirroring`: 是否使用镜像增强
- `perform_everything_on_gpu`: 是否在GPU上执行所有操作
- `save_probabilities`: 是否保存概率图
- `overwrite`: 是否覆盖已存在的输出文件

### 处理配置 (processing)

- `num_processes_preprocessing`: 预处理进程数
- `num_processes_segmentation_export`: 分割导出进程数
- `allow_tqdm`: 是否显示进度条
- `verbose`: 是否显示详细信息

## 命令行参数

### 必需参数

- `--model_folder`: 模型文件夹路径
- `--input` 或 `--input_folder`: 输入图像或文件夹
- `--output` 或 `--output_folder`: 输出路径或文件夹

### 可选参数

- `--config`: 配置文件路径
- `--checkpoint`: 检查点文件名 (默认: checkpoint_best.pth)
- `--device`: 计算设备 (默认: cuda)
- `--use_teacher`: 使用教师模型权重
- `--tile_step_size`: 滑动窗口步长 (默认: 0.5)
- `--use_gaussian`: 使用高斯权重 (默认: True)
- `--use_mirroring`: 使用镜像增强 (默认: True)
- `--save_probabilities`: 保存概率图
- `--overwrite`: 覆盖已存在文件
- `--verbose`: 显示详细信息

## 使用示例

### 示例1: 基本单图像推理

```bash
python semi_supervised_inference.py \
    --input /path/to/test_image.nii.gz \
    --output /path/to/prediction.nii.gz \
    --model_folder "c:/Users/Administrator/Desktop/nnUNet_master/DATASET/nnUNet_trained_models/Dataset102_quan/SemiSupervisedTrainer__nnUNetPlans__3d_fullres"
```

### 示例2: 批量推理带自定义参数

```bash
python semi_supervised_inference.py \
    --input_folder /path/to/test_images \
    --output_folder /path/to/predictions \
    --model_folder "c:/Users/Administrator/Desktop/nnUNet_master/DATASET/nnUNet_trained_models/Dataset102_quan/SemiSupervisedTrainer__nnUNetPlans__3d_fullres" \
    --checkpoint checkpoint_latest.pth \
    --device cuda \
    --tile_step_size 0.3 \
    --save_probabilities \
    --verbose
```

### 示例3: 使用教师模型权重

```bash
python semi_supervised_inference.py \
    --input /path/to/test_image.nii.gz \
    --output /path/to/prediction.nii.gz \
    --model_folder "c:/Users/Administrator/Desktop/nnUNet_master/DATASET/nnUNet_trained_models/Dataset102_quan/SemiSupervisedTrainer__nnUNetPlans__3d_fullres" \
    --use_teacher \
    --verbose
```

## 输出说明

- 推理结果将保存为NIfTI格式 (.nii.gz)
- 如果启用 `save_probabilities`，还会保存概率图
- 输出文件名会自动添加适当的后缀

## 注意事项

1. **内存要求**: 3D图像推理需要较大内存，建议使用GPU
2. **输入格式**: 输入图像必须是NIfTI格式 (.nii.gz)
3. **预处理**: 输入图像会自动进行预处理以匹配训练时的格式
4. **模型兼容性**: 确保使用的检查点文件与当前代码版本兼容

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小 `tile_step_size` 值
   - 设置 `perform_everything_on_gpu=false`

2. **找不到模型文件**
   - 检查 `model_folder` 路径是否正确
   - 确认检查点文件存在

3. **推理速度慢**
   - 增加 `tile_step_size` 值 (减少重叠)
   - 使用GPU加速
   - 增加处理进程数

### 性能优化建议

- 使用 `checkpoint_best.pth` 获得最佳性能
- 在GPU上运行以获得最快速度
- 根据可用内存调整 `tile_step_size`
- 对于批量处理，使用多进程并行

## 技术支持

如遇到问题，请检查：
1. 模型路径和文件是否存在
2. 输入图像格式是否正确
3. CUDA环境是否正确配置
4. 依赖包是否完整安装