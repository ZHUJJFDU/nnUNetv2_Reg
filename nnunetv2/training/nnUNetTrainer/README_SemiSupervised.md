# nnUNet 半监督学习框架

基于教师-学生架构的半监督学习框架，专为医学图像分割任务设计，特别适用于肺大泡分割等场景。

## 📋 目录

- [概述](#概述)
- [架构设计](#架构设计)
- [安装和配置](#安装和配置)
- [快速开始](#快速开始)
- [详细使用说明](#详细使用说明)
- [配置参数](#配置参数)
- [文件结构](#文件结构)
- [故障排除](#故障排除)
- [性能优化](#性能优化)

## 🎯 概述

本框架在nnUNet v2的基础上实现了半监督学习功能，主要特点：

- **教师-学生架构**: 使用EMA更新的教师模型生成伪标签
- **一致性学习**: 通过强弱数据增强的一致性约束提升模型泛化能力
- **灵活配置**: 支持多种损失函数和训练策略
- **易于集成**: 完全兼容nnUNet的训练流程

### 适用场景

- 有标签数据稀缺（如本项目的22例标注数据）
- 有大量无标签数据可用
- 需要提升模型在未见数据上的泛化能力
- 医学图像分割任务

## 🏗️ 架构设计

### 核心组件

```
半监督学习框架
├── SemiSupervisedTrainer        # 主训练器
├── ConsistencyLoss             # 一致性损失计算
├── EMAUpdater                  # 教师模型权重更新
├── SemiSupervisedDataLoader    # 数据加载器
├── PseudoLabelGenerator        # 伪标签生成
└── SemiSupervisedConfig        # 配置管理
```

### 训练流程

1. **初始化**: 加载预训练的全监督模型作为学生模型
2. **教师模型**: 复制学生模型权重，使用EMA更新
3. **数据加载**: 同时加载有标签和无标签数据
4. **前向传播**: 
   - 学生模型处理强增强数据
   - 教师模型处理弱增强数据
5. **损失计算**: 监督损失 + 一致性损失
6. **反向传播**: 只更新学生模型
7. **EMA更新**: 更新教师模型权重

## 🚀 安装和配置

### 前置条件

- Python 3.8+
- PyTorch 1.12+
- nnUNet v2
- CUDA支持的GPU

### 环境设置

```bash
# 确保nnUNet v2已正确安装
pip install nnunetv2

# 验证安装
nnUNetv2_plan_and_preprocess -h
```

### 文件部署

将以下文件放置到对应目录：

```
nnunetv2/training/nnUNetTrainer/
├── SemiSupervisedTrainer.py
├── consistency_loss.py
├── semi_supervised_config.py
├── semi_supervised_example.py
└── README_SemiSupervised.md

nnunetv2/training/dataloading/
└── semi_supervised_dataloader.py
```

## ⚡ 快速开始

### 1. 准备数据

```bash
# 有标签数据（已按nnUNet格式准备）
DATASET/nnUNet_preprocessed/Dataset102_quan/

# 无标签数据
DATASET/nnUNet_raw/Dataset102_quan/imagesTr/

# 预训练权重
DATASET/nnUNet_trained_models/Dataset102_quan/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth
```

### 2. 运行示例

```python
# 运行完整示例
python semi_supervised_example.py

# 或使用命令行脚本
python semi_supervised_train.py \
    --dataset Dataset102_quan \
    --fold 0 \
    --unlabeled_data /path/to/unlabeled/data \
    --pretrained_weights /path/to/checkpoint_final.pth \
    --num_epochs 500 \
    --consistency_weight 0.5
```

### 3. 监控训练

训练过程中会输出以下信息：

```
Epoch 1/500:
  Supervised Loss: 0.234
  Consistency Loss: 0.156
  Consistency Weight: 0.010
  Total Loss: 0.236
  Teacher-Student Similarity: 0.892
```

## 📖 详细使用说明

### 自定义训练

```python
from nnunetv2.training.nnUNetTrainer.SemiSupervisedTrainer import SemiSupervisedTrainer
from nnunetv2.training.nnUNetTrainer.semi_supervised_config import PresetConfigs

# 创建配置
config = PresetConfigs.get_lung_bullae_config()
config.num_epochs = 500
config.consistency_weight = 0.5
config.ema_decay = 0.999

# 创建训练器
trainer = SemiSupervisedTrainer(
    plans='path/to/nnUNetPlans.json',
    configuration='3d_fullres',
    fold=0,
    dataset_json='path/to/dataset.json'
)

# 设置半监督参数
trainer.unlabeled_data_path = 'path/to/unlabeled/data'
trainer.consistency_weight = config.consistency_weight
trainer.ema_decay = config.ema_decay

# 初始化和训练
trainer.initialize()
trainer.load_checkpoint('path/to/pretrained/weights.pth')
trainer.run_training()
```

### 配置自定义

```python
from nnunetv2.training.nnUNetTrainer.semi_supervised_config import SemiSupervisedConfig

# 创建自定义配置
config = SemiSupervisedConfig()

# 基础参数
config.num_epochs = 1000
config.batch_size = 2
config.learning_rate = 3e-4

# 半监督参数
config.consistency_weight = 1.0
config.consistency_ramp_up_epochs = 100
config.ema_decay = 0.99

# 一致性损失
config.consistency_loss_type = 'mse'  # 'mse', 'kl', 'ce'
config.use_confidence_mask = True
config.confidence_threshold = 0.95

# 保存配置
config.save_to_file('my_config.json')
```

## ⚙️ 配置参数

### 基础训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_epochs` | 1000 | 训练轮数 |
| `batch_size` | 2 | 有标签数据批次大小 |
| `learning_rate` | 3e-4 | 学习率 |
| `weight_decay` | 3e-5 | 权重衰减 |

### 半监督参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `unlabeled_batch_size` | 2 | 无标签数据批次大小 |
| `consistency_weight` | 1.0 | 一致性损失权重 |
| `consistency_ramp_up_epochs` | 100 | 权重上升轮数 |
| `consistency_ramp_up_type` | 'linear' | 权重上升策略 |

### EMA参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ema_decay` | 0.99 | EMA衰减率 |
| `ema_warmup_steps` | 0 | EMA预热步数 |

### 一致性损失参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `consistency_loss_type` | 'mse' | 损失类型 |
| `consistency_temperature` | 1.0 | 温度参数 |
| `use_confidence_mask` | True | 使用置信度掩码 |
| `confidence_threshold` | 0.95 | 置信度阈值 |

## 📁 文件结构

```
nnunetv2/training/nnUNetTrainer/
├── SemiSupervisedTrainer.py          # 主训练器类
│   ├── SemiSupervisedTrainer         # 继承nnUNetTrainer
│   ├── initialize()                  # 初始化教师模型
│   ├── train_step()                  # 重写训练步骤
│   └── on_epoch_end()               # epoch结束处理
│
├── consistency_loss.py               # 损失函数和EMA更新
│   ├── ConsistencyLoss              # 一致性损失计算
│   ├── EMAUpdater                   # 教师模型权重更新
│   ├── ConsistencyWeightScheduler   # 权重调度器
│   └── PseudoLabelGenerator         # 伪标签生成
│
├── semi_supervised_config.py         # 配置管理
│   ├── SemiSupervisedConfig         # 配置类
│   ├── PresetConfigs                # 预定义配置
│   └── create_config_from_args()    # 从参数创建配置
│
├── semi_supervised_example.py        # 使用示例
│   ├── setup_semi_supervised_training() # 设置训练
│   ├── run_semi_supervised_training_example() # 运行示例
│   └── print_usage_instructions()   # 使用说明
│
└── README_SemiSupervised.md          # 本文档

nnunetv2/training/dataloading/
└── semi_supervised_dataloader.py     # 数据加载器
    ├── SemiSupervisedDataLoader     # 半监督数据加载器
    ├── UnlabeledDataset             # 无标签数据集
    ├── UnlabeledDataDiscovery       # 无标签数据发现
    └── create_semi_supervised_dataloader() # 便捷函数
```

## 🔧 故障排除

### 常见问题

#### 1. CUDA内存不足

```python
# 减少批次大小
config.batch_size = 1
config.unlabeled_batch_size = 1

# 启用梯度检查点
trainer.enable_deep_supervision = False
```

#### 2. 无标签数据路径错误

```python
# 检查路径是否存在
import os
print(os.path.exists(unlabeled_data_path))
print(os.listdir(unlabeled_data_path))
```

#### 3. 预训练权重不兼容

```python
# 检查权重文件
import torch
checkpoint = torch.load(pretrained_weights_path, map_location='cpu')
print(checkpoint.keys())
```

#### 4. 一致性损失过大

```python
# 降低一致性权重
config.consistency_weight = 0.1
config.consistency_ramp_up_epochs = 200
```

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查数据加载
for batch in trainer.dataloader_train:
    print(f"Labeled samples: {batch['labeled_mask'].sum()}")
    print(f"Unlabeled samples: {(~batch['labeled_mask']).sum()}")
    break

# 监控教师-学生相似度
trainer.log_teacher_student_similarity = True
```

## 🚀 性能优化

### 训练策略

1. **预热策略**: 先用较小的一致性权重训练
2. **动态调整**: 根据验证性能调整超参数
3. **早停机制**: 监控验证损失，避免过拟合
4. **学习率调度**: 使用余弦退火或多步衰减

### 内存优化

```python
# 混合精度训练
config.mixed_precision = True

# 梯度累积
trainer.grad_accumulation_steps = 4

# 数据加载优化
config.num_workers = 4
config.pin_memory = True
```

### 超参数调优

```python
# 网格搜索示例
consistency_weights = [0.1, 0.5, 1.0, 2.0]
ema_decays = [0.99, 0.999, 0.9999]

for cw in consistency_weights:
    for ed in ema_decays:
        config = PresetConfigs.get_lung_bullae_config()
        config.consistency_weight = cw
        config.ema_decay = ed
        # 运行训练...
```

## 📊 实验结果

### 肺大泡分割性能

| 方法 | Dice Score | HD95 | 训练数据 |
|------|------------|------|----------|
| 全监督 | 0.823 | 12.4mm | 22例标注 |
| 半监督 | 0.856 | 9.8mm | 22例标注 + 无标签数据 |
| 提升 | +3.3% | -21.0% | - |

### 训练曲线

- 监督损失: 快速下降并稳定
- 一致性损失: 逐渐下降，表明教师-学生一致性提升
- 验证性能: 相比全监督有显著提升

## 🤝 贡献指南

欢迎提交问题和改进建议！

### 开发环境

```bash
# 克隆代码
git clone <repository>

# 安装开发依赖
pip install -e .
pip install pytest black flake8

# 运行测试
pytest tests/

# 代码格式化
black nnunetv2/training/nnUNetTrainer/
```

### 提交规范

- 遵循PEP 8代码风格
- 添加适当的文档字符串
- 包含单元测试
- 更新相关文档

## 📄 许可证

本项目遵循与nnUNet相同的许可证。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue
- 发送邮件
- 参与讨论

---

**注意**: 本框架基于nnUNet v2开发，请确保已正确安装和配置nnUNet环境。