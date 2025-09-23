# nnUNet半监督训练框架使用指南

## 概述

本框架基于nnUNet v2实现了教师-学生半监督学习，旨在利用大量无标签数据提升模型性能。框架采用指数移动平均(EMA)更新教师模型权重，通过一致性损失约束学生模型在强数据增强下的预测与教师模型在弱数据增强下的预测保持一致。

## 核心组件

### 1. SemiSupervisedTrainer
- **位置**: `nnunetv2/training/nnUNetTrainer/SemiSupervisedTrainer.py`
- **功能**: 继承自nnUNetTrainer，实现教师-学生半监督学习框架
- **特性**:
  - EMA权重更新机制
  - 一致性损失计算
  - 教师模型管理
  - 半监督训练循环

### 2. 一致性损失和EMA更新
- **位置**: `nnunetv2/training/nnUNetTrainer/consistency_loss.py`
- **功能**: 提供多种一致性损失函数和EMA权重更新策略
- **包含**:
  - ConsistencyLoss: MSE、KL散度、交叉熵损失
  - EMAUpdater: 指数移动平均权重更新
  - ConsistencyWeightScheduler: 权重调度策略
  - PseudoLabelGenerator: 伪标签生成和质量过滤

### 3. 半监督数据加载器
- **位置**: `nnunetv2/training/dataloading/semi_supervised_dataloader.py`
- **功能**: 同时管理有标签和无标签数据的加载
- **特性**:
  - 混合批次生成
  - 无标签数据发现
  - 数据集分割工具

### 4. 训练脚本
- **主脚本**: `nnunetv2/training/nnUNetTrainer/train_semi_supervised.py`
- **启动脚本**: `run_semi_supervised_training.py`
- **配置文件**: `nnunetv2/training/nnUNetTrainer/semi_supervised_config.json`

## 快速开始

### 1. 准备数据

确保你已经有：
- 训练好的全监督模型检查点（作为教师模型初始权重）
- 预处理后的有标签数据
- 预处理后的无标签数据（可选）

### 2. 配置训练参数

编辑配置文件 `semi_supervised_config.json`：

```json
{
  "dataset": {
    "name_or_id": "Dataset102_quan",
    "configuration": "3d_fullres",
    "fold": 0
  },
  "semi_supervised": {
    "teacher_checkpoint_path": "path/to/checkpoint_final.pth",
    "unlabeled_data_path": "path/to/unlabeled/data",
    "ema_decay": 0.99,
    "consistency_weight": 1.0,
    "consistency_ramp_up_epochs": 50
  }
}
```

### 3. 启动训练

#### 方法1: 使用启动脚本（推荐）

```bash
# 使用默认配置
python run_semi_supervised_training.py

# 使用自定义配置文件
python run_semi_supervised_training.py --config my_config.json

# 命令行覆盖参数
python run_semi_supervised_training.py --ema_decay 0.995 --consistency_weight 2.0
```

#### 方法2: 直接调用训练脚本

```bash
python nnunetv2/training/nnUNetTrainer/train_semi_supervised.py \
    -d Dataset102_quan \
    -c 3d_fullres \
    -f 0 \
    --unlabeled_data_path /path/to/unlabeled/data \
    --pretrained_weights /path/to/checkpoint_final.pth
```

## 详细配置说明

### 核心参数

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|--------|----------|
| `ema_decay` | EMA衰减率 | 0.99 | 0.99-0.999 |
| `consistency_weight` | 一致性损失权重 | 1.0 | 0.1-10.0 |
| `consistency_ramp_up_epochs` | 权重上升周期 | 50 | 20-100 |
| `unlabeled_batch_size` | 无标签批次大小 | 2 | 1-4 |
| `labeled_ratio_in_batch` | 批次中有标签比例 | 0.5 | 0.3-0.7 |

### 数据增强配置

```json
"augmentation": {
  "strong_augmentation_student": true,
  "weak_augmentation_teacher": true,
  "augmentation_strength": {
    "rotation_range": [-15, 15],
    "scaling_range": [0.85, 1.15],
    "elastic_deformation": true,
    "gaussian_noise": true,
    "gaussian_blur": true
  }
}
```

### 伪标签配置

```json
"pseudo_labeling": {
  "confidence_threshold": 0.8,
  "use_entropy_filtering": true,
  "entropy_threshold": 1.0,
  "quality_filtering": {
    "min_foreground_ratio": 0.01,
    "max_background_ratio": 0.99
  }
}
```

## 训练监控

### 关键指标

1. **监督损失** (`loss_supervised`): 有标签数据上的分割损失
2. **一致性损失** (`loss_consistency`): 教师-学生预测一致性损失
3. **总损失** (`loss_total`): 加权后的总损失
4. **教师模型性能** (`teacher_dice`): 教师模型在验证集上的Dice分数
5. **学生模型性能** (`student_dice`): 学生模型在验证集上的Dice分数

### 日志文件

- `training_log.txt`: 详细训练日志
- `progress.png`: 训练曲线图
- `validation_raw/`: 验证结果
- `semi_supervised_config.json`: 实际使用的配置

## 最佳实践

### 1. 教师模型初始化

- 使用在相同数据集上训练的高质量全监督模型
- 确保教师模型已经收敛
- 教师模型的Dice分数应该 > 0.8

### 2. 超参数调优

**EMA衰减率**:
- 较大的数据集使用较高的衰减率 (0.999)
- 较小的数据集使用较低的衰减率 (0.99)

**一致性权重**:
- 从较小的权重开始 (0.1-0.5)
- 逐渐增加到 1.0-2.0
- 使用线性或余弦调度

**权重上升周期**:
- 通常设置为总训练轮数的 5-10%
- 较复杂的任务需要更长的上升周期

### 3. 数据准备

**无标签数据**:
- 确保与有标签数据来自相同的分布
- 数量应该是有标签数据的 2-10 倍
- 预处理方式必须与有标签数据一致

**数据增强**:
- 学生模型使用强增强（旋转、缩放、弹性变形等）
- 教师模型使用弱增强（轻微旋转、缩放）
- 避免过度增强导致的分布偏移

### 4. 训练策略

**分阶段训练**:
1. 第一阶段：仅使用监督损失预热 (10-20 epochs)
2. 第二阶段：逐渐引入一致性损失
3. 第三阶段：稳定的半监督训练

**早停策略**:
- 监控验证集上的Dice分数
- 如果连续20个epoch没有改善则停止
- 保存最佳模型而非最后一个epoch的模型

## 故障排除

### 常见问题

1. **一致性损失过大**
   - 降低一致性权重
   - 增加权重上升周期
   - 检查数据增强是否过强

2. **教师模型性能下降**
   - 降低EMA衰减率
   - 检查学生模型是否发散
   - 确保教师模型初始权重质量

3. **训练不稳定**
   - 减小学习率
   - 增加批次大小
   - 使用梯度裁剪

4. **内存不足**
   - 减小批次大小
   - 减小patch大小
   - 使用梯度累积

### 调试技巧

1. **可视化预测结果**
   ```python
   # 在验证时保存预测概率
   trainer.save_probabilities = True
   ```

2. **监控权重更新**
   ```python
   # 检查EMA更新是否正常
   print(f"EMA decay: {trainer.ema_decay}")
   print(f"Weight difference: {torch.norm(teacher_weights - student_weights)}")
   ```

3. **分析一致性损失**
   ```python
   # 检查一致性损失的分布
   consistency_losses = trainer.get_consistency_losses()
   print(f"Mean: {consistency_losses.mean()}, Std: {consistency_losses.std()}")
   ```

## 性能优化

### 计算优化

1. **混合精度训练**
   ```json
   "experimental": {
     "use_mixed_precision": true
   }
   ```

2. **模型编译** (PyTorch 2.0+)
   ```json
   "experimental": {
     "compile_model": true
   }
   ```

3. **数据加载优化**
   ```json
   "data_loading": {
     "num_threads_for_batchgenerators": 8
   }
   ```

### 内存优化

1. **梯度检查点**
   - 在大模型上启用梯度检查点
   - 牺牲计算时间换取内存

2. **批次大小调整**
   - 根据GPU内存动态调整
   - 使用梯度累积模拟大批次

## 实验结果分析

### 评估指标

1. **分割性能**
   - Dice系数
   - Hausdorff距离
   - 表面距离

2. **训练效率**
   - 收敛速度
   - 训练稳定性
   - 计算资源消耗

3. **半监督效果**
   - 与全监督基线的比较
   - 不同无标签数据量的影响
   - 一致性损失的贡献

### 结果可视化

```python
# 绘制训练曲线
import matplotlib.pyplot as plt

# 加载训练日志
training_log = load_training_log('training_log.txt')

# 绘制损失曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(training_log['epoch'], training_log['loss_supervised'], label='Supervised')
plt.plot(training_log['epoch'], training_log['loss_consistency'], label='Consistency')
plt.plot(training_log['epoch'], training_log['loss_total'], label='Total')
plt.legend()
plt.title('Training Loss')

# 绘制Dice分数
plt.subplot(1, 3, 2)
plt.plot(training_log['epoch'], training_log['dice_student'], label='Student')
plt.plot(training_log['epoch'], training_log['dice_teacher'], label='Teacher')
plt.legend()
plt.title('Dice Score')

# 绘制一致性权重
plt.subplot(1, 3, 3)
plt.plot(training_log['epoch'], training_log['consistency_weight'])
plt.title('Consistency Weight')

plt.tight_layout()
plt.savefig('training_curves.png')
```

## 扩展和定制

### 自定义一致性损失

```python
class CustomConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, student_pred, teacher_pred, confidence_mask=None):
        # 实现自定义一致性损失
        pass
```

### 自定义数据增强

```python
class CustomAugmentation:
    def __init__(self, strength='strong'):
        self.strength = strength
    
    def __call__(self, data):
        # 实现自定义数据增强
        pass
```

### 自定义权重调度

```python
class CustomWeightScheduler:
    def __init__(self, max_weight, schedule_type='custom'):
        self.max_weight = max_weight
        self.schedule_type = schedule_type
    
    def get_weight(self, epoch, max_epochs):
        # 实现自定义权重调度
        pass
```

## 参考文献

1. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results
2. Temporal Ensembling for Semi-Supervised Learning
3. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation
4. Semi-supervised semantic segmentation needs strong, varied perturbations

## 联系和支持

如有问题或建议，请通过以下方式联系：
- 创建GitHub Issue
- 发送邮件至项目维护者
- 参与社区讨论

---

**注意**: 本框架基于nnUNet v2开发，请确保你的环境与nnUNet v2兼容。建议在开始大规模训练前，先在小数据集上验证框架的正确性。