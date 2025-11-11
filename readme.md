

## 三阶段训练指南（本仓库扩展）

本仓库在nnU-Net v2基础上，提供“全监督分割预训练 → 半监督师生训练 → 双解码器回归训练”的三阶段流程。以下为在 Windows PowerShell 环境下的实操说明。

### 前置准备
- 设置环境变量（建议使用绝对路径）：

```
$env:nnUNet_raw='C:\Users\Administrator\Desktop\nnUNet_master\DATASET\nnUNet_raw'
$env:nnUNet_preprocessed='C:\Users\Administrator\Desktop\nnUNet_master\DATASET\nnUNet_preprocessed'
$env:nnUNet_results='C:\Users\Administrator\Desktop\nnUNet_master\DATASET\nnUNet_trained_models'
```

- 数据计划与预处理（任选其一）：

```
# 命令行入口（若已安装nnUNet v2 CLI）
nnUNetv2_plan_and_preprocess -d 102 -c 3d_fullres

# 纯脚本方式（不依赖CLI安装）
python -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints nnUNetv2_plan_and_preprocess -d 102 -c 3d_fullres
```

- 可选：指定使用的GPU（例如只用第0张）：

```
$env:CUDA_VISIBLE_DEVICES='0'
```

> 说明：文中 `Dataset102_quan` 为示例数据集名，你也可以用数据集ID（例如 `102`）。配置名如 `3d_fullres`、`2d` 等请按你的数据规划结果选择。

### 阶段1：全监督分割预训练（训练教师模型）
- 作用：使用有标签数据训练鲁棒的分割教师模型。
- 单折训练示例：

```
python nnunetv2/run/run_training.py Dataset102_quan 3d_fullres 0
```

- 全折训练（5折）：

```
python nnunetv2/run/run_training.py Dataset102_quan 3d_fullres all
```

- 典型输出：

```
C:\Users\Administrator\Desktop\nnUNet_master\DATASET\nnUNet_trained_models\Dataset102_quan\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_0\checkpoint_final.pth
```

### 阶段2：半监督训练（师生一致性框架）
- 作用：结合大量无标签数据，通过一致性损失与EMA教师更新提升学生模型泛化。
- 便捷封装入口（推荐）：

```
python run_semi_supervised_training.py \
  -d Dataset102_quan -c 3d_fullres -f 0 \
  -t C:\Users\Administrator\Desktop\nnUNet_master\DATASET\nnUNet_trained_models\Dataset102_quan\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_0\checkpoint_final.pth \
  -u C:\path\to\unlabeled_images \
  --ema_decay 0.995 --consistency_weight 0.5 --consistency_ramp_up 50 \
  --epochs 300 --device cuda -o C:\Users\Administrator\Desktop\nnUNet_master\DATASET\nnUNet_trained_models
```

- 原始脚本入口（最小示例）：

```
python nnunetv2/training/nnUNetTrainer/train_semi_supervised.py -d 102 -c 3d_fullres -f 0 --unlabeled_data_path C:\path\to\unlabeled_images
```

- 关键参数说明：
  - `-t/--teacher_checkpoint`：阶段1教师模型权重路径。
  - `-u/--unlabeled_data`：无标签数据文件夹。
  - `--ema_decay`：EMA衰减（常用 0.99–0.999）。
  - `--consistency_weight` / `--consistency_ramp_up`：一致性损失强度及暖启动轮数。
  - `--epochs`：训练轮数；`--device`：`cuda` 或 `cpu`。

- 典型输出：

```
C:\Users\Administrator\Desktop\nnUNet_master\DATASET\nnUNet_trained_models\Dataset102_quan\SemiSupervisedTrainer__nnUNetPlans__3d_fullres\fold_0\checkpoint_latest.pth
```

### 阶段3：双解码器回归训练（保留分割能力同时学习回归）
- 作用：加载阶段2学生模型权重，冻结分割解码器，仅训练回归解码器与交叉注意力模块。

- 方案A（示例脚本，便于快速起步）：

```
python .\train_dual_decoder_regression_example.py
```

> 使用前请在脚本中替换：`pretrained_checkpoint_path` 指向阶段2输出；将 `DummyRegressionDataset` 替换为你的真实数据集；根据任务调整 `num_classes`、`regression_dim`、损失权重等。

- 方案B（通用入口，更易融入现有流程）：

```
python nnunetv2/run/run_regression_training.py Dataset102_quan 3d_fullres 0 \
  -tr PretrainedDualDecoderRegressionTrainer \
  -pretrained_weights C:\Users\Administrator\Desktop\nnUNet_master\DATASET\nnUNet_trained_models\Dataset102_quan\SemiSupervisedTrainer__nnUNetPlans__3d_fullres\fold_0\checkpoint_latest.pth \
  -reg_weight 1.0 -reg_loss mse --disable_amp -device cuda
```

- 多卡训练（例如2卡）：

```
$env:CUDA_VISIBLE_DEVICES='0,1'
python nnunetv2/run/run_regression_training.py Dataset102_quan 3d_fullres 0 -tr PretrainedDualDecoderRegressionTrainer -num_gpus 2
```

- 典型输出：

```
C:\Users\Administrator\Desktop\nnUNet_master\DATASET\nnUNet_trained_models\Dataset102_quan\PretrainedDualDecoderRegressionTrainer__nnUNetPlans__3d_fullres\fold_0\checkpoint_final.pth
```

### 常见路径与注意事项
- 所有训练输出按 `数据集/训练器/配置/折` 组织在 `$env:nnUNet_results` 下。
- 半监督阶段需可用的教师权重与无标签数据目录；一致性损失建议随训练逐步加权。
- 双解码器阶段默认冻结分割解码器；如需联合微调可在训练器内取消冻结。
- 如遇数据类型不匹配或AMP相关报错，可在回归阶段加 `--disable_amp`。
- 回归数值如从外部文件加载，确保键名与 `-reg_key` 一致（默认 `reg`），并准备 `regression_values.json`。

如需我基于你的实际数据路径与配置，生成三阶段的精确命令，请告知数据集名称/ID、折数、无标签数据目录与目标回归维度等信息。
