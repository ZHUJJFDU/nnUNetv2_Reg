# nnUNet_Reg 三阶段训练与实现说明

## 概述
- 扩展 nnU-Net v2 实现三阶段：全监督分割预训练 → 半监督师生训练 → 双解码器回归训练
- 任务：肺大泡分割与厚度回归；支持不确定性指导的鲁棒知识蒸馏与第三阶段知识锁定

## 已实现
- 半监督（阶段1 → 阶段2）
  - 师生双路无标签增强：教师弱增强、学生强增强
  - EMA 教师权重与一致性损失
  - 不确定性掩码：置信度或熵过滤低质量伪标签（仅高置信区域蒸馏）
  - 强制 `unlabeled_data_path`；或按数据集自动推断到 `nnUNet_preprocessed/<dataset>/SeminnUNet`
  - 多线程增广封装与原生一致
- 双解码器回归（阶段2 → 阶段3）
  - 切换网络为 `DualDecoderRegCBAMUNet`，编码器与分割解码器键名对齐原生（`encoder.*`、`decoder.*`）
  - 加载第二阶段检查点仅载入编码器与分割解码器权重，回归解码器随机初始化
  - 知识锁定：冻结分割解码器（或单解码器的 `decoder.seg_layers`）
  - 回归数据集/加载器统一：`nnunetv2/training/dataloading/reg_dataloader.py`
  - 回归损失迁移：`nnunetv2/training/loss/reg_loss.py`
- 推理
  - 单文件入口：`regression_inference.py`（滑窗与镜像集成，输出回归结果 JSON）

## 待完善
- 解剖学先验引导注意力（边缘导向）：将“肺大泡边缘”编码为空间先验调制跨任务注意力
- 特征对齐损失（阶段1→2、2→3）：编码器/中间层 L2 或余弦对齐

## 数据集组织
- 环境变量（绝对路径）
```
$env:nnUNet_raw='C:\path\to\DATASET\nnUNet_raw'
$env:nnUNet_preprocessed='C:\path\to\DATASET\nnUNet_preprocessed'
$env:nnUNet_results='C:\path\to\DATASET\nnUNet_trained_models'
```
- 预处理（任选其一）
```
nnUNetv2_plan_and_preprocess -d 102 -c 3d_fullres
python -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints nnUNetv2_plan_and_preprocess -d 102 -c 3d_fullres
```
- 无标签数据（默认路径）
  - `nnUNet_preprocessed/<dataset>/SeminnUNet` 下 `.npz/.pkl` 结构与原生一致
- 回归数值文件
  - `nnUNet_preprocessed/<dataset>/regression_values.json`
  - 支持 `{ "case_id": value }` 或 `{ "case_id": { "bulla_thickness": value } }`（用 `--reg_key` 指定键名）

## 训练流程
- 阶段1：全监督分割
```
python nnunetv2/run/run_training.py Dataset102_quan 3d_fullres 0
# 或 5折
python nnunetv2/run/run_training.py Dataset102_quan 3d_fullres all
```
输出：`$env:nnUNet_results/Dataset102_quan/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth`

- 阶段2：半监督师生（鲁棒KD）
```
python run_semi_training.py Dataset102_quan 3d_fullres 0 -teacher_checkpoint C:\Users\Administrator\Desktop\nnUNet_master\DATASET\nnUNet_trained_models\Dataset102_quan\nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres\fold_0\checkpoint_best.pth -u C:\Users\Administrator\Desktop\nnUNet_master\DATASET\nnUNet_preprocessed\Dataset102_quan\SeminnUNet

```
输出：`$env:nnUNet_results/Dataset102_quan/SemiSupervisedTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_latest.pth`

- 阶段3：双解码器回归（知识锁定）
```
python run_regression_training.py -d Dataset102_quan -c 3d_fullres -f 0 --device cuda --pretrained_weights C:\Users\Administrator\Desktop\nnUNet_master\DATASET\nnUNet_trained_models\Dataset102_quan\SemiSupervisedTrainer__nnUNetPlans__3d_fullres\fold_0\checkpoint_latest.pth --reg_weight 1.0 --reg_key bulla_thickness

```
说明：第三阶段切换 `DualDecoderRegCBAMUNet`，仅加载编码器与分割解码器权重并冻结分割解码器。

## 推理
```
nnUNetv2_predict -d 102 -i C:\Users\Administrator\Desktop\nnUNet_master\DATASET\nnUNet_raw\Dataset102_quan\imagesTs -o C:\Users\Administrator\Desktop\nnUNet_master\DATASET\nnUNet_raw\Dataset102_quan\output -f 0 -tr nnUNetTrainer_500epochs -c 3d_fullres -p nnUNetPlans --save_probabilities



python regression_inference.py -d 102 -i C:\Users\Administrator\Desktop\nnUNet_master\DATASET\nnUNet_raw\Dataset102_quan\imagesTs -o C:\Users\Administrator\Desktop\nnUNet_master\DATASET\nnUNet_raw\Dataset102_quan\Regoutput -f 0      
```
输出：分割结果与 `<output_folder>/regression_results.json`。

## 注意事项
- 所有输出按 `数据集/训练器/配置/折` 组织在 `$env:nnUNet_results` 下
- 半监督阶段一致性权重建议暖启动；可启用熵过滤
- 第三阶段默认冻结分割解码器；如需联合微调可在训练器中取消冻结
- 遇 AMP 或类型不匹配报错可在第三阶段禁用 AMP

