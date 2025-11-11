#!/usr/bin/env python3
"""
双解码器回归网络训练示例
基于半监督训练的预训练权重，添加回归解码器进行联合训练

使用方法:
python train_dual_decoder_regression_example.py

作者: Assistant
日期: 2024
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path

# 添加动态网络架构路径
sys.path.append(r'c:\Users\Administrator\Desktop\nnUNet_master\dynamic-network-architectures')

from dynamic_network_architectures.architectures.pretrained_dual_decoder_regression_unet import PretrainedDualDecoderRegressionUNet
from nnunetv2.training.nnUNetTrainer.PretrainedDualDecoderRegressionTrainer import PretrainedDualDecoderRegressionTrainer


class DummyRegressionDataset(Dataset):
    """
    示例回归数据集
    在实际使用中，请替换为您的真实数据集
    """
    def __init__(self, num_samples=100, input_size=(1, 32, 32, 32)):
        self.num_samples = num_samples
        self.input_size = input_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机输入数据
        x = torch.randn(*self.input_size)
        
        # 生成随机分割标签 (多类别)
        seg_label = torch.randint(0, 3, self.input_size[1:])  # 3类分割
        
        # 生成随机回归标签 (例如：体积、密度等)
        reg_label = torch.randn(1)  # 单个回归值
        
        return {
            'data': x,
            'seg': seg_label,
            'regression': reg_label
        }


def create_dual_decoder_network():
    """
    创建双解码器回归网络
    """
    print("创建双解码器回归网络...")
    
    # 网络配置参数
    network_config = {
        'input_channels': 1,
        'n_stages': 6,
        'features_per_stage': [32, 64, 128, 256, 320, 320],
        'conv_op': nn.Conv3d,
        'kernel_sizes': [[3, 3, 3]] * 6,
        'strides': [[1, 1, 1]] + [[2, 2, 2]] * 5,
        'n_conv_per_stage': [2, 2, 2, 2, 2, 2],
        'num_classes': 3,  # 分割类别数
        'n_conv_per_stage_decoder': [2, 2, 2, 2, 2],
        'conv_bias': True,
        'norm_op': nn.InstanceNorm3d,
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'dropout_op': None,
        'dropout_op_kwargs': None,
        'nonlin': nn.LeakyReLU,
        'nonlin_kwargs': {'inplace': True},
        'deep_supervision': True,
        'nonlin_first': False,
        'regression_dim': 1,  # 回归输出维度
        'enable_cross_attention': True,
        'cross_attention_stages': [0, 1, 2],  # 在前3个阶段使用交叉注意力
        'pretrained_checkpoint_path': r'c:\Users\Administrator\Desktop\nnUNet_master\DATASET\nnUNet_trained_models\Dataset102_quan\SemiSupervisedTrainer__nnUNetPlans__3d_fullres\fold_0\checkpoint_latest.pth',
        'freeze_segmentation_decoder': True
    }
    
    # 创建网络
    network = PretrainedDualDecoderRegressionUNet(**network_config)
    
    print(f"网络创建成功!")
    print(f"总参数数量: {sum(p.numel() for p in network.parameters()):,}")
    print(f"可训练参数数量: {sum(p.numel() for p in network.parameters() if p.requires_grad):,}")
    
    return network


def train_dual_decoder_network():
    """
    训练双解码器回归网络
    """
    print("开始训练双解码器回归网络...")
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建网络
    network = create_dual_decoder_network()
    network = network.to(device)
    
    # 创建数据集和数据加载器
    train_dataset = DummyRegressionDataset(num_samples=200)
    val_dataset = DummyRegressionDataset(num_samples=50)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # 优化器 - 只优化回归解码器参数
    regression_params = []
    for name, param in network.named_parameters():
        if param.requires_grad:
            regression_params.append(param)
    
    optimizer = optim.Adam(regression_params, lr=1e-4, weight_decay=1e-5)
    
    # 损失函数
    seg_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()
    
    # 训练循环
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # 训练阶段
        network.train()
        train_seg_loss = 0.0
        train_reg_loss = 0.0
        train_total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # 数据移到设备
            inputs = batch['data'].to(device)
            seg_labels = batch['seg'].to(device)
            reg_labels = batch['regression'].to(device)
            
            # 前向传播
            seg_outputs, reg_outputs = network(inputs)
            
            # 计算损失
            seg_loss = seg_loss_fn(seg_outputs[0], seg_labels)  # 使用主要输出
            reg_loss = reg_loss_fn(reg_outputs, reg_labels)
            total_loss = seg_loss + reg_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 累计损失
            train_seg_loss += seg_loss.item()
            train_reg_loss += reg_loss.item()
            train_total_loss += total_loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx}: Seg Loss: {seg_loss.item():.4f}, "
                      f"Reg Loss: {reg_loss.item():.4f}, Total: {total_loss.item():.4f}")
        
        # 计算平均训练损失
        avg_train_seg_loss = train_seg_loss / len(train_loader)
        avg_train_reg_loss = train_reg_loss / len(train_loader)
        avg_train_total_loss = train_total_loss / len(train_loader)
        
        print(f"训练损失 - 分割: {avg_train_seg_loss:.4f}, "
              f"回归: {avg_train_reg_loss:.4f}, 总计: {avg_train_total_loss:.4f}")
        
        # 验证阶段
        network.eval()
        val_seg_loss = 0.0
        val_reg_loss = 0.0
        val_total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['data'].to(device)
                seg_labels = batch['seg'].to(device)
                reg_labels = batch['regression'].to(device)
                
                seg_outputs, reg_outputs = network(inputs)
                
                seg_loss = seg_loss_fn(seg_outputs[0], seg_labels)
                reg_loss = reg_loss_fn(reg_outputs, reg_labels)
                total_loss = seg_loss + reg_loss
                
                val_seg_loss += seg_loss.item()
                val_reg_loss += reg_loss.item()
                val_total_loss += total_loss.item()
        
        # 计算平均验证损失
        avg_val_seg_loss = val_seg_loss / len(val_loader)
        avg_val_reg_loss = val_reg_loss / len(val_loader)
        avg_val_total_loss = val_total_loss / len(val_loader)
        
        print(f"验证损失 - 分割: {avg_val_seg_loss:.4f}, "
              f"回归: {avg_val_reg_loss:.4f}, 总计: {avg_val_total_loss:.4f}")
    
    print("\n训练完成!")
    
    # 保存模型
    output_dir = Path("dual_decoder_regression_output")
    output_dir.mkdir(exist_ok=True)
    
    checkpoint_path = output_dir / "dual_decoder_regression_model.pth"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_total_loss,
        'val_loss': avg_val_total_loss,
    }, checkpoint_path)
    
    print(f"模型已保存到: {checkpoint_path}")
    
    return network


def test_inference():
    """
    测试推理功能
    """
    print("\n测试推理功能...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建网络
    network = create_dual_decoder_network()
    network = network.to(device)
    network.eval()
    
    # 创建测试数据
    test_input = torch.randn(1, 1, 32, 32, 32).to(device)
    
    with torch.no_grad():
        seg_output, reg_output = network(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"分割输出形状: {[out.shape for out in seg_output]}")
    print(f"回归输出形状: {reg_output.shape}")
    print(f"回归预测值: {reg_output.cpu().numpy()}")
    
    print("推理测试完成!")


def main():
    """
    主函数
    """
    print("=" * 80)
    print("双解码器回归网络训练示例")
    print("=" * 80)
    
    try:
        # 检查预训练权重文件是否存在
        pretrained_path = r'c:\Users\Administrator\Desktop\nnUNet_master\DATASET\nnUNet_trained_models\Dataset102_quan\SemiSupervisedTrainer__nnUNetPlans__3d_fullres\fold_0\checkpoint_latest.pth'
        if not os.path.exists(pretrained_path):
            print(f"警告: 预训练权重文件不存在: {pretrained_path}")
            print("将创建随机初始化的网络进行演示")
        
        # 1. 测试推理
        test_inference()
        
        # 2. 训练网络
        trained_network = train_dual_decoder_network()
        
        print("\n" + "=" * 80)
        print("示例运行完成!")
        print("=" * 80)
        
        print("\n使用说明:")
        print("1. 替换 DummyRegressionDataset 为您的真实数据集")
        print("2. 调整网络配置参数以匹配您的数据")
        print("3. 修改损失函数权重以平衡分割和回归任务")
        print("4. 根据需要调整训练超参数")
        print("5. 添加更多的评估指标和可视化")
        
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()