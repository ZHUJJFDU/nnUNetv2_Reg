from typing import Union, Type, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures
)
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim, get_matching_convtransp
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


# CBAM注意力模块已移除，使用更轻量的单向注意力机制


class LightweightCrossAttention(nn.Module):
    """
    轻量级单向交叉注意力模块
    只让分割解码器向回归解码器提供信息，使用简化的全局注意力机制
    """
    def __init__(self, seg_channels, reg_channels, reduction_ratio=16):
        super(LightweightCrossAttention, self).__init__()
        self.seg_channels = seg_channels
        self.reg_channels = reg_channels
        self.reduction_ratio = reduction_ratio
        
        # 使用更大的reduction_ratio来降低计算复杂度
        self.hidden_dim = max(min(seg_channels, reg_channels) // reduction_ratio, 4)
        
        # 只保留必要的投影层
        self.seg_key_conv = None  # 将在set_conv_dim中设置
        self.reg_query_conv = None
        self.gamma = nn.Parameter(torch.tensor(0.01))  # 降低初始权重
        
        # 简化的输出投影
        self.reg_out_conv = None
        
    def set_conv_dim(self, conv_op):
        """根据卷积操作设置相应的卷积层"""
        # 简化的投影层设置
        self.seg_key_conv = conv_op(self.seg_channels, self.hidden_dim, 1, bias=False)
        self.reg_query_conv = conv_op(self.reg_channels, self.hidden_dim, 1, bias=False)
        
        # 简化的输出投影，直接使用1x1卷积
        if self.seg_channels != self.reg_channels:
            self.seg_to_reg_proj = conv_op(self.seg_channels, self.reg_channels, 1, bias=False)
        else:
            self.seg_to_reg_proj = None
        
    def forward(self, seg_feat, reg_feat):
        """
        轻量级单向交叉注意力：只让分割解码器向回归解码器提供信息
        使用简化的全局注意力机制，降低计算复杂度
        Args:
            seg_feat: 分割解码器特征 (B, C_seg, ...)
            reg_feat: 回归解码器特征 (B, C_reg, ...)
        Returns:
            seg_feat, enhanced_reg_feat (分割特征保持不变，只增强回归特征)
        """
        batch_size = seg_feat.size(0)
        spatial_dims = seg_feat.size()[2:]
        
        # 如果回归特征的空间维度不匹配，需要进行插值
        if reg_feat.size()[2:] != seg_feat.size()[2:]:
            reg_feat = F.interpolate(reg_feat, size=spatial_dims, mode='nearest')
        
        # 简化的全局注意力：只使用全局平均池化
        seg_global = F.adaptive_avg_pool3d(seg_feat, 1) if len(spatial_dims) == 3 else F.adaptive_avg_pool2d(seg_feat, 1)
        reg_global = F.adaptive_avg_pool3d(reg_feat, 1) if len(spatial_dims) == 3 else F.adaptive_avg_pool2d(reg_feat, 1)
        
        # 简化的注意力计算
        reg_query_global = self.reg_query_conv(reg_global).view(batch_size, self.hidden_dim)
        seg_key_global = self.seg_key_conv(seg_global).view(batch_size, self.hidden_dim)
        
        # 计算简化的注意力权重
        attention_score = torch.sum(reg_query_global * seg_key_global, dim=1, keepdim=True)  # B x 1
        attention_weight = torch.sigmoid(attention_score).view(batch_size, 1, *([1] * len(spatial_dims)))
        
        # 简化的特征融合：直接使用投影后的分割特征
        if self.seg_to_reg_proj is not None:
            seg_feat_projected = self.seg_to_reg_proj(seg_feat)
        else:
            seg_feat_projected = seg_feat
        
        # 轻量级特征增强：降低融合权重
        enhanced_reg_feat = reg_feat + self.gamma * attention_weight * seg_feat_projected
        
        return seg_feat, enhanced_reg_feat


class RegressionDecoder(nn.Module):
    """
    专门用于回归任务的解码器
    """
    def __init__(self,
                 encoder: PlainConvEncoder,
                 regression_dim: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None
                 ):
        super().__init__()
        
        self.encoder = encoder
        self.regression_dim = regression_dim
        
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        
        # 获取转置卷积操作
        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs

        # 构建回归解码器阶段
        stages = []
        transpconvs = []
        
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))
            
            # 输入特征是跳跃连接特征的2倍（因为要concat）
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first
            ))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        
        # 最终的回归头
        final_features = encoder.output_channels[0]
        conv_dim = convert_conv_op_to_dim(encoder.conv_op)
        
        # 全局池化
        if conv_dim == 2:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.global_pool = nn.AdaptiveAvgPool3d(1)
            
        # 回归头
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, regression_dim)
        )

    def forward(self, skips):
        """
        Args:
            skips: 编码器的跳跃连接特征
        Returns:
            regression_features: 每个阶段的特征（用于交叉注意力）
            regression_output: 最终的回归预测
        """
        lres_input = skips[-1]
        regression_features = []
        
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            regression_features.append(x)
            lres_input = x
        
        # 生成最终回归输出
        pooled_features = self.global_pool(lres_input)
        regression_output = self.regression_head(pooled_features)
        
        # 反转特征列表以匹配分割解码器的顺序
        regression_features = regression_features[::-1]
        
        return regression_features, regression_output


class DualDecoderRegCBAMUNet(AbstractDynamicNetworkArchitectures):
    """
    轻量级双解码器回归U-Net
    包含：
    1. 共享编码器（无CBAM注意力）
    2. 分割解码器
    3. 独立的回归解码器  
    4. 轻量级单向交叉注意力机制
    """
    
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        # CBAM参数已移除
        regression_dim: int = 1,
        enable_cross_attention: bool = True,
        cross_attention_stages: List[int] = None  # 指定在哪些阶段使用交叉注意力
    ):
        super().__init__()
        
        # 设置框架要求的键
        self.key_to_encoder = "encoder.stages"
        self.key_to_stem = "encoder.stages.0"
        self.keys_to_in_proj = (
            "encoder.stages.0.0.convs.0.all_modules.0",
            "encoder.stages.0.0.convs.0.conv",
        )
        
        # 参数处理
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
            
        # 创建编码器
        self.encoder = PlainConvEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_conv_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            nonlin_first=nonlin_first
        )
        
        # CBAM模块已移除，使用轻量级设计
        
        # 创建分割解码器（命名为decoder以与PlainConvUNet保持一致）
        segmentation_decoder = UNetDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision,
            nonlin_first,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            conv_bias
        )
        # 移除嵌套的encoder注册，避免state_dict中出现decoder.encoder.*
        if 'encoder' in segmentation_decoder._modules:
            del segmentation_decoder._modules['encoder']
        self.decoder = segmentation_decoder
        
        # 创建回归解码器
        self.reg_decoder = RegressionDecoder(
            self.encoder,
            regression_dim,
            n_conv_per_stage_decoder,
            nonlin_first,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            conv_bias
        )
        
        # 交叉注意力模块
        self.enable_cross_attention = enable_cross_attention
        if enable_cross_attention:
            if cross_attention_stages is None:
                # 默认在所有解码器阶段使用交叉注意力
                cross_attention_stages = list(range(n_stages - 1))
            
            self.cross_attention_modules = nn.ModuleList()
            # 获取分割解码器和回归解码器的输出通道数
            
            for stage_idx in range(n_stages - 1):
                if stage_idx in cross_attention_stages:
                    # 获取分割解码器该阶段的输出通道数
                    seg_channels = self.decoder.stages[stage_idx].output_channels
                    # 获取回归解码器该阶段的输出通道数
                    # 注意：回归解码器返回的特征是反序的，所以需要调整索引
                    reg_stage_idx = (n_stages - 1) - 1 - stage_idx  # 反向映射
                    reg_channels = self.reg_decoder.stages[reg_stage_idx].output_channels
                    
                    cross_attn = LightweightCrossAttention(seg_channels, reg_channels)
                    cross_attn.set_conv_dim(conv_op)
                    self.cross_attention_modules.append(cross_attn)
                else:
                    self.cross_attention_modules.append(None)
        
        # 存储deep supervision设置
        self.deep_supervision = deep_supervision
        
    def forward(self, x):
        """
        前向传播
        Returns:
            如果deep_supervision=True: (seg_outputs_tuple, regression_output)
            否则: (seg_output, regression_output)
        """
        # 编码器提取特征
        skips = self.encoder(x)
        
        # 直接使用编码器特征，无需CBAM增强
        enhanced_skips = skips
        
        if self.enable_cross_attention:
            # 同时进行分割和回归解码，并应用交叉注意力
            seg_features = []
            reg_features, reg_output = self.reg_decoder(enhanced_skips)
            
            # 分割解码过程（复制UNetDecoder的逻辑但添加交叉注意力）
            lres_input = enhanced_skips[-1]
            seg_outputs = []
            
            for s in range(len(self.decoder.stages)):
                x_seg = self.decoder.transpconvs[s](lres_input)
                x_seg = torch.cat((x_seg, enhanced_skips[-(s+2)]), 1)
                x_seg = self.decoder.stages[s](x_seg)
                if (s < len(self.cross_attention_modules) and 
                    self.cross_attention_modules[s] is not None and
                    s < len(reg_features)):
                    x_seg, reg_features[s] = self.cross_attention_modules[s](x_seg, reg_features[s])
                if self.decoder.deep_supervision:
                    seg_outputs.append(self.decoder.seg_layers[s](x_seg))
                elif s == (len(self.decoder.stages) - 1):
                    seg_outputs.append(self.decoder.seg_layers[-1](x_seg))
                lres_input = x_seg
                seg_features.append(x_seg)
            
            # 反转分割输出顺序
            seg_outputs = seg_outputs[::-1]
            
        else:
            # 独立解码（无交叉注意力）
            seg_outputs = self.decoder(enhanced_skips)
            _, reg_output = self.reg_decoder(enhanced_skips)
        
        # 返回结果
        if isinstance(seg_outputs, list) and len(seg_outputs) > 1:
            return tuple(seg_outputs), reg_output
        else:
            return seg_outputs[0] if isinstance(seg_outputs, list) else seg_outputs, reg_output
    
    def compute_conv_feature_map_size(self, input_size):
        """计算卷积特征图大小"""
        seg_size = self.decoder.compute_conv_feature_map_size(input_size)
        
        # 简单估算回归解码器的大小（与分割解码器类似）
        reg_size = seg_size * 0.8  # 回归解码器稍小
        
        # 如果启用轻量级交叉注意力，增加少量额外开销
        if self.enable_cross_attention:
            attention_overhead = seg_size * 0.05  # 轻量级注意力开销更小
            return seg_size + reg_size + attention_overhead
        
        return seg_size + reg_size
            
    @staticmethod
    def initialize(module):
        """初始化网络权重"""
        InitWeights_He(1e-2)(module)
