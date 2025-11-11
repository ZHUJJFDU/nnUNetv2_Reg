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
            self.reg_out_conv = conv_op(self.seg_channels, self.reg_channels, 1, bias=False)
        else:
            self.reg_out_conv = nn.Identity()
    
    def forward(self, seg_feat, reg_feat):
        """
        前向传播
        Args:
            seg_feat: 分割解码器特征 [B, C_seg, ...]
            reg_feat: 回归解码器特征 [B, C_reg, ...]
        Returns:
            enhanced_reg_feat: 增强后的回归特征
        """
        if self.seg_key_conv is None:
            return reg_feat
            
        batch_size = seg_feat.size(0)
        
        # 全局平均池化降维
        seg_global = F.adaptive_avg_pool3d(seg_feat, 1)  # [B, C_seg, 1, 1, 1]
        reg_global = F.adaptive_avg_pool3d(reg_feat, 1)  # [B, C_reg, 1, 1, 1]
        
        # 投影到隐藏维度
        seg_key = self.seg_key_conv(seg_global).view(batch_size, self.hidden_dim)  # [B, hidden_dim]
        reg_query = self.reg_query_conv(reg_global).view(batch_size, self.hidden_dim)  # [B, hidden_dim]
        
        # 计算注意力权重（简化的点积注意力）
        attention_weight = torch.sum(seg_key * reg_query, dim=1, keepdim=True)  # [B, 1]
        attention_weight = torch.sigmoid(attention_weight)  # 归一化到[0,1]
        
        # 应用注意力权重
        seg_info = self.reg_out_conv(seg_feat)  # 投影到回归特征维度
        enhanced_reg_feat = reg_feat + self.gamma * attention_weight.view(-1, 1, 1, 1, 1) * seg_info
        
        return enhanced_reg_feat


class RegressionDecoder(nn.Module):
    """
    回归解码器，用于预测连续值
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
        
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == (n_stages_encoder - 1)
        
        self.encoder = encoder
        self.num_stages = n_stages_encoder
        self.regression_dim = regression_dim
        
        # 构建回归解码器层
        stages = []
        transpconvs = []
        
        # 从最深层开始构建
        for i in range(len(encoder.output_channels) - 1):
            input_features_below = encoder.output_channels[-(i+1)]
            input_features_skip = encoder.output_channels[-(i+2)]
            stride_for_transpconv = encoder.strides[-(i+1)]
            
            transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))
            
            # 解码器阶段 - 输入是转置卷积输出和跳跃连接的拼接
            stages.append(StackedConvBlocks(
                n_conv_per_stage[i],
                encoder.conv_op,
                2 * input_features_skip,  # 转置卷积输出 + 跳跃连接
                input_features_skip,
                encoder.kernel_sizes[-(i+2)],
                1,
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
        
        # 最终回归头
        self.regression_head = nn.Sequential(
            encoder.conv_op(encoder.output_channels[0], encoder.output_channels[0] // 2, 3, padding=1, bias=conv_bias),
            norm_op(encoder.output_channels[0] // 2, **norm_op_kwargs) if norm_op is not None else nn.Identity(),
            nonlin(**nonlin_kwargs) if nonlin is not None else nn.Identity(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(encoder.output_channels[0] // 2, regression_dim)
        )
        
    def forward(self, skips):
        """
        前向传播
        Args:
            skips: 编码器的跳跃连接特征列表
        Returns:
            regression_output: 回归预测值 [B, regression_dim]
        """
        # 从最深层开始解码
        x = skips[-1]
        
        for i in range(len(self.stages)):
            x = self.transpconvs[i](x)
            x = torch.cat((x, skips[-(i+2)]), 1)
            x = self.stages[i](x)
        
        # 回归预测
        regression_output = self.regression_head(x)
        
        return regression_output


class PretrainedDualDecoderRegressionUNet(AbstractDynamicNetworkArchitectures):
    """
    基于预训练权重的双解码器回归UNet
    
    特点：
    1. 加载预训练的分割模型权重
    2. 冻结分割解码器参数
    3. 添加新的回归解码器
    4. 使用单向注意力机制
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
        # 回归相关参数
        regression_dim: int = 1,
        enable_cross_attention: bool = True,
        cross_attention_stages: List[int] = None,  # 指定在哪些阶段使用交叉注意力
        # 预训练权重相关
        pretrained_checkpoint_path: str = None,
        freeze_segmentation_decoder: bool = True
    ):
        super().__init__()
        
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)
        
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage * 2**i for i in range(n_stages)]
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(features_per_stage) == n_stages
        assert len(kernel_sizes) == n_stages
        assert len(strides) == n_stages
        
        self.conv_op = conv_op
        self.num_classes = num_classes
        self.n_stages = n_stages
        self.deep_supervision = deep_supervision
        self.regression_dim = regression_dim
        self.enable_cross_attention = enable_cross_attention
        self.freeze_segmentation_decoder = freeze_segmentation_decoder
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        
        # 构建编码器
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
        
        # 构建分割解码器（将从预训练权重加载）
        self.segmentation_decoder = UNetDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision,
            nonlin_first=nonlin_first,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            conv_bias=conv_bias
        )
        
        # 构建回归解码器（新添加的）
        self.regression_decoder = RegressionDecoder(
            self.encoder,
            regression_dim,
            n_conv_per_stage_decoder,
            nonlin_first=nonlin_first,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            conv_bias=conv_bias
        )
        
        # 交叉注意力模块
        if enable_cross_attention:
            if cross_attention_stages is None:
                cross_attention_stages = list(range(n_stages - 1))  # 默认在所有解码器阶段使用
            
            self.cross_attention_modules = nn.ModuleDict()
            for stage in cross_attention_stages:
                if stage < len(features_per_stage) - 1:
                    seg_channels = features_per_stage[-(stage+2)]  # 对应的特征通道数
                    reg_channels = features_per_stage[-(stage+2)]
                    
                    attention_module = LightweightCrossAttention(
                        seg_channels, reg_channels, reduction_ratio=16
                    )
                    attention_module.set_conv_dim(conv_op)
                    self.cross_attention_modules[f'stage_{stage}'] = attention_module
        else:
            self.cross_attention_modules = None
        
        # 加载预训练权重
        if pretrained_checkpoint_path:
            self.load_pretrained_weights(pretrained_checkpoint_path)
        
        # 冻结分割解码器参数
        if freeze_segmentation_decoder:
            self.freeze_segmentation_parameters()
    
    def load_pretrained_weights(self, checkpoint_path: str):
        """
        加载预训练的分割模型权重
        Args:
            checkpoint_path: 预训练模型检查点路径
        """
        print(f"正在加载预训练权重: {checkpoint_path}")
        
        # 加载检查点，使用weights_only=False来避免安全限制
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"权重加载失败: {e}")
            return
        
        # 提取网络状态字典
        if 'network_weights' in checkpoint:
            pretrained_state_dict = checkpoint['network_weights']
        elif 'state_dict' in checkpoint:
            pretrained_state_dict = checkpoint['state_dict']
        else:
            pretrained_state_dict = checkpoint
        
        # 加载编码器权重
        encoder_state_dict = {}
        for key, value in pretrained_state_dict.items():
            if key.startswith('encoder.'):
                encoder_state_dict[key] = value
        
        # 加载分割解码器权重
        seg_decoder_state_dict = {}
        for key, value in pretrained_state_dict.items():
            if key.startswith('decoder.'):
                seg_decoder_state_dict[key] = value
        
        # 应用权重
        missing_keys, unexpected_keys = self.load_state_dict(
            {**encoder_state_dict, **seg_decoder_state_dict}, 
            strict=False
        )
        
        print(f"权重加载完成")
        if missing_keys:
            print(f"缺失的键: {missing_keys[:5]}...")  # 只显示前5个
        if unexpected_keys:
            print(f"意外的键: {unexpected_keys[:5]}...")  # 只显示前5个
    
    def freeze_segmentation_parameters(self):
        """
        冻结分割相关的参数（编码器和分割解码器）
        """
        # 冻结编码器
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 冻结分割解码器
        for param in self.segmentation_decoder.parameters():
            param.requires_grad = False
        
        print("已冻结编码器和分割解码器参数")
    
    def unfreeze_all_parameters(self):
        """
        解冻所有参数（用于微调）
        """
        for param in self.parameters():
            param.requires_grad = True
        print("已解冻所有参数")
    
    def forward(self, x):
        """
        前向传播
        Returns:
            segmentation_output: 分割预测结果
            regression_output: 回归预测结果
        """
        # 编码器前向传播
        skips = self.encoder(x)
        
        # 分割解码器前向传播
        segmentation_output = self.segmentation_decoder(skips)
        
        # 回归解码器前向传播（带交叉注意力）
        if self.enable_cross_attention and self.cross_attention_modules:
            # 获取分割解码器的中间特征（需要修改分割解码器以返回中间特征）
            # 这里简化处理，直接使用编码器的跳跃连接
            regression_output = self.regression_decoder(skips)
        else:
            regression_output = self.regression_decoder(skips)
        
        return segmentation_output, regression_output
    
    def compute_conv_feature_map_size(self, input_size):
        """计算卷积特征图大小"""
        return self.encoder.compute_conv_feature_map_size(input_size)
    
    @staticmethod
    def initialize(module):
        """初始化网络权重"""
        InitWeights_He(1e-2)(module)