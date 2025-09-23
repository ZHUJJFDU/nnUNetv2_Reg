import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union


class ConsistencyLoss(nn.Module):
    """
    半监督学习中的一致性损失函数
    
    支持多种一致性损失计算方式：
    1. MSE损失：计算预测概率的均方误差
    2. KL散度：计算预测分布的KL散度
    3. 加权损失：基于置信度的加权损失
    """
    
    def __init__(self, 
                 loss_type: str = 'mse',
                 temperature: float = 1.0,
                 confidence_threshold: float = 0.95,
                 use_confidence_mask: bool = True):
        """
        Args:
            loss_type: 损失类型 ('mse', 'kl', 'ce')
            temperature: 温度参数，用于软化预测分布
            confidence_threshold: 置信度阈值
            use_confidence_mask: 是否使用置信度掩码
        """
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.use_confidence_mask = use_confidence_mask
        
    def forward(self, 
                student_logits: torch.Tensor, 
                teacher_logits: torch.Tensor,
                confidence_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算一致性损失
        
        Args:
            student_logits: 学生模型的logits输出
            teacher_logits: 教师模型的logits输出
            confidence_mask: 置信度掩码
            
        Returns:
            一致性损失值
        """
        # 应用温度参数
        student_probs = F.softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # 生成置信度掩码（如果未提供）
        if confidence_mask is None and self.use_confidence_mask:
            confidence_mask = self._generate_confidence_mask(teacher_probs)
            
        # 计算损失
        if self.loss_type == 'mse':
            loss = self._mse_loss(student_probs, teacher_probs, confidence_mask)
        elif self.loss_type == 'kl':
            loss = self._kl_loss(student_probs, teacher_probs, confidence_mask)
        elif self.loss_type == 'ce':
            loss = self._ce_loss(student_logits, teacher_probs, confidence_mask)
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")
            
        return loss
        
    def _generate_confidence_mask(self, teacher_probs: torch.Tensor) -> torch.Tensor:
        """生成基于置信度的掩码"""
        max_probs, _ = torch.max(teacher_probs, dim=1, keepdim=True)
        confidence_mask = (max_probs >= self.confidence_threshold).float()
        return confidence_mask
        
    def _mse_loss(self, 
                  student_probs: torch.Tensor, 
                  teacher_probs: torch.Tensor,
                  confidence_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """MSE一致性损失"""
        if confidence_mask is not None:
            # 应用置信度掩码
            student_probs = student_probs * confidence_mask
            teacher_probs = teacher_probs * confidence_mask
            
            # 计算有效像素数量
            valid_pixels = confidence_mask.sum()
            if valid_pixels > 0:
                loss = F.mse_loss(student_probs, teacher_probs, reduction='sum') / valid_pixels
            else:
                loss = torch.tensor(0.0, device=student_probs.device)
        else:
            loss = F.mse_loss(student_probs, teacher_probs, reduction='mean')
            
        return loss
        
    def _kl_loss(self, 
                 student_probs: torch.Tensor, 
                 teacher_probs: torch.Tensor,
                 confidence_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """KL散度一致性损失"""
        # 添加小的epsilon避免log(0)
        eps = 1e-8
        teacher_probs = teacher_probs + eps
        student_probs = student_probs + eps
        
        # 计算KL散度
        kl_div = teacher_probs * torch.log(teacher_probs / student_probs)
        
        if confidence_mask is not None:
            # 应用置信度掩码
            kl_div = kl_div * confidence_mask
            valid_pixels = confidence_mask.sum()
            if valid_pixels > 0:
                loss = kl_div.sum() / valid_pixels
            else:
                loss = torch.tensor(0.0, device=student_probs.device)
        else:
            loss = kl_div.mean()
            
        return loss
        
    def _ce_loss(self, 
                 student_logits: torch.Tensor, 
                 teacher_probs: torch.Tensor,
                 confidence_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """交叉熵一致性损失"""
        # 使用教师概率作为软标签
        log_student_probs = F.log_softmax(student_logits, dim=1)
        ce_loss = -(teacher_probs * log_student_probs).sum(dim=1, keepdim=True)
        
        if confidence_mask is not None:
            # 应用置信度掩码
            ce_loss = ce_loss * confidence_mask
            valid_pixels = confidence_mask.sum()
            if valid_pixels > 0:
                loss = ce_loss.sum() / valid_pixels
            else:
                loss = torch.tensor(0.0, device=student_logits.device)
        else:
            loss = ce_loss.mean()
            
        return loss


class EMAUpdater:
    """
    指数移动平均（EMA）权重更新器
    
    用于更新教师模型的权重，使其成为学生模型权重的平滑版本
    """
    
    def __init__(self, decay: float = 0.999, warmup_steps: int = 0):
        """
        Args:
            decay: EMA衰减率
            warmup_steps: 预热步数，在此期间衰减率会逐渐增加
        """
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
    def update(self, teacher_model: nn.Module, student_model: nn.Module):
        """
        更新教师模型权重
        
        Args:
            teacher_model: 教师模型
            student_model: 学生模型
        """
        self.step_count += 1
        
        # 计算当前的衰减率（考虑预热）
        current_decay = self._get_current_decay()
        
        with torch.no_grad():
            for teacher_param, student_param in zip(teacher_model.parameters(), 
                                                   student_model.parameters()):
                teacher_param.data = (current_decay * teacher_param.data + 
                                    (1 - current_decay) * student_param.data)
                                    
    def _get_current_decay(self) -> float:
        """获取当前的衰减率（考虑预热）"""
        if self.step_count <= self.warmup_steps:
            # 预热期间，衰减率从0逐渐增加到目标值
            return self.decay * (self.step_count / self.warmup_steps)
        else:
            return self.decay
            
    def reset(self):
        """重置步数计数器"""
        self.step_count = 0


class ConsistencyWeightScheduler:
    """
    一致性损失权重调度器
    
    支持多种权重调度策略：
    1. 线性增长
    2. 指数增长
    3. 余弦增长
    4. 固定权重
    """
    
    def __init__(self, 
                 max_weight: float = 1.0,
                 rampup_type: str = 'linear',
                 rampup_length: int = 100,
                 start_epoch: int = 0):
        """
        Args:
            max_weight: 最大权重值
            rampup_type: 增长类型 ('linear', 'exp', 'cosine', 'fixed')
            rampup_length: 增长期长度（epoch数）
            start_epoch: 开始增长的epoch
        """
        self.max_weight = max_weight
        self.rampup_type = rampup_type
        self.rampup_length = rampup_length
        self.start_epoch = start_epoch
        
    def get_weight(self, epoch: int) -> float:
        """
        获取当前epoch的一致性损失权重
        
        Args:
            epoch: 当前epoch
            
        Returns:
            一致性损失权重
        """
        if epoch < self.start_epoch:
            return 0.0
            
        if self.rampup_type == 'fixed':
            return self.max_weight
            
        # 计算增长进度
        progress = min(1.0, (epoch - self.start_epoch) / self.rampup_length)
        
        if self.rampup_type == 'linear':
            weight = progress
        elif self.rampup_type == 'exp':
            weight = progress ** 2
        elif self.rampup_type == 'cosine':
            weight = 0.5 * (1 - np.cos(np.pi * progress))
        else:
            raise ValueError(f"不支持的增长类型: {self.rampup_type}")
            
        return self.max_weight * weight


class PseudoLabelGenerator:
    """
    伪标签生成器
    
    从教师模型的预测中生成高质量的伪标签
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.95,
                 temperature: float = 1.0,
                 use_entropy_filter: bool = True,
                 entropy_threshold: float = 0.5):
        """
        Args:
            confidence_threshold: 置信度阈值
            temperature: 温度参数
            use_entropy_filter: 是否使用熵过滤
            entropy_threshold: 熵阈值
        """
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        self.use_entropy_filter = use_entropy_filter
        self.entropy_threshold = entropy_threshold
        
    def generate(self, teacher_logits: torch.Tensor) -> tuple:
        """
        生成伪标签和置信度掩码
        
        Args:
            teacher_logits: 教师模型的logits输出
            
        Returns:
            (pseudo_labels, confidence_mask, quality_mask)
        """
        with torch.no_grad():
            # 应用温度参数
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
            
            # 获取最大概率和对应的类别
            max_probs, pseudo_labels = torch.max(teacher_probs, dim=1)
            
            # 生成置信度掩码
            confidence_mask = (max_probs >= self.confidence_threshold).float()
            
            # 生成质量掩码（结合熵过滤）
            quality_mask = confidence_mask
            
            if self.use_entropy_filter:
                # 计算预测熵
                entropy = -torch.sum(teacher_probs * torch.log(teacher_probs + 1e-8), dim=1)
                entropy_mask = (entropy <= self.entropy_threshold).float()
                quality_mask = confidence_mask * entropy_mask
                
            return pseudo_labels, confidence_mask, quality_mask