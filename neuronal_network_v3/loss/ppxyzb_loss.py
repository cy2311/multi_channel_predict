"""PPXYZBLoss实现

6通道损失函数，用于DECODE网络的训练：
- 检测损失：BCEWithLogitsLoss用于概率预测
- 回归损失：MSELoss用于光子数、坐标和背景预测
支持通道权重和像素权重。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, List


class PPXYZBLoss(nn.Module):
    """PPXYZB损失函数（6通道）
    
    损失组成：
    - 检测损失：BCEWithLogitsLoss用于概率预测（通道0）
    - 回归损失：MSELoss用于其他通道（光子数、坐标、背景）
    
    Args:
        chweight_stat: 通道权重，长度为6的列表或张量
        p_fg_weight: 前景权重，用于平衡正负样本
        reduction: 损失缩减方式 ('mean', 'sum', 'none')
        eps: 小常数，防止数值不稳定
        use_focal_loss: 是否使用Focal Loss替代BCE
        focal_alpha: Focal Loss的alpha参数
        focal_gamma: Focal Loss的gamma参数
    """
    
    def __init__(self,
                 chweight_stat: Optional[Union[List[float], torch.Tensor]] = None,
                 p_fg_weight: float = 1.0,
                 reduction: str = 'mean',
                 eps: float = 1e-6,
                 use_focal_loss: bool = False,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        super().__init__()
        
        self.reduction = reduction
        self.eps = eps
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # 通道权重
        if chweight_stat is None:
            chweight_stat = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        
        if isinstance(chweight_stat, list):
            chweight_stat = torch.tensor(chweight_stat, dtype=torch.float32)
        
        self.register_buffer('chweight_stat', chweight_stat)
        
        # 前景权重
        if p_fg_weight != 1.0:
            self.register_buffer('pos_weight', torch.tensor(p_fg_weight))
        else:
            self.pos_weight = None
        
        # 损失函数
        if use_focal_loss:
            self.detection_loss_fn = self._focal_loss
        else:
            self.detection_loss_fn = nn.BCEWithLogitsLoss(
                reduction='none',
                pos_weight=self.pos_weight
            )
        
        self.regression_loss_fn = nn.MSELoss(reduction='none')
    
    def forward(self,
                output: torch.Tensor,
                target: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播
        
        Args:
            output: 模型输出，形状为 (B, 6, H, W)
            target: 目标张量，形状为 (B, 6, H, W)
            weight: 像素权重，形状为 (B, 1, H, W) 或 (B, 6, H, W)
            
        Returns:
            如果reduction='none'，返回详细损失字典
            否则返回标量损失
        """
        if output.size(1) != 6 or target.size(1) != 6:
            raise ValueError("PPXYZBLoss expects 6-channel input")
        
        batch_size = output.size(0)
        
        # 检测损失（通道0）
        p_output = output[:, 0:1]  # (B, 1, H, W)
        p_target = target[:, 0:1]  # (B, 1, H, W)
        
        if self.use_focal_loss:
            p_loss = self.detection_loss_fn(p_output, p_target)
        else:
            p_loss = self.detection_loss_fn(p_output, p_target)
        
        # 回归损失（通道1-5）
        reg_output = output[:, 1:6]  # (B, 5, H, W)
        reg_target = target[:, 1:6]  # (B, 5, H, W)
        reg_loss = self.regression_loss_fn(reg_output, reg_target)  # (B, 5, H, W)
        
        # 合并损失
        total_loss = torch.cat([p_loss, reg_loss], dim=1)  # (B, 6, H, W)
        
        # 应用通道权重
        ch_weight = self.chweight_stat.view(1, 6, 1, 1).to(total_loss.device)
        total_loss = total_loss * ch_weight
        
        # 应用像素权重
        if weight is not None:
            if weight.size(1) == 1:
                # 广播到所有通道
                weight = weight.expand(-1, 6, -1, -1)
            total_loss = total_loss * weight
        
        # 损失缩减
        if self.reduction == 'none':
            return {
                'total_loss': total_loss,
                'detection_loss': p_loss * ch_weight[:, 0:1],
                'regression_loss': reg_loss * ch_weight[:, 1:6],
                'per_channel_loss': total_loss
            }
        elif self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
    
    def _focal_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal Loss实现"""
        # 计算BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(
            output, target, reduction='none'
        )
        
        # 计算概率
        p = torch.sigmoid(output)
        p_t = p * target + (1 - p) * (1 - target)
        
        # 计算alpha权重
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        
        # 计算focal权重
        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma
        
        # 应用focal权重
        focal_loss = focal_weight * bce_loss
        
        return focal_loss
    
    def get_loss_weights(self) -> Dict[str, torch.Tensor]:
        """获取损失权重信息"""
        return {
            'channel_weights': self.chweight_stat,
            'pos_weight': self.pos_weight if self.pos_weight is not None else torch.tensor(1.0)
        }
    
    def set_channel_weights(self, weights: Union[List[float], torch.Tensor]):
        """设置通道权重"""
        if isinstance(weights, list):
            weights = torch.tensor(weights, dtype=torch.float32)
        
        if len(weights) != 6:
            raise ValueError("Channel weights must have length 6")
        
        self.chweight_stat = weights.to(self.chweight_stat.device)


class AdaptivePPXYZBLoss(PPXYZBLoss):
    """自适应PPXYZB损失
    
    在标准PPXYZBLoss基础上添加：
    - 动态权重调整
    - 难样本挖掘
    - 类别平衡
    """
    
    def __init__(self,
                 *args,
                 adaptive_weight: bool = True,
                 hard_mining: bool = True,
                 mining_ratio: float = 0.3,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.adaptive_weight = adaptive_weight
        self.hard_mining = hard_mining
        self.mining_ratio = mining_ratio
        
        # 动态权重历史
        self.register_buffer('weight_history', torch.ones(6))
        self.register_buffer('loss_history', torch.zeros(6))
        self.update_count = 0
    
    def forward(self,
                output: torch.Tensor,
                target: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """自适应前向传播"""
        # 计算基础损失
        loss_dict = super().forward(output, target, weight)
        
        if isinstance(loss_dict, dict):
            total_loss = loss_dict['total_loss']
            per_channel_loss = loss_dict['per_channel_loss']
        else:
            # 如果返回标量，重新计算详细损失
            self.reduction = 'none'
            loss_dict = super().forward(output, target, weight)
            total_loss = loss_dict['total_loss']
            per_channel_loss = loss_dict['per_channel_loss']
        
        # 难样本挖掘
        if self.hard_mining:
            total_loss = self._hard_mining(total_loss)
        
        # 更新动态权重
        if self.adaptive_weight:
            self._update_adaptive_weights(per_channel_loss)
        
        if self.reduction == 'none':
            loss_dict['total_loss'] = total_loss
            return loss_dict
        elif self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
    
    def _hard_mining(self, loss: torch.Tensor) -> torch.Tensor:
        """难样本挖掘"""
        B, C, H, W = loss.shape
        
        # 计算每个像素的总损失
        pixel_loss = loss.sum(dim=1)  # (B, H, W)
        
        # 选择最难的样本
        num_hard = int(H * W * self.mining_ratio)
        
        for b in range(B):
            pixel_loss_flat = pixel_loss[b].flatten()  # (H*W,)
            _, hard_indices = torch.topk(pixel_loss_flat, num_hard)
            
            # 创建掩码
            mask = torch.zeros_like(pixel_loss_flat)
            mask[hard_indices] = 1.0
            mask = mask.view(H, W)
            
            # 应用掩码
            loss[b] = loss[b] * mask.unsqueeze(0)
        
        return loss
    
    def _update_adaptive_weights(self, per_channel_loss: torch.Tensor):
        """更新自适应权重"""
        # 计算每个通道的平均损失
        channel_loss = per_channel_loss.mean(dim=(0, 2, 3))  # (6,)
        
        # 更新历史
        self.loss_history = 0.9 * self.loss_history + 0.1 * channel_loss
        
        # 计算新权重（损失越大，权重越小）
        if self.update_count > 10:  # 等待一些更新后再调整
            max_loss = self.loss_history.max()
            new_weights = max_loss / (self.loss_history + self.eps)
            
            # 平滑更新
            self.chweight_stat = 0.95 * self.chweight_stat + 0.05 * new_weights
        
        self.update_count += 1


class MultiScalePPXYZBLoss(nn.Module):
    """多尺度PPXYZB损失
    
    在不同尺度上计算损失，提高多尺度特征学习
    """
    
    def __init__(self,
                 scales: List[float] = [1.0, 0.5, 0.25],
                 scale_weights: Optional[List[float]] = None,
                 **loss_kwargs):
        super().__init__()
        
        self.scales = scales
        
        if scale_weights is None:
            scale_weights = [1.0] * len(scales)
        
        self.register_buffer('scale_weights', torch.tensor(scale_weights))
        
        # 为每个尺度创建损失函数
        self.loss_functions = nn.ModuleList([
            PPXYZBLoss(**loss_kwargs) for _ in scales
        ])
    
    def forward(self,
                output: torch.Tensor,
                target: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """多尺度前向传播"""
        total_loss = 0.0
        
        for i, (scale, loss_fn) in enumerate(zip(self.scales, self.loss_functions)):
            if scale != 1.0:
                # 下采样到指定尺度
                scaled_output = F.interpolate(
                    output, scale_factor=scale, mode='bilinear', align_corners=False
                )
                scaled_target = F.interpolate(
                    target, scale_factor=scale, mode='bilinear', align_corners=False
                )
                
                if weight is not None:
                    scaled_weight = F.interpolate(
                        weight, scale_factor=scale, mode='bilinear', align_corners=False
                    )
                else:
                    scaled_weight = None
            else:
                scaled_output = output
                scaled_target = target
                scaled_weight = weight
            
            # 计算该尺度的损失
            scale_loss = loss_fn(scaled_output, scaled_target, scaled_weight)
            
            # 加权累加
            total_loss += self.scale_weights[i] * scale_loss
        
        return total_loss