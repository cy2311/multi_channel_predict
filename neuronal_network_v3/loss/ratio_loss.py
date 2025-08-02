"""比例预测损失函数

基于GaussianNLLLoss的比例预测损失函数，支持不确定性量化和物理约束。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union


class RatioGaussianNLLLoss(nn.Module):
    """基于GaussianNLLLoss的比例预测损失函数，支持不确定性量化
    
    Args:
        conservation_weight: 光子数守恒约束权重
        consistency_weight: 比例一致性约束权重
        eps: 数值稳定性参数
        reduction: 损失缩减方式
        full: 是否计算完整的负对数似然
    """
    
    def __init__(self, 
                 conservation_weight: float = 0.1, 
                 consistency_weight: float = 0.05,
                 eps: float = 1e-6, 
                 reduction: str = 'mean',
                 full: bool = False):
        super().__init__()
        
        self.gaussian_nll = nn.GaussianNLLLoss(eps=eps, reduction=reduction, full=full)
        self.conservation_weight = conservation_weight
        self.consistency_weight = consistency_weight
        
    def forward(self, 
                ratio_mean: torch.Tensor, 
                ratio_log_var: torch.Tensor, 
                target_ratio: torch.Tensor,
                photons_ch1: Optional[torch.Tensor] = None, 
                photons_ch2: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """前向传播
        
        Args:
            ratio_mean: 预测的比例均值，形状为 (B, 1)
            ratio_log_var: 预测的比例对数方差，形状为 (B, 1)
            target_ratio: 目标比例，形状为 (B, 1)
            photons_ch1: 通道1的光子数（可选），用于物理约束
            photons_ch2: 通道2的光子数（可选），用于物理约束
            
        Returns:
            总损失和损失组件字典
        """
        # 主要的高斯负对数似然损失
        ratio_var = torch.exp(ratio_log_var)
        nll_loss = self.gaussian_nll(ratio_mean, target_ratio, ratio_var)
        
        total_loss = nll_loss
        loss_dict = {'nll_loss': nll_loss.item() if nll_loss.dim() == 0 else nll_loss.mean().item()}
        
        # 可选的物理约束正则项
        if photons_ch1 is not None and photons_ch2 is not None:
            # 光子数守恒约束
            total_photons = photons_ch1 + photons_ch2
            predicted_ch1_from_ratio = total_photons * ratio_mean.squeeze()
            conservation_loss = F.mse_loss(predicted_ch1_from_ratio, photons_ch1)
            
            # 比例一致性约束
            ratio_from_photons = photons_ch1 / (total_photons + 1e-8)
            consistency_loss = F.mse_loss(ratio_mean.squeeze(), ratio_from_photons)
            
            total_loss = (nll_loss + 
                         self.conservation_weight * conservation_loss +
                         self.consistency_weight * consistency_loss)
            
            loss_dict.update({
                'conservation_loss': conservation_loss.item(),
                'consistency_loss': consistency_loss.item()
            })
        
        return total_loss, loss_dict


class MultiChannelLossWithGaussianRatio(nn.Module):
    """集成双通道独立损失和基于GaussianNLL的比例预测损失
    
    Args:
        loss_type: 单通道损失函数类型 ('PPXYZBLoss' 或 'GaussianMMLoss')
        ratio_loss_weight: 比例损失权重
        conservation_weight: 光子数守恒约束权重
        consistency_weight: 比例一致性约束权重
    """
    
    def __init__(self, 
                 loss_type: str = 'PPXYZBLoss', 
                 ratio_loss_weight: float = 0.1,
                 conservation_weight: float = 0.1,
                 consistency_weight: float = 0.05):
        super().__init__()
        
        self.ch1_loss = self._create_loss(loss_type)
        self.ch2_loss = self._create_loss(loss_type)
        self.ratio_loss = RatioGaussianNLLLoss(
            conservation_weight=conservation_weight,
            consistency_weight=consistency_weight
        )
        self.ratio_loss_weight = ratio_loss_weight
        
    def _create_loss(self, loss_type: str):
        """创建损失函数实例"""
        if loss_type == 'PPXYZBLoss':
            from .ppxyzb_loss import PPXYZBLoss
            return PPXYZBLoss()
        elif loss_type == 'GaussianMMLoss':
            from .gaussian_mm_loss import GaussianMMLoss
            return GaussianMMLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
    def forward(self, 
                pred_ch1: torch.Tensor, 
                pred_ch2: torch.Tensor, 
                ratio_mean: torch.Tensor, 
                ratio_log_var: torch.Tensor,
                target_ch1: torch.Tensor, 
                target_ch2: torch.Tensor, 
                target_ratio: torch.Tensor,
                weight_ch1: Optional[torch.Tensor] = None,
                weight_ch2: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """前向传播
        
        Args:
            pred_ch1: 通道1预测结果
            pred_ch2: 通道2预测结果
            ratio_mean: 比例均值预测
            ratio_log_var: 比例对数方差预测
            target_ch1: 通道1目标
            target_ch2: 通道2目标
            target_ratio: 目标比例
            weight_ch1: 通道1权重（可选）
            weight_ch2: 通道2权重（可选）
            
        Returns:
            总损失和损失组件字典
        """
        # 双通道独立损失
        if weight_ch1 is not None:
            loss_ch1 = self.ch1_loss(pred_ch1, target_ch1, weight_ch1)
        else:
            loss_ch1 = self.ch1_loss(pred_ch1, target_ch1)
            
        if weight_ch2 is not None:
            loss_ch2 = self.ch2_loss(pred_ch2, target_ch2, weight_ch2)
        else:
            loss_ch2 = self.ch2_loss(pred_ch2, target_ch2)
        
        # 处理损失函数返回值
        if isinstance(loss_ch1, dict):
            loss_ch1_value = loss_ch1['total_loss'] if 'total_loss' in loss_ch1 else loss_ch1['mean_total_loss']
        else:
            loss_ch1_value = loss_ch1
            
        if isinstance(loss_ch2, dict):
            loss_ch2_value = loss_ch2['total_loss'] if 'total_loss' in loss_ch2 else loss_ch2['mean_total_loss']
        else:
            loss_ch2_value = loss_ch2
        
        # 提取光子数用于物理约束（假设在通道1）
        photons_ch1 = pred_ch1[:, 1] if pred_ch1.shape[1] > 1 else None
        photons_ch2 = pred_ch2[:, 1] if pred_ch2.shape[1] > 1 else None
        
        # 比例预测损失（包含不确定性量化）
        ratio_loss, ratio_loss_dict = self.ratio_loss(
            ratio_mean, ratio_log_var, target_ratio, 
            photons_ch1, photons_ch2
        )
        
        # 总损失
        total_loss = loss_ch1_value + loss_ch2_value + self.ratio_loss_weight * ratio_loss
        
        # 构建损失字典
        loss_dict = {
            'total_loss': total_loss.item() if total_loss.dim() == 0 else total_loss.mean().item(),
            'ch1_loss': loss_ch1_value.item() if hasattr(loss_ch1_value, 'item') else float(loss_ch1_value),
            'ch2_loss': loss_ch2_value.item() if hasattr(loss_ch2_value, 'item') else float(loss_ch2_value),
            'ratio_loss': ratio_loss.item() if ratio_loss.dim() == 0 else ratio_loss.mean().item()
        }
        
        # 添加比例损失的详细信息
        for k, v in ratio_loss_dict.items():
            loss_dict[f'ratio_{k}'] = v
        
        return total_loss, loss_dict


class PhotonConservationLoss(nn.Module):
    """光子数守恒约束损失
    
    确保两个通道的光子数总和与预期的总光子数一致。
    
    Args:
        weight: 损失权重
        eps: 数值稳定性参数
    """
    
    def __init__(self, weight: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.weight = weight
        self.eps = eps
        
    def forward(self, 
                photons_ch1: torch.Tensor, 
                photons_ch2: torch.Tensor, 
                ratio: torch.Tensor, 
                total_photons: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            photons_ch1: 通道1的光子数
            photons_ch2: 通道2的光子数
            ratio: 通道1的光子数比例
            total_photons: 总光子数
            
        Returns:
            守恒损失
        """
        # 计算预测的总光子数
        pred_total = photons_ch1 + photons_ch2
        
        # 计算比例一致性
        ratio_consistency = torch.abs(
            photons_ch1 / (photons_ch1 + photons_ch2 + self.eps) - ratio
        )
        
        # 总光子数一致性
        total_consistency = torch.abs(pred_total - total_photons)
        
        return self.weight * (ratio_consistency.mean() + total_consistency.mean())