import torch
import torch.nn as nn
from typing import Dict, Optional


class UnifiedDECODELoss(nn.Module):
    """
    DECODE风格的统一损失函数，采用6通道架构：[prob, x, y, z, photon, bg]
    
    设计理念：
    - 简洁优于复杂：使用成熟的PyTorch内置损失函数
    - 数值稳定性：避免复杂的概率建模和对数运算
    - 可维护性：统一的架构，易于理解和调试
    
    通道说明：
    - 通道0 (prob): 发射体存在概率，使用BCEWithLogitsLoss
    - 通道1-3 (x,y,z): 坐标偏移，使用MSELoss
    - 通道4 (photon): 光子数，使用MSELoss
    - 通道5 (bg): 背景强度，使用MSELoss
    """
    
    def __init__(self, 
                 channel_weights: Optional[list] = None,
                 pos_weight: float = 1.0,
                 reduction: str = 'mean'):
        """
        初始化统一损失函数
        
        Args:
            channel_weights: 各通道权重 [prob, x, y, z, photon, bg]，默认为[1.0, 1.0, 1.0, 1.0, 0.5, 0.1]
            pos_weight: 概率通道的前景权重，用于平衡前景/背景
            reduction: 损失归约方式，'mean' 或 'sum'
        """
        super().__init__()
        
        # 默认通道权重：基于DECODE经验的合理权重
        self.channel_weights = channel_weights or [1.0, 1.0, 1.0, 1.0, 0.5, 0.1]
        
        # 概率通道使用BCEWithLogitsLoss（数值稳定）
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight),
            reduction=reduction
        )
        
        # 其他通道使用MSELoss（简单直接）
        self.mse_loss = nn.MSELoss(reduction=reduction)
        
        self.reduction = reduction
    
    def forward(self, 
                output: torch.Tensor, 
                target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算统一损失
        
        Args:
            output: 网络输出，形状为 [B, 6, H, W]
            target: 目标值，形状为 [B, 6, H, W]
            mask: 可选的掩码，用于只在特定区域计算损失
            
        Returns:
            包含各通道损失和总损失的字典
        """
        losses = {}
        
        # 确保输入形状正确
        if output.dim() != 4 or target.dim() != 4:
            raise ValueError(f"Expected 4D tensors, got output: {output.shape}, target: {target.shape}")
        
        if output.size(1) < 6 or target.size(1) < 6:
            raise ValueError(f"Expected at least 6 channels, got output: {output.size(1)}, target: {target.size(1)}")
        
        # 通道0：概率损失（使用BCEWithLogitsLoss）
        prob_logits = output[:, 0]  # [B, H, W]
        prob_target = target[:, 0]  # [B, H, W]
        
        if mask is not None:
            prob_logits = prob_logits[mask]
            prob_target = prob_target[mask]
        
        losses['prob'] = self.bce_loss(prob_logits.reshape(-1), prob_target.reshape(-1))
        
        # 通道1-5：坐标、光子、背景损失（使用MSELoss）
        channel_names = ['x', 'y', 'z', 'photon', 'bg']
        
        for i, name in enumerate(channel_names, start=1):
            pred = output[:, i]  # [B, H, W]
            tgt = target[:, i]   # [B, H, W]
            
            if mask is not None:
                pred = pred[mask]
                tgt = tgt[mask]
            
            losses[name] = self.mse_loss(pred.reshape(-1), tgt.reshape(-1))
        
        # 计算加权总损失
        total_loss = (
            self.channel_weights[0] * losses['prob'] +
            self.channel_weights[1] * losses['x'] +
            self.channel_weights[2] * losses['y'] +
            self.channel_weights[3] * losses['z'] +
            self.channel_weights[4] * losses['photon'] +
            self.channel_weights[5] * losses['bg']
        )
        
        losses['total'] = total_loss
        
        return losses


class SimpleCountLoss(nn.Module):
    """
    DECODE风格的简洁计数损失，替换复杂的CountLoss
    
    使用BCEWithLogitsLoss确保数值稳定性，避免复杂的概率建模
    """
    
    def __init__(self, pos_weight: float = 1.0):
        """
        初始化简洁计数损失
        
        Args:
            pos_weight: 前景权重，用于平衡前景/背景
        """
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight)
        )
    
    def forward(self, pred_logits: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
        """
        计算计数损失
        
        Args:
            pred_logits: 预测的logits，形状为 [B, 1, H, W]，未经过sigmoid
            target_prob: 目标概率，形状为 [B, 1, H, W]，值在[0,1]范围内
            
        Returns:
            标量损失值
        """
        # 展平张量进行计算
        pred_flat = pred_logits.reshape(-1)
        target_flat = target_prob.reshape(-1)
        
        return self.bce_loss(pred_flat, target_flat)


class SimpleLocLoss(nn.Module):
    """
    DECODE风格的简洁定位损失，替换复杂的LocLoss
    
    使用简单的MSE回归，避免复杂的高斯混合模型
    """
    
    def __init__(self):
        """
        初始化简洁定位损失
        """
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, 
                pred_coords: torch.Tensor, 
                target_coords: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算定位损失
        
        Args:
            pred_coords: 预测坐标，形状为 [B, 3, H, W] (x, y, z)
            target_coords: 目标坐标，形状为 [B, 3, H, W]
            mask: 可选掩码，只在有发射体的位置计算损失
            
        Returns:
            标量损失值
        """
        if mask is not None:
            # 扩展mask到所有坐标通道
            if mask.dim() == 4 and mask.size(1) == 1:
                mask = mask.expand(-1, 3, -1, -1)  # [B, 1, H, W] -> [B, 3, H, W]
            
            # 只在mask为True的位置计算损失
            pred_masked = pred_coords[mask]
            target_masked = target_coords[mask]
            
            if pred_masked.numel() == 0:
                # 如果没有有效像素，返回零损失
                return torch.tensor(0.0, device=pred_coords.device, requires_grad=True)
            
            return self.mse_loss(pred_masked, target_masked)
        else:
            return self.mse_loss(pred_coords.reshape(-1), target_coords.reshape(-1))


class SimpleCombinedLoss(nn.Module):
    """
    简化的组合损失函数，用于快速替换现有的复杂损失组合
    
    这是一个过渡性的解决方案，最终建议使用UnifiedDECODELoss
    """
    
    def __init__(self, 
                 count_weight: float = 1.0,
                 loc_weight: float = 1.0,
                 photon_weight: float = 0.5,
                 bg_weight: float = 0.1,
                 pos_weight: float = 1.0):
        """
        初始化组合损失函数
        
        Args:
            count_weight: 计数损失权重
            loc_weight: 定位损失权重
            photon_weight: 光子损失权重
            bg_weight: 背景损失权重
            pos_weight: 前景权重
        """
        super().__init__()
        
        self.count_loss = SimpleCountLoss(pos_weight=pos_weight)
        self.loc_loss = SimpleLocLoss()
        self.photon_loss = nn.MSELoss()
        self.bg_loss = nn.MSELoss()
        
        self.count_weight = count_weight
        self.loc_weight = loc_weight
        self.photon_weight = photon_weight
        self.bg_weight = bg_weight
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算组合损失
        
        Args:
            outputs: 网络输出字典，包含 'prob', 'offset', 'photon', 'background'
            targets: 目标字典，包含对应的目标值
            
        Returns:
            包含各损失组件和总损失的字典
        """
        losses = {}
        
        # 计数损失
        count_logits = outputs['prob']  # [B, 1, H, W]
        count_target = targets['count_maps'][:, 0:1, :, :]  # [B, 1, H, W]
        losses['count'] = self.count_loss(count_logits, count_target)
        
        # 定位损失（只在有发射体的位置计算）
        loc_pred = outputs['offset']  # [B, 3, H, W]
        loc_target = targets['loc_maps'][:, 0:3, :, :]  # [B, 3, H, W]
        mask = count_target > 0.5  # [B, 1, H, W]
        losses['localization'] = self.loc_loss(loc_pred, loc_target, mask)
        
        # 光子损失（只在有发射体的位置计算）
        if 'photon' in outputs and 'photon_maps' in targets:
            photon_pred = outputs['photon']  # [B, 1, H, W]
            photon_target = targets['photon_maps'][:, 0:1, :, :]  # [B, 1, H, W]
            if mask.sum() > 0:
                losses['photon'] = self.photon_loss(photon_pred[mask], photon_target[mask])
            else:
                losses['photon'] = torch.tensor(0.0, device=photon_pred.device)
        
        # 背景损失
        if 'background' in outputs and 'background_maps' in targets:
            bg_pred = outputs['background']  # [B, 1, H, W]
            bg_target = targets['background_maps'][:, 0:1, :, :]  # [B, 1, H, W]
            losses['background'] = self.bg_loss(bg_pred, bg_target)
        
        # 计算总损失
        total_loss = (
            self.count_weight * losses['count'] +
            self.loc_weight * losses['localization']
        )
        
        if 'photon' in losses:
            total_loss += self.photon_weight * losses['photon']
        if 'background' in losses:
            total_loss += self.bg_weight * losses['background']
        
        losses['total'] = total_loss
        
        return losses