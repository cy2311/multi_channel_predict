import torch
import torch.nn as nn


class BackgroundLoss(nn.Module):
    """
    背景损失函数，计算预测背景图和真实背景图之间的均方误差。
    
    损失定义为：
    L_bg = Σ(B_k^GT - B_k^pred)²
    
    其中：
    - B_k^GT 是第k个像素的真实背景值
    - B_k^pred 是第k个像素的预测背景值
    """
    
    def __init__(self, eps: float = 1e-6):
        """
        初始化背景损失函数
        
        Args:
            eps: 小常数，用于防止数值不稳定性
        """
        super().__init__()
        self.eps = eps
        
    def forward(self, pred_background: torch.Tensor, true_background: torch.Tensor) -> torch.Tensor:
        """
        计算背景损失
        
        Args:
            pred_background: 预测的背景图，形状为 (B, 1, H, W)
            true_background: 真实的背景图，形状为 (B, 1, H, W)
            
        Returns:
            标量损失值（批次平均）
        """
        # 确保输入张量有梯度
        if not pred_background.requires_grad:
            pred_background.requires_grad_(True)
        
        # 计算均方误差
        mse = torch.mean((true_background - pred_background) ** 2)
        
        return mse