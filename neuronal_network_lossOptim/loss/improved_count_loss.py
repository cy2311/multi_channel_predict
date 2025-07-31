import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class ImprovedCountLoss(nn.Module):
    """
    改进的计数损失函数，使用BCEWithLogitsLoss替代BCELoss以提高数值稳定性，
    并支持前景权重机制来处理类别不平衡问题。
    
    主要改进：
    1. 使用BCEWithLogitsLoss替代BCELoss，避免sigmoid饱和问题
    2. 支持pos_weight参数处理前景/背景不平衡
    3. 支持像素级权重图进行精细控制
    4. 支持通道级权重进行多尺度平衡
    """
    
    def __init__(self, 
                 pos_weight: Optional[float] = None,
                 pixel_weight_strategy: str = 'none',
                 channel_weights: Optional[torch.Tensor] = None,
                 eps: float = 1e-6):
        """
        初始化改进的计数损失函数
        
        Args:
            pos_weight: 前景权重，用于平衡正负样本。None表示不使用
            pixel_weight_strategy: 像素权重策略 ('none', 'distance', 'density', 'adaptive')
            channel_weights: 通道级权重，形状为[C]，用于多通道平衡
            eps: 小常数，防止数值不稳定
        """
        super().__init__()
        self.eps = eps
        self.pixel_weight_strategy = pixel_weight_strategy
        
        # 前景权重
        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor(pos_weight))
        else:
            self.pos_weight = None
            
        # 通道权重
        if channel_weights is not None:
            self.register_buffer('channel_weights', channel_weights)
        else:
            self.channel_weights = None
            
        # BCE损失函数（使用logits版本）
        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction='none',
            pos_weight=self.pos_weight
        )
        
    def generate_pixel_weights(self, 
                              target: torch.Tensor, 
                              strategy: str = 'distance') -> torch.Tensor:
        """
        生成像素级权重图
        
        Args:
            target: 目标张量，形状为[B, C, H, W]
            strategy: 权重生成策略
            
        Returns:
            权重图，形状为[B, C, H, W]
        """
        if strategy == 'none' or strategy is None:
            return torch.ones_like(target)
            
        elif strategy == 'distance':
            # 基于距离的权重：离前景像素越近权重越高
            weights = torch.ones_like(target)
            B, C, H, W = target.shape
            
            for b in range(B):
                for c in range(C):
                    target_slice = target[b, c]  # [H, W]
                    # 找到前景像素位置
                    fg_positions = torch.nonzero(target_slice > 0.5, as_tuple=False)  # [N, 2]
                    
                    if len(fg_positions) > 0:
                        # 创建坐标网格
                        y_coords, x_coords = torch.meshgrid(
                            torch.arange(H, device=target.device),
                            torch.arange(W, device=target.device),
                            indexing='ij'
                        )
                        coords = torch.stack([y_coords, x_coords], dim=-1)  # [H, W, 2]
                        
                        # 计算到最近前景像素的距离
                        min_distances = torch.full((H, W), float('inf'), device=target.device)
                        for pos in fg_positions:
                            dist = torch.norm(coords - pos.float(), dim=-1)  # [H, W]
                            min_distances = torch.min(min_distances, dist)
                        
                        # 转换为权重（距离越近权重越高）
                        weights[b, c] = torch.exp(-min_distances / 5.0)  # 可调参数
                        
            return weights
            
        elif strategy == 'density':
            # 基于密度的权重：前景像素密度越高权重越高
            weights = torch.ones_like(target)
            kernel_size = 5
            padding = kernel_size // 2
            
            # 使用卷积计算局部密度
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=target.device) / (kernel_size ** 2)
            density = F.conv2d(target, kernel, padding=padding)
            
            # 归一化密度作为权重
            weights = 1.0 + 2.0 * density  # 基础权重1.0，密度贡献最多2.0
            
            return weights
            
        elif strategy == 'adaptive':
            # 自适应权重：结合距离和密度
            distance_weights = self.generate_pixel_weights(target, 'distance')
            density_weights = self.generate_pixel_weights(target, 'density')
            
            # 加权组合
            weights = 0.6 * distance_weights + 0.4 * density_weights
            
            return weights
            
        else:
            raise ValueError(f"未知的权重策略: {strategy}")
    
    def forward(self, 
                logits: torch.Tensor, 
                target: torch.Tensor,
                pixel_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算改进的计数损失
        
        Args:
            logits: 网络输出的logits，形状为[B, C, H, W]，未经过sigmoid
            target: 目标概率图，形状为[B, C, H, W]，值在[0,1]之间
            pixel_weights: 可选的像素级权重，形状为[B, C, H, W]
            
        Returns:
            标量损失值
        """
        # 确保输入张量有梯度
        if not logits.requires_grad:
            logits.requires_grad_(True)
            
        # 计算基础BCE损失（使用logits版本）
        bce_loss = self.bce_loss(logits, target)  # [B, C, H, W]
        
        # 应用像素级权重
        if pixel_weights is None and self.pixel_weight_strategy != 'none':
            pixel_weights = self.generate_pixel_weights(target, self.pixel_weight_strategy)
            
        if pixel_weights is not None:
            bce_loss = bce_loss * pixel_weights
            
        # 应用通道级权重
        if self.channel_weights is not None:
            # 确保通道权重维度匹配
            C = bce_loss.size(1)
            if len(self.channel_weights) != C:
                raise ValueError(f"通道权重维度({len(self.channel_weights)})与输入通道数({C})不匹配")
                
            # 广播通道权重
            channel_weights = self.channel_weights.view(1, -1, 1, 1)  # [1, C, 1, 1]
            bce_loss = bce_loss * channel_weights
            
        # 计算平均损失
        loss = torch.mean(bce_loss)
        
        # 数值稳定性检查
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN or Inf detected in improved count loss")
            # 返回一个有效的损失值
            return torch.zeros(1, device=logits.device, requires_grad=True)
            
        return loss


class MultiLevelLoss(nn.Module):
    """
    多层次损失函数，整合计数、定位、光子和背景损失，
    支持DECODE原始库风格的多级权重机制。
    """
    
    def __init__(self,
                 count_pos_weight: Optional[float] = 2.0,
                 pixel_weight_strategy: str = 'adaptive',
                 channel_weights: Optional[torch.Tensor] = None,
                 loss_weights: Optional[dict] = None):
        """
        初始化多层次损失函数
        
        Args:
            count_pos_weight: 计数损失的前景权重
            pixel_weight_strategy: 像素权重策略
            channel_weights: 通道级权重
            loss_weights: 各损失组件的权重字典
        """
        super().__init__()
        
        # 改进的计数损失
        self.count_loss = ImprovedCountLoss(
            pos_weight=count_pos_weight,
            pixel_weight_strategy=pixel_weight_strategy,
            channel_weights=channel_weights
        )
        
        # 其他损失函数
        self.loc_loss = nn.MSELoss()
        self.photon_loss = nn.MSELoss()
        self.background_loss = nn.MSELoss()
        
        # 损失权重
        self.loss_weights = loss_weights or {
            'count': 1.0,
            'localization': 1.0,
            'photon': 0.5,
            'background': 0.1
        }
        
    def forward(self, 
                outputs: dict, 
                targets: dict,
                pixel_weights: Optional[torch.Tensor] = None) -> dict:
        """
        计算多层次损失
        
        Args:
            outputs: 网络输出字典
            targets: 目标字典
            pixel_weights: 可选的像素级权重
            
        Returns:
            损失字典，包含各组件损失和总损失
        """
        losses = {}
        
        # 计数损失（使用logits）
        if 'prob' in outputs and 'count_maps' in targets:
            count_logits = outputs['prob']  # 未经sigmoid的logits
            count_target = targets['count_maps'][:, 0:1, :, :]  # 取第一帧
            losses['count'] = self.count_loss(count_logits, count_target, pixel_weights)
        
        # 定位损失（仅在有发射器的位置计算）
        if 'offset' in outputs and 'loc_maps' in targets:
            loc_pred = outputs['offset']
            loc_target = targets['loc_maps'][:, 0:3, :, :]
            
            # 创建掩码
            if 'count_maps' in targets:
                mask = targets['count_maps'][:, 0:1, :, :] > 0.5
                mask = mask.expand(-1, 3, -1, -1)
                
                if mask.sum() > 0:
                    losses['localization'] = self.loc_loss(loc_pred[mask], loc_target[mask])
                else:
                    losses['localization'] = torch.tensor(0.0, device=loc_pred.device)
            else:
                losses['localization'] = self.loc_loss(loc_pred, loc_target)
        
        # 光子损失
        if 'photon' in outputs and 'photon_maps' in targets:
            photon_pred = outputs['photon']
            photon_target = targets['photon_maps'][:, 0:1, :, :]
            
            # 创建掩码（仅在有发射器的位置计算）
            if 'count_maps' in targets:
                mask = targets['count_maps'][:, 0:1, :, :] > 0.5
                
                if mask.sum() > 0:
                    losses['photon'] = self.photon_loss(photon_pred[mask], photon_target[mask])
                else:
                    losses['photon'] = torch.tensor(0.0, device=photon_pred.device)
            else:
                losses['photon'] = self.photon_loss(photon_pred, photon_target)
        
        # 背景损失
        if 'background' in outputs and 'background_maps' in targets:
            bg_pred = outputs['background']
            bg_target = targets['background_maps'][:, 0:1, :, :]
            losses['background'] = self.background_loss(bg_pred, bg_target)
        
        # 计算总损失
        total_loss = torch.tensor(0.0, device=next(iter(outputs.values())).device)
        for loss_name, loss_value in losses.items():
            if loss_name in self.loss_weights:
                total_loss += self.loss_weights[loss_name] * loss_value
        
        losses['total'] = total_loss
        
        return losses


class WeightGenerator:
    """
    权重生成器，用于动态生成像素级和通道级权重
    """
    
    @staticmethod
    def generate_adaptive_pos_weight(target: torch.Tensor, 
                                   min_weight: float = 1.0, 
                                   max_weight: float = 10.0) -> float:
        """
        根据目标分布自适应生成前景权重
        
        Args:
            target: 目标张量
            min_weight: 最小权重
            max_weight: 最大权重
            
        Returns:
            自适应前景权重
        """
        # 计算前景像素比例
        fg_ratio = torch.mean((target > 0.5).float())
        
        if fg_ratio < 1e-6:  # 避免除零
            return max_weight
            
        # 计算权重：前景越少，权重越高
        bg_ratio = 1.0 - fg_ratio
        adaptive_weight = bg_ratio / fg_ratio
        
        # 限制在合理范围内
        adaptive_weight = torch.clamp(adaptive_weight, min_weight, max_weight)
        
        return adaptive_weight.item()
    
    @staticmethod
    def generate_channel_weights(num_channels: int, 
                               strategy: str = 'equal') -> torch.Tensor:
        """
        生成通道级权重
        
        Args:
            num_channels: 通道数
            strategy: 权重策略 ('equal', 'decreasing', 'custom')
            
        Returns:
            通道权重张量
        """
        if strategy == 'equal':
            return torch.ones(num_channels)
        elif strategy == 'decreasing':
            # 递减权重，第一个通道权重最高
            weights = torch.linspace(1.0, 0.5, num_channels)
            return weights
        else:
            raise ValueError(f"未知的通道权重策略: {strategy}")