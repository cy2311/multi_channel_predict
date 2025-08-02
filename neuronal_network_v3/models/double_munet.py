"""DoubleMUnet实现

DECODE的核心双重U-Net架构，包含：
- 共享U-Net：处理单个输入通道
- 联合U-Net：处理多通道特征融合
支持1或3个输入通道，可配置的深度参数。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

from .unet2d import UNet2d, DoubleConvBlock


class DoubleMUnet(nn.Module):
    """双重U-Net架构
    
    包含两个U-Net：
    1. 共享U-Net：处理每个输入通道，生成特征表示
    2. 联合U-Net：融合所有通道的特征，生成最终输出
    
    Args:
        channels_in: 输入通道数 (1 或 3)
        channels_out: 输出通道数
        depth_shared: 共享U-Net的深度
        depth_union: 联合U-Net的深度
        initial_features: 初始特征数
        inter_features: 中间特征数（共享U-Net的输出通道数）
        norm: 归一化方法
        activation: 激活函数
        pool_mode: 池化模式
        upsample_mode: 上采样模式
        final_activation: 最终激活函数
    """
    
    def __init__(self,
                 channels_in: int = 1,
                 channels_out: int = 10,
                 depth_shared: int = 3,
                 depth_union: int = 3,
                 initial_features: int = 64,
                 inter_features: int = 64,
                 norm: str = 'GroupNorm',
                 activation: str = 'ReLU',
                 pool_mode: str = 'StrideConv',
                 upsample_mode: str = 'bilinear',
                 final_activation: Optional[str] = None):
        super().__init__()
        
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.depth_shared = depth_shared
        self.depth_union = depth_union
        
        # 共享U-Net：处理每个输入通道
        self.shared_unet = UNet2d(
            channels_in=1,  # 每次处理单个通道
            channels_out=inter_features,
            depth=depth_shared,
            initial_features=initial_features,
            norm=norm,
            activation=activation,
            pool_mode=pool_mode,
            upsample_mode=upsample_mode,
            final_activation=None  # 中间特征不需要激活
        )
        
        # 联合U-Net：融合所有通道特征
        union_input_channels = channels_in * inter_features
        self.union_unet = UNet2d(
            channels_in=union_input_channels,
            channels_out=channels_out,
            depth=depth_union,
            initial_features=initial_features,
            norm=norm,
            activation=activation,
            pool_mode=pool_mode,
            upsample_mode=upsample_mode,
            final_activation=final_activation
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 (B, C, H, W)，其中 C = channels_in
            
        Returns:
            输出张量，形状为 (B, channels_out, H, W)
        """
        batch_size, channels, height, width = x.shape
        
        if channels != self.channels_in:
            raise ValueError(f"Expected {self.channels_in} input channels, got {channels}")
        
        # 第一阶段：共享U-Net处理每个通道
        shared_features = []
        
        for i in range(self.channels_in):
            # 提取单个通道 (B, 1, H, W)
            single_channel = x[:, i:i+1, :, :]
            
            # 通过共享U-Net处理
            features = self.shared_unet(single_channel)  # (B, inter_features, H, W)
            shared_features.append(features)
        
        # 第二阶段：联合U-Net融合所有特征
        if self.channels_in == 1:
            # 单通道输入，直接使用特征
            union_input = shared_features[0]
        else:
            # 多通道输入，沿通道维度连接
            union_input = torch.cat(shared_features, dim=1)  # (B, channels_in * inter_features, H, W)
        
        # 通过联合U-Net生成最终输出
        output = self.union_unet(union_input)  # (B, channels_out, H, W)
        
        return output
    
    def get_shared_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """获取共享U-Net的特征表示（用于分析和可视化）
        
        Args:
            x: 输入张量，形状为 (B, C, H, W)
            
        Returns:
            每个通道的特征列表，每个元素形状为 (B, inter_features, H, W)
        """
        batch_size, channels, height, width = x.shape
        
        if channels != self.channels_in:
            raise ValueError(f"Expected {self.channels_in} input channels, got {channels}")
        
        shared_features = []
        
        for i in range(self.channels_in):
            single_channel = x[:, i:i+1, :, :]
            features = self.shared_unet(single_channel)
            shared_features.append(features)
        
        return shared_features
    
    def get_parameter_count(self) -> dict:
        """获取模型参数统计信息"""
        shared_params = sum(p.numel() for p in self.shared_unet.parameters())
        union_params = sum(p.numel() for p in self.union_unet.parameters())
        total_params = shared_params + union_params
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'shared_unet_parameters': shared_params,
            'union_unet_parameters': union_params,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_sharing_ratio': shared_params / total_params if total_params > 0 else 0
        }
    
    def get_model_info(self) -> dict:
        """获取模型详细信息"""
        param_info = self.get_parameter_count()
        
        # 计算模型大小
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        return {
            **param_info,
            'model_size_mb': param_size / 1024 / 1024,
            'architecture': {
                'channels_in': self.channels_in,
                'channels_out': self.channels_out,
                'depth_shared': self.depth_shared,
                'depth_union': self.depth_union,
                'shared_unet_info': self.shared_unet.get_parameter_count(),
                'union_unet_info': self.union_unet.get_parameter_count()
            }
        }


class AdaptiveDoubleMUnet(DoubleMUnet):
    """自适应双重U-Net
    
    在DoubleMUnet基础上添加：
    - 自适应特征融合
    - 注意力机制
    - 残差连接
    """
    
    def __init__(self,
                 channels_in: int = 1,
                 channels_out: int = 10,
                 depth_shared: int = 3,
                 depth_union: int = 3,
                 initial_features: int = 64,
                 inter_features: int = 64,
                 norm: str = 'GroupNorm',
                 activation: str = 'ReLU',
                 pool_mode: str = 'StrideConv',
                 upsample_mode: str = 'bilinear',
                 final_activation: Optional[str] = None,
                 use_attention: bool = True,
                 use_residual: bool = True):
        super().__init__(
            channels_in=channels_in,
            channels_out=channels_out,
            depth_shared=depth_shared,
            depth_union=depth_union,
            initial_features=initial_features,
            inter_features=inter_features,
            norm=norm,
            activation=activation,
            pool_mode=pool_mode,
            upsample_mode=upsample_mode,
            final_activation=final_activation
        )
        
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # 注意力机制
        if use_attention and channels_in > 1:
            self.attention = ChannelAttention(inter_features * channels_in)
        
        # 残差连接的投影层
        if use_residual:
            self.residual_proj = nn.Conv2d(channels_in, channels_out, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播（带注意力和残差连接）"""
        batch_size, channels, height, width = x.shape
        
        if channels != self.channels_in:
            raise ValueError(f"Expected {self.channels_in} input channels, got {channels}")
        
        # 保存原始输入用于残差连接
        residual = x
        
        # 共享U-Net处理
        shared_features = []
        for i in range(self.channels_in):
            single_channel = x[:, i:i+1, :, :]
            features = self.shared_unet(single_channel)
            shared_features.append(features)
        
        # 特征融合
        if self.channels_in == 1:
            union_input = shared_features[0]
        else:
            union_input = torch.cat(shared_features, dim=1)
            
            # 应用注意力机制
            if self.use_attention:
                union_input = self.attention(union_input)
        
        # 联合U-Net处理
        output = self.union_unet(union_input)
        
        # 残差连接
        if self.use_residual:
            residual_proj = self.residual_proj(residual)
            # 确保尺寸匹配
            if residual_proj.size()[2:] != output.size()[2:]:
                residual_proj = F.interpolate(
                    residual_proj, size=output.size()[2:], 
                    mode='bilinear', align_corners=False
                )
            output = output + residual_proj
        
        return output


class ChannelAttention(nn.Module):
    """通道注意力机制"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention