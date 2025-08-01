"""基础U-Net 2D实现

这个模块实现了标准的U-Net架构，作为DECODE网络的基础组件。
支持可配置的深度、特征数、激活函数、归一化方法等。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List


class ConvBlock(nn.Module):
    """基础卷积块：Conv2d + Norm + Activation"""
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 norm: str = 'GroupNorm',
                 activation: str = 'ReLU',
                 num_groups: int = 8):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        
        # 归一化层
        if norm == 'BatchNorm':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'GroupNorm':
            # 确保组数不超过通道数
            num_groups = min(num_groups, out_channels)
            self.norm = nn.GroupNorm(num_groups, out_channels)
        elif norm == 'InstanceNorm':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'LayerNorm':
            self.norm = nn.LayerNorm([out_channels])
        else:
            self.norm = nn.Identity()
            
        # 激活函数
        if activation == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'ELU':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'GELU':
            self.activation = nn.GELU()
        elif activation == 'Swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class DoubleConvBlock(nn.Module):
    """双卷积块：两个连续的ConvBlock"""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm: str = 'GroupNorm',
                 activation: str = 'ReLU'):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, out_channels, norm=norm, activation=activation)
        self.conv2 = ConvBlock(out_channels, out_channels, norm=norm, activation=activation)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DownBlock(nn.Module):
    """下采样块：DoubleConvBlock + Pooling"""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 pool_mode: str = 'MaxPool',
                 norm: str = 'GroupNorm',
                 activation: str = 'ReLU'):
        super().__init__()
        
        self.conv = DoubleConvBlock(in_channels, out_channels, norm=norm, activation=activation)
        
        if pool_mode == 'MaxPool':
            self.pool = nn.MaxPool2d(2)
        elif pool_mode == 'AvgPool':
            self.pool = nn.AvgPool2d(2)
        elif pool_mode == 'StrideConv':
            self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)
        else:
            raise ValueError(f"Unsupported pool_mode: {pool_mode}")
            
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        skip = x
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    """上采样块：Upsample + DoubleConvBlock"""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 upsample_mode: str = 'bilinear',
                 norm: str = 'GroupNorm',
                 activation: str = 'ReLU'):
        super().__init__()
        
        self.upsample_mode = upsample_mode
        
        if upsample_mode == 'transpose':
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            conv_in_channels = out_channels + out_channels  # skip connection
        else:
            self.up = None
            conv_in_channels = in_channels + out_channels  # skip connection
            
        self.conv = DoubleConvBlock(conv_in_channels, out_channels, norm=norm, activation=activation)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.up is not None:
            x = self.up(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode, align_corners=False)
            
        # 确保尺寸匹配
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=False)
            
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNet2d(nn.Module):
    """标准U-Net 2D实现
    
    Args:
        channels_in: 输入通道数
        channels_out: 输出通道数
        depth: U-Net深度（下采样层数）
        initial_features: 初始特征数
        norm: 归一化方法 ('BatchNorm', 'GroupNorm', 'InstanceNorm', 'LayerNorm', 'None')
        activation: 激活函数 ('ReLU', 'LeakyReLU', 'ELU', 'GELU', 'Swish')
        pool_mode: 池化模式 ('MaxPool', 'AvgPool', 'StrideConv')
        upsample_mode: 上采样模式 ('bilinear', 'nearest', 'transpose')
        final_activation: 最终激活函数 (None, 'sigmoid', 'tanh', 'softmax')
    """
    
    def __init__(self,
                 channels_in: int = 1,
                 channels_out: int = 1,
                 depth: int = 3,
                 initial_features: int = 64,
                 norm: str = 'GroupNorm',
                 activation: str = 'ReLU',
                 pool_mode: str = 'MaxPool',
                 upsample_mode: str = 'bilinear',
                 final_activation: Optional[str] = None):
        super().__init__()
        
        self.depth = depth
        self.final_activation = final_activation
        
        # 计算每层的特征数
        features = [initial_features * (2 ** i) for i in range(depth + 1)]
        
        # 编码器
        self.encoder = nn.ModuleList()
        in_ch = channels_in
        for i in range(depth):
            self.encoder.append(DownBlock(
                in_ch, features[i], pool_mode=pool_mode, 
                norm=norm, activation=activation
            ))
            in_ch = features[i]
            
        # 瓶颈层
        self.bottleneck = DoubleConvBlock(
            features[depth-1], features[depth], 
            norm=norm, activation=activation
        )
        
        # 解码器
        self.decoder = nn.ModuleList()
        for i in range(depth):
            self.decoder.append(UpBlock(
                features[depth-i], features[depth-i-1],
                upsample_mode=upsample_mode,
                norm=norm, activation=activation
            ))
            
        # 最终输出层
        self.final_conv = nn.Conv2d(features[0], channels_out, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码器
        skips = []
        for encoder_block in self.encoder:
            x, skip = encoder_block(x)
            skips.append(skip)
            
        # 瓶颈层
        x = self.bottleneck(x)
        
        # 解码器
        for i, decoder_block in enumerate(self.decoder):
            skip = skips[-(i+1)]  # 反向使用skip连接
            x = decoder_block(x, skip)
            
        # 最终输出
        x = self.final_conv(x)
        
        # 应用最终激活函数
        if self.final_activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.final_activation == 'tanh':
            x = torch.tanh(x)
        elif self.final_activation == 'softmax':
            x = F.softmax(x, dim=1)
            
        return x
    
    def get_parameter_count(self) -> dict:
        """获取模型参数统计信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }
    
    def get_model_size(self) -> dict:
        """获取模型大小信息"""
        param_size = 0
        buffer_size = 0
        
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
            
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        model_size = param_size + buffer_size
        
        return {
            'model_size_mb': model_size / 1024 / 1024,
            'param_size_mb': param_size / 1024 / 1024,
            'buffer_size_mb': buffer_size / 1024 / 1024
        }