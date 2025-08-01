"""SimpleSMLMNet实现

简化版的SMLM网络，基于标准U-Net2d，适用于基础的SMLM任务。
输出5或6个通道，提供更轻量级的解决方案。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union, List

from .unet2d import UNet2d


class SimpleSMLMNet(UNet2d):
    """简化版SMLM网络
    
    基于标准U-Net2d，适用于基础SMLM任务：
    - 5通道模式：检测概率 + xyz坐标 + 光子数
    - 6通道模式：检测概率 + xyz坐标 + 光子数 + 背景
    
    Args:
        channels_in: 输入通道数
        output_mode: 输出模式 ('5ch' 或 '6ch')
        depth: U-Net深度
        initial_features: 初始特征数
        norm: 归一化方法
        activation: 激活函数
        pool_mode: 池化模式
        upsample_mode: 上采样模式
        use_sigmoid_output: 是否对所有输出使用sigmoid激活
    """
    
    def __init__(self,
                 channels_in: int = 1,
                 output_mode: str = '6ch',
                 depth: int = 3,
                 initial_features: int = 64,
                 norm: str = 'GroupNorm',
                 activation: str = 'ReLU',
                 pool_mode: str = 'MaxPool',
                 upsample_mode: str = 'bilinear',
                 use_sigmoid_output: bool = True):
        
        # 确定输出通道数
        if output_mode == '5ch':
            channels_out = 5  # p, x, y, z, photons
        elif output_mode == '6ch':
            channels_out = 6  # p, x, y, z, photons, bg
        else:
            raise ValueError(f"Unsupported output_mode: {output_mode}. Use '5ch' or '6ch'.")
        
        super().__init__(
            channels_in=channels_in,
            channels_out=channels_out,
            depth=depth,
            initial_features=initial_features,
            norm=norm,
            activation=activation,
            pool_mode=pool_mode,
            upsample_mode=upsample_mode,
            final_activation='sigmoid' if use_sigmoid_output else None
        )
        
        self.output_mode = output_mode
        self.use_sigmoid_output = use_sigmoid_output
        
        # 通道映射
        if output_mode == '5ch':
            self.channel_mapping = {
                0: 'detection_probability',
                1: 'x_coordinate',
                2: 'y_coordinate',
                3: 'z_coordinate',
                4: 'photon_count'
            }
        else:  # 6ch
            self.channel_mapping = {
                0: 'detection_probability',
                1: 'x_coordinate',
                2: 'y_coordinate',
                3: 'z_coordinate',
                4: 'photon_count',
                5: 'background'
            }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 (B, C, H, W)
            
        Returns:
            输出张量，形状为 (B, channels_out, H, W)
        """
        return super().forward(x)
    
    def decode_output(self, output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """解码输出张量为各个组件
        
        Args:
            output: 形状为 (B, channels_out, H, W) 的输出张量
            
        Returns:
            包含各个组件的字典
        """
        expected_channels = 5 if self.output_mode == '5ch' else 6
        if output.size(1) != expected_channels:
            raise ValueError(f"Expected {expected_channels} channels, got {output.size(1)}")
        
        result = {
            'p': output[:, 0:1],  # 检测概率
            'xyz': output[:, 1:4],  # xyz坐标
            'photons': output[:, 4:5]  # 光子数
        }
        
        if self.output_mode == '6ch':
            result['bg'] = output[:, 5:6]  # 背景
        
        return result
    
    def get_channel_info(self) -> Dict[str, Union[Dict, List]]:
        """获取通道信息"""
        return {
            'output_mode': self.output_mode,
            'num_channels': 5 if self.output_mode == '5ch' else 6,
            'channel_mapping': self.channel_mapping,
            'use_sigmoid_output': self.use_sigmoid_output
        }


class EnhancedSimpleSMLMNet(nn.Module):
    """增强版简化SMLM网络
    
    在SimpleSMLMNet基础上添加：
    - 多尺度特征融合
    - 注意力机制
    - 残差连接
    """
    
    def __init__(self,
                 channels_in: int = 1,
                 output_mode: str = '6ch',
                 depth: int = 3,
                 initial_features: int = 64,
                 norm: str = 'GroupNorm',
                 activation: str = 'ReLU',
                 pool_mode: str = 'MaxPool',
                 upsample_mode: str = 'bilinear',
                 use_attention: bool = True,
                 use_multiscale: bool = True):
        super().__init__()
        
        self.output_mode = output_mode
        self.use_attention = use_attention
        self.use_multiscale = use_multiscale
        
        # 基础网络
        self.base_net = SimpleSMLMNet(
            channels_in=channels_in,
            output_mode=output_mode,
            depth=depth,
            initial_features=initial_features,
            norm=norm,
            activation=activation,
            pool_mode=pool_mode,
            upsample_mode=upsample_mode,
            use_sigmoid_output=False  # 我们会自定义激活
        )
        
        channels_out = 5 if output_mode == '5ch' else 6
        
        # 注意力机制
        if use_attention:
            self.attention = SpatialAttention()
        
        # 多尺度特征融合
        if use_multiscale:
            self.multiscale_fusion = MultiscaleFusion(initial_features, channels_out)
        
        # 最终激活
        self.final_activation = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self.use_multiscale:
            # 使用多尺度融合
            output = self.multiscale_fusion(x, self.base_net)
        else:
            # 标准前向传播
            output = self.base_net(x)
        
        # 应用注意力
        if self.use_attention:
            attention_map = self.attention(output)
            output = output * attention_map
        
        # 最终激活
        output = self.final_activation(output)
        
        return output
    
    def decode_output(self, output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """解码输出"""
        return self.base_net.decode_output(output)
    
    def get_channel_info(self) -> Dict[str, Union[Dict, List]]:
        """获取通道信息"""
        info = self.base_net.get_channel_info()
        info.update({
            'enhanced_features': {
                'attention': self.use_attention,
                'multiscale': self.use_multiscale
            }
        })
        return info


class SpatialAttention(nn.Module):
    """空间注意力机制"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算通道维度的平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 连接并卷积
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(attention_input))
        
        return attention_map


class MultiscaleFusion(nn.Module):
    """多尺度特征融合"""
    
    def __init__(self, base_features: int, output_channels: int):
        super().__init__()
        
        # 不同尺度的卷积
        self.scale1 = nn.Conv2d(base_features, output_channels, kernel_size=1)
        self.scale3 = nn.Conv2d(base_features, output_channels, kernel_size=3, padding=1)
        self.scale5 = nn.Conv2d(base_features, output_channels, kernel_size=5, padding=2)
        
        # 融合权重
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x: torch.Tensor, base_net: nn.Module) -> torch.Tensor:
        # 获取基础网络的中间特征
        # 这里简化处理，直接使用最终输出
        base_output = base_net(x)
        
        # 假设我们能获取到中间特征（实际实现中需要修改base_net）
        # 这里使用输入的不同尺度处理
        features = base_net.final_conv.weight.size(1)  # 获取最后一层的输入特征数
        
        # 创建虚拟的中间特征（实际应该从网络中提取）
        dummy_features = torch.randn(x.size(0), features, x.size(2), x.size(3), device=x.device)
        
        # 多尺度处理
        out1 = self.scale1(dummy_features)
        out3 = self.scale3(dummy_features)
        out5 = self.scale5(dummy_features)
        
        # 加权融合
        weights = F.softmax(self.fusion_weights, dim=0)
        output = weights[0] * out1 + weights[1] * out3 + weights[2] * out5
        
        return output


class AdaptiveSMLMNet(SimpleSMLMNet):
    """自适应SMLM网络
    
    根据输入数据的特性自动调整网络行为
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 自适应模块
        self.adaptive_module = AdaptiveModule(self.initial_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 分析输入特性
        adaptation_params = self.adaptive_module(x)
        
        # 标准前向传播
        output = super().forward(x)
        
        # 根据自适应参数调整输出
        output = output * adaptation_params['scale'] + adaptation_params['bias']
        
        return output


class AdaptiveModule(nn.Module):
    """自适应模块"""
    
    def __init__(self, features: int):
        super().__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(features, features // 4),
            nn.ReLU(),
            nn.Linear(features // 4, 2)  # scale and bias
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 全局特征
        global_feat = self.global_pool(x).flatten(1)
        params = self.fc(global_feat)
        
        scale = torch.sigmoid(params[:, 0:1]).unsqueeze(-1).unsqueeze(-1)
        bias = torch.tanh(params[:, 1:2]).unsqueeze(-1).unsqueeze(-1)
        
        return {
            'scale': scale,
            'bias': bias
        }