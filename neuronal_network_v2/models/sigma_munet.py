"""SigmaMUNet实现

DECODE的主要模型，继承自DoubleMUnet，输出10个通道：
- p head: 1个通道 (检测概率)
- phot,xyz_mu head: 4个通道 (光子数和xyz坐标均值)
- phot,xyz_sig head: 4个通道 (光子数和xyz坐标标准差)
- bg head: 1个通道 (背景)

支持不同的激活函数配置和输出头分离。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union

from .double_munet import DoubleMUnet


class SigmaMUNet(DoubleMUnet):
    """SigmaMUNet - DECODE的主要网络架构
    
    基于DoubleMUnet，输出10个通道，支持不同的激活函数配置：
    - 通道0: 检测概率 (p) - sigmoid
    - 通道1-4: 光子数和xyz坐标均值 (pxyz_mu) - sigmoid/tanh
    - 通道5-8: 光子数和xyz坐标标准差 (pxyz_sig) - sigmoid
    - 通道9: 背景 (bg) - sigmoid
    
    Args:
        channels_in: 输入通道数 (1 或 3)
        depth_shared: 共享U-Net深度
        depth_union: 联合U-Net深度
        initial_features: 初始特征数
        inter_features: 中间特征数
        norm: 归一化方法
        activation: 激活函数
        pool_mode: 池化模式
        upsample_mode: 上采样模式
        sigmoid_ch_ix: 使用sigmoid激活的通道索引
        tanh_ch_ix: 使用tanh激活的通道索引
        use_separate_heads: 是否使用分离的输出头
    """
    
    def __init__(self,
                 channels_in: int = 1,
                 depth_shared: int = 3,
                 depth_union: int = 3,
                 initial_features: int = 64,
                 inter_features: int = 64,
                 norm: str = 'GroupNorm',
                 activation: str = 'ReLU',
                 pool_mode: str = 'StrideConv',
                 upsample_mode: str = 'bilinear',
                 sigmoid_ch_ix: Optional[List[int]] = None,
                 tanh_ch_ix: Optional[List[int]] = None,
                 use_separate_heads: bool = False):
        
        # 固定输出通道数为10
        super().__init__(
            channels_in=channels_in,
            channels_out=10,
            depth_shared=depth_shared,
            depth_union=depth_union,
            initial_features=initial_features,
            inter_features=inter_features,
            norm=norm,
            activation=activation,
            pool_mode=pool_mode,
            upsample_mode=upsample_mode,
            final_activation=None  # 我们会自定义激活
        )
        
        # 默认激活函数配置
        if sigmoid_ch_ix is None:
            sigmoid_ch_ix = [0, 1, 5, 6, 7, 8, 9]  # p, phot_mu, pxyz_sig, bg
        if tanh_ch_ix is None:
            tanh_ch_ix = [2, 3, 4]  # xyz_mu
            
        self.sigmoid_ch_ix = sigmoid_ch_ix
        self.tanh_ch_ix = tanh_ch_ix
        self.use_separate_heads = use_separate_heads
        
        # 分离的输出头（可选）
        if use_separate_heads:
            # 获取联合U-Net的最后一层特征数
            last_features = initial_features
            
            self.p_head = nn.Conv2d(last_features, 1, kernel_size=1)  # 检测概率
            self.phot_xyz_mu_head = nn.Conv2d(last_features, 4, kernel_size=1)  # 光子数和xyz均值
            self.phot_xyz_sig_head = nn.Conv2d(last_features, 4, kernel_size=1)  # 光子数和xyz标准差
            self.bg_head = nn.Conv2d(last_features, 1, kernel_size=1)  # 背景
            
            # 移除原来的最终卷积层
            self.union_unet.final_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播
        
        Args:
            x: 输入张量，形状为 (B, C, H, W)
            
        Returns:
            如果use_separate_heads=False:
                输出张量，形状为 (B, 10, H, W)，已应用相应的激活函数
            如果use_separate_heads=True:
                字典，包含各个输出头的结果
        """
        if self.use_separate_heads:
            return self._forward_separate_heads(x)
        else:
            return self._forward_unified(x)
    
    def _forward_unified(self, x: torch.Tensor) -> torch.Tensor:
        """统一输出的前向传播"""
        # 通过父类获取原始输出
        output = super().forward(x)  # (B, 10, H, W)
        
        # 应用通道特定的激活函数
        activated_output = torch.zeros_like(output)
        
        # Sigmoid激活的通道
        for ch_ix in self.sigmoid_ch_ix:
            if ch_ix < output.size(1):
                activated_output[:, ch_ix] = torch.sigmoid(output[:, ch_ix])
        
        # Tanh激活的通道
        for ch_ix in self.tanh_ch_ix:
            if ch_ix < output.size(1):
                activated_output[:, ch_ix] = torch.tanh(output[:, ch_ix])
        
        # 其他通道保持原样
        for ch_ix in range(output.size(1)):
            if ch_ix not in self.sigmoid_ch_ix and ch_ix not in self.tanh_ch_ix:
                activated_output[:, ch_ix] = output[:, ch_ix]
        
        return activated_output
    
    def _forward_separate_heads(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """分离输出头的前向传播"""
        # 获取共享特征
        shared_features = self.get_shared_features(x)
        
        # 特征融合
        if self.channels_in == 1:
            union_input = shared_features[0]
        else:
            union_input = torch.cat(shared_features, dim=1)
        
        # 通过联合U-Net获取特征（不包括最终卷积）
        features = self.union_unet(union_input)  # 现在返回特征而不是最终输出
        
        # 分别通过各个输出头
        p = torch.sigmoid(self.p_head(features))  # (B, 1, H, W)
        phot_xyz_mu = self.phot_xyz_mu_head(features)  # (B, 4, H, W)
        phot_xyz_sig = torch.sigmoid(self.phot_xyz_sig_head(features))  # (B, 4, H, W)
        bg = torch.sigmoid(self.bg_head(features))  # (B, 1, H, W)
        
        # 对坐标均值应用tanh激活
        phot_mu = torch.sigmoid(phot_xyz_mu[:, 0:1])  # 光子数均值
        xyz_mu = torch.tanh(phot_xyz_mu[:, 1:4])  # xyz坐标均值
        phot_xyz_mu = torch.cat([phot_mu, xyz_mu], dim=1)
        
        return {
            'p': p,
            'phot_xyz_mu': phot_xyz_mu,
            'phot_xyz_sig': phot_xyz_sig,
            'bg': bg,
            'combined': torch.cat([p, phot_xyz_mu, phot_xyz_sig, bg], dim=1)
        }
    
    def get_raw_output(self, x: torch.Tensor) -> torch.Tensor:
        """获取未经激活函数处理的原始输出（用于损失计算）"""
        return super().forward(x)
    
    def decode_output(self, output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """解码输出张量为各个组件
        
        Args:
            output: 形状为 (B, 10, H, W) 的输出张量
            
        Returns:
            包含各个组件的字典
        """
        if output.size(1) != 10:
            raise ValueError(f"Expected 10 channels, got {output.size(1)}")
        
        return {
            'p': output[:, 0:1],  # 检测概率
            'phot_mu': output[:, 1:2],  # 光子数均值
            'xyz_mu': output[:, 2:5],  # xyz坐标均值
            'phot_sig': output[:, 5:6],  # 光子数标准差
            'xyz_sig': output[:, 6:9],  # xyz坐标标准差
            'bg': output[:, 9:10]  # 背景
        }
    
    def get_channel_info(self) -> Dict[str, Dict]:
        """获取通道信息"""
        return {
            'channel_mapping': {
                0: 'detection_probability',
                1: 'photon_mean',
                2: 'x_coordinate_mean',
                3: 'y_coordinate_mean', 
                4: 'z_coordinate_mean',
                5: 'photon_std',
                6: 'x_coordinate_std',
                7: 'y_coordinate_std',
                8: 'z_coordinate_std',
                9: 'background'
            },
            'activation_functions': {
                'sigmoid_channels': self.sigmoid_ch_ix,
                'tanh_channels': self.tanh_ch_ix
            },
            'output_heads': {
                'p_head': [0],
                'phot_xyz_mu_head': [1, 2, 3, 4],
                'phot_xyz_sig_head': [5, 6, 7, 8],
                'bg_head': [9]
            }
        }


class SigmaMUNetWithUncertainty(SigmaMUNet):
    """带不确定性量化的SigmaMUNet
    
    在标准SigmaMUNet基础上添加：
    - 不确定性量化
    - 置信度估计
    - 预测区间
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 不确定性估计头
        if hasattr(self, 'union_unet'):
            last_features = self.initial_features
            self.uncertainty_head = nn.Conv2d(last_features, 10, kernel_size=1)
            self.confidence_head = nn.Conv2d(last_features, 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播（包含不确定性）"""
        # 获取标准输出
        if self.use_separate_heads:
            output = super().forward(x)
        else:
            standard_output = super().forward(x)
            output = {'combined': standard_output}
        
        # 计算不确定性
        if hasattr(self, 'uncertainty_head'):
            # 获取特征
            shared_features = self.get_shared_features(x)
            if self.channels_in == 1:
                union_input = shared_features[0]
            else:
                union_input = torch.cat(shared_features, dim=1)
            
            features = self.union_unet(union_input)
            
            # 不确定性和置信度
            uncertainty = F.softplus(self.uncertainty_head(features))  # 确保为正
            confidence = torch.sigmoid(self.confidence_head(features))
            
            output.update({
                'uncertainty': uncertainty,
                'confidence': confidence
            })
        
        return output
    
    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 10) -> Dict[str, torch.Tensor]:
        """使用Monte Carlo Dropout进行不确定性估计"""
        self.train()  # 启用dropout
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred['combined'] if 'combined' in pred else pred)
        
        self.eval()  # 恢复评估模式
        
        # 计算统计量
        predictions = torch.stack(predictions, dim=0)  # (num_samples, B, C, H, W)
        
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'samples': predictions
        }