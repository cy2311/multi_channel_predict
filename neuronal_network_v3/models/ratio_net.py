"""RatioNet实现

用于预测同一个emitter在两个通道间的光子数分配比例ratio_e的网络。
支持不确定性量化，能够同时预测比例的均值和方差。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class RatioNet(nn.Module):
    """比例预测网络
    
    用于预测同一个emitter在两个通道间的光子数分配比例。
    支持不确定性量化，输出比例的均值和对数方差。
    
    Args:
        input_features: 输入特征维度（两个通道特征拼接后的维度）
        hidden_dim: 隐藏层维度
        dropout_rate: Dropout比率
        min_variance: 最小方差值
        max_variance: 最大方差值
    """
    
    def __init__(self,
                 input_features: int = 20,
                 hidden_dim: int = 64,
                 dropout_rate: float = 0.1,
                 min_variance: float = 1e-6,
                 max_variance: float = 1.0):
        super().__init__()
        
        self.min_variance = min_variance
        self.max_variance = max_variance
        
        # 共享特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 均值预测头
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # 输出0-1之间的比例均值
        )
        
        # 对数方差预测头
        self.log_var_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)  # 输出对数方差，无激活函数
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, ch1_features: torch.Tensor, ch2_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            ch1_features: 通道1的特征，形状为 (B, F1)
            ch2_features: 通道2的特征，形状为 (B, F2)
            
        Returns:
            ratio_mean: 比例均值，形状为 (B, 1)
            ratio_log_var: 比例对数方差，形状为 (B, 1)
        """
        # 拼接两个通道的特征
        combined = torch.cat([ch1_features, ch2_features], dim=1)
        
        # 共享特征提取
        shared_features = self.shared_layers(combined)
        
        # 预测均值和对数方差
        ratio_mean = self.mean_head(shared_features)
        ratio_log_var = self.log_var_head(shared_features)
        
        # 限制方差范围
        ratio_log_var = torch.clamp(ratio_log_var, 
                                   math.log(self.min_variance), 
                                   math.log(self.max_variance))
        
        return ratio_mean, ratio_log_var
    
    def predict_with_uncertainty(self, ch1_features: torch.Tensor, ch2_features: torch.Tensor) -> dict:
        """预测比例及其不确定性
        
        Args:
            ch1_features: 通道1的特征
            ch2_features: 通道2的特征
            
        Returns:
            包含均值、方差、标准差和置信区间的字典
        """
        ratio_mean, ratio_log_var = self.forward(ch1_features, ch2_features)
        
        ratio_var = torch.exp(ratio_log_var)
        ratio_std = torch.sqrt(ratio_var)
        
        # 95%置信区间
        z_score = 1.96
        lower_bound = torch.clamp(ratio_mean - z_score * ratio_std, 0, 1)
        upper_bound = torch.clamp(ratio_mean + z_score * ratio_std, 0, 1)
        
        return {
            'mean': ratio_mean,
            'variance': ratio_var,
            'std': ratio_std,
            'log_var': ratio_log_var,
            'confidence_interval_95': (lower_bound, upper_bound)
        }


class FeatureExtractor(nn.Module):
    """特征提取器
    
    从SigmaMUNet的输出中提取用于比例预测的特征。
    """
    
    def __init__(self, input_channels: int = 10, output_features: int = 10):
        super().__init__()
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 特征变换
        self.feature_transform = nn.Sequential(
            nn.Linear(input_channels, output_features),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征
        
        Args:
            x: 输入张量，形状为 (B, C, H, W)
            
        Returns:
            特征向量，形状为 (B, output_features)
        """
        # 全局平均池化
        pooled = self.global_pool(x)  # (B, C, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, C)
        
        # 特征变换
        features = self.feature_transform(pooled)
        
        return features


import math  # 添加缺失的导入