"""数据变换模块

包含SMLM数据的变换和增强功能。
"""

import torch
import numpy as np
from typing import Dict, Any, List
import random


class RandomRotation:
    """随机旋转变换"""
    
    def __init__(self, degrees=None, angle_range=(-180, 180)):
        if degrees is not None:
            # 兼容旧的degrees参数
            if isinstance(degrees, (int, float)):
                self.angles = [0, degrees, 2*degrees, 3*degrees] if degrees == 90 else [0, degrees]
            else:
                self.angles = degrees if isinstance(degrees, list) else [0, degrees]
        else:
            # 默认使用90度的倍数
            self.angles = [0, 90, 180, 270]
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        angle = random.choice(self.angles)
        k = angle // 90
        
        if k > 0:
            if 'input' in sample:
                sample['input'] = torch.rot90(sample['input'], k, dims=[-2, -1])
            if 'target' in sample:
                sample['target'] = torch.rot90(sample['target'], k, dims=[-2, -1])
        
        return sample


class RandomFlip:
    """随机翻转变换"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.p:
            # 水平翻转
            if 'input' in sample:
                sample['input'] = torch.flip(sample['input'], dims=[-1])
            if 'target' in sample:
                sample['target'] = torch.flip(sample['target'], dims=[-1])
        
        if random.random() < self.p:
            # 垂直翻转
            if 'input' in sample:
                sample['input'] = torch.flip(sample['input'], dims=[-2])
            if 'target' in sample:
                sample['target'] = torch.flip(sample['target'], dims=[-2])
        
        return sample


class GaussianNoise:
    """高斯噪声变换"""
    
    def __init__(self, std=None, noise_std: float = 0.1):
        if std is not None:
            self.noise_std = std
        else:
            self.noise_std = noise_std
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if 'input' in sample:
            noise = torch.randn_like(sample['input']) * self.noise_std
            sample['input'] = sample['input'] + noise
        return sample


class Normalize:
    """归一化变换"""
    
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if 'input' in sample:
            sample['input'] = (sample['input'] - self.mean) / self.std
        return sample