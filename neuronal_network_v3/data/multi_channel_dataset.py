"""多通道数据集

实现多通道DECODE数据的加载、预处理和增强，支持：
- 双通道图像数据加载
- 光子数比例计算和目标生成
- 物理约束数据验证
- 数据增强和预处理流水线
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
import h5py
import json
from scipy import ndimage
import logging

from .dataset import SMLMDataset
from .transforms import RandomRotation, RandomFlip, GaussianNoise, Normalize


class MultiChannelSMLMDataset(Dataset):
    """多通道SMLM数据集
    
    支持双通道DECODE数据的加载和处理，包括：
    - 双通道图像数据
    - 光子数分配比例
    - 物理约束验证
    - 数据增强
    
    Args:
        data_path: 数据文件路径
        config: 数据配置字典
        mode: 数据模式 ('train', 'val', 'test')
        transform: 数据变换函数
    """
    
    def __init__(self, 
                 data_path: str,
                 config: Dict[str, Any],
                 mode: str = 'train',
                 transform: Optional[Callable] = None):
        self.data_path = Path(data_path)
        self.config = config
        self.mode = mode
        self.transform = transform
        
        self.logger = logging.getLogger(__name__)
        
        # 数据参数
        self.patch_size = config.get('patch_size', 64)
        self.pixel_size = config.get('pixel_size', 100)  # nm
        self.photon_threshold = config.get('photon_threshold', 50)
        
        # 比例计算参数
        self.ratio_method = config.get('ratio_method', 'photon_based')
        self.add_noise_to_ratio = config.get('add_noise_to_ratio', True)
        self.ratio_noise_std = config.get('ratio_noise_std', 0.01)
        
        # 物理约束参数
        self.enforce_conservation = config.get('enforce_conservation', True)
        self.conservation_tolerance = config.get('conservation_tolerance', 0.05)
        
        # 加载数据
        self._load_data()
        
        # 设置数据变换
        if transform is None:
            self.transform = self._get_default_transform()
        
        self.logger.info(f"MultiChannelSMLMDataset initialized: {len(self)} samples, mode: {mode}")
    
    def _load_data(self):
        """加载数据文件"""
        if self.data_path.suffix == '.h5':
            self._load_h5_data()
        elif self.data_path.suffix == '.npz':
            self._load_npz_data()
        else:
            raise ValueError(f"Unsupported data format: {self.data_path.suffix}")
    
    def _load_h5_data(self):
        """加载HDF5格式数据"""
        with h5py.File(self.data_path, 'r') as f:
            # 双通道图像数据
            self.channel1_images = f[f'{self.mode}/channel1/images'][:]
            self.channel2_images = f[f'{self.mode}/channel2/images'][:]
            
            # 目标数据
            self.channel1_targets = f[f'{self.mode}/channel1/targets'][:]
            self.channel2_targets = f[f'{self.mode}/channel2/targets'][:]
            
            # 光子数数据
            self.channel1_photons = f[f'{self.mode}/channel1/photons'][:]
            self.channel2_photons = f[f'{self.mode}/channel2/photons'][:]
            
            # 元数据
            if f'{self.mode}/metadata' in f:
                self.metadata = json.loads(f[f'{self.mode}/metadata'][()])
            else:
                self.metadata = {}
        
        # 计算比例数据
        self._compute_ratios()
        
        # 验证数据一致性
        self._validate_data()
    
    def _load_npz_data(self):
        """加载NPZ格式数据"""
        data = np.load(self.data_path)
        
        # 双通道图像数据
        self.channel1_images = data[f'{self.mode}_channel1_images']
        self.channel2_images = data[f'{self.mode}_channel2_images']
        
        # 目标数据
        self.channel1_targets = data[f'{self.mode}_channel1_targets']
        self.channel2_targets = data[f'{self.mode}_channel2_targets']
        
        # 光子数数据
        self.channel1_photons = data[f'{self.mode}_channel1_photons']
        self.channel2_photons = data[f'{self.mode}_channel2_photons']
        
        # 元数据
        if f'{self.mode}_metadata' in data:
            self.metadata = data[f'{self.mode}_metadata'].item()
        else:
            self.metadata = {}
        
        # 计算比例数据
        self._compute_ratios()
        
        # 验证数据一致性
        self._validate_data()
    
    def _compute_ratios(self):
        """计算光子数分配比例"""
        if self.ratio_method == 'photon_based':
            # 基于光子数计算比例
            total_photons = self.channel1_photons + self.channel2_photons
            # 避免除零
            valid_mask = total_photons > self.photon_threshold
            
            self.ratios = np.zeros_like(self.channel1_photons, dtype=np.float32)
            self.ratios[valid_mask] = self.channel1_photons[valid_mask] / total_photons[valid_mask]
            
            # 对于低光子数的情况，使用默认比例
            self.ratios[~valid_mask] = 0.5
            
        elif self.ratio_method == 'intensity_based':
            # 基于强度计算比例
            ch1_intensity = np.sum(self.channel1_images, axis=(1, 2))
            ch2_intensity = np.sum(self.channel2_images, axis=(1, 2))
            total_intensity = ch1_intensity + ch2_intensity
            
            valid_mask = total_intensity > 0
            self.ratios = np.zeros_like(ch1_intensity, dtype=np.float32)
            self.ratios[valid_mask] = ch1_intensity[valid_mask] / total_intensity[valid_mask]
            self.ratios[~valid_mask] = 0.5
            
        else:
            raise ValueError(f"Unsupported ratio method: {self.ratio_method}")
        
        # 添加噪声（用于训练时的数据增强）
        if self.add_noise_to_ratio and self.mode == 'train':
            noise = np.random.normal(0, self.ratio_noise_std, self.ratios.shape)
            self.ratios = np.clip(self.ratios + noise, 0, 1)
        
        # 计算总光子数（用于物理约束）
        self.total_photons = self.channel1_photons + self.channel2_photons
    
    def _validate_data(self):
        """验证数据一致性"""
        n_samples = len(self.channel1_images)
        
        # 检查数据形状一致性
        assert len(self.channel2_images) == n_samples, "Channel data length mismatch"
        assert len(self.channel1_targets) == n_samples, "Target data length mismatch"
        assert len(self.channel2_targets) == n_samples, "Target data length mismatch"
        assert len(self.ratios) == n_samples, "Ratio data length mismatch"
        
        # 检查图像尺寸
        assert self.channel1_images.shape[1:] == self.channel2_images.shape[1:], "Image size mismatch"
        
        # 检查比例范围
        assert np.all((self.ratios >= 0) & (self.ratios <= 1)), "Invalid ratio values"
        
        # 物理约束检查
        if self.enforce_conservation:
            predicted_total = self.channel1_photons + self.channel2_photons
            conservation_error = np.abs(predicted_total - self.total_photons) / (self.total_photons + 1e-8)
            
            if np.any(conservation_error > self.conservation_tolerance):
                self.logger.warning(f"Conservation violation detected in {np.sum(conservation_error > self.conservation_tolerance)} samples")
        
        self.logger.info(f"Data validation completed: {n_samples} samples")
    
    def _get_default_transform(self):
        """获取默认数据变换"""
        if self.mode == 'train':
            return MultiChannelTransform([
                RandomRotation(degrees=90),
                RandomFlip(p=0.5),
                GaussianNoise(std=0.01),
                Normalize()
            ])
        else:
            return MultiChannelTransform([
                Normalize()
            ])
    
    def __len__(self) -> int:
        return len(self.channel1_images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        # 获取原始数据
        ch1_image = self.channel1_images[idx].astype(np.float32)
        ch2_image = self.channel2_images[idx].astype(np.float32)
        ch1_target = self.channel1_targets[idx].astype(np.float32)
        ch2_target = self.channel2_targets[idx].astype(np.float32)
        ratio = self.ratios[idx]
        total_photons = self.total_photons[idx]
        
        # 构建样本字典
        sample = {
            'channel1_input': ch1_image,
            'channel2_input': ch2_image,
            'channel1_target': ch1_target,
            'channel2_target': ch2_target,
            'ratio': ratio,
            'total_photons': total_photons,
            'channel1_photons': self.channel1_photons[idx],
            'channel2_photons': self.channel2_photons[idx]
        }
        
        # 应用数据变换
        if self.transform:
            sample = self.transform(sample)
        
        # 转换为tensor
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                sample[key] = torch.from_numpy(value)
            elif isinstance(value, (int, float)):
                sample[key] = torch.tensor(value, dtype=torch.float32)
        
        return sample
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        stats = {
            'n_samples': len(self),
            'image_shape': self.channel1_images.shape[1:],
            'target_shape': self.channel1_targets.shape[1:],
            
            # 光子数统计
            'photon_stats': {
                'channel1': {
                    'mean': float(np.mean(self.channel1_photons)),
                    'std': float(np.std(self.channel1_photons)),
                    'min': float(np.min(self.channel1_photons)),
                    'max': float(np.max(self.channel1_photons))
                },
                'channel2': {
                    'mean': float(np.mean(self.channel2_photons)),
                    'std': float(np.std(self.channel2_photons)),
                    'min': float(np.min(self.channel2_photons)),
                    'max': float(np.max(self.channel2_photons))
                },
                'total': {
                    'mean': float(np.mean(self.total_photons)),
                    'std': float(np.std(self.total_photons)),
                    'min': float(np.min(self.total_photons)),
                    'max': float(np.max(self.total_photons))
                }
            },
            
            # 比例统计
            'ratio_stats': {
                'mean': float(np.mean(self.ratios)),
                'std': float(np.std(self.ratios)),
                'min': float(np.min(self.ratios)),
                'max': float(np.max(self.ratios)),
                'median': float(np.median(self.ratios))
            },
            
            # 图像强度统计
            'intensity_stats': {
                'channel1': {
                    'mean': float(np.mean(self.channel1_images)),
                    'std': float(np.std(self.channel1_images))
                },
                'channel2': {
                    'mean': float(np.mean(self.channel2_images)),
                    'std': float(np.std(self.channel2_images))
                }
            }
        }
        
        return stats


class MultiChannelTransform:
    """多通道数据变换
    
    对双通道数据同时应用相同的变换，保持数据一致性。
    
    Args:
        transforms: 变换函数列表
    """
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """应用变换"""
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class MultiChannelRandomRotation:
    """多通道随机旋转"""
    
    def __init__(self, degrees: Union[float, Tuple[float, float]] = 90):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
    
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # 随机选择旋转角度
        angle = np.random.uniform(self.degrees[0], self.degrees[1])
        
        # 对图像数据应用旋转
        for key in ['channel1_input', 'channel2_input', 'channel1_target', 'channel2_target']:
            if key in sample and len(sample[key].shape) >= 2:
                sample[key] = ndimage.rotate(sample[key], angle, reshape=False, order=1)
        
        return sample


class MultiChannelRandomFlip:
    """多通道随机翻转"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # 水平翻转
        if np.random.random() < self.p:
            for key in ['channel1_input', 'channel2_input', 'channel1_target', 'channel2_target']:
                if key in sample and len(sample[key].shape) >= 2:
                    sample[key] = np.fliplr(sample[key])
        
        # 垂直翻转
        if np.random.random() < self.p:
            for key in ['channel1_input', 'channel2_input', 'channel1_target', 'channel2_target']:
                if key in sample and len(sample[key].shape) >= 2:
                    sample[key] = np.flipud(sample[key])
        
        return sample


class MultiChannelGaussianNoise:
    """多通道高斯噪声"""
    
    def __init__(self, std: float = 0.01):
        self.std = std
    
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # 只对输入图像添加噪声
        for key in ['channel1_input', 'channel2_input']:
            if key in sample:
                noise = np.random.normal(0, self.std, sample[key].shape)
                sample[key] = sample[key] + noise.astype(sample[key].dtype)
        
        return sample


class MultiChannelNormalize:
    """多通道归一化"""
    
    def __init__(self, 
                 mean: Optional[Dict[str, float]] = None,
                 std: Optional[Dict[str, float]] = None):
        self.mean = mean or {'channel1': 0.0, 'channel2': 0.0}
        self.std = std or {'channel1': 1.0, 'channel2': 1.0}
    
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # 归一化输入图像
        if 'channel1_input' in sample:
            sample['channel1_input'] = (sample['channel1_input'] - self.mean['channel1']) / self.std['channel1']
        
        if 'channel2_input' in sample:
            sample['channel2_input'] = (sample['channel2_input'] - self.mean['channel2']) / self.std['channel2']
        
        return sample


def create_multi_channel_dataloader(data_path: str,
                                   config: Dict[str, Any],
                                   mode: str = 'train',
                                   batch_size: int = 16,
                                   shuffle: bool = True,
                                   num_workers: int = 4) -> DataLoader:
    """创建多通道数据加载器
    
    Args:
        data_path: 数据文件路径
        config: 数据配置
        mode: 数据模式
        batch_size: 批大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
        
    Returns:
        DataLoader对象
    """
    dataset = MultiChannelSMLMDataset(
        data_path=data_path,
        config=config,
        mode=mode
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == 'train')
    )
    
    return dataloader


def collate_multi_channel_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """多通道批数据整理函数"""
    # 获取所有键
    keys = batch[0].keys()
    
    # 整理每个键的数据
    collated = {}
    for key in keys:
        values = [sample[key] for sample in batch]
        
        if key in ['channel1_input', 'channel2_input', 'channel1_target', 'channel2_target']:
            # 图像数据需要添加通道维度
            stacked = torch.stack(values)
            if len(stacked.shape) == 3:  # [B, H, W] -> [B, 1, H, W]
                stacked = stacked.unsqueeze(1)
            collated[key] = stacked
        else:
            # 标量数据直接堆叠
            collated[key] = torch.stack(values)
    
    return collated


class MultiChannelDataModule:
    """多通道数据模块
    
    封装数据加载、预处理和验证的完整流程。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 数据路径
        self.train_path = config['data']['train_path']
        self.val_path = config['data']['val_path']
        self.test_path = config.get('data', {}).get('test_path')
        
        # 数据加载参数
        self.batch_size = config.get('batch_size', 16)
        self.num_workers = config.get('num_workers', 4)
        
        # 数据集
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # 数据加载器
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def setup(self):
        """设置数据集和数据加载器"""
        # 创建数据集
        self.train_dataset = MultiChannelSMLMDataset(
            data_path=self.train_path,
            config=self.config['data'],
            mode='train'
        )
        
        self.val_dataset = MultiChannelSMLMDataset(
            data_path=self.val_path,
            config=self.config['data'],
            mode='val'
        )
        
        if self.test_path:
            self.test_dataset = MultiChannelSMLMDataset(
                data_path=self.test_path,
                config=self.config['data'],
                mode='test'
            )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_multi_channel_batch
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_multi_channel_batch
        )
        
        if self.test_dataset:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=collate_multi_channel_batch
            )
        
        self.logger.info(f"Data module setup completed: "
                        f"Train: {len(self.train_dataset)}, "
                        f"Val: {len(self.val_dataset)}, "
                        f"Test: {len(self.test_dataset) if self.test_dataset else 0}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        stats = {}
        
        if self.train_dataset:
            stats['train'] = self.train_dataset.get_statistics()
        
        if self.val_dataset:
            stats['val'] = self.val_dataset.get_statistics()
        
        if self.test_dataset:
            stats['test'] = self.test_dataset.get_statistics()
        
        return stats