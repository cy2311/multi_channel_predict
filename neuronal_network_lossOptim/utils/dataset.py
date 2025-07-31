#!/usr/bin/env python3
"""
DECODE数据集加载器

用于加载TIFF图像和对应的emitters.h5标注文件进行训练
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tifffile as tiff
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import random


class DECODEDataset(Dataset):
    """DECODE数据集类
    
    加载TIFF图像序列和对应的emitters.h5标注文件
    """
    
    def __init__(self, 
                 data_root: str,
                 sample_list: Optional[List[str]] = None,
                 frames_per_sample: int = 200,
                 consecutive_frames: int = 3,
                 image_size: int = 256,
                 transform=None):
        """
        Parameters
        ----------
        data_root : str
            数据根目录路径
        sample_list : list, optional
            要使用的样本列表，如果为None则使用所有样本
        frames_per_sample : int
            每个样本的帧数
        consecutive_frames : int
            连续帧数（用于第一级网络）
        image_size : int
            图像尺寸
        transform : callable, optional
            数据变换函数
        """
        self.data_root = Path(data_root)
        self.frames_per_sample = frames_per_sample
        self.consecutive_frames = consecutive_frames
        self.image_size = image_size
        self.transform = transform
        
        # 获取所有样本
        if sample_list is None:
            self.samples = self._get_all_samples()
        else:
            self.samples = sample_list
            
        # 验证样本存在性
        self.samples = self._validate_samples()
        
        print(f"找到 {len(self.samples)} 个有效样本")
        
        # 创建数据索引
        self.data_indices = self._create_data_indices()
        
        print(f"总共 {len(self.data_indices)} 个训练样本")
    
    def _get_all_samples(self) -> List[str]:
        """获取所有样本文件夹"""
        samples = []
        for item in self.data_root.iterdir():
            if item.is_dir() and item.name.startswith('sample_'):
                samples.append(item.name)
        return sorted(samples)
    
    def _validate_samples(self) -> List[str]:
        """验证样本的有效性"""
        valid_samples = []
        for sample in self.samples:
            sample_dir = self.data_root / sample
            tiff_file = sample_dir / f"{sample}.ome.tiff"
            h5_file = sample_dir / "emitters.h5"
            
            if tiff_file.exists() and h5_file.exists():
                valid_samples.append(sample)
            else:
                print(f"警告: 样本 {sample} 缺少必要文件")
        
        return valid_samples
    
    def _create_data_indices(self) -> List[Tuple[str, int]]:
        """创建数据索引
        
        Returns
        -------
        indices : list
            (sample_name, start_frame) 的列表
        """
        indices = []
        
        for sample in self.samples:
            # 每个原始样本只生成一个训练样本，从第0帧开始
            start_frame = 0
            indices.append((sample, start_frame))
        
        return indices
    
    def __len__(self) -> int:
        return len(self.data_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个数据样本
        
        Returns
        -------
        data : dict
            包含以下键的字典:
            - 'frames': 连续帧图像 [consecutive_frames, H, W]
            - 'emitter_positions': 发射器位置 [N, 3] (x, y, z)
            - 'emitter_photons': 发射器光子数 [N]
            - 'emitter_frame_ids': 发射器所在帧ID [N]
            - 'sample_name': 样本名称
            - 'start_frame': 起始帧
        """
        sample_name, start_frame = self.data_indices[idx]
        
        # 加载TIFF图像
        frames = self._load_frames(sample_name, start_frame)
        
        # 加载emitters数据
        emitter_data = self._load_emitters(sample_name, start_frame)
        
        # 应用变换
        if self.transform:
            frames = self.transform(frames)
        
        return {
            'frames': frames,
            'emitter_positions': emitter_data['positions'],
            'emitter_photons': emitter_data['photons'],
            'emitter_frame_ids': emitter_data['frame_ids'],
            'sample_name': sample_name,
            'start_frame': start_frame
        }
    
    def _load_frames(self, sample_name: str, start_frame: int) -> torch.Tensor:
        """加载连续帧图像"""
        sample_dir = self.data_root / sample_name
        tiff_file = sample_dir / f"{sample_name}.ome.tiff"
        
        # 读取TIFF文件
        with tiff.TiffFile(tiff_file) as tif:
            images = tif.asarray()
        
        # 提取连续帧
        end_frame = start_frame + self.consecutive_frames
        frames = images[start_frame:end_frame]
        
        # 转换为torch tensor并归一化
        frames = torch.from_numpy(frames.astype(np.float32))
        
        # 简单归一化到[0,1]
        frames = frames / (frames.max() + 1e-8)
        
        # 调整图像尺寸到指定大小
        if self.image_size != frames.shape[1] or self.image_size != frames.shape[2]:
            # 使用双线性插值调整尺寸
            frames = torch.nn.functional.interpolate(
                frames.unsqueeze(0), 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        return frames
    
    def _load_emitters(self, sample_name: str, start_frame: int) -> Dict[str, torch.Tensor]:
        """加载emitters标注数据"""
        sample_dir = self.data_root / sample_name
        h5_file = sample_dir / "emitters.h5"
        
        with h5py.File(h5_file, 'r') as f:
            # 读取记录数据
            frame_ix = np.array(f['records/frame_ix'])
            xyz_rec = np.array(f['records/xyz'])
            phot_rec = np.array(f['records/phot'])
        
        # 筛选当前帧范围内的发射器
        end_frame = start_frame + self.consecutive_frames
        mask = (frame_ix >= start_frame) & (frame_ix < end_frame)
        
        # 提取相关数据
        positions = xyz_rec[mask]  # [N, 3]
        photons = phot_rec[mask]   # [N]
        frame_ids = frame_ix[mask] - start_frame  # 相对帧ID [N]
        
        return {
            'positions': torch.from_numpy(positions.astype(np.float32)),
            'photons': torch.from_numpy(photons.astype(np.float32)),
            'frame_ids': torch.from_numpy(frame_ids.astype(np.int64))
        }


def create_train_val_split(data_root: str, 
                          train_ratio: float = 0.8,
                          seed: int = 42) -> Tuple[List[str], List[str]]:
    """创建训练和验证集划分
    
    Parameters
    ----------
    data_root : str
        数据根目录
    train_ratio : float
        训练集比例
    seed : int
        随机种子
        
    Returns
    -------
    train_samples : list
        训练样本列表
    val_samples : list
        验证样本列表
    """
    data_root = Path(data_root)
    
    # 获取所有样本
    all_samples = []
    for item in data_root.iterdir():
        if item.is_dir() and item.name.startswith('sample_'):
            all_samples.append(item.name)
    
    all_samples = sorted(all_samples)
    
    # 随机划分
    random.seed(seed)
    random.shuffle(all_samples)
    
    split_idx = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    print(f"训练样本: {len(train_samples)}, 验证样本: {len(val_samples)}")
    
    return train_samples, val_samples


def custom_collate_fn(batch):
    """自定义collate函数，处理变长的emitter数据"""
    # 分离不同类型的数据
    frames = torch.stack([item['frames'] for item in batch])
    sample_names = [item['sample_name'] for item in batch]
    start_frames = [item['start_frame'] for item in batch]
    
    # 处理变长的emitter数据
    batch_size = len(batch)
    max_emitters = max(len(item['emitter_positions']) for item in batch)
    
    if max_emitters == 0:
        # 如果没有发射器，创建空张量
        emitter_positions = torch.zeros(batch_size, 0, 3)
        emitter_photons = torch.zeros(batch_size, 0)
        emitter_frame_ids = torch.zeros(batch_size, 0, dtype=torch.long)
    else:
        # 填充到相同长度
        emitter_positions = torch.zeros(batch_size, max_emitters, 3)
        emitter_photons = torch.zeros(batch_size, max_emitters)
        emitter_frame_ids = torch.zeros(batch_size, max_emitters, dtype=torch.long)
        
        for i, item in enumerate(batch):
            n_emitters = len(item['emitter_positions'])
            if n_emitters > 0:
                emitter_positions[i, :n_emitters] = item['emitter_positions']
                emitter_photons[i, :n_emitters] = item['emitter_photons']
                emitter_frame_ids[i, :n_emitters] = item['emitter_frame_ids']
    
    return {
        'frames': frames,
        'emitter_positions': emitter_positions,
        'emitter_photons': emitter_photons,
        'emitter_frame_ids': emitter_frame_ids,
        'sample_name': sample_names,
        'start_frame': start_frames
    }


def create_dataloaders(data_root: str,
                      batch_size: int = 4,
                      num_workers: int = 4,
                      train_ratio: float = 0.8,
                      sample_subset: Optional[int] = None,
                      **dataset_kwargs) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器
    
    Parameters
    ----------
    data_root : str
        数据根目录
    batch_size : int
        批次大小
    num_workers : int
        数据加载进程数
    train_ratio : float
        训练集比例
    sample_subset : int, optional
        使用的样本子集数量（用于测试不同样本数的影响）
    **dataset_kwargs
        传递给Dataset的其他参数
        
    Returns
    -------
    train_loader : DataLoader
        训练数据加载器
    val_loader : DataLoader
        验证数据加载器
    """
    # 创建训练验证划分
    train_samples, val_samples = create_train_val_split(data_root, train_ratio)
    
    # 如果指定了样本子集，则截取
    if sample_subset is not None:
        train_subset_size = int(sample_subset * train_ratio)
        val_subset_size = sample_subset - train_subset_size
        
        train_samples = train_samples[:train_subset_size]
        val_samples = val_samples[:val_subset_size]
        
        print(f"使用样本子集: 训练 {len(train_samples)}, 验证 {len(val_samples)}")
    
    # 创建数据集
    train_dataset = DECODEDataset(data_root, train_samples, **dataset_kwargs)
    val_dataset = DECODEDataset(data_root, val_samples, **dataset_kwargs)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # 测试数据加载器
    data_root = "/home/guest/Others/DECODE_rewrite/simulation_zmap2tiff/outputs_100samples_256"
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        data_root=data_root,
        batch_size=2,
        num_workers=2,
        sample_subset=10  # 使用10个样本进行测试
    )
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    
    # 测试加载一个批次
    for batch in train_loader:
        print(f"帧形状: {batch['frames'].shape}")
        print(f"发射器位置形状: {batch['emitter_positions'].shape}")
        print(f"发射器光子数形状: {batch['emitter_photons'].shape}")
        print(f"发射器帧ID形状: {batch['emitter_frame_ids'].shape}")
        break