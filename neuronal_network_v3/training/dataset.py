"""数据集和数据加载器实现

包含SMLM数据的处理、加载和预处理功能。
"""

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import random
from scipy import ndimage
from skimage import transform
import warnings


class SMLMStaticDataset(Dataset):
    """SMLM静态数据集
    
    用于训练的静态数据集，支持：
    - HDF5文件加载
    - 数据增强
    - 多帧处理
    - 目标生成
    - 权重计算
    
    Args:
        data_path: 数据文件路径
        target_generator: 目标生成器
        weight_generator: 权重生成器
        transform: 数据变换
        frame_window: 帧窗口大小
        cache_size: 缓存大小
    """
    
    def __init__(self,
                 data_path: Union[str, Path, List[str]],
                 target_generator: Optional[Callable] = None,
                 weight_generator: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 frame_window: int = 3,
                 cache_size: int = 1000,
                 preload: bool = False):
        
        self.data_paths = self._parse_data_paths(data_path)
        self.target_generator = target_generator
        self.weight_generator = weight_generator
        self.transform = transform
        self.frame_window = frame_window
        self.cache_size = cache_size
        self.preload = preload
        
        # 数据索引
        self.data_index = []
        self.data_cache = {}
        
        # 构建数据索引
        self._build_index()
        
        # 预加载数据
        if self.preload:
            self._preload_data()
    
    def _parse_data_paths(self, data_path: Union[str, Path, List[str]]) -> List[Path]:
        """解析数据路径"""
        if isinstance(data_path, (str, Path)):
            path = Path(data_path)
            if path.is_file():
                return [path]
            elif path.is_dir():
                return list(path.glob('*.h5')) + list(path.glob('*.hdf5'))
            else:
                raise ValueError(f"Data path does not exist: {path}")
        elif isinstance(data_path, list):
            return [Path(p) for p in data_path]
        else:
            raise ValueError("Invalid data_path type")
    
    def _build_index(self):
        """构建数据索引"""
        for file_path in self.data_paths:
            try:
                with h5py.File(file_path, 'r') as f:
                    # 检查数据结构：emitters, records
                    if 'emitters' in f and 'records' in f:
                        # 每个文件只创建一个样本，从第0帧开始
                        self.data_index.append({
                            'file_path': file_path,
                            'frame_start': 0,
                            'frame_end': self.frame_window
                        })
                    else:
                        print(f"Warning: Invalid data structure in {file_path}")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        print(f"Built index with {len(self.data_index)} samples from {len(self.data_paths)} files")
    
    def _preload_data(self):
        """预加载数据到内存"""
        print("Preloading data...")
        for file_path in self.data_paths:
            try:
                with h5py.File(file_path, 'r') as f:
                    if 'frames' in f:
                        self.data_cache[str(file_path)] = {
                            'frames': f['frames'][:],
                            'emitters': {
                                'xyz': f['emitters']['xyz'][:],
                                'intensity': f['emitters']['intensity'][:],
                                'id': f['emitters']['id'][:]
                            } if 'emitters' in f else None,
                            'metadata': dict(f.attrs) if f.attrs else {}
                        }
                    else:
                        # 对于新数据结构，暂时不预加载（避免内存问题）
                        print(f"Skipping preload for new format file: {file_path}")
            except Exception as e:
                print(f"Error preloading {file_path}: {e}")
        print(f"Preloaded {len(self.data_cache)} files")
    
    def __len__(self) -> int:
        return len(self.data_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取数据样本"""
        if idx >= len(self.data_index):
            raise IndexError(f"Index {idx} out of range")
        
        sample_info = self.data_index[idx]
        file_path = sample_info['file_path']
        frame_start = sample_info['frame_start']
        frame_end = sample_info['frame_end']
        
        # 加载数据
        if str(file_path) in self.data_cache:
            # 从缓存加载
            data = self.data_cache[str(file_path)]
            frames = data['frames'][frame_start:frame_end]
            emitters = data['emitters']
            metadata = data['metadata']
        else:
            # 从文件加载
            with h5py.File(file_path, 'r') as f:
                if 'frames' in f:
                    frames = f['frames'][frame_start:frame_end]
                    emitters = f['emitters'][:] if 'emitters' in f else None
                else:
                    # 新的数据结构：从records重建帧数据
                    records = f['records']
                    frame_indices = records['frame_ix'][:]
                    
                    # 获取指定帧范围的记录
                    mask = (frame_indices >= frame_start) & (frame_indices < frame_end)
                    if not np.any(mask):
                        # 如果没有记录，创建空帧
                        frames = np.zeros((frame_end - frame_start, 64, 64), dtype=np.float32)
                    else:
                        # 从记录重建帧（简化版本，实际可能需要更复杂的重建逻辑）
                        frames = np.zeros((frame_end - frame_start, 64, 64), dtype=np.float32)
                        # 这里可以根据records中的xyz和phot信息重建帧
                        # 暂时使用简化版本
                    
                    # emitters是一个组，包含多个数据集
                    emitters = {
                        'xyz': f['emitters']['xyz'][:],
                        'intensity': f['emitters']['intensity'][:],
                        'id': f['emitters']['id'][:]
                    } if 'emitters' in f else None
                metadata = dict(f.attrs) if f.attrs else {}
        
        # 转换为torch张量
        frames = torch.from_numpy(frames.astype(np.float32))
        
        # 数据归一化（关键修复）
        if frames.max() > 1.0:  # 如果数据不在[0,1]范围内
            frames = frames / frames.max()  # 简单的最大值归一化
        
        # 处理多帧：(T, H, W) -> (T, H, W) 或 (3*H, W)
        if self.frame_window == 1:
            input_tensor = frames.squeeze(0)  # (H, W)
        elif self.frame_window == 3:
            # 可以选择堆叠或连接
            input_tensor = frames  # (3, H, W)
        else:
            input_tensor = frames  # (T, H, W)
        
        # 确保输入是3D: (C, H, W)
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(0)  # (1, H, W)
        
        sample = {
            'input': input_tensor,
            'frame_indices': torch.tensor([frame_start, frame_end]),
            'file_path': str(file_path),
            'metadata': metadata
        }
        
        # 生成目标
        if self.target_generator is not None and emitters is not None:
            # 过滤当前帧窗口的发射器
            frame_emitters = self._filter_emitters_by_frame(emitters, frame_start, frame_end)
            target = self.target_generator(frame_emitters, input_tensor.shape[-2:])
            sample['target'] = target
            sample['emitters'] = frame_emitters
        
        # 生成权重
        if self.weight_generator is not None:
            if 'emitters' in sample:
                weight = self.weight_generator(sample['emitters'], input_tensor.shape[-2:])
            else:
                weight = self.weight_generator(None, input_tensor.shape[-2:])
            sample['weight'] = weight
        
        # 应用变换
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def _filter_emitters_by_frame(self, emitters, frame_start: int, frame_end: int):
        """根据帧范围过滤发射器"""
        if emitters is None:
            return np.array([])
        
        # 处理新的数据结构（字典格式）
        if isinstance(emitters, dict):
            if 'xyz' in emitters and 'intensity' in emitters:
                # 创建标准格式的emitters数组：[frame, x, y, z, photons]
                xyz = emitters['xyz']
                intensity = emitters['intensity']
                
                if len(xyz) == 0:
                    return np.array([])
                
                # 假设所有发射器都在当前帧范围内（简化处理）
                n_emitters = len(xyz)
                frame_col = np.full(n_emitters, frame_start)  # 使用frame_start作为帧号
                
                # 组合成标准格式：[frame, x, y, z, photons]
                filtered_emitters = np.column_stack([
                    frame_col,
                    xyz[:, 0],  # x
                    xyz[:, 1],  # y
                    xyz[:, 2] if xyz.shape[1] > 2 else np.zeros(n_emitters),  # z
                    intensity   # photons
                ])
                
                return filtered_emitters
            else:
                return np.array([])
        
        # 处理原始格式（数组）
        elif isinstance(emitters, np.ndarray):
            if len(emitters) == 0:
                return np.array([])
            
            if emitters.ndim == 2 and emitters.shape[1] >= 5:
                frame_mask = (emitters[:, 0] >= frame_start) & (emitters[:, 0] < frame_end)
                return emitters[frame_mask]
            else:
                return emitters
        
        return np.array([])
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """获取样本信息"""
        return self.data_index[idx]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        stats = {
            'num_samples': len(self.data_index),
            'num_files': len(self.data_paths),
            'frame_window': self.frame_window,
            'cache_size': len(self.data_cache),
            'preloaded': self.preload
        }
        
        # 计算数据统计
        if self.preload and self.data_cache:
            all_frames = []
            all_emitters = []
            
            for data in self.data_cache.values():
                if data['frames'] is not None:
                    all_frames.append(data['frames'])
                if data['emitters'] is not None:
                    all_emitters.append(data['emitters'])
            
            if all_frames:
                frames_concat = np.concatenate(all_frames, axis=0)
                stats.update({
                    'frame_shape': frames_concat.shape[1:],
                    'frame_mean': float(frames_concat.mean()),
                    'frame_std': float(frames_concat.std()),
                    'frame_min': float(frames_concat.min()),
                    'frame_max': float(frames_concat.max())
                })
            
            if all_emitters:
                emitters_concat = np.concatenate(all_emitters, axis=0)
                stats.update({
                    'num_emitters': len(emitters_concat),
                    'emitters_per_frame': len(emitters_concat) / len(frames_concat) if all_frames else 0
                })
        
        return stats


class InferenceDataset(Dataset):
    """推理数据集
    
    用于推理的数据集，支持：
    - 实时数据流
    - 批量推理
    - 内存映射
    
    Args:
        data_source: 数据源（文件路径或数据数组）
        frame_window: 帧窗口大小
        overlap: 帧重叠
        transform: 数据变换
    """
    
    def __init__(self,
                 data_source: Union[str, Path, np.ndarray, torch.Tensor],
                 frame_window: int = 3,
                 overlap: int = 1,
                 transform: Optional[Callable] = None,
                 memory_map: bool = True):
        
        self.frame_window = frame_window
        self.overlap = overlap
        self.transform = transform
        self.memory_map = memory_map
        
        # 加载数据
        self.data = self._load_data(data_source)
        
        # 计算样本数量
        self.num_frames = self.data.shape[0]
        self.stride = self.frame_window - self.overlap
        self.num_samples = max(1, (self.num_frames - self.frame_window) // self.stride + 1)
    
    def _load_data(self, data_source: Union[str, Path, np.ndarray, torch.Tensor]) -> np.ndarray:
        """加载数据"""
        if isinstance(data_source, (str, Path)):
            path = Path(data_source)
            if path.suffix in ['.h5', '.hdf5']:
                with h5py.File(path, 'r') as f:
                    if self.memory_map:
                        # 使用内存映射
                        return f['frames']
                    else:
                        return f['frames'][:]
            elif path.suffix in ['.npy']:
                if self.memory_map:
                    return np.load(path, mmap_mode='r')
                else:
                    return np.load(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        elif isinstance(data_source, np.ndarray):
            return data_source
        elif isinstance(data_source, torch.Tensor):
            return data_source.numpy()
        else:
            raise ValueError("Invalid data_source type")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取推理样本"""
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range")
        
        # 计算帧范围
        frame_start = idx * self.stride
        frame_end = min(frame_start + self.frame_window, self.num_frames)
        
        # 提取帧
        frames = self.data[frame_start:frame_end]
        
        # 处理不足的帧（填充）
        if frames.shape[0] < self.frame_window:
            padding = np.zeros((self.frame_window - frames.shape[0],) + frames.shape[1:], 
                             dtype=frames.dtype)
            frames = np.concatenate([frames, padding], axis=0)
        
        # 转换为torch张量
        frames = torch.from_numpy(frames.astype(np.float32))
        
        # 处理输入格式
        if self.frame_window == 1:
            input_tensor = frames.squeeze(0)  # (H, W)
        else:
            input_tensor = frames  # (T, H, W)
        
        # 确保输入是3D: (C, H, W)
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(0)  # (1, H, W)
        
        sample = {
            'input': input_tensor,
            'frame_indices': torch.tensor([frame_start, frame_end]),
            'sample_idx': idx
        }
        
        # 应用变换
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def get_frame_range(self, idx: int) -> Tuple[int, int]:
        """获取样本对应的帧范围"""
        frame_start = idx * self.stride
        frame_end = min(frame_start + self.frame_window, self.num_frames)
        return frame_start, frame_end


class SMLMDataLoader:
    """SMLM数据加载器工厂
    
    提供便捷的数据加载器创建方法
    """
    
    @staticmethod
    def create_train_loader(dataset: Dataset,
                          batch_size: int = 32,
                          shuffle: bool = True,
                          num_workers: int = 4,
                          pin_memory: bool = True,
                          drop_last: bool = True,
                          distributed: bool = False,
                          **kwargs) -> DataLoader:
        """创建训练数据加载器"""
        sampler = None
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False  # sampler已经处理了shuffle
        
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=SMLMDataLoader._collate_fn,
            **kwargs
        )
    
    @staticmethod
    def create_val_loader(dataset: Dataset,
                        batch_size: int = 32,
                        num_workers: int = 4,
                        pin_memory: bool = True,
                        distributed: bool = False,
                        **kwargs) -> DataLoader:
        """创建验证数据加载器"""
        sampler = None
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=False)
        
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=SMLMDataLoader._collate_fn,
            **kwargs
        )
    
    @staticmethod
    def create_inference_loader(dataset: Dataset,
                              batch_size: int = 1,
                              num_workers: int = 1,
                              pin_memory: bool = True,
                              **kwargs) -> DataLoader:
        """创建推理数据加载器"""
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=SMLMDataLoader._collate_fn,
            **kwargs
        )
    
    @staticmethod
    def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """自定义批处理函数"""
        if not batch:
            return {}
        
        # 获取所有键
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            values = [sample[key] for sample in batch]
            
            if key == 'input' or key == 'target' or key == 'weight':
                # 张量数据
                if all(isinstance(v, torch.Tensor) for v in values):
                    collated[key] = torch.stack(values, dim=0)
                else:
                    collated[key] = values
            elif key in ['frame_indices', 'sample_idx']:
                # 整数张量
                if all(isinstance(v, torch.Tensor) for v in values):
                    collated[key] = torch.stack(values, dim=0)
                else:
                    collated[key] = torch.tensor(values)
            elif key in ['emitters']:
                # 发射器数据（可能长度不同）
                collated[key] = values
            else:
                # 其他数据（字符串、字典等）
                collated[key] = values
        
        return collated


# 数据变换类
class SMLMTransform:
    """SMLM数据变换基类"""
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class RandomFlip(SMLMTransform):
    """随机翻转"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.p:
            # 水平翻转
            sample['input'] = torch.flip(sample['input'], dims=[-1])
            if 'target' in sample:
                sample['target'] = torch.flip(sample['target'], dims=[-1])
            if 'weight' in sample:
                sample['weight'] = torch.flip(sample['weight'], dims=[-1])
        
        if random.random() < self.p:
            # 垂直翻转
            sample['input'] = torch.flip(sample['input'], dims=[-2])
            if 'target' in sample:
                sample['target'] = torch.flip(sample['target'], dims=[-2])
            if 'weight' in sample:
                sample['weight'] = torch.flip(sample['weight'], dims=[-2])
        
        return sample


class RandomRotation(SMLMTransform):
    """随机旋转"""
    
    def __init__(self, angles: List[int] = [0, 90, 180, 270]):
        self.angles = angles
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        angle = random.choice(self.angles)
        k = angle // 90
        
        if k > 0:
            sample['input'] = torch.rot90(sample['input'], k, dims=[-2, -1])
            if 'target' in sample:
                sample['target'] = torch.rot90(sample['target'], k, dims=[-2, -1])
            if 'weight' in sample:
                sample['weight'] = torch.rot90(sample['weight'], k, dims=[-2, -1])
        
        return sample


class AddNoise(SMLMTransform):
    """添加噪声"""
    
    def __init__(self, noise_std: float = 0.1):
        self.noise_std = noise_std
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        noise = torch.randn_like(sample['input']) * self.noise_std
        sample['input'] = sample['input'] + noise
        return sample


class Normalize(SMLMTransform):
    """归一化"""
    
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample['input'] = (sample['input'] - self.mean) / self.std
        return sample


class Compose(SMLMTransform):
    """组合变换"""
    
    def __init__(self, transforms: List[SMLMTransform]):
        self.transforms = transforms
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for transform in self.transforms:
            sample = transform(sample)
        return sample