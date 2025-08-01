"""数据处理工具模块

该模块提供了数据处理相关的工具函数，包括：
- 数据归一化和标准化
- 数据增强
- 数据集分割
- 数据格式转换
- 数据统计和验证
"""

import numpy as np
import h5py
import torch
from torch.utils.data import random_split, Subset
from typing import Tuple, Dict, Any, List, Optional, Union
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from scipy import ndimage
import cv2

logger = logging.getLogger(__name__)


def normalize_data(data: np.ndarray, 
                  method: str = "zscore", 
                  percentile: Tuple[float, float] = (1, 99),
                  axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """数据归一化
    
    Args:
        data: 输入数据
        method: 归一化方法 (zscore, minmax, percentile, robust)
        percentile: 百分位数范围（用于percentile方法）
        axis: 计算统计量的轴
        
    Returns:
        Tuple[np.ndarray, Dict]: 归一化后的数据和统计信息
    """
    stats = {}
    
    if method == "zscore":
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        std = np.where(std == 0, 1, std)  # 避免除零
        normalized = (data - mean) / std
        stats = {"mean": mean, "std": std, "method": "zscore"}
        
    elif method == "minmax":
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)  # 避免除零
        normalized = (data - min_val) / range_val
        stats = {"min": min_val, "max": max_val, "method": "minmax"}
        
    elif method == "percentile":
        p_low, p_high = percentile
        low = np.percentile(data, p_low, axis=axis, keepdims=True)
        high = np.percentile(data, p_high, axis=axis, keepdims=True)
        range_val = high - low
        range_val = np.where(range_val == 0, 1, range_val)  # 避免除零
        normalized = np.clip((data - low) / range_val, 0, 1)
        stats = {"low": low, "high": high, "percentile": percentile, "method": "percentile"}
        
    elif method == "robust":
        median = np.median(data, axis=axis, keepdims=True)
        mad = np.median(np.abs(data - median), axis=axis, keepdims=True)
        mad = np.where(mad == 0, 1, mad)  # 避免除零
        normalized = (data - median) / (1.4826 * mad)  # 1.4826是正态分布的MAD缩放因子
        stats = {"median": median, "mad": mad, "method": "robust"}
        
    else:
        raise ValueError(f"不支持的归一化方法: {method}")
    
    logger.info(f"数据归一化完成，方法: {method}")
    return normalized, stats


def denormalize_data(data: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    """数据反归一化
    
    Args:
        data: 归一化后的数据
        stats: 归一化统计信息
        
    Returns:
        np.ndarray: 反归一化后的数据
    """
    method = stats["method"]
    
    if method == "zscore":
        return data * stats["std"] + stats["mean"]
    elif method == "minmax":
        return data * (stats["max"] - stats["min"]) + stats["min"]
    elif method == "percentile":
        return data * (stats["high"] - stats["low"]) + stats["low"]
    elif method == "robust":
        return data * (1.4826 * stats["mad"]) + stats["median"]
    else:
        raise ValueError(f"不支持的反归一化方法: {method}")


def augment_data(data: np.ndarray, 
                augmentation_params: Dict[str, Any]) -> np.ndarray:
    """数据增强
    
    Args:
        data: 输入数据 (H, W) 或 (C, H, W)
        augmentation_params: 增强参数
        
    Returns:
        np.ndarray: 增强后的数据
    """
    augmented = data.copy()
    
    # 翻转
    if augmentation_params.get("flip_horizontal", False):
        if np.random.random() < augmentation_params.get("flip_prob", 0.5):
            augmented = np.flip(augmented, axis=-1)
    
    if augmentation_params.get("flip_vertical", False):
        if np.random.random() < augmentation_params.get("flip_prob", 0.5):
            augmented = np.flip(augmented, axis=-2)
    
    # 旋转
    if augmentation_params.get("rotation", False):
        if np.random.random() < augmentation_params.get("rotation_prob", 0.3):
            angle = np.random.uniform(-augmentation_params.get("max_angle", 15),
                                    augmentation_params.get("max_angle", 15))
            if len(augmented.shape) == 2:
                augmented = ndimage.rotate(augmented, angle, reshape=False, mode='reflect')
            else:
                for i in range(augmented.shape[0]):
                    augmented[i] = ndimage.rotate(augmented[i], angle, reshape=False, mode='reflect')
    
    # 缩放
    if augmentation_params.get("scaling", False):
        if np.random.random() < augmentation_params.get("scaling_prob", 0.3):
            scale = np.random.uniform(augmentation_params.get("min_scale", 0.9),
                                    augmentation_params.get("max_scale", 1.1))
            if len(augmented.shape) == 2:
                augmented = ndimage.zoom(augmented, scale, mode='reflect')
            else:
                for i in range(augmented.shape[0]):
                    augmented[i] = ndimage.zoom(augmented[i], scale, mode='reflect')
    
    # 添加噪声
    if augmentation_params.get("noise", False):
        if np.random.random() < augmentation_params.get("noise_prob", 0.2):
            noise_std = augmentation_params.get("noise_std", 0.1)
            noise = np.random.normal(0, noise_std, augmented.shape)
            augmented = augmented + noise
    
    # 亮度调整
    if augmentation_params.get("brightness", False):
        if np.random.random() < augmentation_params.get("brightness_prob", 0.3):
            factor = np.random.uniform(augmentation_params.get("min_brightness", 0.8),
                                     augmentation_params.get("max_brightness", 1.2))
            augmented = augmented * factor
    
    # 对比度调整
    if augmentation_params.get("contrast", False):
        if np.random.random() < augmentation_params.get("contrast_prob", 0.3):
            factor = np.random.uniform(augmentation_params.get("min_contrast", 0.8),
                                     augmentation_params.get("max_contrast", 1.2))
            mean = np.mean(augmented)
            augmented = (augmented - mean) * factor + mean
    
    return augmented


def split_dataset(dataset, 
                 train_ratio: float = 0.8, 
                 val_ratio: float = 0.1, 
                 test_ratio: float = 0.1,
                 random_seed: int = 42) -> Tuple:
    """数据集分割
    
    Args:
        dataset: 数据集
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
        
    Returns:
        Tuple: (训练集, 验证集, 测试集)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须等于1"
    
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # 使用PyTorch的random_split
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    logger.info(f"数据集分割完成: 训练集={train_size}, 验证集={val_size}, 测试集={test_size}")
    return train_dataset, val_dataset, test_dataset


def balance_dataset(dataset, labels: np.ndarray, method: str = "oversample") -> Subset:
    """数据集平衡
    
    Args:
        dataset: 数据集
        labels: 标签数组
        method: 平衡方法 (oversample, undersample, smote)
        
    Returns:
        Subset: 平衡后的数据集
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    if method == "oversample":
        # 过采样到最大类别的数量
        max_count = np.max(counts)
        balanced_indices = []
        
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            # 重复采样到max_count
            oversampled_indices = np.random.choice(
                label_indices, size=max_count, replace=True
            )
            balanced_indices.extend(oversampled_indices)
        
    elif method == "undersample":
        # 欠采样到最小类别的数量
        min_count = np.min(counts)
        balanced_indices = []
        
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            # 随机选择min_count个样本
            undersampled_indices = np.random.choice(
                label_indices, size=min_count, replace=False
            )
            balanced_indices.extend(undersampled_indices)
    
    else:
        raise ValueError(f"不支持的平衡方法: {method}")
    
    balanced_indices = np.array(balanced_indices)
    np.random.shuffle(balanced_indices)
    
    logger.info(f"数据集平衡完成，方法: {method}, 样本数: {len(balanced_indices)}")
    return Subset(dataset, balanced_indices)


def calculate_dataset_stats(data: np.ndarray) -> Dict[str, Any]:
    """计算数据集统计信息
    
    Args:
        data: 输入数据
        
    Returns:
        Dict[str, Any]: 统计信息
    """
    stats = {
        "shape": data.shape,
        "dtype": str(data.dtype),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "median": float(np.median(data)),
        "percentile_1": float(np.percentile(data, 1)),
        "percentile_99": float(np.percentile(data, 99)),
        "zeros_ratio": float(np.mean(data == 0)),
        "nan_count": int(np.sum(np.isnan(data))),
        "inf_count": int(np.sum(np.isinf(data)))
    }
    
    logger.info(f"数据集统计完成: shape={stats['shape']}, mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    return stats


def load_hdf5_data(file_path: Union[str, Path], 
                  keys: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """加载HDF5数据
    
    Args:
        file_path: HDF5文件路径
        keys: 要加载的键列表，None表示加载所有
        
    Returns:
        Dict[str, np.ndarray]: 数据字典
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"HDF5文件不存在: {file_path}")
    
    data = {}
    
    with h5py.File(file_path, 'r') as f:
        if keys is None:
            keys = list(f.keys())
        
        for key in keys:
            if key in f:
                data[key] = f[key][:]
                logger.debug(f"加载数据: {key}, shape={data[key].shape}")
            else:
                logger.warning(f"键不存在: {key}")
    
    logger.info(f"HDF5数据加载完成: {file_path}, 键数量={len(data)}")
    return data


def save_hdf5_data(data: Dict[str, np.ndarray], 
                  file_path: Union[str, Path],
                  compression: str = "gzip") -> None:
    """保存HDF5数据
    
    Args:
        data: 数据字典
        file_path: 保存路径
        compression: 压缩方法
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(file_path, 'w') as f:
        for key, value in data.items():
            f.create_dataset(key, data=value, compression=compression)
            logger.debug(f"保存数据: {key}, shape={value.shape}")
    
    logger.info(f"HDF5数据保存完成: {file_path}")


def convert_data_format(data: np.ndarray, 
                       from_format: str, 
                       to_format: str) -> np.ndarray:
    """数据格式转换
    
    Args:
        data: 输入数据
        from_format: 源格式 (CHW, HWC, NCHW, NHWC)
        to_format: 目标格式
        
    Returns:
        np.ndarray: 转换后的数据
    """
    from_format = from_format.upper()
    to_format = to_format.upper()
    
    if from_format == to_format:
        return data
    
    # 定义转换映射
    conversions = {
        ("CHW", "HWC"): (1, 2, 0),
        ("HWC", "CHW"): (2, 0, 1),
        ("NCHW", "NHWC"): (0, 2, 3, 1),
        ("NHWC", "NCHW"): (0, 3, 1, 2),
    }
    
    if (from_format, to_format) in conversions:
        axes = conversions[(from_format, to_format)]
        converted = np.transpose(data, axes)
        logger.info(f"数据格式转换: {from_format} -> {to_format}")
        return converted
    else:
        raise ValueError(f"不支持的格式转换: {from_format} -> {to_format}")


def validate_data_format(data: np.ndarray, 
                        expected_format: str,
                        expected_shape: Optional[Tuple[int, ...]] = None) -> bool:
    """验证数据格式
    
    Args:
        data: 输入数据
        expected_format: 期望格式
        expected_shape: 期望形状
        
    Returns:
        bool: 是否符合期望格式
    """
    # 检查维度
    expected_dims = len(expected_format)
    if len(data.shape) != expected_dims:
        logger.error(f"维度不匹配: 期望{expected_dims}维，实际{len(data.shape)}维")
        return False
    
    # 检查形状
    if expected_shape is not None:
        if data.shape != expected_shape:
            logger.error(f"形状不匹配: 期望{expected_shape}，实际{data.shape}")
            return False
    
    # 检查数据类型
    if not np.isfinite(data).all():
        logger.error("数据包含NaN或Inf值")
        return False
    
    logger.info(f"数据格式验证通过: {expected_format}, shape={data.shape}")
    return True


def create_data_splits(data_path: Union[str, Path],
                      output_dir: Union[str, Path],
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.1,
                      random_seed: int = 42) -> None:
    """创建数据分割文件
    
    Args:
        data_path: 原始数据路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
    """
    # 加载数据
    data = load_hdf5_data(data_path)
    
    # 获取样本数量
    sample_keys = list(data.keys())
    num_samples = len(data[sample_keys[0]])
    
    # 创建索引
    indices = np.arange(num_samples)
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    # 分割索引
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存分割数据
    splits = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices
    }
    
    for split_name, split_indices in splits.items():
        split_data = {}
        for key, value in data.items():
            split_data[key] = value[split_indices]
        
        split_path = output_dir / f"{split_name}.h5"
        save_hdf5_data(split_data, split_path)
        logger.info(f"保存{split_name}数据: {len(split_indices)}个样本")


def apply_transforms(data: np.ndarray, 
                    transforms: List[Dict[str, Any]]) -> np.ndarray:
    """应用变换序列
    
    Args:
        data: 输入数据
        transforms: 变换列表
        
    Returns:
        np.ndarray: 变换后的数据
    """
    result = data.copy()
    
    for transform in transforms:
        transform_type = transform.get("type")
        params = transform.get("params", {})
        
        if transform_type == "normalize":
            result, _ = normalize_data(result, **params)
        elif transform_type == "augment":
            result = augment_data(result, params)
        elif transform_type == "convert_format":
            result = convert_data_format(result, **params)
        else:
            logger.warning(f"未知的变换类型: {transform_type}")
    
    return result