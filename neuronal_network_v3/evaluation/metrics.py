"""评估指标模块

包含各种性能指标的计算，如检测精度、定位精度、光子数估计精度等。
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import warnings


@dataclass
class SegmentationMetrics:
    """分割指标
    
    Attributes:
        iou: 交并比
        dice: Dice系数
        pixel_accuracy: 像素准确率
        mean_iou: 平均交并比
    """
    iou: float = 0.0
    dice: float = 0.0
    pixel_accuracy: float = 0.0
    mean_iou: float = 0.0


@dataclass
class RegressionMetrics:
    """回归指标
    
    Attributes:
        mse: 均方误差
        mae: 平均绝对误差
        rmse: 均方根误差
        r2_score: R²分数
    """
    mse: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    r2_score: float = 0.0


@dataclass
class DetectionMetrics:
    """检测指标
    
    Attributes:
        precision: 精确率
        recall: 召回率
        f1_score: F1分数
        accuracy: 准确率
        specificity: 特异性
        false_positive_rate: 假阳性率
        false_negative_rate: 假阴性率
        true_positives: 真阳性数量
        false_positives: 假阳性数量
        true_negatives: 真阴性数量
        false_negatives: 假阴性数量
        jaccard_index: Jaccard指数
        matthews_correlation: Matthews相关系数
    """
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    specificity: float
    false_positive_rate: float
    false_negative_rate: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    jaccard_index: float
    matthews_correlation: float
    
    def to_dict(self) -> Dict[str, Union[float, int]]:
        """转换为字典"""
        return {
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'accuracy': self.accuracy,
            'specificity': self.specificity,
            'false_positive_rate': self.false_positive_rate,
            'false_negative_rate': self.false_negative_rate,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives,
            'jaccard_index': self.jaccard_index,
            'matthews_correlation': self.matthews_correlation
        }


@dataclass
class LocalizationMetrics:
    """定位指标
    
    Attributes:
        rmse_x: X方向均方根误差
        rmse_y: Y方向均方根误差
        rmse_z: Z方向均方根误差
        rmse_3d: 3D均方根误差
        mae_x: X方向平均绝对误差
        mae_y: Y方向平均绝对误差
        mae_z: Z方向平均绝对误差
        mae_3d: 3D平均绝对误差
        bias_x: X方向偏差
        bias_y: Y方向偏差
        bias_z: Z方向偏差
        std_x: X方向标准差
        std_y: Y方向标准差
        std_z: Z方向标准差
        lateral_precision: 横向精度
        axial_precision: 轴向精度
        efficiency: 定位效率
    """
    rmse_x: float
    rmse_y: float
    rmse_z: float
    rmse_3d: float
    mae_x: float
    mae_y: float
    mae_z: float
    mae_3d: float
    bias_x: float
    bias_y: float
    bias_z: float
    std_x: float
    std_y: float
    std_z: float
    lateral_precision: float
    axial_precision: float
    efficiency: float
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'rmse_x': self.rmse_x,
            'rmse_y': self.rmse_y,
            'rmse_z': self.rmse_z,
            'rmse_3d': self.rmse_3d,
            'mae_x': self.mae_x,
            'mae_y': self.mae_y,
            'mae_z': self.mae_z,
            'mae_3d': self.mae_3d,
            'bias_x': self.bias_x,
            'bias_y': self.bias_y,
            'bias_z': self.bias_z,
            'std_x': self.std_x,
            'std_y': self.std_y,
            'std_z': self.std_z,
            'lateral_precision': self.lateral_precision,
            'axial_precision': self.axial_precision,
            'efficiency': self.efficiency
        }


@dataclass
class PhotonMetrics:
    """光子数指标
    
    Attributes:
        rmse: 均方根误差
        mae: 平均绝对误差
        mape: 平均绝对百分比误差
        bias: 偏差
        std: 标准差
        correlation: 相关系数
        r_squared: 决定系数
        relative_error: 相对误差
        efficiency: 估计效率
    """
    rmse: float
    mae: float
    mape: float
    bias: float
    std: float
    correlation: float
    r_squared: float
    relative_error: float
    efficiency: float
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'rmse': self.rmse,
            'mae': self.mae,
            'mape': self.mape,
            'bias': self.bias,
            'std': self.std,
            'correlation': self.correlation,
            'r_squared': self.r_squared,
            'relative_error': self.relative_error,
            'efficiency': self.efficiency
        }


@dataclass
class ComprehensiveMetrics:
    """综合指标
    
    Attributes:
        detection: 检测指标
        localization: 定位指标
        photon: 光子数指标
        overall_score: 总体评分
        weighted_score: 加权评分
    """
    detection: DetectionMetrics
    localization: LocalizationMetrics
    photon: PhotonMetrics
    overall_score: float
    weighted_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'detection': self.detection.to_dict(),
            'localization': self.localization.to_dict(),
            'photon': self.photon.to_dict(),
            'overall_score': self.overall_score,
            'weighted_score': self.weighted_score
        }


def calculate_detection_metrics(predicted_positions: np.ndarray,
                              ground_truth_positions: np.ndarray,
                              tolerance: float = 3.0,
                              image_shape: Optional[Tuple[int, int]] = None) -> DetectionMetrics:
    """计算检测指标
    
    Args:
        predicted_positions: 预测位置，形状为 (N, 2) 或 (N, 3)
        ground_truth_positions: 真实位置，形状为 (M, 2) 或 (M, 3)
        tolerance: 匹配容差（像素）
        image_shape: 图像形状，用于计算真阴性
        
    Returns:
        检测指标
    """
    if len(predicted_positions) == 0 and len(ground_truth_positions) == 0:
        # 特殊情况：没有预测也没有真实值
        return DetectionMetrics(
            precision=1.0, recall=1.0, f1_score=1.0, accuracy=1.0,
            specificity=1.0, false_positive_rate=0.0, false_negative_rate=0.0,
            true_positives=0, false_positives=0, true_negatives=0, false_negatives=0,
            jaccard_index=1.0, matthews_correlation=1.0
        )
    
    # 匹配预测和真实位置
    matches = _match_positions(predicted_positions, ground_truth_positions, tolerance)
    
    # 计算混淆矩阵元素
    true_positives = len(matches)
    false_positives = len(predicted_positions) - true_positives
    false_negatives = len(ground_truth_positions) - true_positives
    
    # 估计真阴性（如果提供了图像形状）
    if image_shape is not None:
        total_pixels = image_shape[0] * image_shape[1]
        # 简化估计：总像素数减去所有检测到的位置
        true_negatives = max(0, total_pixels - len(predicted_positions) - len(ground_truth_positions))
    else:
        true_negatives = 0
    
    # 计算基本指标
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    total = true_positives + false_positives + true_negatives + false_negatives
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0
    
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
    false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0
    false_negative_rate = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0.0
    
    # Jaccard指数
    jaccard_index = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0.0
    
    # Matthews相关系数
    numerator = (true_positives * true_negatives) - (false_positives * false_negatives)
    denominator = np.sqrt((true_positives + false_positives) * (true_positives + false_negatives) * 
                         (true_negatives + false_positives) * (true_negatives + false_negatives))
    matthews_correlation = numerator / denominator if denominator > 0 else 0.0
    
    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        accuracy=accuracy,
        specificity=specificity,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        jaccard_index=jaccard_index,
        matthews_correlation=matthews_correlation
    )


def calculate_localization_metrics(predicted_positions: np.ndarray,
                                  ground_truth_positions: np.ndarray,
                                  tolerance: float = 3.0,
                                  pixel_size: float = 100.0) -> LocalizationMetrics:
    """计算定位指标
    
    Args:
        predicted_positions: 预测位置，形状为 (N, 2) 或 (N, 3)
        ground_truth_positions: 真实位置，形状为 (M, 2) 或 (M, 3)
        tolerance: 匹配容差（像素）
        pixel_size: 像素大小（nm）
        
    Returns:
        定位指标
    """
    if len(predicted_positions) == 0 or len(ground_truth_positions) == 0:
        # 没有匹配的位置
        return LocalizationMetrics(
            rmse_x=float('inf'), rmse_y=float('inf'), rmse_z=float('inf'), rmse_3d=float('inf'),
            mae_x=float('inf'), mae_y=float('inf'), mae_z=float('inf'), mae_3d=float('inf'),
            bias_x=0.0, bias_y=0.0, bias_z=0.0,
            std_x=0.0, std_y=0.0, std_z=0.0,
            lateral_precision=float('inf'), axial_precision=float('inf'),
            efficiency=0.0
        )
    
    # 匹配位置
    matches = _match_positions(predicted_positions, ground_truth_positions, tolerance)
    
    if len(matches) == 0:
        # 没有匹配的位置
        return LocalizationMetrics(
            rmse_x=float('inf'), rmse_y=float('inf'), rmse_z=float('inf'), rmse_3d=float('inf'),
            mae_x=float('inf'), mae_y=float('inf'), mae_z=float('inf'), mae_3d=float('inf'),
            bias_x=0.0, bias_y=0.0, bias_z=0.0,
            std_x=0.0, std_y=0.0, std_z=0.0,
            lateral_precision=float('inf'), axial_precision=float('inf'),
            efficiency=0.0
        )
    
    # 提取匹配的位置
    pred_matched = predicted_positions[matches[:, 0]]
    gt_matched = ground_truth_positions[matches[:, 1]]
    
    # 计算误差
    errors = pred_matched - gt_matched
    
    # 确保至少有2D坐标
    if errors.shape[1] < 2:
        raise ValueError("Positions must have at least 2 dimensions (x, y)")
    
    # X, Y误差
    error_x = errors[:, 0] * pixel_size  # 转换为nm
    error_y = errors[:, 1] * pixel_size
    
    # Z误差（如果存在）
    if errors.shape[1] >= 3:
        error_z = errors[:, 2] * pixel_size
    else:
        error_z = np.zeros_like(error_x)
    
    # 3D误差
    error_3d = np.sqrt(error_x**2 + error_y**2 + error_z**2)
    
    # RMSE
    rmse_x = np.sqrt(np.mean(error_x**2))
    rmse_y = np.sqrt(np.mean(error_y**2))
    rmse_z = np.sqrt(np.mean(error_z**2)) if errors.shape[1] >= 3 else 0.0
    rmse_3d = np.sqrt(np.mean(error_3d**2))
    
    # MAE
    mae_x = np.mean(np.abs(error_x))
    mae_y = np.mean(np.abs(error_y))
    mae_z = np.mean(np.abs(error_z)) if errors.shape[1] >= 3 else 0.0
    mae_3d = np.mean(error_3d)
    
    # 偏差
    bias_x = np.mean(error_x)
    bias_y = np.mean(error_y)
    bias_z = np.mean(error_z) if errors.shape[1] >= 3 else 0.0
    
    # 标准差
    std_x = np.std(error_x)
    std_y = np.std(error_y)
    std_z = np.std(error_z) if errors.shape[1] >= 3 else 0.0
    
    # 横向和轴向精度
    lateral_precision = np.sqrt(rmse_x**2 + rmse_y**2)
    axial_precision = rmse_z
    
    # 定位效率
    efficiency = len(matches) / len(ground_truth_positions)
    
    return LocalizationMetrics(
        rmse_x=rmse_x, rmse_y=rmse_y, rmse_z=rmse_z, rmse_3d=rmse_3d,
        mae_x=mae_x, mae_y=mae_y, mae_z=mae_z, mae_3d=mae_3d,
        bias_x=bias_x, bias_y=bias_y, bias_z=bias_z,
        std_x=std_x, std_y=std_y, std_z=std_z,
        lateral_precision=lateral_precision, axial_precision=axial_precision,
        efficiency=efficiency
    )


def calculate_photon_metrics(predicted_photons: np.ndarray,
                           ground_truth_photons: np.ndarray,
                           predicted_positions: np.ndarray,
                           ground_truth_positions: np.ndarray,
                           tolerance: float = 3.0) -> PhotonMetrics:
    """计算光子数指标
    
    Args:
        predicted_photons: 预测光子数
        ground_truth_photons: 真实光子数
        predicted_positions: 预测位置
        ground_truth_positions: 真实位置
        tolerance: 匹配容差
        
    Returns:
        光子数指标
    """
    if len(predicted_photons) == 0 or len(ground_truth_photons) == 0:
        return PhotonMetrics(
            rmse=float('inf'), mae=float('inf'), mape=float('inf'),
            bias=0.0, std=0.0, correlation=0.0, r_squared=0.0,
            relative_error=float('inf'), efficiency=0.0
        )
    
    # 匹配位置
    matches = _match_positions(predicted_positions, ground_truth_positions, tolerance)
    
    if len(matches) == 0:
        return PhotonMetrics(
            rmse=float('inf'), mae=float('inf'), mape=float('inf'),
            bias=0.0, std=0.0, correlation=0.0, r_squared=0.0,
            relative_error=float('inf'), efficiency=0.0
        )
    
    # 提取匹配的光子数
    pred_photons = predicted_photons[matches[:, 0]]
    gt_photons = ground_truth_photons[matches[:, 1]]
    
    # 计算误差
    errors = pred_photons - gt_photons
    
    # RMSE
    rmse = np.sqrt(np.mean(errors**2))
    
    # MAE
    mae = np.mean(np.abs(errors))
    
    # MAPE
    mape = np.mean(np.abs(errors / gt_photons)) * 100 if np.all(gt_photons > 0) else float('inf')
    
    # 偏差和标准差
    bias = np.mean(errors)
    std = np.std(errors)
    
    # 相关系数
    correlation = np.corrcoef(pred_photons, gt_photons)[0, 1] if len(pred_photons) > 1 else 0.0
    
    # 决定系数 (R²)
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((gt_photons - np.mean(gt_photons))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # 相对误差
    relative_error = np.mean(np.abs(errors / gt_photons)) if np.all(gt_photons > 0) else float('inf')
    
    # 估计效率
    efficiency = len(matches) / len(ground_truth_photons)
    
    return PhotonMetrics(
        rmse=rmse, mae=mae, mape=mape,
        bias=bias, std=std, correlation=correlation, r_squared=r_squared,
        relative_error=relative_error, efficiency=efficiency
    )


def calculate_comprehensive_metrics(predicted_positions: np.ndarray,
                                  ground_truth_positions: np.ndarray,
                                  predicted_photons: Optional[np.ndarray] = None,
                                  ground_truth_photons: Optional[np.ndarray] = None,
                                  tolerance: float = 3.0,
                                  pixel_size: float = 100.0,
                                  image_shape: Optional[Tuple[int, int]] = None,
                                  weights: Optional[Dict[str, float]] = None) -> ComprehensiveMetrics:
    """计算综合指标
    
    Args:
        predicted_positions: 预测位置
        ground_truth_positions: 真实位置
        predicted_photons: 预测光子数
        ground_truth_photons: 真实光子数
        tolerance: 匹配容差
        pixel_size: 像素大小
        image_shape: 图像形状
        weights: 指标权重
        
    Returns:
        综合指标
    """
    # 默认权重
    if weights is None:
        weights = {'detection': 0.4, 'localization': 0.4, 'photon': 0.2}
    
    # 计算各项指标
    detection_metrics = calculate_detection_metrics(
        predicted_positions, ground_truth_positions, tolerance, image_shape
    )
    
    localization_metrics = calculate_localization_metrics(
        predicted_positions, ground_truth_positions, tolerance, pixel_size
    )
    
    if predicted_photons is not None and ground_truth_photons is not None:
        photon_metrics = calculate_photon_metrics(
            predicted_photons, ground_truth_photons,
            predicted_positions, ground_truth_positions, tolerance
        )
    else:
        # 创建默认光子指标
        photon_metrics = PhotonMetrics(
            rmse=0.0, mae=0.0, mape=0.0, bias=0.0, std=0.0,
            correlation=1.0, r_squared=1.0, relative_error=0.0, efficiency=1.0
        )
    
    # 计算总体评分
    detection_score = detection_metrics.f1_score
    localization_score = 1.0 / (1.0 + localization_metrics.rmse_3d / 100.0)  # 归一化定位分数
    photon_score = photon_metrics.correlation if not np.isnan(photon_metrics.correlation) else 0.0
    
    overall_score = (detection_score + localization_score + photon_score) / 3.0
    
    # 加权评分
    weighted_score = (
        weights['detection'] * detection_score +
        weights['localization'] * localization_score +
        weights['photon'] * photon_score
    )
    
    return ComprehensiveMetrics(
        detection=detection_metrics,
        localization=localization_metrics,
        photon=photon_metrics,
        overall_score=overall_score,
        weighted_score=weighted_score
    )


def _match_positions(predicted_positions: np.ndarray,
                    ground_truth_positions: np.ndarray,
                    tolerance: float) -> np.ndarray:
    """匹配预测和真实位置
    
    使用匈牙利算法进行最优匹配。
    
    Args:
        predicted_positions: 预测位置
        ground_truth_positions: 真实位置
        tolerance: 匹配容差
        
    Returns:
        匹配索引，形状为 (N, 2)，每行为 [pred_idx, gt_idx]
    """
    if len(predicted_positions) == 0 or len(ground_truth_positions) == 0:
        return np.empty((0, 2), dtype=int)
    
    # 计算距离矩阵
    distances = cdist(predicted_positions, ground_truth_positions)
    
    # 使用匈牙利算法进行最优分配
    pred_indices, gt_indices = linear_sum_assignment(distances)
    
    # 过滤超出容差的匹配
    valid_matches = distances[pred_indices, gt_indices] <= tolerance
    
    matches = np.column_stack([
        pred_indices[valid_matches],
        gt_indices[valid_matches]
    ])
    
    return matches


def calculate_precision_recall_curve(predicted_probs: np.ndarray,
                                    ground_truth_binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算精确率-召回率曲线
    
    Args:
        predicted_probs: 预测概率
        ground_truth_binary: 二值化真实标签
        
    Returns:
        精确率、召回率、阈值
    """
    precision, recall, thresholds = precision_recall_curve(ground_truth_binary, predicted_probs)
    return precision, recall, thresholds


def calculate_roc_curve(predicted_probs: np.ndarray,
                       ground_truth_binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """计算ROC曲线
    
    Args:
        predicted_probs: 预测概率
        ground_truth_binary: 二值化真实标签
        
    Returns:
        假阳性率、真阳性率、阈值、AUC
    """
    fpr, tpr, thresholds = roc_curve(ground_truth_binary, predicted_probs)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, thresholds, auc_score


@dataclass
class RegressionMetrics:
    """回归指标
    
    Attributes:
        mse: 均方误差
        rmse: 均方根误差
        mae: 平均绝对误差
        r2: 决定系数
        correlation: 相关系数
    """
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    correlation: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'mse': self.mse,
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2,
            'correlation': self.correlation
        }