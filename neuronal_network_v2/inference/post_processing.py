"""后处理模块

包含峰值检测、发射器提取、非极大值抑制等后处理功能。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage.morphology import local_maxima as peak_local_maxima
from skimage.segmentation import watershed
from skimage.morphology import local_maxima
import cv2


class PostProcessor:
    """后处理器基类
    
    定义后处理的通用接口
    """
    
    def __call__(self, model_output: Union[torch.Tensor, List, Tuple]) -> Dict[str, Any]:
        """处理模型输出
        
        Args:
            model_output: 模型输出
            
        Returns:
            处理后的结果
        """
        raise NotImplementedError


class PeakFinder:
    """峰值检测器
    
    用于在概率图中检测局部最大值，支持多种检测方法。
    
    Args:
        min_distance: 峰值间最小距离
        threshold_abs: 绝对阈值
        threshold_rel: 相对阈值
        method: 检测方法 ('local_maxima', 'watershed', 'nms')
        exclude_border: 是否排除边界
    """
    
    def __init__(self,
                 min_distance: int = 3,
                 threshold_abs: float = 0.3,
                 threshold_rel: float = 0.1,
                 method: str = 'local_maxima',
                 exclude_border: bool = True,
                 nms_threshold: float = 0.5):
        
        self.min_distance = min_distance
        self.threshold_abs = threshold_abs
        self.threshold_rel = threshold_rel
        self.method = method
        self.exclude_border = exclude_border
        self.nms_threshold = nms_threshold
    
    def find_peaks(self, prob_map: Union[torch.Tensor, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """检测峰值
        
        Args:
            prob_map: 概率图，形状为 (H, W)
            
        Returns:
            峰值坐标 (N, 2) 和对应的概率值 (N,)
        """
        # 转换为numpy数组
        if isinstance(prob_map, torch.Tensor):
            prob_map = prob_map.detach().cpu().numpy()
        
        if self.method == 'local_maxima':
            return self._find_peaks_local_maxima(prob_map)
        elif self.method == 'watershed':
            return self._find_peaks_watershed(prob_map)
        elif self.method == 'nms':
            return self._find_peaks_nms(prob_map)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _find_peaks_local_maxima(self, prob_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """使用局部最大值检测峰值"""
        # 计算动态阈值
        threshold = max(self.threshold_abs, self.threshold_rel * prob_map.max())
        
        # 检测局部最大值
        coordinates = peak_local_maxima(
            prob_map,
            min_distance=self.min_distance,
            threshold_abs=threshold,
            exclude_border=self.exclude_border
        )
        
        if len(coordinates) == 0:
            return np.empty((0, 2)), np.empty(0)
        
        # 获取峰值坐标和概率
        peak_coords = np.column_stack(coordinates)
        peak_probs = prob_map[coordinates]
        
        return peak_coords, peak_probs
    
    def _find_peaks_watershed(self, prob_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """使用分水岭算法检测峰值"""
        # 计算动态阈值
        threshold = max(self.threshold_abs, self.threshold_rel * prob_map.max())
        
        # 二值化
        binary = prob_map > threshold
        
        # 距离变换
        distance = ndimage.distance_transform_edt(binary)
        
        # 检测局部最大值作为种子点
        local_max = local_maxima(distance)
        markers = ndimage.label(local_max)[0]
        
        # 分水岭分割
        labels = watershed(-distance, markers, mask=binary)
        
        # 提取峰值坐标
        peak_coords = []
        peak_probs = []
        
        for label_id in range(1, labels.max() + 1):
            mask = labels == label_id
            if mask.sum() == 0:
                continue
            
            # 找到该区域内的最大值位置
            region_probs = prob_map * mask
            max_idx = np.unravel_index(np.argmax(region_probs), region_probs.shape)
            
            peak_coords.append(max_idx)
            peak_probs.append(prob_map[max_idx])
        
        if len(peak_coords) == 0:
            return np.empty((0, 2)), np.empty(0)
        
        return np.array(peak_coords), np.array(peak_probs)
    
    def _find_peaks_nms(self, prob_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """使用非极大值抑制检测峰值"""
        # 计算动态阈值
        threshold = max(self.threshold_abs, self.threshold_rel * prob_map.max())
        
        # 找到所有候选峰值
        candidates = np.where(prob_map > threshold)
        if len(candidates[0]) == 0:
            return np.empty((0, 2)), np.empty(0)
        
        candidate_coords = np.column_stack(candidates)
        candidate_probs = prob_map[candidates]
        
        # 按概率排序
        sorted_indices = np.argsort(candidate_probs)[::-1]
        candidate_coords = candidate_coords[sorted_indices]
        candidate_probs = candidate_probs[sorted_indices]
        
        # 非极大值抑制
        keep = []
        for i, coord in enumerate(candidate_coords):
            # 检查是否与已选择的峰值太近
            if len(keep) == 0:
                keep.append(i)
                continue
            
            distances = np.sqrt(np.sum((candidate_coords[keep] - coord) ** 2, axis=1))
            if np.min(distances) >= self.min_distance:
                keep.append(i)
        
        return candidate_coords[keep], candidate_probs[keep]


class EmitterExtractor:
    """发射器提取器
    
    从模型输出中提取发射器的完整信息（位置、光子数、背景等）。
    
    Args:
        peak_finder: 峰值检测器
        output_format: 输出格式 ('ppxyzb', 'sigma_mu', 'simple')
        subpixel_localization: 是否进行亚像素定位
        fitting_window: 拟合窗口大小
    """
    
    def __init__(self,
                 peak_finder: PeakFinder,
                 output_format: str = 'ppxyzb',
                 subpixel_localization: bool = True,
                 fitting_window: int = 3,
                 min_photons: float = 50.0,
                 max_photons: float = 10000.0):
        
        self.peak_finder = peak_finder
        self.output_format = output_format
        self.subpixel_localization = subpixel_localization
        self.fitting_window = fitting_window
        self.min_photons = min_photons
        self.max_photons = max_photons
    
    def extract_emitters(self, model_output: Union[torch.Tensor, List, Tuple]) -> Dict[str, np.ndarray]:
        """提取发射器信息
        
        Args:
            model_output: 模型输出
            
        Returns:
            发射器信息字典
        """
        # 解析模型输出
        output_dict = self._parse_model_output(model_output)
        
        # 检测峰值
        prob_map = output_dict['prob']
        peak_coords, peak_probs = self.peak_finder.find_peaks(prob_map)
        
        if len(peak_coords) == 0:
            return self._empty_result()
        
        # 提取发射器信息
        emitters = self._extract_emitter_properties(peak_coords, output_dict)
        
        # 过滤无效发射器
        emitters = self._filter_emitters(emitters)
        
        return emitters
    
    def _parse_model_output(self, model_output: Union[torch.Tensor, List, Tuple]) -> Dict[str, np.ndarray]:
        """解析模型输出"""
        if isinstance(model_output, torch.Tensor):
            output = model_output.detach().cpu().numpy()
        elif isinstance(model_output, (list, tuple)):
            output = [out.detach().cpu().numpy() if isinstance(out, torch.Tensor) else out 
                     for out in model_output]
        else:
            output = model_output
        
        if self.output_format == 'ppxyzb':
            return self._parse_ppxyzb_output(output)
        elif self.output_format == 'sigma_mu':
            return self._parse_sigma_mu_output(output)
        elif self.output_format == 'simple':
            return self._parse_simple_output(output)
        else:
            raise ValueError(f"Unknown output format: {self.output_format}")
    
    def _parse_ppxyzb_output(self, output: np.ndarray) -> Dict[str, np.ndarray]:
        """解析PPXYZB格式输出"""
        if output.shape[0] == 5:
            return {
                'prob': output[0],
                'photons': output[1],
                'x': output[2],
                'y': output[3],
                'z': output[4]
            }
        elif output.shape[0] == 6:
            return {
                'prob': output[0],
                'photons': output[1],
                'x': output[2],
                'y': output[3],
                'z': output[4],
                'bg': output[5]
            }
        else:
            raise ValueError(f"Invalid PPXYZB output shape: {output.shape}")
    
    def _parse_sigma_mu_output(self, output: np.ndarray) -> Dict[str, np.ndarray]:
        """解析Sigma-MU格式输出"""
        if output.shape[0] == 10:
            return {
                'prob': output[0],
                'photons_mu': output[1],
                'x_mu': output[2],
                'y_mu': output[3],
                'z_mu': output[4],
                'photons_sigma': output[5],
                'x_sigma': output[6],
                'y_sigma': output[7],
                'z_sigma': output[8],
                'bg': output[9]
            }
        else:
            raise ValueError(f"Invalid Sigma-MU output shape: {output.shape}")
    
    def _parse_simple_output(self, output: np.ndarray) -> Dict[str, np.ndarray]:
        """解析简单格式输出"""
        if output.shape[0] == 5:
            return {
                'prob': output[0],
                'x': output[1],
                'y': output[2],
                'z': output[3],
                'photons': output[4]
            }
        else:
            raise ValueError(f"Invalid simple output shape: {output.shape}")
    
    def _extract_emitter_properties(self, peak_coords: np.ndarray, output_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """提取发射器属性"""
        num_emitters = len(peak_coords)
        
        # 初始化结果
        emitters = {
            'x': np.zeros(num_emitters),
            'y': np.zeros(num_emitters),
            'z': np.zeros(num_emitters),
            'photons': np.zeros(num_emitters),
            'prob': np.zeros(num_emitters)
        }
        
        if 'bg' in output_dict:
            emitters['bg'] = np.zeros(num_emitters)
        
        # 添加不确定性信息（如果可用）
        if 'x_sigma' in output_dict:
            emitters['x_sigma'] = np.zeros(num_emitters)
            emitters['y_sigma'] = np.zeros(num_emitters)
            emitters['z_sigma'] = np.zeros(num_emitters)
            emitters['photons_sigma'] = np.zeros(num_emitters)
        
        # 提取每个发射器的属性
        for i, (y, x) in enumerate(peak_coords):
            # 基本属性
            emitters['prob'][i] = output_dict['prob'][y, x]
            
            if self.subpixel_localization:
                # 亚像素定位
                refined_coords = self._subpixel_localization(y, x, output_dict)
                emitters['x'][i] = refined_coords[1]
                emitters['y'][i] = refined_coords[0]
            else:
                emitters['x'][i] = x
                emitters['y'][i] = y
            
            # 其他属性
            if 'x' in output_dict:
                emitters['x'][i] += output_dict['x'][y, x]  # 偏移量
            if 'y' in output_dict:
                emitters['y'][i] += output_dict['y'][y, x]  # 偏移量
            
            if 'z' in output_dict:
                emitters['z'][i] = output_dict['z'][y, x]
            elif 'z_mu' in output_dict:
                emitters['z'][i] = output_dict['z_mu'][y, x]
            
            if 'photons' in output_dict:
                emitters['photons'][i] = output_dict['photons'][y, x]
            elif 'photons_mu' in output_dict:
                emitters['photons'][i] = output_dict['photons_mu'][y, x]
            
            if 'bg' in output_dict:
                emitters['bg'][i] = output_dict['bg'][y, x]
            
            # 不确定性
            if 'x_sigma' in output_dict:
                emitters['x_sigma'][i] = output_dict['x_sigma'][y, x]
                emitters['y_sigma'][i] = output_dict['y_sigma'][y, x]
                emitters['z_sigma'][i] = output_dict['z_sigma'][y, x]
                emitters['photons_sigma'][i] = output_dict['photons_sigma'][y, x]
        
        return emitters
    
    def _subpixel_localization(self, y: int, x: int, output_dict: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """亚像素定位"""
        prob_map = output_dict['prob']
        h, w = prob_map.shape
        
        # 定义拟合窗口
        half_window = self.fitting_window // 2
        y_min = max(0, y - half_window)
        y_max = min(h, y + half_window + 1)
        x_min = max(0, x - half_window)
        x_max = min(w, x + half_window + 1)
        
        # 提取窗口
        window = prob_map[y_min:y_max, x_min:x_max]
        
        if window.size == 0:
            return float(y), float(x)
        
        # 计算质心
        y_indices, x_indices = np.mgrid[y_min:y_max, x_min:x_max]
        total_intensity = window.sum()
        
        if total_intensity > 0:
            centroid_y = (y_indices * window).sum() / total_intensity
            centroid_x = (x_indices * window).sum() / total_intensity
            return centroid_y, centroid_x
        else:
            return float(y), float(x)
    
    def _filter_emitters(self, emitters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """过滤无效发射器"""
        if len(emitters['photons']) == 0:
            return emitters
        
        # 光子数过滤
        valid_mask = (
            (emitters['photons'] >= self.min_photons) &
            (emitters['photons'] <= self.max_photons) &
            (emitters['prob'] > 0)
        )
        
        # 应用过滤
        filtered_emitters = {}
        for key, values in emitters.items():
            filtered_emitters[key] = values[valid_mask]
        
        return filtered_emitters
    
    def _empty_result(self) -> Dict[str, np.ndarray]:
        """返回空结果"""
        result = {
            'x': np.empty(0),
            'y': np.empty(0),
            'z': np.empty(0),
            'photons': np.empty(0),
            'prob': np.empty(0)
        }
        
        if self.output_format == 'ppxyzb' or self.output_format == 'sigma_mu':
            result['bg'] = np.empty(0)
        
        if self.output_format == 'sigma_mu':
            result['x_sigma'] = np.empty(0)
            result['y_sigma'] = np.empty(0)
            result['z_sigma'] = np.empty(0)
            result['photons_sigma'] = np.empty(0)
        
        return result


class StandardPostProcessor(PostProcessor):
    """标准后处理器
    
    结合峰值检测和发射器提取的完整后处理流程。
    """
    
    def __init__(self,
                 peak_finder: Optional[PeakFinder] = None,
                 emitter_extractor: Optional[EmitterExtractor] = None,
                 output_format: str = 'ppxyzb'):
        
        self.peak_finder = peak_finder or PeakFinder()
        self.emitter_extractor = emitter_extractor or EmitterExtractor(
            self.peak_finder, output_format=output_format
        )
    
    def __call__(self, model_output: Union[torch.Tensor, List, Tuple]) -> Dict[str, Any]:
        """执行后处理"""
        # 提取发射器
        emitters = self.emitter_extractor.extract_emitters(model_output)
        
        # 添加统计信息
        result = {
            'emitters': emitters,
            'num_emitters': len(emitters['x']),
            'total_photons': emitters['photons'].sum() if len(emitters['photons']) > 0 else 0.0
        }
        
        # 添加密度信息
        if len(emitters['x']) > 0:
            # 计算发射器密度（每平方像素）
            if isinstance(model_output, torch.Tensor):
                h, w = model_output.shape[-2:]
            else:
                h, w = model_output[0].shape[-2:] if isinstance(model_output, (list, tuple)) else (64, 64)
            
            result['density'] = len(emitters['x']) / (h * w)
            
            # 计算平均光子数
            result['avg_photons'] = emitters['photons'].mean()
            result['std_photons'] = emitters['photons'].std()
        else:
            result['density'] = 0.0
            result['avg_photons'] = 0.0
            result['std_photons'] = 0.0
        
        return result


class AdaptivePostProcessor(PostProcessor):
    """自适应后处理器
    
    根据图像特性自动调整处理参数。
    """
    
    def __init__(self,
                 base_threshold: float = 0.3,
                 adaptive_threshold: bool = True,
                 min_distance_factor: float = 1.0,
                 output_format: str = 'ppxyzb'):
        
        self.base_threshold = base_threshold
        self.adaptive_threshold = adaptive_threshold
        self.min_distance_factor = min_distance_factor
        self.output_format = output_format
    
    def __call__(self, model_output: Union[torch.Tensor, List, Tuple]) -> Dict[str, Any]:
        """自适应后处理"""
        # 解析输出获取概率图
        if isinstance(model_output, torch.Tensor):
            prob_map = model_output[0].detach().cpu().numpy()
        elif isinstance(model_output, (list, tuple)):
            prob_map = model_output[0][0].detach().cpu().numpy() if isinstance(model_output[0], torch.Tensor) else model_output[0][0]
        else:
            prob_map = model_output[0]
        
        # 自适应参数调整
        if self.adaptive_threshold:
            # 基于图像统计调整阈值
            threshold = self._adaptive_threshold(prob_map)
            min_distance = self._adaptive_min_distance(prob_map)
        else:
            threshold = self.base_threshold
            min_distance = 3
        
        # 创建自适应的峰值检测器和发射器提取器
        peak_finder = PeakFinder(
            min_distance=min_distance,
            threshold_abs=threshold,
            threshold_rel=0.1
        )
        
        emitter_extractor = EmitterExtractor(
            peak_finder=peak_finder,
            output_format=self.output_format
        )
        
        # 执行处理
        emitters = emitter_extractor.extract_emitters(model_output)
        
        return {
            'emitters': emitters,
            'num_emitters': len(emitters['x']),
            'adaptive_threshold': threshold,
            'adaptive_min_distance': min_distance
        }
    
    def _adaptive_threshold(self, prob_map: np.ndarray) -> float:
        """自适应阈值计算"""
        # 基于图像的统计特性调整阈值
        mean_prob = prob_map.mean()
        std_prob = prob_map.std()
        max_prob = prob_map.max()
        
        # 如果图像很暗，降低阈值
        if max_prob < 0.5:
            return max(0.1, self.base_threshold * 0.5)
        
        # 如果图像对比度很低，降低阈值
        if std_prob < 0.1:
            return max(0.15, self.base_threshold * 0.7)
        
        # 如果图像很亮，提高阈值
        if mean_prob > 0.3:
            return min(0.8, self.base_threshold * 1.5)
        
        return self.base_threshold
    
    def _adaptive_min_distance(self, prob_map: np.ndarray) -> int:
        """自适应最小距离计算"""
        # 估计发射器密度
        high_prob_pixels = (prob_map > self.base_threshold).sum()
        total_pixels = prob_map.size
        density = high_prob_pixels / total_pixels
        
        # 根据密度调整最小距离
        if density > 0.1:  # 高密度
            return max(2, int(3 * self.min_distance_factor * 0.7))
        elif density < 0.01:  # 低密度
            return int(3 * self.min_distance_factor * 1.3)
        else:
            return int(3 * self.min_distance_factor)