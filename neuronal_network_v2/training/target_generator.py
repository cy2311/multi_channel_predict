"""目标生成器和权重生成器实现

用于将发射器位置转换为训练目标和权重。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import math
from scipy import ndimage
from scipy.spatial.distance import cdist


class UnifiedEmbeddingTarget:
    """统一嵌入目标生成器
    
    将发射器位置转换为训练目标，支持多种输出格式：
    - 检测概率图
    - 坐标回归目标
    - 光子数目标
    - 背景目标
    
    Args:
        target_size: 目标图像尺寸 (H, W)
        sigma: 高斯核标准差
        output_format: 输出格式 ('ppxyzb', 'sigma_mu', 'simple')
        coordinate_system: 坐标系统 ('pixel', 'subpixel')
        background_value: 背景值
    """
    
    def __init__(self,
                 target_size: Tuple[int, int],
                 sigma: float = 1.0,
                 output_format: str = 'ppxyzb',
                 coordinate_system: str = 'subpixel',
                 background_value: float = 0.0,
                 max_emitters_per_pixel: int = 1):
        
        self.target_size = target_size
        self.sigma = sigma
        self.output_format = output_format
        self.coordinate_system = coordinate_system
        self.background_value = background_value
        self.max_emitters_per_pixel = max_emitters_per_pixel
        
        # 预计算高斯核
        self.kernel_size = int(6 * sigma + 1)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        
        self.gaussian_kernel = self._create_gaussian_kernel()
    
    def __call__(self, emitters: np.ndarray, image_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """生成目标张量
        
        Args:
            emitters: 发射器数组，形状为 (N, 5+) [frame, x, y, z, photons, ...]
            image_shape: 图像形状，如果提供则覆盖target_size
            
        Returns:
            目标张量，形状取决于output_format
        """
        if image_shape is not None:
            target_size = image_shape
        else:
            target_size = self.target_size
        
        H, W = target_size
        
        if emitters is None or len(emitters) == 0:
            return self._create_empty_target(target_size)
        
        # 解析发射器信息
        if emitters.ndim == 1:
            emitters = emitters.reshape(1, -1)
        
        if emitters.shape[1] < 5:
            raise ValueError("Emitters must have at least 5 columns: [frame, x, y, z, photons]")
        
        positions = emitters[:, 1:4]  # x, y, z
        photons = emitters[:, 4]      # photon counts
        
        # 生成不同格式的目标
        if self.output_format == 'ppxyzb':
            return self._generate_ppxyzb_target(positions, photons, target_size)
        elif self.output_format == 'sigma_mu':
            return self._generate_sigma_mu_target(positions, photons, target_size)
        elif self.output_format == 'simple':
            return self._generate_simple_target(positions, photons, target_size)
        else:
            raise ValueError(f"Unknown output format: {self.output_format}")
    
    def _create_gaussian_kernel(self) -> np.ndarray:
        """创建高斯核"""
        size = self.kernel_size
        center = size // 2
        
        y, x = np.ogrid[-center:size-center, -center:size-center]
        kernel = np.exp(-(x*x + y*y) / (2 * self.sigma**2))
        kernel = kernel / kernel.sum()
        
        return kernel
    
    def _create_empty_target(self, target_size: Tuple[int, int]) -> torch.Tensor:
        """创建空目标"""
        H, W = target_size
        
        if self.output_format == 'ppxyzb':
            # 6通道：[p, photons, x, y, z, bg]
            target = torch.zeros(6, H, W)
            target[5] = self.background_value  # 背景通道
        elif self.output_format == 'sigma_mu':
            # 10通道：[p, p_mu, x_mu, y_mu, z_mu, p_sig, x_sig, y_sig, z_sig, bg]
            target = torch.zeros(10, H, W)
            target[9] = self.background_value  # 背景通道
        elif self.output_format == 'simple':
            # 5通道：[p, x, y, z, photons]
            target = torch.zeros(5, H, W)
        else:
            target = torch.zeros(1, H, W)
        
        return target
    
    def _generate_ppxyzb_target(self, positions: np.ndarray, photons: np.ndarray, target_size: Tuple[int, int]) -> torch.Tensor:
        """生成PPXYZB格式目标"""
        H, W = target_size
        target = torch.zeros(6, H, W)  # [p, photons, x, y, z, bg]
        
        # 设置背景
        target[5] = self.background_value
        
        for i, (pos, photon) in enumerate(zip(positions, photons)):
            x, y, z = pos
            
            # 转换坐标
            if self.coordinate_system == 'pixel':
                px, py = int(round(x)), int(round(y))
                dx, dy = 0.0, 0.0
            else:  # subpixel
                px, py = int(x), int(y)
                dx, dy = x - px, y - py
            
            # 检查边界
            if 0 <= px < W and 0 <= py < H:
                # 检测概率
                target[0, py, px] = 1.0
                
                # 光子数（归一化到合理范围）
                target[1, py, px] = min(photon / 1000.0, 1.0)  # 归一化光子数
                
                # 坐标偏移（归一化到[-1,1]范围）
                target[2, py, px] = np.clip(dx, -1.0, 1.0)
                target[3, py, px] = np.clip(dy, -1.0, 1.0)
                target[4, py, px] = np.clip(z / 400.0, -1.0, 1.0)  # z坐标归一化
        
        return target
    
    def _generate_sigma_mu_target(self, positions: np.ndarray, photons: np.ndarray, target_size: Tuple[int, int]) -> torch.Tensor:
        """生成SigmaMU格式目标（10通道）"""
        H, W = target_size
        target = torch.zeros(10, H, W)  # [p, p_mu, x_mu, y_mu, z_mu, p_sig, x_sig, y_sig, z_sig, bg]
        
        # 设置背景
        target[9] = self.background_value
        
        # 设置默认标准差
        target[5:9] = 0.1  # 小的默认标准差
        
        for i, (pos, photon) in enumerate(zip(positions, photons)):
            x, y, z = pos
            
            # 转换坐标
            if self.coordinate_system == 'pixel':
                px, py = int(round(x)), int(round(y))
                dx, dy = 0.0, 0.0
            else:  # subpixel
                px, py = int(x), int(y)
                dx, dy = x - px, y - py
            
            # 检查边界
            if 0 <= px < W and 0 <= py < H:
                # 检测概率
                target[0, py, px] = 1.0
                
                # 参数均值
                target[1, py, px] = photon  # 光子数均值
                target[2, py, px] = dx      # x偏移均值
                target[3, py, px] = dy      # y偏移均值
                target[4, py, px] = z       # z坐标均值
                
                # 参数标准差（基于光子数的不确定性）
                photon_uncertainty = max(0.01, 1.0 / math.sqrt(max(1, photon)))
                target[5, py, px] = photon_uncertainty  # 光子数标准差
                target[6, py, px] = 0.1     # x偏移标准差
                target[7, py, px] = 0.1     # y偏移标准差
                target[8, py, px] = 0.2     # z坐标标准差
        
        return target
    
    def _generate_simple_target(self, positions: np.ndarray, photons: np.ndarray, target_size: Tuple[int, int]) -> torch.Tensor:
        """生成简单格式目标（5通道）"""
        H, W = target_size
        target = torch.zeros(5, H, W)  # [p, x, y, z, photons]
        
        for i, (pos, photon) in enumerate(zip(positions, photons)):
            x, y, z = pos
            
            # 转换坐标
            if self.coordinate_system == 'pixel':
                px, py = int(round(x)), int(round(y))
                dx, dy = 0.0, 0.0
            else:  # subpixel
                px, py = int(x), int(y)
                dx, dy = x - px, y - py
            
            # 检查边界
            if 0 <= px < W and 0 <= py < H:
                # 检测概率
                target[0, py, px] = 1.0
                
                # 坐标
                target[1, py, px] = dx
                target[2, py, px] = dy
                target[3, py, px] = z
                
                # 光子数
                target[4, py, px] = photon
        
        return target
    
    def generate_gaussian_target(self, positions: np.ndarray, photons: np.ndarray, target_size: Tuple[int, int]) -> torch.Tensor:
        """生成高斯分布目标（用于密度估计）"""
        H, W = target_size
        target = torch.zeros(1, H, W)
        
        for i, (pos, photon) in enumerate(zip(positions, photons)):
            x, y, z = pos
            
            # 像素坐标
            px, py = int(round(x)), int(round(y))
            
            # 计算高斯分布范围
            half_size = self.kernel_size // 2
            y_min = max(0, py - half_size)
            y_max = min(H, py + half_size + 1)
            x_min = max(0, px - half_size)
            x_max = min(W, px + half_size + 1)
            
            # 计算核的对应区域
            ky_min = max(0, half_size - py)
            ky_max = ky_min + (y_max - y_min)
            kx_min = max(0, half_size - px)
            kx_max = kx_min + (x_max - x_min)
            
            # 应用高斯核
            if y_max > y_min and x_max > x_min:
                kernel_region = self.gaussian_kernel[ky_min:ky_max, kx_min:kx_max]
                target[0, y_min:y_max, x_min:x_max] += torch.from_numpy(kernel_region * photon)
        
        return target


class SimpleWeight:
    """简单权重生成器
    
    生成训练权重，支持：
    - 常数权重
    - 基于光子数的权重
    - 基于距离的权重
    - 类别平衡权重
    
    Args:
        weight_type: 权重类型 ('constant', 'photon', 'distance', 'balanced')
        base_weight: 基础权重值
        photon_weight_factor: 光子权重因子
        distance_sigma: 距离权重的标准差
    """
    
    def __init__(self,
                 weight_type: str = 'constant',
                 base_weight: float = 1.0,
                 photon_weight_factor: float = 1.0,
                 distance_sigma: float = 2.0,
                 fg_weight: float = 1.0,
                 bg_weight: float = 1.0):
        
        self.weight_type = weight_type
        self.base_weight = base_weight
        self.photon_weight_factor = photon_weight_factor
        self.distance_sigma = distance_sigma
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
    
    def __call__(self, emitters: Optional[np.ndarray], image_shape: Tuple[int, int]) -> torch.Tensor:
        """生成权重张量
        
        Args:
            emitters: 发射器数组
            image_shape: 图像形状
            
        Returns:
            权重张量
        """
        H, W = image_shape
        
        if self.weight_type == 'constant':
            return self._generate_constant_weight(image_shape)
        elif self.weight_type == 'photon':
            return self._generate_photon_weight(emitters, image_shape)
        elif self.weight_type == 'distance':
            return self._generate_distance_weight(emitters, image_shape)
        elif self.weight_type == 'balanced':
            return self._generate_balanced_weight(emitters, image_shape)
        else:
            raise ValueError(f"Unknown weight type: {self.weight_type}")
    
    def _generate_constant_weight(self, image_shape: Tuple[int, int]) -> torch.Tensor:
        """生成常数权重"""
        H, W = image_shape
        return torch.full((1, H, W), self.base_weight)
    
    def _generate_photon_weight(self, emitters: Optional[np.ndarray], image_shape: Tuple[int, int]) -> torch.Tensor:
        """生成基于光子数的权重"""
        H, W = image_shape
        weight = torch.full((1, H, W), self.bg_weight)
        
        if emitters is not None and len(emitters) > 0:
            for emitter in emitters:
                if len(emitter) >= 5:
                    x, y, z, photon = emitter[1:5]
                    px, py = int(round(x)), int(round(y))
                    
                    if 0 <= px < W and 0 <= py < H:
                        # 基于光子数的权重
                        photon_weight = self.base_weight + self.photon_weight_factor * math.log(max(1, photon))
                        weight[0, py, px] = photon_weight
        
        return weight
    
    def _generate_distance_weight(self, emitters: Optional[np.ndarray], image_shape: Tuple[int, int]) -> torch.Tensor:
        """生成基于距离的权重"""
        H, W = image_shape
        weight = torch.full((1, H, W), self.bg_weight)
        
        if emitters is not None and len(emitters) > 0:
            # 提取位置
            positions = emitters[:, 1:3]  # x, y
            
            # 计算距离矩阵
            if len(positions) > 1:
                distances = cdist(positions, positions)
                min_distances = np.min(distances + np.eye(len(distances)) * 1e6, axis=1)
            else:
                min_distances = np.array([float('inf')])
            
            for i, emitter in enumerate(emitters):
                if len(emitter) >= 4:
                    x, y = emitter[1:3]
                    px, py = int(round(x)), int(round(y))
                    
                    if 0 <= px < W and 0 <= py < H:
                        # 基于最近邻距离的权重
                        min_dist = min_distances[i]
                        if min_dist < float('inf'):
                            dist_weight = self.base_weight * math.exp(-min_dist**2 / (2 * self.distance_sigma**2))
                            weight[0, py, px] = max(self.fg_weight, dist_weight)
                        else:
                            weight[0, py, px] = self.fg_weight
        
        return weight
    
    def _generate_balanced_weight(self, emitters: Optional[np.ndarray], image_shape: Tuple[int, int]) -> torch.Tensor:
        """生成类别平衡权重"""
        H, W = image_shape
        weight = torch.full((1, H, W), self.bg_weight)
        
        if emitters is not None and len(emitters) > 0:
            # 计算前景像素数量
            fg_pixels = len(emitters)
            bg_pixels = H * W - fg_pixels
            
            if fg_pixels > 0 and bg_pixels > 0:
                # 计算平衡权重
                total_pixels = H * W
                fg_weight = total_pixels / (2 * fg_pixels)
                bg_weight = total_pixels / (2 * bg_pixels)
                
                # 设置背景权重
                weight.fill_(bg_weight)
                
                # 设置前景权重
                for emitter in emitters:
                    if len(emitter) >= 3:
                        x, y = emitter[1:3]
                        px, py = int(round(x)), int(round(y))
                        
                        if 0 <= px < W and 0 <= py < H:
                            weight[0, py, px] = fg_weight
        
        return weight


class AdaptiveWeight:
    """自适应权重生成器
    
    根据训练进度和损失动态调整权重
    """
    
    def __init__(self,
                 base_generator: SimpleWeight,
                 adaptation_rate: float = 0.1,
                 min_weight: float = 0.1,
                 max_weight: float = 10.0):
        
        self.base_generator = base_generator
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # 权重历史
        self.weight_history = []
        self.loss_history = []
    
    def __call__(self, emitters: Optional[np.ndarray], image_shape: Tuple[int, int], 
                 loss_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """生成自适应权重"""
        # 获取基础权重
        base_weight = self.base_generator(emitters, image_shape)
        
        if loss_map is None:
            return base_weight
        
        # 根据损失调整权重
        adaptive_weight = base_weight.clone()
        
        # 高损失区域增加权重
        high_loss_mask = loss_map > loss_map.mean() + loss_map.std()
        adaptive_weight[high_loss_mask] *= (1 + self.adaptation_rate)
        
        # 低损失区域减少权重
        low_loss_mask = loss_map < loss_map.mean() - loss_map.std()
        adaptive_weight[low_loss_mask] *= (1 - self.adaptation_rate)
        
        # 限制权重范围
        adaptive_weight = torch.clamp(adaptive_weight, self.min_weight, self.max_weight)
        
        return adaptive_weight
    
    def update_history(self, weights: torch.Tensor, losses: torch.Tensor):
        """更新权重和损失历史"""
        self.weight_history.append(weights.mean().item())
        self.loss_history.append(losses.mean().item())
        
        # 保持历史长度
        if len(self.weight_history) > 100:
            self.weight_history.pop(0)
            self.loss_history.pop(0)


class MultiScaleTarget:
    """多尺度目标生成器
    
    在不同尺度上生成目标，用于多尺度训练
    """
    
    def __init__(self,
                 base_generator: UnifiedEmbeddingTarget,
                 scales: List[float] = [1.0, 0.5, 0.25]):
        
        self.base_generator = base_generator
        self.scales = scales
    
    def __call__(self, emitters: np.ndarray, image_shape: Tuple[int, int]) -> List[torch.Tensor]:
        """生成多尺度目标"""
        targets = []
        
        for scale in self.scales:
            # 缩放图像尺寸
            scaled_shape = (int(image_shape[0] * scale), int(image_shape[1] * scale))
            
            # 缩放发射器位置
            if emitters is not None and len(emitters) > 0:
                scaled_emitters = emitters.copy()
                scaled_emitters[:, 1:3] *= scale  # 缩放x, y坐标
            else:
                scaled_emitters = emitters
            
            # 生成目标
            target = self.base_generator(scaled_emitters, scaled_shape)
            targets.append(target)
        
        return targets