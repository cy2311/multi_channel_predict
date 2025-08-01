"""数学工具模块

该模块提供了数学计算相关的工具函数，包括：
- 高斯函数
- 统计计算
- 数组处理
- 信号处理
- 拟合算法
"""

import numpy as np
from scipy import ndimage, optimize, interpolate
from scipy.stats import norm
from typing import Tuple, Optional, Union, List
import logging

logger = logging.getLogger(__name__)


def gaussian_2d(x: np.ndarray, y: np.ndarray, 
               x0: float, y0: float, 
               sigma_x: float, sigma_y: float = None,
               amplitude: float = 1.0, 
               offset: float = 0.0,
               theta: float = 0.0) -> np.ndarray:
    """2D高斯函数
    
    Args:
        x, y: 坐标网格
        x0, y0: 中心位置
        sigma_x: X方向标准差
        sigma_y: Y方向标准差，None表示与sigma_x相同
        amplitude: 振幅
        offset: 偏移量
        theta: 旋转角度（弧度）
        
    Returns:
        np.ndarray: 2D高斯分布值
    """
    if sigma_y is None:
        sigma_y = sigma_x
    
    # 旋转坐标
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    x_rot = cos_theta * (x - x0) + sin_theta * (y - y0)
    y_rot = -sin_theta * (x - x0) + cos_theta * (y - y0)
    
    # 计算高斯函数
    gaussian = amplitude * np.exp(
        -(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2))
    ) + offset
    
    return gaussian


def gaussian_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray,
               x0: float, y0: float, z0: float,
               sigma_x: float, sigma_y: float = None, sigma_z: float = None,
               amplitude: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """3D高斯函数
    
    Args:
        x, y, z: 坐标网格
        x0, y0, z0: 中心位置
        sigma_x, sigma_y, sigma_z: 各方向标准差
        amplitude: 振幅
        offset: 偏移量
        
    Returns:
        np.ndarray: 3D高斯分布值
    """
    if sigma_y is None:
        sigma_y = sigma_x
    if sigma_z is None:
        sigma_z = sigma_x
    
    gaussian = amplitude * np.exp(
        -((x - x0)**2 / (2 * sigma_x**2) + 
          (y - y0)**2 / (2 * sigma_y**2) + 
          (z - z0)**2 / (2 * sigma_z**2))
    ) + offset
    
    return gaussian


def calculate_fwhm(sigma: float) -> float:
    """计算半高全宽(FWHM)
    
    Args:
        sigma: 高斯分布标准差
        
    Returns:
        float: FWHM值
    """
    return 2 * np.sqrt(2 * np.log(2)) * sigma


def fit_gaussian(data: np.ndarray, 
                initial_guess: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """拟合高斯分布
    
    Args:
        data: 输入数据 (1D或2D)
        initial_guess: 初始参数猜测
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 拟合参数和协方差矩阵
    """
    if data.ndim == 1:
        # 1D高斯拟合
        x = np.arange(len(data))
        
        def gaussian_1d(x, amplitude, x0, sigma, offset):
            return amplitude * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset
        
        if initial_guess is None:
            # 自动估计初始参数
            amplitude = np.max(data) - np.min(data)
            x0 = np.argmax(data)
            sigma = len(data) / 6  # 粗略估计
            offset = np.min(data)
            initial_guess = [amplitude, x0, sigma, offset]
        
        popt, pcov = optimize.curve_fit(gaussian_1d, x, data, p0=initial_guess)
        
    elif data.ndim == 2:
        # 2D高斯拟合
        h, w = data.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        def gaussian_2d_flat(coords, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
            x, y = coords
            return gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, amplitude, offset, theta).ravel()
        
        if initial_guess is None:
            # 自动估计初始参数
            amplitude = np.max(data) - np.min(data)
            y0, x0 = np.unravel_index(np.argmax(data), data.shape)
            sigma_x = w / 6
            sigma_y = h / 6
            theta = 0.0
            offset = np.min(data)
            initial_guess = [amplitude, x0, y0, sigma_x, sigma_y, theta, offset]
        
        popt, pcov = optimize.curve_fit(
            gaussian_2d_flat, (x, y), data.ravel(), p0=initial_guess
        )
    
    else:
        raise ValueError("只支持1D和2D数据的高斯拟合")
    
    return popt, pcov


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """计算信噪比
    
    Args:
        signal: 信号数据
        noise: 噪声数据
        
    Returns:
        float: 信噪比（dB）
    """
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    
    if noise_power == 0:
        return float('inf')
    
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def apply_noise(data: np.ndarray, 
               noise_type: str = "gaussian",
               noise_level: float = 0.1,
               **kwargs) -> np.ndarray:
    """添加噪声
    
    Args:
        data: 输入数据
        noise_type: 噪声类型 (gaussian, poisson, uniform)
        noise_level: 噪声水平
        **kwargs: 额外参数
        
    Returns:
        np.ndarray: 添加噪声后的数据
    """
    if noise_type == "gaussian":
        noise = np.random.normal(0, noise_level, data.shape)
        noisy_data = data + noise
        
    elif noise_type == "poisson":
        # 泊松噪声，数据需要是非负的
        data_scaled = np.maximum(data / noise_level, 0)
        noisy_data = np.random.poisson(data_scaled) * noise_level
        
    elif noise_type == "uniform":
        noise = np.random.uniform(-noise_level, noise_level, data.shape)
        noisy_data = data + noise
        
    elif noise_type == "salt_pepper":
        noisy_data = data.copy()
        # 盐噪声
        salt_coords = np.random.random(data.shape) < noise_level / 2
        noisy_data[salt_coords] = np.max(data)
        # 胡椒噪声
        pepper_coords = np.random.random(data.shape) < noise_level / 2
        noisy_data[pepper_coords] = np.min(data)
        
    else:
        raise ValueError(f"不支持的噪声类型: {noise_type}")
    
    return noisy_data


def normalize_array(arr: np.ndarray, 
                   method: str = "minmax",
                   axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
    """数组归一化
    
    Args:
        arr: 输入数组
        method: 归一化方法 (minmax, zscore, l2)
        axis: 归一化轴
        
    Returns:
        np.ndarray: 归一化后的数组
    """
    if method == "minmax":
        min_val = np.min(arr, axis=axis, keepdims=True)
        max_val = np.max(arr, axis=axis, keepdims=True)
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        normalized = (arr - min_val) / range_val
        
    elif method == "zscore":
        mean = np.mean(arr, axis=axis, keepdims=True)
        std = np.std(arr, axis=axis, keepdims=True)
        std = np.where(std == 0, 1, std)
        normalized = (arr - mean) / std
        
    elif method == "l2":
        norm = np.linalg.norm(arr, axis=axis, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        normalized = arr / norm
        
    else:
        raise ValueError(f"不支持的归一化方法: {method}")
    
    return normalized


def standardize_array(arr: np.ndarray, 
                     axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Tuple[np.ndarray, float, float]:
    """数组标准化
    
    Args:
        arr: 输入数组
        axis: 标准化轴
        
    Returns:
        Tuple[np.ndarray, float, float]: 标准化数组、均值、标准差
    """
    mean = np.mean(arr, axis=axis, keepdims=True)
    std = np.std(arr, axis=axis, keepdims=True)
    std = np.where(std == 0, 1, std)
    
    standardized = (arr - mean) / std
    
    return standardized, mean, std


def clip_array(arr: np.ndarray, 
              percentile: Tuple[float, float] = (1, 99)) -> np.ndarray:
    """数组裁剪
    
    Args:
        arr: 输入数组
        percentile: 裁剪百分位数
        
    Returns:
        np.ndarray: 裁剪后的数组
    """
    low, high = np.percentile(arr, percentile)
    clipped = np.clip(arr, low, high)
    return clipped


def smooth_array(arr: np.ndarray, 
                method: str = "gaussian",
                **kwargs) -> np.ndarray:
    """数组平滑
    
    Args:
        arr: 输入数组
        method: 平滑方法 (gaussian, median, uniform)
        **kwargs: 平滑参数
        
    Returns:
        np.ndarray: 平滑后的数组
    """
    if method == "gaussian":
        sigma = kwargs.get("sigma", 1.0)
        smoothed = ndimage.gaussian_filter(arr, sigma=sigma)
        
    elif method == "median":
        size = kwargs.get("size", 3)
        smoothed = ndimage.median_filter(arr, size=size)
        
    elif method == "uniform":
        size = kwargs.get("size", 3)
        smoothed = ndimage.uniform_filter(arr, size=size)
        
    else:
        raise ValueError(f"不支持的平滑方法: {method}")
    
    return smoothed


def interpolate_array(arr: np.ndarray, 
                     scale_factor: Union[float, Tuple[float, ...]],
                     method: str = "linear") -> np.ndarray:
    """数组插值
    
    Args:
        arr: 输入数组
        scale_factor: 缩放因子
        method: 插值方法 (linear, cubic, nearest)
        
    Returns:
        np.ndarray: 插值后的数组
    """
    if isinstance(scale_factor, (int, float)):
        scale_factor = [scale_factor] * arr.ndim
    
    # 创建新的坐标
    old_coords = [np.arange(s) for s in arr.shape]
    new_shape = [int(s * f) for s, f in zip(arr.shape, scale_factor)]
    new_coords = [np.linspace(0, s-1, ns) for s, ns in zip(arr.shape, new_shape)]
    
    # 执行插值
    if arr.ndim == 1:
        f = interpolate.interp1d(old_coords[0], arr, kind=method, 
                               bounds_error=False, fill_value='extrapolate')
        interpolated = f(new_coords[0])
    elif arr.ndim == 2:
        f = interpolate.interp2d(old_coords[1], old_coords[0], arr, kind=method)
        interpolated = f(new_coords[1], new_coords[0])
    else:
        # 对于高维数组，使用scipy的zoom
        interpolated = ndimage.zoom(arr, scale_factor, order=1 if method=='linear' else 3)
    
    return interpolated


def calculate_statistics(arr: np.ndarray, 
                        axis: Optional[Union[int, Tuple[int, ...]]] = None) -> dict:
    """计算数组统计信息
    
    Args:
        arr: 输入数组
        axis: 计算轴
        
    Returns:
        dict: 统计信息字典
    """
    stats = {
        "mean": np.mean(arr, axis=axis),
        "std": np.std(arr, axis=axis),
        "var": np.var(arr, axis=axis),
        "min": np.min(arr, axis=axis),
        "max": np.max(arr, axis=axis),
        "median": np.median(arr, axis=axis),
        "q25": np.percentile(arr, 25, axis=axis),
        "q75": np.percentile(arr, 75, axis=axis),
        "skewness": calculate_skewness(arr, axis=axis),
        "kurtosis": calculate_kurtosis(arr, axis=axis)
    }
    
    return stats


def calculate_skewness(arr: np.ndarray, 
                      axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
    """计算偏度
    
    Args:
        arr: 输入数组
        axis: 计算轴
        
    Returns:
        np.ndarray: 偏度值
    """
    mean = np.mean(arr, axis=axis, keepdims=True)
    std = np.std(arr, axis=axis, keepdims=True)
    
    # 避免除零
    std = np.where(std == 0, 1, std)
    
    skewness = np.mean(((arr - mean) / std) ** 3, axis=axis)
    return skewness


def calculate_kurtosis(arr: np.ndarray, 
                      axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
    """计算峰度
    
    Args:
        arr: 输入数组
        axis: 计算轴
        
    Returns:
        np.ndarray: 峰度值
    """
    mean = np.mean(arr, axis=axis, keepdims=True)
    std = np.std(arr, axis=axis, keepdims=True)
    
    # 避免除零
    std = np.where(std == 0, 1, std)
    
    kurtosis = np.mean(((arr - mean) / std) ** 4, axis=axis) - 3
    return kurtosis


def find_peaks(arr: np.ndarray, 
              height: Optional[float] = None,
              distance: Optional[int] = None,
              prominence: Optional[float] = None) -> Tuple[np.ndarray, dict]:
    """寻找峰值
    
    Args:
        arr: 输入数组（1D）
        height: 最小峰值高度
        distance: 峰值间最小距离
        prominence: 最小峰值突出度
        
    Returns:
        Tuple[np.ndarray, dict]: 峰值位置和属性
    """
    from scipy.signal import find_peaks as scipy_find_peaks
    
    peaks, properties = scipy_find_peaks(
        arr, height=height, distance=distance, prominence=prominence
    )
    
    return peaks, properties


def calculate_correlation(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """计算相关系数
    
    Args:
        arr1, arr2: 输入数组
        
    Returns:
        float: 相关系数
    """
    # 展平数组
    flat1 = arr1.flatten()
    flat2 = arr2.flatten()
    
    # 计算皮尔逊相关系数
    correlation = np.corrcoef(flat1, flat2)[0, 1]
    
    return correlation


def calculate_mutual_information(arr1: np.ndarray, arr2: np.ndarray, bins: int = 50) -> float:
    """计算互信息
    
    Args:
        arr1, arr2: 输入数组
        bins: 直方图bins数量
        
    Returns:
        float: 互信息值
    """
    # 展平数组
    flat1 = arr1.flatten()
    flat2 = arr2.flatten()
    
    # 计算联合直方图
    hist_2d, _, _ = np.histogram2d(flat1, flat2, bins=bins)
    
    # 计算边缘分布
    hist_1 = np.sum(hist_2d, axis=1)
    hist_2 = np.sum(hist_2d, axis=0)
    
    # 归一化为概率
    p_xy = hist_2d / np.sum(hist_2d)
    p_x = hist_1 / np.sum(hist_1)
    p_y = hist_2 / np.sum(hist_2)
    
    # 计算互信息
    mi = 0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
    
    return mi


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int = 2) -> np.ndarray:
    """多项式拟合
    
    Args:
        x, y: 输入数据点
        degree: 多项式阶数
        
    Returns:
        np.ndarray: 多项式系数
    """
    coeffs = np.polyfit(x, y, degree)
    return coeffs


def moving_average(arr: np.ndarray, window_size: int) -> np.ndarray:
    """移动平均
    
    Args:
        arr: 输入数组
        window_size: 窗口大小
        
    Returns:
        np.ndarray: 移动平均结果
    """
    if window_size > len(arr):
        return np.full_like(arr, np.mean(arr))
    
    # 使用卷积计算移动平均
    kernel = np.ones(window_size) / window_size
    padded = np.pad(arr, (window_size//2, window_size//2), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    
    return smoothed


def calculate_gradient(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    """计算梯度
    
    Args:
        arr: 输入数组
        axis: 计算梯度的轴
        
    Returns:
        np.ndarray: 梯度数组
    """
    gradient = np.gradient(arr, axis=axis)
    return gradient


def calculate_laplacian(arr: np.ndarray) -> np.ndarray:
    """计算拉普拉斯算子
    
    Args:
        arr: 输入数组（2D）
        
    Returns:
        np.ndarray: 拉普拉斯结果
    """
    laplacian = ndimage.laplace(arr)
    return laplacian