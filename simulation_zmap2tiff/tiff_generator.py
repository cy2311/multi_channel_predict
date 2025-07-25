#!/usr/bin/env python3
"""
完整的TIFF图像生成模块

从包含Zernike系数的发射器数据生成多帧TIFF图像
"""

import os
import glob
import json
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import tifffile as tiff
from scipy.fft import ifft2, ifftshift, fftshift
from skimage.transform import resize
from tqdm import tqdm

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))


def load_config() -> Tuple[float, float, float, float]:
    """加载光学参数配置
    
    Returns
    -------
    wavelength_nm : float
        波长（纳米）
    pixel_size_nm_x : float
        X方向像素大小（纳米）
    pixel_size_nm_y : float
        Y方向像素大小（纳米）
    NA : float
        数值孔径
    """
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    opt_cfg = cfg["optical"]
    return (
        float(opt_cfg["wavelength_nm"]),
        float(opt_cfg["pixel_size_nm_x"]),
        float(opt_cfg["pixel_size_nm_y"]),
        float(opt_cfg["NA"]),
    )


def load_zernike_basis() -> np.ndarray:
    """加载21个Zernike基函数
    
    Returns
    -------
    basis : ndarray
        Zernike基函数，形状为 (21, 128, 128)
    """
    zernike_dir = os.path.join(os.path.dirname(__file__), "..", "simulated_data", "zernike_polynomials")
    pattern = os.path.join(zernike_dir, "zernike_*_n*_m*.npy")
    files = sorted(glob.glob(pattern))
    if len(files) < 21:
        raise RuntimeError(f"期望至少21个Zernike .npy文件，但只找到 {len(files)} 个")
    basis = np.stack([np.load(fp) for fp in files[:21]], axis=0).astype(np.float32)
    return basis


def build_pupil_mask(N: int, pixel_size_x: float, pixel_size_y: float, 
                    NA: float, wavelength: float) -> np.ndarray:
    """构建瞳孔掩码
    
    Parameters
    ----------
    N : int
        网格大小
    pixel_size_x : float
        X方向像素大小（米）
    pixel_size_y : float
        Y方向像素大小（米）
    NA : float
        数值孔径
    wavelength : float
        波长（米）
        
    Returns
    -------
    mask : ndarray
        二值瞳孔掩码
    """
    fx = np.fft.fftfreq(N, d=pixel_size_x)
    fy = np.fft.fftfreq(N, d=pixel_size_y)
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    rho = np.sqrt(FX ** 2 + FY ** 2)
    f_max = NA / wavelength
    mask = (rho <= f_max).astype(np.float32)
    return mask


def construct_pupil(coeff_mag: np.ndarray, coeff_phase: np.ndarray, 
                   basis: np.ndarray, pupil_mask: np.ndarray) -> np.ndarray:
    """从Zernike系数构建复数瞳函数
    
    Parameters
    ----------
    coeff_mag : ndarray
        幅度系数 (21,)
    coeff_phase : ndarray
        相位系数 (21,)
    basis : ndarray
        Zernike基函数 (21, N, N)
    pupil_mask : ndarray
        瞳孔掩码 (N, N)
        
    Returns
    -------
    pupil : ndarray
        复数瞳函数
    """
    # 检查输入系数是否包含异常值
    if np.any(np.isnan(coeff_mag)) or np.any(np.isinf(coeff_mag)):
        print(f"警告: 幅度系数包含NaN或无穷大值")
        print(f"幅度系数范围: [{np.nanmin(coeff_mag):.6f}, {np.nanmax(coeff_mag):.6f}]")
        coeff_mag = np.nan_to_num(coeff_mag, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.any(np.isnan(coeff_phase)) or np.any(np.isinf(coeff_phase)):
        print(f"警告: 相位系数包含NaN或无穷大值")
        print(f"相位系数范围: [{np.nanmin(coeff_phase):.6f}, {np.nanmax(coeff_phase):.6f}]")
        coeff_phase = np.nan_to_num(coeff_phase, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 振幅：1 + Zernike基函数的线性组合（限制为非负）
    amplitude = 1.0 + np.sum(coeff_mag[:, None, None] * basis, axis=0)
    amplitude = np.clip(amplitude, 0.0, np.inf)
    
    # 相位：Zernike基函数的线性组合
    phase = np.sum(coeff_phase[:, None, None] * basis, axis=0)
    
    # 复数瞳函数
    pupil = amplitude * pupil_mask * np.exp(1j * phase)
    return pupil


def apply_defocus(pupil: np.ndarray, z_nm: float, wavelength_m: float, 
                 pixel_size_x: float, pixel_size_y: float) -> np.ndarray:
    """应用离焦效应
    
    Parameters
    ----------
    pupil : ndarray
        瞳函数
    z_nm : float
        Z位置（纳米）
    wavelength_m : float
        波长（米）
    pixel_size_x : float
        X方向像素大小（米）
    pixel_size_y : float
        Y方向像素大小（米）
        
    Returns
    -------
    pupil_defocus : ndarray
        包含离焦的瞳函数
    """
    N = pupil.shape[0]
    fx = np.fft.fftfreq(N, d=pixel_size_x)
    fy = np.fft.fftfreq(N, d=pixel_size_y)
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    RHO2 = FX ** 2 + FY ** 2
    
    # 离焦相位因子
    defocus_factor = np.exp(1j * np.pi * wavelength_m * (z_nm * 1e-9) * RHO2)
    pupil_defocus = pupil * defocus_factor
    
    return pupil_defocus


def generate_psf(pupil: np.ndarray) -> np.ndarray:
    """从瞳函数生成PSF
    
    Parameters
    ----------
    pupil : ndarray
        复数瞳函数
        
    Returns
    -------
    psf : ndarray
        归一化的PSF
    """
    # 检查瞳函数是否包含NaN或无穷大值
    if np.any(np.isnan(pupil)) or np.any(np.isinf(pupil)):
        print(f"警告: 瞳函数包含NaN或无穷大值")
        print(f"瞳函数统计: min={np.nanmin(np.abs(pupil)):.6f}, max={np.nanmax(np.abs(pupil)):.6f}")
        # 用零替换NaN和无穷大值
        pupil = np.nan_to_num(pupil, nan=0.0, posinf=0.0, neginf=0.0)
    
    field = ifft2(ifftshift(pupil))
    psf = np.abs(fftshift(field)) ** 2
    
    # 检查PSF总和是否为零或NaN
    psf_sum = psf.sum()
    if psf_sum == 0 or np.isnan(psf_sum):
        print(f"警告: PSF总和为零或NaN: {psf_sum}")
        # 返回一个中心有单个像素的默认PSF
        psf = np.zeros_like(psf)
        center = psf.shape[0] // 2
        psf[center, center] = 1.0
        psf_sum = 1.0
    
    psf /= psf_sum  # 归一化
    return psf.astype(np.float32)


def add_camera_noise(image: np.ndarray, background: float = 100, 
                    readout_noise: float = 10, shot_noise: bool = True) -> np.ndarray:
    """添加相机噪声模型
    
    Parameters
    ----------
    image : ndarray
        输入图像
    background : float
        背景水平
    readout_noise : float
        读出噪声标准差
    shot_noise : bool
        是否添加散粒噪声
        
    Returns
    -------
    noisy_image : ndarray
        添加噪声后的图像
    """
    # 添加背景
    image_with_bg = image + background
    
    # 散粒噪声（泊松噪声）
    if shot_noise:
        # 限制泊松分布的lambda值以避免数值问题
        # 当lambda > 1e4时，泊松分布近似为正态分布
        max_lambda = 1e4
        mask_large = image_with_bg > max_lambda
        
        if np.any(mask_large):
            # 对于大值使用正态分布近似
            large_values = image_with_bg[mask_large]
            noise_large = np.random.normal(large_values, np.sqrt(large_values))
            image_with_bg[mask_large] = noise_large
            
            # 对于小值使用泊松分布
            if np.any(~mask_large):
                small_values = image_with_bg[~mask_large]
                noise_small = np.random.poisson(small_values)
                image_with_bg[~mask_large] = noise_small
        else:
            # 所有值都不太大，直接使用泊松分布
            image_with_bg = np.random.poisson(image_with_bg)
        
        image_with_bg = image_with_bg.astype(np.float32)
    
    # 读出噪声（高斯噪声）
    readout = np.random.normal(0, readout_noise, image_with_bg.shape)
    noisy_image = image_with_bg + readout
    
    # 确保非负
    noisy_image = np.clip(noisy_image, 0, np.inf)
    
    return noisy_image.astype(np.float32)


def simulate_frame_direct(frame_idx: int, emitters_data: Dict[str, np.ndarray], 
                          basis: np.ndarray, pupil_mask: np.ndarray, 
                          wavelength_m: float, pixel_size_x: float, pixel_size_y: float,
                          roi_size: int = 1200, add_noise: bool = True, 
                          noise_params: Optional[Dict] = None) -> np.ndarray:
    """直接生成目标分辨率的单帧图像（避免高分辨率渲染和降采样）
    
    Parameters
    ----------
    frame_idx : int
        帧索引
    emitters_data : dict
        发射器数据字典
    basis : ndarray
        Zernike基函数
    pupil_mask : ndarray
        瞳孔掩码
    wavelength_m : float
        波长（米）
    pixel_size_x : float
        X方向像素大小（米）
    pixel_size_y : float
        Y方向像素大小（米）
    roi_size : int
        目标图像大小
    add_noise : bool
        是否添加噪声
    noise_params : dict, optional
        噪声参数
        
    Returns
    -------
    frame : ndarray
        模拟的帧图像
    """
    if noise_params is None:
        noise_params = {'background': 100, 'readout_noise': 10, 'shot_noise': True}
    
    # 获取该帧的活跃发射器
    frame_mask = emitters_data['frame_ix'] == frame_idx
    if not np.any(frame_mask):
        # 没有活跃发射器，返回空白帧
        frame = np.zeros((roi_size, roi_size), dtype=np.float32)
        if add_noise:
            frame = add_camera_noise(frame, **noise_params)
        return frame
    
    # 获取该帧的发射器数据
    active_ids = emitters_data['ids_rec'][frame_mask]
    active_xyz = emitters_data['xyz_rec'][frame_mask]
    active_phot = emitters_data['phot_rec'][frame_mask]
    
    # 获取对应的Zernike系数
    coeff_mag = emitters_data['coeff_mag_all'][active_ids]
    coeff_phase = emitters_data['coeff_phase_all'][active_ids]
    
    # 创建目标分辨率画布
    canvas = np.zeros((roi_size, roi_size), dtype=np.float32)
    half_psf = basis.shape[1] // 2
    
    # 为每个活跃发射器生成PSF并添加到画布
    for i in range(len(active_ids)):
        # 构建瞳函数
        pupil = construct_pupil(coeff_mag[i], coeff_phase[i], basis, pupil_mask)
        
        # 应用离焦
        z_nm = active_xyz[i, 2] * 1000  # 转换为纳米
        pupil_defocus = apply_defocus(pupil, z_nm, wavelength_m, pixel_size_x, pixel_size_y)
        
        # 生成PSF
        psf = generate_psf(pupil_defocus)
        
        # 缩放PSF强度
        psf_scaled = psf * active_phot[i]
        
        # 计算在目标画布上的亚像素位置
        cx_float = active_xyz[i, 0]  # 保持浮点精度
        cy_float = active_xyz[i, 1]
        
        # 计算整数像素位置和亚像素偏移
        cx_int = int(np.floor(cx_float))
        cy_int = int(np.floor(cy_float))
        dx = cx_float - cx_int  # X方向亚像素偏移
        dy = cy_float - cy_int  # Y方向亚像素偏移
        
        # 使用双线性插值处理亚像素位置
        # 计算四个相邻像素的权重
        w00 = (1 - dx) * (1 - dy)  # 左上
        w01 = (1 - dx) * dy        # 左下
        w10 = dx * (1 - dy)        # 右上
        w11 = dx * dy              # 右下
        
        # 为四个相邻位置添加PSF
        positions = [
            (cy_int, cx_int, w00),
            (cy_int + 1, cx_int, w01),
            (cy_int, cx_int + 1, w10),
            (cy_int + 1, cx_int + 1, w11)
        ]
        
        for py, px, weight in positions:
            if weight < 1e-6:  # 跳过权重很小的位置
                continue
                
            # 计算PSF在画布上的边界
            r0 = py - half_psf
            r1 = py + half_psf
            c0 = px - half_psf
            c1 = px + half_psf
            
            # 处理边界情况
            psf_r0 = psf_c0 = 0
            if r0 < 0:
                psf_r0 = -r0
                r0 = 0
            if c0 < 0:
                psf_c0 = -c0
                c0 = 0
            if r1 > roi_size:
                r1 = roi_size
            if c1 > roi_size:
                c1 = roi_size
            
            psf_r1 = psf_r0 + (r1 - r0)
            psf_c1 = psf_c0 + (c1 - c0)
            
            # 添加加权PSF到画布
            if r1 > r0 and c1 > c0:
                canvas[r0:r1, c0:c1] += weight * psf_scaled[psf_r0:psf_r1, psf_c0:psf_c1]
    
    # 添加噪声
    if add_noise:
        # 调试信息：检查像素值范围
        print(f"帧 {frame_idx}: 像素值范围 [{canvas.min():.2f}, {canvas.max():.2f}]")
        canvas = add_camera_noise(canvas, **noise_params)
    
    return canvas.astype(np.float32)


def simulate_frame(frame_idx: int, emitters_data: Dict[str, np.ndarray], 
                  basis: np.ndarray, pupil_mask: np.ndarray, 
                  wavelength_m: float, pixel_size_x: float, pixel_size_y: float,
                  roi_size: int = 1200, hr_size: int = 6144,
                  add_noise: bool = True, noise_params: Optional[Dict] = None,
                  use_direct_rendering: bool = True) -> np.ndarray:
    """模拟单帧图像
    
    Parameters
    ----------
    frame_idx : int
        帧索引
    emitters_data : dict
        发射器数据字典
    basis : ndarray
        Zernike基函数
    pupil_mask : ndarray
        瞳孔掩码
    wavelength_m : float
        波长（米）
    pixel_size_x : float
        X方向像素大小（米）
    pixel_size_y : float
        Y方向像素大小（米）
    roi_size : int
        最终图像大小
    hr_size : int
        高分辨率渲染大小（仅在use_direct_rendering=False时使用）
    add_noise : bool
        是否添加噪声
    noise_params : dict, optional
        噪声参数
    use_direct_rendering : bool
        是否使用直接渲染（推荐，避免高分辨率渲染和降采样）
        
    Returns
    -------
    frame : ndarray
        模拟的帧图像
    """
    if use_direct_rendering:
        # 使用直接渲染方法（推荐）
        return simulate_frame_direct(
            frame_idx, emitters_data, basis, pupil_mask,
            wavelength_m, pixel_size_x, pixel_size_y,
            roi_size, add_noise, noise_params
        )
    
    # 原始的高分辨率渲染方法（保留用于兼容性）
    if noise_params is None:
        noise_params = {'background': 100, 'readout_noise': 10, 'shot_noise': True}
    
    # 获取该帧的活跃发射器
    frame_mask = emitters_data['frame_ix'] == frame_idx
    if not np.any(frame_mask):
        # 没有活跃发射器，返回空白帧
        frame = np.zeros((roi_size, roi_size), dtype=np.float32)
        if add_noise:
            frame = add_camera_noise(frame, **noise_params)
        return frame
    
    # 获取该帧的发射器数据
    active_ids = emitters_data['ids_rec'][frame_mask]
    active_xyz = emitters_data['xyz_rec'][frame_mask]
    active_phot = emitters_data['phot_rec'][frame_mask]
    
    # 获取对应的Zernike系数
    coeff_mag = emitters_data['coeff_mag_all'][active_ids]
    coeff_phase = emitters_data['coeff_phase_all'][active_ids]
    
    # 创建高分辨率画布
    canvas = np.zeros((hr_size, hr_size), dtype=np.float32)
    upscale = hr_size / roi_size
    half_psf = basis.shape[1] // 2
    
    # 为每个活跃发射器生成PSF并添加到画布
    for i in range(len(active_ids)):
        # 构建瞳函数
        pupil = construct_pupil(coeff_mag[i], coeff_phase[i], basis, pupil_mask)
        
        # 应用离焦
        z_nm = active_xyz[i, 2] * 1000  # 转换为纳米
        pupil_defocus = apply_defocus(pupil, z_nm, wavelength_m, pixel_size_x, pixel_size_y)
        
        # 生成PSF
        psf = generate_psf(pupil_defocus)
        
        # 缩放PSF强度
        psf_scaled = psf * active_phot[i]
        
        # 计算在高分辨率画布上的位置
        cx = int(round(active_xyz[i, 0] * upscale))
        cy = int(round(active_xyz[i, 1] * upscale))
        
        # 计算PSF在画布上的边界
        r0 = cy - half_psf
        r1 = cy + half_psf
        c0 = cx - half_psf
        c1 = cx + half_psf
        
        # 处理边界情况
        psf_r0 = psf_c0 = 0
        if r0 < 0:
            psf_r0 = -r0
            r0 = 0
        if c0 < 0:
            psf_c0 = -c0
            c0 = 0
        if r1 > hr_size:
            r1 = hr_size
        if c1 > hr_size:
            c1 = hr_size
        
        psf_r1 = psf_r0 + (r1 - r0)
        psf_c1 = psf_c0 + (c1 - c0)
        
        # 添加PSF到画布
        if r1 > r0 and c1 > c0:
            canvas[r0:r1, c0:c1] += psf_scaled[psf_r0:psf_r1, psf_c0:psf_c1]
    
    # 下采样到目标分辨率
    frame = resize(canvas, (roi_size, roi_size), anti_aliasing=True, preserve_range=True)
    frame = frame.astype(np.float32)
    
    # 添加噪声
    if add_noise:
        frame = add_camera_noise(frame, **noise_params)
    
    return frame


def generate_tiff_stack(h5_path: str, output_path: str, 
                       config: Optional[Dict[str, Any]] = None) -> None:
    """生成完整的TIFF图像堆栈
    
    Parameters
    ----------
    h5_path : str
        输入的HDF5文件路径
    output_path : str
        输出TIFF文件路径
    config : dict, optional
        配置参数
    """
    if config is None:
        config = {}
    
    tiff_config = config.get('tiff', {})
    
    print(f"从 {h5_path} 加载发射器数据...")
    
    # 加载光学参数
    wavelength_nm, pix_x_nm, pix_y_nm, NA = load_config()
    wavelength_m = wavelength_nm * 1e-9
    pix_x_m = pix_x_nm * 1e-9
    pix_y_m = pix_y_nm * 1e-9
    
    print(f"光学参数: λ={wavelength_nm}nm, NA={NA}, 像素={pix_x_nm}x{pix_y_nm}nm")
    
    # 加载Zernike基函数和瞳孔掩码
    basis = load_zernike_basis()
    N = basis.shape[1]
    pupil_mask = build_pupil_mask(N, pix_x_m, pix_y_m, NA, wavelength_m)
    
    print(f"Zernike基函数形状: {basis.shape}")
    
    # 从HDF5文件加载数据
    with h5py.File(h5_path, 'r') as f:
        # 记录数据
        frame_ix = np.array(f['records/frame_ix'])
        ids_rec = np.array(f['records/id'])
        xyz_rec = np.array(f['records/xyz'])
        phot_rec = np.array(f['records/phot'])
        
        # Zernike系数
        coeff_mag_all = np.array(f['zernike_coeffs/mag'])
        coeff_phase_all = np.array(f['zernike_coeffs/phase'])
        
        print(f"总记录数: {len(frame_ix)}")
        print(f"Zernike系数形状: mag={coeff_mag_all.shape}, phase={coeff_phase_all.shape}")
    
    # 组织数据
    emitters_data = {
        'frame_ix': frame_ix,
        'ids_rec': ids_rec,
        'xyz_rec': xyz_rec,
        'phot_rec': phot_rec,
        'coeff_mag_all': coeff_mag_all,
        'coeff_phase_all': coeff_phase_all
    }
    
    # 获取参数
    roi_size = tiff_config.get('roi_size', 1200)
    hr_size = tiff_config.get('hr_size', 6144)
    use_direct_rendering = tiff_config.get('use_direct_rendering', True)
    add_noise = tiff_config.get('add_noise', True)
    noise_params = tiff_config.get('noise_params', {
        'background': 100,
        'readout_noise': 10,
        'shot_noise': True
    })
    
    # 获取所有帧
    unique_frames = np.unique(frame_ix)
    num_frames = len(unique_frames)
    
    print(f"生成 {num_frames} 帧图像，大小: {roi_size}x{roi_size}")
    if use_direct_rendering:
        print("使用直接渲染方法（推荐）")
    else:
        print(f"使用高分辨率渲染和降采样方法，高分辨率大小: {hr_size}x{hr_size}")
    
    # 生成所有帧
    frames = []
    for frame_idx in tqdm(unique_frames, desc="生成帧"):
        frame = simulate_frame(
            frame_idx, emitters_data, basis, pupil_mask,
            wavelength_m, pix_x_m, pix_y_m,
            roi_size, hr_size, add_noise, noise_params,
            use_direct_rendering
        )
        frames.append(frame)
    
    # 转换为numpy数组
    frames_array = np.stack(frames, axis=0)
    
    print(f"最终图像堆栈形状: {frames_array.shape}")
    print(f"像素值范围: [{frames_array.min():.2f}, {frames_array.max():.2f}]")
    
    # 保存为OME-TIFF
    print(f"保存TIFF文件到: {output_path}")
    
    # 创建OME-XML元数据
    metadata = {
        'axes': 'TYX',
        'PhysicalSizeX': pix_x_nm / 1000,  # 转换为微米
        'PhysicalSizeY': pix_y_nm / 1000,
        'PhysicalSizeXUnit': 'µm',
        'PhysicalSizeYUnit': 'µm'
    }
    
    tiff.imwrite(
        output_path,
        frames_array,
        metadata=metadata,
        compression='lzw'
    )
    
    print(f"TIFF文件已保存: {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='生成TIFF图像堆栈')
    parser.add_argument('--h5', type=str, required=True, help='输入HDF5文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出TIFF文件路径')
    parser.add_argument('--config', type=str, help='配置文件路径')
    
    args = parser.parse_args()
    
    config = None
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    generate_tiff_stack(args.h5, args.output, config)