#!/usr/bin/env python3
"""
tiff2h5_phase_retrieval.py

从原始的OME-TIFF文件中检测发射体，提取PSF补丁，进行相位恢复，计算Zernike系数，
并生成全视场的Zernike系数图。整个流程的结果保存在HDF5文件中。

使用方法：
    python tiff2h5_phase_retrieval.py
"""

import os
import json
import math
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from scipy.ndimage import zoom, gaussian_filter
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from numpy.linalg import lstsq
from skimage.feature import peak_local_max
from sklearn.neighbors import KDTree
import tifffile as tiff

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量和配置
CONFIG_PATH = os.path.join('..', 'configs', 'default_config.json')
CAMERA_PARAMS_PATH = os.path.join('..', 'beads', 'spool_100mW_30ms_3D_1_2', 'camera_parameters.json')
TIFF_PATH = os.path.join('..', 'beads', 'spool_100mW_30ms_3D_1_2', 'spool_100mW_30ms_3D_1_2_MMStack_Default.ome.tif')
OUTPUT_DIR = os.path.join('result')
OUTPUT_H5 = os.path.join(OUTPUT_DIR, 'result.h5')
ZERNIKE_DIR = os.path.join('..', 'simulated_data', 'zernike_polynomials')

# 补丁提取参数
PATCH_Z = 201  # Z轴帧数
PATCH_XY = 25  # 补丁XY尺寸
MIN_DISTANCE = 25  # 发射体之间的最小距离

# 相位恢复参数
CENTER_SLICE =101  # 中心帧索引
STEP = 2  # 帧间隔
N_EACH_SIDE = 20  # 中心两侧各取的帧数
ITER_MAX = 50  # 最大迭代次数
NCC_THRESHOLD = 0.85  # 早停NCC阈值
SIGMA_OTF = 1  # 可视化时的高斯模糊sigma

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------

def load_config(path):
    """加载JSON配置文件"""
    with open(path, 'r') as f:
        return json.load(f)


def load_tiff_stack(path):
    """使用内存映射加载OME-TIFF堆栈"""
    with tiff.TiffFile(path) as tif:
        # out='memmap'返回numpy.memmap - 行为类似ndarray但是是惰性的
        arr = tif.asarray(out='memmap')
    # 确保数组是(Z, Y, X)。对于许多OME-TIFF采集，第一个轴可能是时间或通道；
    # 我们挤压单例维度。
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D stack (Z,Y,X) but got shape {arr.shape}")
    return arr


def detect_emitters(stack, n_emitters=100, min_distance=MIN_DISTANCE):
    """使用MIP + 局部最大值检测候选发射体(x,y)位置。
    
    参数
    ----------
    stack : np.ndarray
        3-D体积 (Z,Y,X)。
    n_emitters : int
        要保留的最大发射体数量。
    min_distance : int
        峰值之间的最小距离。
    
    返回
    -------
    List[(y,x)] 按峰值强度排序的坐标。
    """
    # 沿Z轴的最大强度投影
    mip = stack.max(axis=0)
    # 平滑以抑制像素噪声
    mip_smooth = gaussian_filter(mip.astype(float), sigma=1)
    # 使用skimage定位峰值
    coords = peak_local_max(
        mip_smooth,
        min_distance=min_distance,
        threshold_abs=np.percentile(mip_smooth, 98),
        num_peaks=n_emitters,
    )
    # peak_local_max返回(row,col)
    return [tuple(coord) for coord in coords]


def filter_close_emitters(coords, min_distance=MIN_DISTANCE):
    """使用KDTree过滤掉彼此太近的发射体"""
    if len(coords) <= 1:
        return coords
    
    # 转换为numpy数组以便KDTree处理
    coords_array = np.array(coords)
    
    # 构建KDTree
    tree = KDTree(coords_array)
    
    # 查询每个点的邻居
    indices_to_keep = []
    remaining_indices = set(range(len(coords)))
    
    while remaining_indices:
        # 选择第一个剩余的索引
        idx = min(remaining_indices)
        indices_to_keep.append(idx)
        
        # 查找太近的点
        close_indices = tree.query_radius([coords_array[idx]], r=min_distance)[0]
        
        # 从剩余集合中移除这些点
        remaining_indices -= set(close_indices)
    
    # 返回过滤后的坐标
    return [coords[i] for i in indices_to_keep]


def extract_patch(stack, center, patch_z=PATCH_Z, patch_xy=PATCH_XY):
    """提取以给定(y,x)为中心的(Z, PATCH_XY, PATCH_XY)补丁。"""
    y, x = center
    half = patch_xy // 2
    y_start, y_end = y - half, y + half + 1  # +1因为end是排他的
    x_start, x_end = x - half, x + half + 1
    # 防止边界情况
    if y_start < 0 or y_end > stack.shape[1] or x_start < 0 or x_end > stack.shape[2]:
        raise ValueError("Emitter too close to border for required patch size")
    patch = stack[:patch_z, y_start:y_end, x_start:x_end]
    if patch.shape != (patch_z, patch_xy, patch_xy):
        raise ValueError(f"Unexpected patch shape {patch.shape}")
    return patch


def photons_per_pixel(patch, e_adu, baseline, qe=None):
    """将补丁中的ADU值转换为每像素光子计数。"""
    # 减去相机基线
    electrons = (patch.astype(float) - baseline) * e_adu
    electrons[electrons < 0] = 0  # 剪裁负值
    if qe is not None and qe > 0:
        photons = electrons / qe
    else:
        photons = electrons
    return photons


def normalize_patch(patch):
    """将补丁全局归一化到[0,1]（跨所有体素）。"""
    vmin = patch.min()
    vmax = patch.max()
    if vmax == vmin:
        return np.zeros_like(patch, dtype=float)
    return (patch - vmin) / (vmax - vmin)


def select_frames(patch, center=CENTER_SLICE, step=STEP, n_each_side=N_EACH_SIDE):
    """以固定步长选择中心周围的帧。"""
    indices = center + np.arange(-n_each_side, n_each_side + 1) * step
    if indices.min() < 0 or indices.max() >= patch.shape[0]:
        raise ValueError("Selected frame indices out of bounds")
    return patch[indices]


def normalized_cross_correlation(a, b):
    """计算两个图像（相同形状）之间的NCC。"""
    a_flat = a.ravel().astype(float)
    b_flat = b.ravel().astype(float)
    a_flat -= a_flat.mean()
    b_flat -= b_flat.mean()
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def load_zernike_basis():
    """加载之前计算的21个Zernike基（128x128）。

    尝试两个位置：
    1. 直接在simulated_data/下
    2. 在simulated_data/zernike_polynomials/下
    """
    import glob
    pattern_root = os.path.join(ZERNIKE_DIR, 'zernike_*_n*_m*.npy')
    files = sorted(glob.glob(pattern_root))
    if len(files) < 21:
        pattern_sub = os.path.join('..', 'simulated_data', 'zernike_*_n*_m*.npy')
        files = sorted(glob.glob(pattern_sub))
    if len(files) < 21:
        # 如果找不到现有的基，则生成新的
        logger.info("Generating new Zernike basis...")
        generate_zernike_basis()
        files = sorted(glob.glob(pattern_root))
    if len(files) < 21:
        raise RuntimeError('Expected 21 Zernike basis files (npy) in simulated_data or simulated_data/zernike_polynomials.')
    basis = np.stack([np.load(fp) for fp in files[:21]], axis=0)  # (21, 128, 128)
    return basis


def unwrap_phase_2d(phase):
    """简单的2-D相位解包。"""
    return np.unwrap(np.unwrap(phase, axis=0), axis=1)


# -----------------------------------------------------------------------------
# OTF高斯低通（INSPR风格）的辅助函数
# -----------------------------------------------------------------------------

def _psf_to_otf(psf):
    """将空间PSF（已移位）转换为复数OTF（已移位）。"""
    return fftshift(ifft2(ifftshift(psf)))


def _gauss2d(flat_coords, amp, sigma, bg):
    """用于curve_fit的各向同性2-D高斯（输入已展平）。"""
    x, y = flat_coords
    r2 = x ** 2 + y ** 2
    return amp * np.exp(-r2 / (2.0 * sigma ** 2)) + bg


def _fit_gaussian_ratio(ratio_crop):
    """将裁剪的比率OTF拟合为高斯，返回(amp, sigma_px, bg)。"""
    n = ratio_crop.shape[0]
    xx, yy = np.meshgrid(np.arange(n) - n // 2, np.arange(n) - n // 2)
    popt, _ = curve_fit(
        _gauss2d,
        (xx.ravel(), yy.ravel()),
        ratio_crop.ravel(),
        p0=(1.0, 3.0, 0.0),
        bounds=([0.0, 0.3, -np.inf], [10.0, 10.0, np.inf]),
    )
    amp, sigma, bg = float(popt[0]), float(popt[1]), float(popt[2])
    return (amp, sigma, bg)


def _build_gauss_filter(N, sigma_px, amp=1.0, bg=0.0):
    """返回以DC为中心的N×N各向同性高斯滤波器。"""
    xx, yy = np.meshgrid(np.arange(N) - N // 2, np.arange(N) - N // 2)
    return amp * np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma_px ** 2)) + bg


# -----------------------------------------------------------------------------
# Zernike多项式生成
# -----------------------------------------------------------------------------

def wyant_l_to_nm(l):
    """将Wyant索引l（0-based）转换为(n, m)。"""
    n = int(math.floor(math.sqrt(l)))
    rem = l - n * n
    mm = math.ceil((2 * n - rem) / 2)  # m的幅度
    if rem % 2 == 0:
        m = mm
    else:
        m = -mm
    return n, m


def radial_poly(n, m, rho):
    """计算径向分量R_n^|m|(rho)。"""
    m = abs(m)
    if (n - m) % 2 != 0:
        # 对于无效的奇偶性，径向分量处处为零
        return np.zeros_like(rho)
    R = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        coeff = ((-1) ** k) * math.factorial(n - k) / (
            math.factorial(k) * math.factorial((n + m) // 2 - k) * math.factorial((n - m) // 2 - k)
        )
        R += coeff * rho ** (n - 2 * k)
    return R


def zernike_nm(n, m, rho, theta):
    """返回rho<=1上的（未归一化）Zernike多项式Z_n^m。"""
    R = radial_poly(n, m, rho)
    if m > 0:
        Z = R * np.cos(m * theta)
    elif m < 0:
        Z = R * np.sin(-m * theta)
    else:
        Z = R  # m == 0
    return Z


def generate_zernike_basis():
    """生成Zernike多项式基并保存为.npy文件。"""
    # 从配置加载像素大小（各向异性）
    cfg = load_config(CONFIG_PATH)
    optical_cfg = cfg.get('optical', {})
    pixel_size_nm_x = optical_cfg.get('pixel_size_nm_x', 101.11)
    pixel_size_nm_y = optical_cfg.get('pixel_size_nm_y', 98.83)

    # 转换为米进行计算（物理网格）
    pixel_size_x = pixel_size_nm_x * 1e-9  # m
    pixel_size_y = pixel_size_nm_y * 1e-9  # m

    # 配置
    n_pixels = 128  # 每个多项式的网格大小（正方形）
    max_l = 21  # 要生成的Wyant索引多项式总数
    output_dir = ZERNIKE_DIR  # .npy和.png文件的目标目录
    os.makedirs(output_dir, exist_ok=True)

    # 网格准备（物理坐标感知）
    # 构建具有各向异性像素间距的物理坐标数组，以0为中心。
    half_idx = (n_pixels - 1) / 2.0
    x_phys = (np.arange(n_pixels) - half_idx) * pixel_size_x  # 米长度
    y_phys = (np.arange(n_pixels) - half_idx) * pixel_size_y

    X, Y = np.meshgrid(x_phys, y_phys)

    # 将物理半径归一化为矩形网格内的单位圆
    r_phys = np.sqrt(X**2 + Y**2)
    r_max = r_phys.max() if r_phys.max() != 0 else 1.0  # 避免除以零

    rho = r_phys / r_max
    theta = np.arctan2(Y, X)

    inside_mask = rho <= 1.0

    # 将单位圆外的rho设为0（值不会被使用）
    rho[~inside_mask] = 0

    # 生成并保存基多项式
    logger.info(f"Generating first {max_l} Wyant-indexed Zernike polynomials on a {n_pixels}×{n_pixels} grid...")
    for l in range(max_l):
        n, m = wyant_l_to_nm(l)
        Z = zernike_nm(n, m, rho, theta)
        # 为清晰起见，将孔径外的值置零
        Z[~inside_mask] = 0

        # 可选的孔径内单位RMS归一化
        rms = np.sqrt(np.mean(Z[inside_mask] ** 2))
        if rms > 0:
            Z /= rms

        # 保存为.npy
        npy_path = os.path.join(output_dir, f'zernike_{l:02d}_n{n}_m{m:+d}.npy')
        np.save(npy_path, Z.astype(np.float32))

        # 可视化并保存图像
        vmax = np.max(np.abs(Z))
        plt.figure(figsize=(3, 3))
        plt.imshow(Z, cmap='seismic', vmin=-vmax, vmax=vmax)
        plt.axis('off')
        plt.title(f'l={l}\n(n={n}, m={m:+d})')
        png_path = os.path.join(output_dir, f'zernike_{l:02d}_n{n}_m{m:+d}.png')
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f'  Saved l={l:02d} (n={n}, m={m:+d}) to {npy_path} & {png_path}')

    logger.info('Zernike basis generation completed.')


# -----------------------------------------------------------------------------
# 相位恢复
# -----------------------------------------------------------------------------

def run_gs_phase_retrieval(meas_psfs, cfg):
    """对测量的PSF堆栈运行Gerchberg-Saxton相位恢复算法。
    
    参数
    ----------
    meas_psfs : np.ndarray
        形状为(41, H, W)的数组，表示41个选定的PSF切片
    cfg : dict
        包含光学参数的配置字典
        
    返回
    -------
    tuple：(P, final_psf_stack, mean_ncc)
        P : 复数瞳函数
        final_psf_stack : 预测的PSF堆栈
        mean_ncc : 最终的平均NCC值
    """
    # 获取光学参数
    wavelength_nm = cfg['optical']['wavelength_nm']
    pixel_size_nm_x = cfg['optical']['pixel_size_nm_x']
    pixel_size_nm_y = cfg['optical']['pixel_size_nm_y']
    NA = cfg['optical']['NA']

    wavelength = wavelength_nm * 1e-9  # m
    pixel_size_x = pixel_size_nm_x * 1e-9  # m
    pixel_size_y = pixel_size_nm_y * 1e-9  # m

    # 预计算测量的振幅（强度的平方根）
    meas_ampls = np.sqrt(np.clip(meas_psfs, 0, None))

    # 使用128x128的频域尺寸
    N = 128  # 直接使用128x128的频域尺寸
    
    # 频率网格用于瞳面操作
    fx = np.fft.fftfreq(N, d=pixel_size_x)  # 每米周期（x轴）
    fy = np.fft.fftfreq(N, d=pixel_size_y)  # 每米周期（y轴）
    FY, FX = np.meshgrid(fy, fx, indexing='ij')
    RHO2 = FX ** 2 + FY ** 2  # 径向空间频率平方

    # 由NA定义的最大频率
    f_max = NA / wavelength  # 每米截止周期
    pupil_mask = (np.sqrt(RHO2) <= f_max).astype(float)

    # 离焦相位因子：
    #   H(z) = exp(i * pi * wavelength * z * (FX^2 + FY^2))  [Fresnel近似]
    z_list_m = (np.arange(-N_EACH_SIDE, N_EACH_SIDE + 1) * 20e-9)  # 20nm间隔
    defocus_phases = [np.exp(1j * math.pi * wavelength * z * RHO2) for z in z_list_m]

    # 初始化
    P = pupil_mask * np.exp(1j * np.zeros((N, N)))  # 初始瞳函数（单位振幅，零相位）

    # 主GS迭代循环
    logger.info('Starting Gerchberg–Saxton iterations...')
    final_mean_ncc = 0.0
    for it in range(ITER_MAX):
        P_estimates = []
        preds_psf = []

        for z_idx, H_z in enumerate(defocus_phases):
            # 前向传播：瞳面 -> 图像（通过逆FFT）
            Pz = P * H_z
            field_img_large = np.asarray(ifft2(ifftshift(Pz)))
            
            # 从大场景中提取中心区域与原始PSF匹配，并将修改后的小场景嵌入到大场景中
            center = N // 2
            half_patch = meas_psfs.shape[1] // 2
            field_img = field_img_large[center-half_patch:center+half_patch+1, center-half_patch:center+half_patch+1]
            
            # 用测量的振幅替换
            new_field_img = meas_ampls[z_idx] * np.exp(1j * np.angle(np.asarray(field_img)))
            
            # 将修改后的小场景嵌入到大场景中
            new_field_img_large = np.zeros((N, N), dtype=complex)
            new_field_img_large[center-half_patch:center+half_patch+1, center-half_patch:center+half_patch+1] = new_field_img
            
            # 反向传播
            Pz_new = np.asarray(fftshift(fft2(new_field_img_large)))
            # 移除离焦
            P_est = Pz_new / H_z
            P_estimates.append(P_est)

            # 保存预测的PSF用于相似度检查（使用小场景进行比较）
            preds_psf.append(np.abs(field_img) ** 2)

        # 计算preds_psf和meas_psfs之间各切片的平均NCC
        ncc_vals = [normalized_cross_correlation(preds_psf[i], meas_psfs[i]) for i in range(len(preds_psf))]
        mean_ncc = float(np.mean(ncc_vals))
        logger.info(f'Iteration {it+1:02d}: mean NCC = {mean_ncc:.4f}')

        # 通过平均复数估计并重新应用掩码来更新瞳函数
        P = np.mean(P_estimates, axis=0)
        # 在掩码内强制瞳函数振幅为1（或保持幅度？）
        P = pupil_mask * np.exp(1j * np.angle(np.asarray(P)))

        final_mean_ncc = mean_ncc
        if mean_ncc >= NCC_THRESHOLD:
            logger.info('Stopping early – similarity threshold reached.')
            break

    # 最终预测的PSF堆栈
    logger.info('Generating final predicted PSFs...')
    pred_psfs_final = []
    for H_z in defocus_phases:
        field_img_large = np.asarray(ifft2(ifftshift(P * H_z)))
        # 提取中心区域与原始PSF大小匹配
        center = N // 2
        half_patch = meas_psfs.shape[1] // 2
        field_img = field_img_large[center-half_patch:center+half_patch+1, center-half_patch:center+half_patch+1]
        pred_psfs_final.append(np.abs(field_img) ** 2)
    final_psf_stack = np.stack(pred_psfs_final)

    return P, final_psf_stack, final_mean_ncc


def apply_otf_gaussian_lowpass(final_psf_stack, meas_psfs):
    """应用OTF高斯低通正则化（INSPR风格）。"""
    logger.info('Applying OTF Gaussian low-pass regularization...')

    # 选择中心切片（meas_psfs列表中的索引）
    center_idx = N_EACH_SIDE

    meas_center = meas_psfs[center_idx]
    pred_center = final_psf_stack[center_idx]

    # 1. 计算OTF的幅度比
    otf_meas = _psf_to_otf(meas_center)
    otf_pred = _psf_to_otf(pred_center)
    ratio_mag = np.abs(otf_meas) / (np.abs(otf_pred) + 1e-12)

    # 2. 裁剪中心窗口并拟合各向同性高斯
    OTF_RATIO_SIZE = 60  # 用于高斯拟合的裁剪大小（像素）
    R = ratio_mag.shape[0]
    half = min(OTF_RATIO_SIZE // 2, R // 2 - 1)  # 确保不超出边界
    crop = ratio_mag[R // 2 - half : R // 2 + half, R // 2 - half : R // 2 + half]
    amp_g, sigma_px, bg_g = _fit_gaussian_ratio(crop)
    logger.info(f'  Fitted Gaussian sigma = {sigma_px:.2f} px')

    # 3. 构建全尺寸高斯滤波器
    gauss_filter = _build_gauss_filter(R, sigma_px, amp=amp_g, bg=bg_g)

    # 4. 将滤波器应用于每个预测的OTF并转换回PSF
    psf_lpf_list = []
    for psf_z in final_psf_stack:
        otf_z = _psf_to_otf(psf_z)
        otf_filt = otf_z * gauss_filter
        psf_mod = np.abs(fft2(ifftshift(otf_filt))) ** 2
        psf_mod /= psf_mod.sum()
        psf_lpf_list.append(psf_mod.astype(np.float32))

    mod_psf_stack = np.stack(psf_lpf_list)
    return mod_psf_stack


def zernike_decompose(P, pupil_mask, basis):
    """将瞳函数分解为Zernike系数。
    
    参数
    ----------
    P : np.ndarray
        复数瞳函数
    pupil_mask : np.ndarray
        瞳孔掩码（布尔值或浮点数）
    basis : np.ndarray
        Zernike基函数，形状为(21, H, W)
        
    返回
    -------
    tuple：(coeff_mag, coeff_phase)
        两个长度为21的数组，分别表示幅度和相位系数
    """
    logger.info('Performing Zernike decomposition...')
    inside_aperture = pupil_mask.astype(bool)

    # 为最小二乘法准备矩阵
    basis_vectors = basis[:, inside_aperture].T  # 形状 (Npix_in, 21)

    # 幅度系数
    mag_target = np.abs(P)[inside_aperture]
    coeff_mag, *_ = lstsq(basis_vectors, mag_target, rcond=None)

    # 强制幅度系数的RMS = 1
    mag_rms = np.linalg.norm(coeff_mag)
    if mag_rms != 0:
        coeff_mag = coeff_mag / mag_rms

    # 相位系数（使用解包的相位）
    phase_target = unwrap_phase_2d(np.angle(np.asarray(P)))[inside_aperture]
    coeff_phase, *_ = lstsq(basis_vectors, phase_target, rcond=None)

    # 强制相位系数的RMS = 1
    phase_rms = np.linalg.norm(coeff_phase)
    if phase_rms != 0:
        coeff_phase = coeff_phase / phase_rms

    return coeff_mag, coeff_phase


def build_grid_maps(coords, coeff_phase, roi_yx=(1200, 1200)):
    """返回形状为(21, H, W)的插值图数组。"""
    N = coords.shape[0]
    if N < 4:
        raise RuntimeError('Too few points for interpolation')
    y_coords, x_coords = coords[:, 0], coords[:, 1]
    H, W = roi_yx
    xi = np.linspace(0, W - 1, W)
    yi = np.linspace(0, H - 1, H)
    XI, YI = np.meshgrid(xi, yi)

    maps = np.empty((21, H, W), dtype=np.float32)
    points = np.column_stack([x_coords, y_coords])

    for j in range(21):
        values = coeff_phase[:, j]
        grid = griddata(points, values, (XI, YI), method='cubic')
        # 使用最近邻插值处理NaN
        nan_mask = np.isnan(grid)
        if np.any(nan_mask):
            grid[nan_mask] = griddata(points, values, (XI[nan_mask], YI[nan_mask]), method='nearest')
        maps[j] = grid.astype(np.float32)
    return maps


def plot_maps(maps, output_path):
    """将21个Zernike系数图绘制为7×3面板图。"""
    n_rows, n_cols = 7, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))

    idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            if idx < 21:
                data = maps[idx]
                vmax_local = np.nanmax(np.abs(data))
                im = ax.imshow(data, cmap='coolwarm', origin='lower', vmin=-vmax_local, vmax=vmax_local)
                ax.set_title(f'Z{idx+1}')
                ax.axis('off')
                # 单独的颜色条
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.01)
            else:
                ax.remove()
            idx += 1
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f'Zernike maps figure saved to {output_path}')


def visualize_psf_comparison(meas_psfs, final_psf_stack, output_dir):
    """可视化原始和预测的PSF比较。"""
    # 中心PSF比较
    orig_center = meas_psfs[N_EACH_SIDE]  # meas_psfs列表中的中心切片索引
    pred_center = final_psf_stack[N_EACH_SIDE]

    # 应用相当于傅里叶域的高斯模糊（这里只是空间域）
    orig_blur = gaussian_filter(np.asarray(orig_center, dtype=float), sigma=SIGMA_OTF)
    pred_blur = gaussian_filter(np.asarray(pred_center, dtype=float), sigma=SIGMA_OTF)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, img, title in zip(axes, [orig_blur, pred_blur], ['Original PSF', 'Predicted PSF (blurred)']):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'psf_comparison.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f'PSF comparison figure saved to {fig_path}')

    # 5帧比较（原始vs预测）
    VIS_OFFSETS = [0, 10, 20, 30, 40]
    fig, axes = plt.subplots(len(VIS_OFFSETS), 2, figsize=(6, 12))
    for row, idx in enumerate(VIS_OFFSETS):
        orig_b = gaussian_filter(np.asarray(meas_psfs[idx], dtype=float), sigma=SIGMA_OTF)
        pred_b = gaussian_filter(np.asarray(final_psf_stack[idx], dtype=float), sigma=SIGMA_OTF)
        axes[row, 0].imshow(orig_b, cmap='gray')
        axes[row, 0].set_title(f'Orig frame {idx+1}')
        axes[row, 0].axis('off')
        axes[row, 1].imshow(pred_b, cmap='gray')
        axes[row, 1].set_title(f'Pred frame {idx+1}')
        axes[row, 1].axis('off')
    plt.tight_layout()
    fig_path2 = os.path.join(output_dir, 'psf_comparison_five.png')
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    logger.info(f'5-frame comparison figure saved to {fig_path2}')


def plot_zernike_coefficients(coeff_mag, coeff_phase, output_dir):
    """绘制单个PSF的Zernike幅度和相位系数。"""
    x = np.arange(1, len(coeff_mag) + 1)  # 为可读性使用1-based索引

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(x, coeff_mag, marker='o')
    axes[0].set_ylabel('Magnitude Coefficient')
    axes[0].set_title('Zernike Magnitude Coefficients')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].plot(x, coeff_phase, marker='o', color='tab:red')
    axes[1].set_xlabel('Coefficient Index (1-based)')
    axes[1].set_ylabel('Phase Coefficient')
    axes[1].set_title('Zernike Phase Coefficients')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'zernike_coefficients.png')
    fig.savefig(fig_path, dpi=300)
    logger.info(f"Zernike coefficients figure saved to {fig_path}")


def plot_zernike_coefficients_dataset(coeff_mag_set, coeff_phase_set, mean_ncc, output_dir, n_samples=10):
    """绘制多个发射体的Zernike系数。"""
    valid_idx = np.where(~np.isnan(mean_ncc))[0]
    if valid_idx.size == 0:
        logger.warning('No valid emitters with NCC >= threshold.')
        return

    import random
    sample_idx = random.sample(list(valid_idx), k=min(n_samples, valid_idx.size))
    sample_idx_sorted = np.sort(sample_idx)
    coeff_mag = np.asarray(coeff_mag_set[sample_idx_sorted])  # (k,21)
    coeff_phase = np.asarray(coeff_phase_set[sample_idx_sorted])

    # 恢复原始随机顺序用于绘图标签
    reorder = np.argsort(np.searchsorted(sample_idx_sorted, sample_idx))
    coeff_mag = coeff_mag[reorder]
    coeff_phase = coeff_phase[reorder]
    sample_idx = [sample_idx[i] for i in reorder]

    x = np.arange(1, coeff_mag.shape[1] + 1)  # 1..21

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    for i, vec in enumerate(coeff_mag):
        axes[0].plot(x, vec, linewidth=0.5, label=f'Emitter {sample_idx[i]}')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title('Zernike Magnitude Coefficients (Random)')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    for i, vec in enumerate(coeff_phase):
        axes[1].plot(x, vec, linewidth=0.5, label=f'Emitter {sample_idx[i]}')
    axes[1].set_xlabel('Coefficient Index (1-based)')
    axes[1].set_ylabel('Phase')
    axes[1].set_title('Zernike Phase Coefficients (Random)')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # 将单个合并的图例放在图外
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5),
               fontsize='small')

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'zernike_coefficients_random.png')
    fig.savefig(fig_path, dpi=300)
    logger.info(f"Random emitters Zernike coefficients figure saved to {fig_path}")


# -----------------------------------------------------------------------------
# 主函数
# -----------------------------------------------------------------------------

def main():
    # 加载配置
    logger.info('Loading configuration...')
    cfg = load_config(CONFIG_PATH)
    
    # 加载相机参数
    logger.info('Loading camera parameters...')
    cam_params = load_config(CAMERA_PARAMS_PATH)
    
    # 更新配置中的像素大小
    cfg['optical']['pixel_size_nm_x'] = cam_params['cam_pxszx']
    cfg['optical']['pixel_size_nm_y'] = cam_params['cam_pxszy']
    
    # 加载TIFF堆栈
    logger.info(f'Loading TIFF stack: {TIFF_PATH}')
    stack = load_tiff_stack(TIFF_PATH)
    logger.info(f'Stack shape: {stack.shape}')
    
    # 检测发射体
    logger.info('Detecting emitters...')
    emitter_coords = detect_emitters(stack, n_emitters=300, min_distance=MIN_DISTANCE)
    logger.info(f'Detected {len(emitter_coords)} emitter candidates')
    
    # 过滤太近的发射体
    filtered_coords = filter_close_emitters(emitter_coords, min_distance=MIN_DISTANCE)
    logger.info(f'After filtering: {len(filtered_coords)} emitters')
    
    # 创建HDF5文件存储结果
    with h5py.File(OUTPUT_H5, 'w') as h5f:
        # 保存坐标
        coords_ds = h5f.create_dataset('coords', data=np.array(filtered_coords), dtype='int32')
        
        # 创建补丁数据集
        n_emitters = len(filtered_coords)
        patches_ds = h5f.create_dataset('patches', shape=(n_emitters, 41, PATCH_XY, PATCH_XY), 
                                       dtype='float32', compression='gzip')
        
        # 创建Zernike组
        zernike_grp = h5f.create_group('zernike')
        coeff_mag_ds = zernike_grp.create_dataset('coeff_mag', shape=(n_emitters, 21), 
                                                dtype='float32', fillvalue=np.nan)
        coeff_phase_ds = zernike_grp.create_dataset('coeff_phase', shape=(n_emitters, 21), 
                                                  dtype='float32', fillvalue=np.nan)
        mean_ncc_ds = zernike_grp.create_dataset('mean_ncc', shape=(n_emitters,), 
                                               dtype='float32', fillvalue=np.nan)
        
        # 加载Zernike基
        Z_basis = load_zernike_basis()  # (21, 128, 128)
        
        # 处理每个发射体
        for idx, coord in enumerate(filtered_coords):
            logger.info(f'Processing emitter {idx+1}/{n_emitters} at {coord}')
            
            try:
                # 提取补丁
                patch = extract_patch(stack, coord, patch_z=PATCH_Z, patch_xy=PATCH_XY)
                
                # 转换为光子
                photons_patch = photons_per_pixel(patch, 
                                                e_adu=cam_params['A2D'], 
                                                baseline=cam_params['offset'])
                
                # 归一化
                patch_norm = normalize_patch(photons_patch)
                
                # 选择41帧
                frames_selected = select_frames(patch_norm, center=CENTER_SLICE, 
                                              step=STEP, n_each_side=N_EACH_SIDE)
                
                # 保存到HDF5
                patches_ds[idx] = frames_selected
                
                # 运行相位恢复
                P, final_psf_stack, final_ncc = run_gs_phase_retrieval(frames_selected, cfg)
                
                # 检查NCC阈值，只有达到0.7才保存
                if final_ncc < 0.7:
                    logger.info(f'Emitter {idx+1} NCC ({final_ncc:.4f}) below threshold 0.7, skipping save')
                    continue
                
                # 跳过OTF高斯低通滤波处理
                # mod_psf_stack = apply_otf_gaussian_lowpass(final_psf_stack, frames_selected)
                logger.info('Skipping OTF Gaussian low-pass regularization...')
                mod_psf_stack = final_psf_stack  # 直接使用原始的PSF堆栈
                
                # 构建瞳孔掩码 - 与相位恢复中使用的掩码相同(128x128)
                N_freq = 128  # 与Zernike基底相同的尺寸
                wavelength = cfg['optical']['wavelength_nm'] * 1e-9
                px_x = cfg['optical']['pixel_size_nm_x'] * 1e-9
                px_y = cfg['optical']['pixel_size_nm_y'] * 1e-9
                NA = cfg['optical']['NA']
                fx = np.fft.fftfreq(N_freq, d=px_x)
                fy = np.fft.fftfreq(N_freq, d=px_y)
                FY, FX = np.meshgrid(fy, fx, indexing='ij')
                RHO2 = FX**2 + FY**2
                pupil_mask = (np.sqrt(RHO2) <= NA / wavelength).astype(np.float32)
                
                # 瞳函数P已经是128x128大小，直接用于Zernike分解
                # Zernike分解
                coeff_mag, coeff_phase = zernike_decompose(P, pupil_mask, Z_basis)
                
                # 保存系数和NCC
                coeff_mag_ds[idx] = coeff_mag
                coeff_phase_ds[idx] = coeff_phase
                mean_ncc_ds[idx] = final_ncc
                
                # 可视化第一个发射体
                if idx == 0:
                    visualize_psf_comparison(frames_selected, final_psf_stack, OUTPUT_DIR)
                    plot_zernike_coefficients(coeff_mag, coeff_phase, OUTPUT_DIR)
                
            except Exception as e:
                logger.warning(f'Error processing emitter {idx}: {e}')
                continue
        
        # 构建全视场Zernike系数图
        logger.info('Building full-field Zernike coefficient maps...')
        try:
            # 过滤有效的发射体
            valid_mask = ~np.isnan(mean_ncc_ds[:])
            if np.sum(valid_mask) >= 4:  # 需要至少4个点进行插值
                valid_coords = np.array(filtered_coords)[valid_mask]
                valid_coeff_phase = coeff_phase_ds[:][valid_mask]
                
                # 构建插值图
                maps = build_grid_maps(valid_coords, valid_coeff_phase, 
                                      roi_yx=(cam_params['chipszh'], cam_params['chipszw']))
                
                # 保存到HDF5
                zm_grp = h5f.create_group('z_maps')
                zm_grp.create_dataset('phase', data=maps, compression='gzip')
                
                # 绘制图
                plot_maps(maps, os.path.join(OUTPUT_DIR, 'zernike_phase_maps.png'))
            else:
                logger.warning('Too few valid emitters for interpolation')
        except Exception as e:
            logger.error(f'Error building Zernike maps: {e}')
        
        # 绘制随机发射体的系数
        try:
            plot_zernike_coefficients_dataset(coeff_mag_ds, coeff_phase_ds, 
                                            mean_ncc_ds, OUTPUT_DIR)
        except Exception as e:
            logger.error(f'Error plotting dataset coefficients: {e}')
    
    logger.info(f'Results saved to {OUTPUT_H5}')
    logger.info('Processing completed.')


if __name__ == '__main__':
    main()