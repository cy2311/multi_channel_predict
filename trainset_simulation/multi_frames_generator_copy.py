import os
import glob
import json
import random
import argparse
from pathlib import Path
from typing import Tuple, List

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import tifffile as tiff  # for OME-TIFF export
from scipy.fft import ifft2, ifftshift, fftshift  # type: ignore
from skimage.transform import resize  # type: ignore
from tqdm import tqdm  # 添加进度条支持


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def load_config() -> Tuple[float, float, float, float]:
    """Load optical parameters from configs/default_config.json.

    Returns
    -------
    wavelength_nm : float
        Wavelength in nanometres.
    pixel_size_nm_x : float
        Pixel size along *x* in nanometres.
    pixel_size_nm_y : float
        Pixel size along *y* in nanometres.
    NA : float
        Numerical aperture.
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
    """Load the 21 × 128 × 128 Zernike basis from .npy files located next to this script."""
    zernike_dir = os.path.join(os.path.dirname(__file__), "..", "simulated_data", "zernike_polynomials")
    pattern = os.path.join(zernike_dir, "zernike_*_n*_m*.npy")
    files = sorted(glob.glob(pattern))
    if len(files) < 21:
        raise RuntimeError("Expected at least 21 Zernike .npy files in the current directory.")
    basis = np.stack([np.load(fp) for fp in files[:21]], axis=0).astype(np.float32)
    return basis  # shape (21, 128, 128)


def construct_pupil(
    coeff_mag: np.ndarray,
    coeff_phase: np.ndarray,
    basis: np.ndarray,
    pupil_mask: np.ndarray,
) -> np.ndarray:
    """Construct complex pupil function from coefficients and basis.

    Parameters
    ----------
    coeff_mag : (21,) ndarray
        Magnitude coefficients.
    coeff_phase : (21,) ndarray
        Phase coefficients (radians).
    basis : (21, N, N) ndarray
        Zernike basis.
    pupil_mask : (N, N) ndarray
        Binary pupil mask (ones inside aperture).
    """
    # Amplitude: 1 + linear combination of basis (clamped to >= 0)
    amplitude = 1.0 + np.sum(coeff_mag[:, None, None] * basis, axis=0)  # type: ignore
    amplitude = np.clip(np.asarray(amplitude), 0.0, np.inf)  # type: ignore

    # Phase: linear combination (radians)
    phase = np.sum(coeff_phase[:, None, None] * basis, axis=0)  # type: ignore[arg-type]

    pupil = amplitude * pupil_mask * np.exp(1j * phase)
    return pupil


def generate_psf(pupil: np.ndarray) -> np.ndarray:
    """Generate intensity PSF (128×128) from pupil function."""
    field = ifft2(ifftshift(pupil))
    psf = np.abs(fftshift(field)) ** 2  # type: ignore[arg-type]  # center bright spot
    psf /= psf.sum()
    return psf.astype(np.float32)


def build_pupil_mask(N: int, pixel_size_x: float, pixel_size_y: float, NA: float, wavelength: float) -> np.ndarray:
    """Return binary pupil mask with cutoff defined by NA."""
    fx = np.fft.fftfreq(N, d=pixel_size_x)
    fy = np.fft.fftfreq(N, d=pixel_size_y)
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    rho = np.sqrt(FX ** 2 + FY ** 2)
    f_max = NA / wavelength
    mask = (rho <= f_max).astype(np.float32)
    return mask


def process_h5_file(h5_path: str, out_dir: str, wavelength_nm: float, pix_x_nm: float, pix_y_nm: float, 
                   NA: float, basis: np.ndarray, pupil_mask: np.ndarray) -> None:
    """处理单个H5文件并生成对应的多帧TIFF结果"""
    # 从文件名中提取集合索引
    h5_filename = os.path.basename(h5_path)
    set_idx = h5_filename.split("_set")[1].split(".")[0]  # 提取"setX"中的X
    out_filename = f"frames_set{set_idx}.ome.tiff"
    out_path = os.path.join(out_dir, out_filename)
    
    # 转换单位
    wavelength_m = wavelength_nm * 1e-9
    pix_x_m = pix_x_nm * 1e-9
    pix_y_m = pix_y_nm * 1e-9
    
    N = basis.shape[1]
    
    # 加载发射器数据
    with h5py.File(h5_path, "r") as f:
        # 检查文件中是否有必要的数据集
        if "records/frame_ix" not in f or "records/id" not in f:
            print(f"警告: {h5_path} 中缺少必要的数据集，跳过处理")
            return
            
        frame_ix = np.asarray(f["records/frame_ix"])  # type: ignore[index]
        ids_all = np.asarray(f["records/id"])  # type: ignore[index]

        on_mask = frame_ix == 0  # first frame ON emitters
        emitter_ids = np.asarray(ids_all[on_mask])  # type: ignore[index]

        if emitter_ids.size == 0:  # type: ignore[attr-defined]
            print(f"警告: {h5_path} 中在第0帧没有找到激活的发射器，跳过处理")
            return

        # 预加载全局每个发射器的数据
        if "zernike_coeffs/mag" not in f or "zernike_coeffs/phase" not in f:
            print(f"警告: {h5_path} 中缺少Zernike系数数据，请先运行compute_zernike_coeffs.py处理此文件")
            return
            
        coeff_mag_all = np.asarray(f["zernike_coeffs/mag"])  # (Ne,21)
        coeff_phase_all = np.asarray(f["zernike_coeffs/phase"])  # (Ne,21)
        emit_xyz_all = np.asarray(f["emitters/xyz"])  # (Ne,3)

        # 每帧的记录
        ids_rec = ids_all
        xyz_rec = np.asarray(f["records/xyz"])  # (Nr,3)
        phot_rec = np.asarray(f["records/phot"])  # (Nr,)

    # 频率网格用于散焦
    fx = np.fft.fftfreq(N, d=pix_x_m)
    fy = np.fft.fftfreq(N, d=pix_y_m)
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    RHO2 = FX ** 2 + FY ** 2

    # 迭代唯一帧并构建低分辨率图像堆栈
    unique_frames = np.unique(frame_ix)
    low_imgs = []

    HR_SIZE = 6144
    ROI_SIZE = 1200.0
    UPSCALE = HR_SIZE / ROI_SIZE
    half_psf = N // 2

    for fr in tqdm(unique_frames, desc=f"处理集合{set_idx}的帧"):
        on_mask = frame_ix == fr
        chosen_ids = ids_rec[on_mask]
        if chosen_ids.size == 0:
            # 此帧中没有活跃的发射器 - 保持空白
            low_imgs.append(np.zeros((int(ROI_SIZE), int(ROI_SIZE)), dtype=np.float32))
            continue

        coeff_mag = coeff_mag_all[chosen_ids]
        coeff_phase = coeff_phase_all[chosen_ids]
        z_nm = xyz_rec[on_mask, 2]
        xy_pix = xyz_rec[on_mask, :2]
        phot_counts = phot_rec[on_mask]

        canvas = np.zeros((HR_SIZE, HR_SIZE), dtype=np.float32)

        for i in range(chosen_ids.size):
            P0 = construct_pupil(coeff_mag[i], coeff_phase[i], basis, pupil_mask)
            pupil_defocus = P0 * np.exp(1j * np.pi * wavelength_m * (z_nm[i] * 1e-9) * RHO2)
            psf_i = generate_psf(pupil_defocus) * phot_counts[i]

            cx = int(round(xy_pix[i,0] * UPSCALE))
            cy = int(round(xy_pix[i,1] * UPSCALE))

            r0 = cy - half_psf; r1 = cy + half_psf
            c0 = cx - half_psf; c1 = cx + half_psf
            psf_r0 = psf_c0 = 0
            if r0 < 0:
                psf_r0 = -r0; r0 = 0
            if c0 < 0:
                psf_c0 = -c0; c0 = 0
            r1 = min(r1, HR_SIZE)
            c1 = min(c1, HR_SIZE)
            canvas[r0:r1, c0:c1] += psf_i[psf_r0:psf_r0+(r1-r0), psf_c0:psf_c0+(c1-c0)]

        # 下采样到相机分辨率
        low_img = resize(canvas, (int(ROI_SIZE), int(ROI_SIZE)), order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
        # 光子守恒
        total_h = float(canvas.sum()); total_l = float(low_img.sum())
        if total_l != 0:
            low_img *= (total_h/total_l)

        low_imgs.append(low_img)

    stack = np.stack(low_imgs, axis=0)  # (T, 1200,1200)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    ome_meta = {
        "Axes": "TYX",
        "PhysicalSizeX": pix_x_nm,
        "PhysicalSizeXUnit": "nm",
        "PhysicalSizeY": pix_y_nm,
        "PhysicalSizeYUnit": "nm",
    }
    tiff.imwrite(out_path, stack, photometric="minisblack", metadata=ome_meta)
    print(f"保存了{stack.shape[0]}帧堆栈到 {out_path}")


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="从发射器H5数据集生成多帧TIFF")
    parser.add_argument("--h5", type=str, help="输入发射器HDF5文件的路径")
    parser.add_argument("--out_dir", type=str, default="../simulated_data_multi_frames", help="输出目录")
    args = parser.parse_args()

    # 检查输入参数
    h5_path = args.h5 if args.h5 else os.path.join(os.path.dirname(__file__), "..", "simulated_data_multi_frames", "emitter_sets")
    h5_path = os.path.abspath(h5_path)
    
    # 加载配置参数
    wavelength_nm, pix_x_nm, pix_y_nm, NA = load_config()
    
    # 基础和瞳孔掩码
    basis = load_zernike_basis()  # shape (21, 128, 128)
    N = basis.shape[1]
    wavelength_m = wavelength_nm * 1e-9
    pix_x_m = pix_x_nm * 1e-9
    pix_y_m = pix_y_nm * 1e-9
    pupil_mask = build_pupil_mask(N, pix_x_m, pix_y_m, NA, wavelength_m)

    # 确保输出目录存在
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    # 检查输入路径是文件还是目录
    if os.path.isfile(h5_path):
        # 处理单个文件
        process_h5_file(h5_path, out_dir, wavelength_nm, pix_x_nm, pix_y_nm, NA, basis, pupil_mask)
    elif os.path.isdir(h5_path):
        # 处理目录中的所有h5文件
        h5_files = sorted(glob.glob(os.path.join(h5_path, "*.h5")))
        if not h5_files:
            print(f"警告: 在目录 {h5_path} 中没有找到h5文件")
            return
        
        print(f"在目录 {h5_path} 中找到 {len(h5_files)} 个h5文件")
        for h5_file in h5_files:
            process_h5_file(h5_file, out_dir, wavelength_nm, pix_x_nm, pix_y_nm, NA, basis, pupil_mask)
    else:
        raise FileNotFoundError(f"指定的路径不存在: {h5_path}")


if __name__ == "__main__":
    main()