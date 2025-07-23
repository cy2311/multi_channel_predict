import argparse
import random
from pathlib import Path
import os

import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.interpolate as interp
import scipy.ndimage as ndi
from tqdm import tqdm


def load_data(patches_path: Path, emitters_path: Path):
    """Load necessary datasets from the two HDF5 files."""
    with h5py.File(patches_path, 'r') as f_patch:
        phase_maps = np.array(f_patch['z_maps/phase'])          # type: ignore[arg-type]
        coords = np.array(f_patch['coords'], dtype=float)       # type: ignore[arg-type]
        coeff_mag_patch = np.array(f_patch['zernike/coeff_mag']) # type: ignore[arg-type]

    with h5py.File(emitters_path, 'r') as f_emit:
        em_xyz = np.array(f_emit['emitters/xyz'])                # type: ignore[arg-type]

    em_xy = em_xyz[:, :2].astype(float)
    return phase_maps, coords, coeff_mag_patch, em_xy


def compute_phase_coeffs(phase_maps: np.ndarray, em_xy: np.ndarray) -> np.ndarray:
    """Bicubic spline interpolation of phase maps at emitter positions."""
    n_coeff = phase_maps.shape[0]
    phase_coeffs = np.zeros((len(em_xy), n_coeff), dtype=np.float32)

    # clamp coordinates to valid range
    x = np.clip(em_xy[:, 0], 0, phase_maps.shape[2] - 1e-3)
    y = np.clip(em_xy[:, 1], 0, phase_maps.shape[1] - 1e-3)

    for idx in range(n_coeff):
        phase_coeffs[:, idx] = ndi.map_coordinates(
            phase_maps[idx], [y, x], order=3, mode='nearest'
        )
    return phase_coeffs


def compute_mag_coeffs(coords: np.ndarray, coeff_mag_patch: np.ndarray, em_xy: np.ndarray) -> np.ndarray:
    """Cubic interpolation (via griddata) of magnitude coefficients at emitter positions."""
    n_coeff = coeff_mag_patch.shape[1]
    mag_coeffs = np.zeros((len(em_xy), n_coeff), dtype=np.float32)

    for idx in tqdm(range(n_coeff), desc='Interpolating magnitude coeffs'):
        vals = interp.griddata(coords, coeff_mag_patch[:, idx], em_xy, method='cubic')
        # fallback to nearest for out-of-hull points
        mask = np.isnan(vals)
        if np.any(mask):
            vals[mask] = interp.griddata(coords, coeff_mag_patch[:, idx], em_xy[mask], method='nearest')
        mag_coeffs[:, idx] = vals.astype(np.float32)
    return mag_coeffs


def visualise_coeffs(phase_coeffs: np.ndarray, mag_coeffs: np.ndarray, output_path: Path, num_plot: int = 10):
    """Plot line charts for random emitters' phase & magnitude coefficients."""
    n_emitters, n_coeff = phase_coeffs.shape
    chosen = random.sample(range(n_emitters), k=min(num_plot, n_emitters))
    coeff_idx = np.arange(n_coeff)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for i, eid in enumerate(chosen):
        color = cm.tab20(i % 20)  # type: ignore[attr-defined]  # consistent coloring
        axes[0].plot(coeff_idx, phase_coeffs[eid], color=color, linewidth=1)
        axes[1].plot(coeff_idx, mag_coeffs[eid],   color=color, linewidth=1)

    axes[0].set_ylabel('Phase coeff')
    axes[1].set_ylabel('Mag coeff')
    axes[1].set_xlabel('Zernike order index')
    axes[0].set_title('Random emitters – phase coefficients')
    axes[1].set_title('Random emitters – magnitude coefficients')
    plt.tight_layout()
    
    # 使用输出路径的文件名作为可视化文件名的一部分
    output_filename = f"{output_path.stem}_zernike_coeffs.png"
    output_dir = output_path.parent
    plt.savefig(output_dir / output_filename)
    plt.close(fig)


def save_coeffs(emitters_path: Path, phase_coeffs: np.ndarray, mag_coeffs: np.ndarray):
    """Add / update group 'zernike_coeffs' in emitters HDF5 file."""
    with h5py.File(emitters_path, 'a') as f:
        if 'zernike_coeffs' in f:
            del f['zernike_coeffs']  # remove old results to avoid shape mismatch
        grp = f.create_group('zernike_coeffs')
        grp.create_dataset('phase', data=phase_coeffs)
        grp.create_dataset('mag', data=mag_coeffs)


def process_single_file(patches_path: Path, emitters_path: Path, num_plot: int):
    """处理单个h5文件"""
    print(f"处理文件: {emitters_path}")
    
    # 加载数据
    phase_maps, coords, coeff_mag_patch, em_xy = load_data(patches_path, emitters_path)

    # 计算系数
    phase_coeffs = compute_phase_coeffs(phase_maps, em_xy)  # type: ignore[arg-type]
    mag_coeffs   = compute_mag_coeffs(coords, coeff_mag_patch, em_xy)  # type: ignore[arg-type]

    # 可视化
    visualise_coeffs(phase_coeffs, mag_coeffs, emitters_path, num_plot)

    # 保存到发射器文件
    save_coeffs(emitters_path, phase_coeffs, mag_coeffs)
    print(f'已更新 {emitters_path} 文件，添加了zernike_coeffs组，包含phase和mag数据集。')


def main():
    parser = argparse.ArgumentParser(description='计算每个发射器的Zernike系数，使用三次插值。')
    parser.add_argument('--patches', type=str, default='simulated_data/patches.h5', help='输入patches HDF5路径')
    parser.add_argument('--emitters', type=str, default='simulated_data/emitters_sets_raw.h5', help='单个发射器HDF5路径更新')
    parser.add_argument('--emitters_dir', type=str, help='包含多个发射器HDF5文件的目录路径')
    parser.add_argument('--num_plot', type=int, default=10, help='要绘制的随机发射器数量')
    args = parser.parse_args()

    patches_path = Path(args.patches)
    
    # 检查patches文件是否存在
    if not patches_path.exists():
        raise FileNotFoundError(f"找不到patches文件: {patches_path}")
    
    # 如果指定了目录，则处理目录中的所有h5文件
    if args.emitters_dir:
        emitters_dir = Path(args.emitters_dir)
        if not emitters_dir.exists() or not emitters_dir.is_dir():
            raise NotADirectoryError(f"指定的目录不存在或不是一个目录: {emitters_dir}")
        
        # 获取目录中的所有h5文件
        h5_files = list(emitters_dir.glob("*.h5"))
        if not h5_files:
            print(f"警告: 在目录 {emitters_dir} 中没有找到h5文件")
            return
        
        print(f"在目录 {emitters_dir} 中找到 {len(h5_files)} 个h5文件")
        for h5_file in tqdm(h5_files, desc="处理h5文件"):
            process_single_file(patches_path, h5_file, args.num_plot)
    
    # 否则处理单个文件
    else:
        emitters_path = Path(args.emitters)
        if not emitters_path.exists():
            raise FileNotFoundError(f"找不到发射器文件: {emitters_path}")
        
        process_single_file(patches_path, emitters_path, args.num_plot)


if __name__ == '__main__':
    main()