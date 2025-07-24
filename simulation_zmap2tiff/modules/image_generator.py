"""图像生成模块
负责PSF生成和图像合成
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import tifffile as tiff
from scipy.fft import ifft2, ifftshift, fftshift
from skimage.transform import resize
from pathlib import Path
from typing import Tuple, List, Dict, Any
from tqdm import tqdm


class ImageGenerator:
    """图像生成器"""
    
    def __init__(self, config):
        self.config = config
        self.optical_params = config.get_optical_params()
        self.sim_params = config.get_simulation_params()
        
        # 加载Zernike基函数
        self.zernike_basis = self._load_zernike_basis()
        
        # 构建瞳孔掩膜
        self.pupil_mask = self._build_pupil_mask()
        
        # 预计算频率网格
        self._setup_frequency_grids()
        
        print(f"图像生成器初始化完成")
        print(f"Zernike基函数形状: {self.zernike_basis.shape}")
        print(f"瞳孔掩膜形状: {self.pupil_mask.shape}")
    
    def _load_zernike_basis(self) -> np.ndarray:
        """生成Zernike基函数"""
        # 使用默认PSF大小
        psf_size = 128  # 默认PSF大小
        
        # 生成坐标网格
        x = np.linspace(-1, 1, psf_size)
        y = np.linspace(-1, 1, psf_size)
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        
        # 单位圆内的掩膜
        mask = rho <= 1.0
        
        # 生成前21个Zernike模式 (Wyant ordering)
        basis = []
        
        # 手动定义前21个Zernike模式的(n,m)对
        zernike_modes = [
            (0, 0),   # Z0: piston
            (1, 1),   # Z1: tip
            (1, -1),  # Z2: tilt
            (2, 0),   # Z3: defocus
            (2, -2),  # Z4: astigmatism
            (2, 2),   # Z5: astigmatism
            (3, -1),  # Z6: coma
            (3, 1),   # Z7: coma
            (3, -3),  # Z8: trefoil
            (3, 3),   # Z9: trefoil
            (4, 0),   # Z10: spherical
            (4, 2),   # Z11
            (4, -2),  # Z12
            (4, 4),   # Z13
            (4, -4),  # Z14
            (5, 1),   # Z15
            (5, -1),  # Z16
            (5, 3),   # Z17
            (5, -3),  # Z18
            (5, 5),   # Z19
            (5, -5),  # Z20
        ]
        
        for n, m in zernike_modes:
            zernike = self._zernike_polynomial(n, m, rho, theta, mask)
            basis.append(zernike)
        
        basis = np.stack(basis, axis=0).astype(np.float32)
        return basis  # shape (21, psf_size, psf_size)
    
    def _zernike_polynomial(self, n: int, m: int, rho: np.ndarray, 
                           theta: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """生成单个Zernike多项式"""
        # 径向多项式
        R = self._radial_polynomial(n, abs(m), rho)
        
        # 角度部分
        if m > 0:
            Z = R * np.cos(m * theta)
        elif m < 0:
            Z = R * np.sin(abs(m) * theta)
        else:
            Z = R
        
        # 应用掩膜
        Z = Z * mask
        
        # 归一化
        if np.sum(mask) > 0:
            Z = Z / np.sqrt(np.mean(Z[mask]**2)) if np.mean(Z[mask]**2) > 0 else Z
        
        return Z
    
    def _radial_polynomial(self, n: int, m: int, rho: np.ndarray) -> np.ndarray:
        """计算径向多项式R_n^m(rho)"""
        if (n - m) % 2 != 0:
            return np.zeros_like(rho)
        
        R = np.zeros_like(rho)
        for k in range((n - m) // 2 + 1):
            coeff = ((-1)**k * np.math.factorial(n - k) / 
                    (np.math.factorial(k) * 
                     np.math.factorial((n + m) // 2 - k) * 
                     np.math.factorial((n - m) // 2 - k)))
            R += coeff * rho**(n - 2*k)
        
        return R
    
    def _build_pupil_mask(self) -> np.ndarray:
        """构建瞳孔掩膜"""
        N = 128  # 默认PSF大小
        pixel_size = self.optical_params['pixel_size']
        pixel_size_x = pixel_size
        pixel_size_y = pixel_size
        NA = self.optical_params['NA']
        wavelength = self.optical_params['wavelength']
        
        fx = np.fft.fftfreq(N, d=pixel_size_x)
        fy = np.fft.fftfreq(N, d=pixel_size_y)
        FX, FY = np.meshgrid(fx, fy, indexing="ij")
        rho = np.sqrt(FX ** 2 + FY ** 2)
        f_max = NA / wavelength
        mask = (rho <= f_max).astype(np.float32)
        return mask
    
    def _setup_frequency_grids(self):
        """设置频率网格"""
        N = 128  # 默认PSF大小
        pixel_size = self.optical_params['pixel_size']
        pixel_size_x = pixel_size
        pixel_size_y = pixel_size
        
        fx = np.fft.fftfreq(N, d=pixel_size_x)
        fy = np.fft.fftfreq(N, d=pixel_size_y)
        self.FX, self.FY = np.meshgrid(fx, fy, indexing="ij")
        self.RHO2 = self.FX ** 2 + self.FY ** 2
    
    def construct_pupil(self, coeff_mag: np.ndarray, coeff_phase: np.ndarray) -> np.ndarray:
        """构建复数瞳孔函数
        
        Args:
            coeff_mag: (21,) 幅度系数
            coeff_phase: (21,) 相位系数 (弧度)
            
        Returns:
            pupil: (N, N) 复数瞳孔函数
        """
        # 幅度: 1 + 基函数的线性组合 (限制 >= 0)
        amplitude = 1.0 + np.sum(coeff_mag[:, None, None] * self.zernike_basis, axis=0)
        amplitude = np.clip(amplitude, 0.0, np.inf)
        
        # 相位: 基函数的线性组合 (弧度)
        phase = np.sum(coeff_phase[:, None, None] * self.zernike_basis, axis=0)
        
        pupil = amplitude * self.pupil_mask * np.exp(1j * phase)
        return pupil
    
    def generate_psf(self, pupil: np.ndarray) -> np.ndarray:
        """从瞳孔函数生成强度PSF
        
        Args:
            pupil: (N, N) 复数瞳孔函数
            
        Returns:
            psf: (N, N) 归一化强度PSF
        """
        field = ifft2(ifftshift(pupil))
        psf = np.abs(fftshift(field)) ** 2  # 中心亮斑
        psf /= psf.sum()  # 归一化
        return psf.astype(np.float32)
    
    def apply_defocus(self, pupil: np.ndarray, z_nm: float) -> np.ndarray:
        """应用离焦
        
        Args:
            pupil: (N, N) 复数瞳孔函数
            z_nm: 离焦距离 (纳米)
            
        Returns:
            defocused_pupil: (N, N) 离焦后的瞳孔函数
        """
        wavelength = self.optical_params['wavelength']
        defocus_factor = np.exp(1j * np.pi * wavelength * (z_nm * 1e-9) * self.RHO2)
        return pupil * defocus_factor
    
    def generate_single_frame(self, emitter_data: Dict[str, np.ndarray], 
                            zernike_coeffs: Dict[str, np.ndarray],
                            frame_idx: int) -> np.ndarray:
        """生成单帧图像
        
        Args:
            emitter_data: 包含frame_ix, xyz, phot, id的字典
            zernike_coeffs: 包含phase, mag的Zernike系数
            frame_idx: 帧索引
            
        Returns:
            frame_image: (roi_size, roi_size) 帧图像
        """
        # 筛选当前帧的发射器
        frame_mask = emitter_data['frame_ix'] == frame_idx
        if hasattr(frame_mask, 'numpy'):
            frame_mask = frame_mask.numpy()
        if not np.any(frame_mask):
            # 没有活跃发射器，返回空白帧
            roi_size = self.sim_params['image_size'][0]  # 使用图像尺寸
            return np.zeros((roi_size, roi_size), dtype=np.float32)
        
        active_ids = emitter_data['id'][frame_mask]
        active_xyz = emitter_data['xyz'][frame_mask]
        active_phot = emitter_data['phot'][frame_mask]
        
        # 转换为numpy数组
        if hasattr(active_ids, 'numpy'):
            active_ids = active_ids.numpy()
        if hasattr(active_xyz, 'numpy'):
            active_xyz = active_xyz.numpy()
        if hasattr(active_phot, 'numpy'):
            active_phot = active_phot.numpy()
        
        # 确保active_ids是整数类型用于索引
        active_ids = active_ids.astype(int)
        
        # 获取对应的Zernike系数
        active_mag_coeffs = zernike_coeffs['mag'][active_ids]
        active_phase_coeffs = zernike_coeffs['phase'][active_ids]
        
        # 转换Zernike系数为numpy数组
        if hasattr(active_mag_coeffs, 'numpy'):
            active_mag_coeffs = active_mag_coeffs.numpy()
        if hasattr(active_phase_coeffs, 'numpy'):
            active_phase_coeffs = active_phase_coeffs.numpy()
        
        # 生成高分辨率画布
        roi_size = self.sim_params['image_size'][0]  # 使用图像尺寸
        upscale = self.sim_params['upsampling_factor']
        hr_size = int(roi_size * upscale)
        canvas = np.zeros((hr_size, hr_size), dtype=np.float32)
        
        half_psf = 128 // 2  # PSF大小的一半
        
        # 为每个发射器生成PSF并添加到画布
        for i in range(len(active_ids)):
            # 构建瞳孔函数
            pupil = self.construct_pupil(active_mag_coeffs[i], active_phase_coeffs[i])
            
            # 应用离焦
            z_nm = active_xyz[i, 2] * 1000  # μm转换为nm
            pupil_defocus = self.apply_defocus(pupil, z_nm)
            
            # 生成PSF
            psf = self.generate_psf(pupil_defocus)
            
            # 应用光子数权重
            weighted_psf = psf * active_phot[i]
            
            # 计算在高分辨率画布上的位置
            x_pix, y_pix = active_xyz[i, 0], active_xyz[i, 1]
            cx = int(round(x_pix * upscale))
            cy = int(round(y_pix * upscale))
            
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
            r1 = min(r1, hr_size)
            c1 = min(c1, hr_size)
            
            # 添加PSF到画布
            if r1 > r0 and c1 > c0:
                psf_h = r1 - r0
                psf_w = c1 - c0
                canvas[r0:r1, c0:c1] += weighted_psf[psf_r0:psf_r0+psf_h, psf_c0:psf_c0+psf_w]
        
        # 下采样到相机分辨率
        low_img = resize(canvas, (int(roi_size), int(roi_size)), 
                        order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
        
        # 光子守恒
        total_h = float(canvas.sum())
        total_l = float(low_img.sum())
        if total_l > 0:
            low_img *= (total_h / total_l)
        
        return low_img
    
    def generate_multi_frame_stack(self, emitter_data: Dict[str, np.ndarray],
                                 zernike_coeffs: Dict[str, np.ndarray]) -> np.ndarray:
        """生成多帧图像堆栈
        
        Args:
            emitter_data: 发射器数据
            zernike_coeffs: Zernike系数
            
        Returns:
            stack: (T, H, W) 图像堆栈
        """
        frame_ix = emitter_data['frame_ix']
        if hasattr(frame_ix, 'numpy'):
            frame_ix = frame_ix.numpy()
        unique_frames = np.unique(frame_ix)
        frames = []
        
        print(f"生成{len(unique_frames)}帧图像")
        
        for frame_idx in tqdm(unique_frames, desc="生成帧"):
            frame_img = self.generate_single_frame(emitter_data, zernike_coeffs, frame_idx)
            frames.append(frame_img)
        
        stack = np.stack(frames, axis=0)
        print(f"生成的图像堆栈形状: {stack.shape}")
        
        return stack
    
    def save_tiff_stack(self, stack: np.ndarray, output_file: Path, 
                       metadata: Dict[str, Any] = None):
        """保存TIFF堆栈
        
        Args:
            stack: (T, H, W) 图像堆栈
            output_file: 输出文件路径
            metadata: 元数据字典
        """
        # 准备OME元数据
        optical_params = self.optical_params
        ome_meta = {
            "Axes": "TYX",
            "PhysicalSizeX": optical_params['pixel_size'] * 1e9,  # 转换为nm
            "PhysicalSizeXUnit": "nm",
            "PhysicalSizeY": optical_params['pixel_size'] * 1e9,  # 转换为nm
            "PhysicalSizeYUnit": "nm",
        }
        
        if metadata:
            ome_meta.update(metadata)
        
        # 确保输出目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存TIFF文件
        tiff.imwrite(output_file, stack, photometric="minisblack", metadata=ome_meta)
        print(f"TIFF堆栈保存到: {output_file}")
    
    def visualize_psfs(self, zernike_coeffs: Dict[str, np.ndarray], 
                      output_dir: Path, num_examples: int = 6, prefix: str = ""):
        """可视化PSF示例
        
        Args:
            zernike_coeffs: Zernike系数
            output_dir: 输出目录
            num_examples: 示例数量
            prefix: 文件名前缀
        """
        n_emitters = len(zernike_coeffs['phase'])
        chosen_ids = np.random.choice(n_emitters, size=min(num_examples, n_emitters), replace=False)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, eid in enumerate(chosen_ids):
            if i >= len(axes):
                break
                
            # 生成PSF
            pupil = self.construct_pupil(zernike_coeffs['mag'][eid], zernike_coeffs['phase'][eid])
            psf = self.generate_psf(pupil)
            
            # 绘制
            im = axes[i].imshow(psf, cmap='hot', norm=LogNorm(vmin=psf.max()*1e-6, vmax=psf.max()))
            axes[i].set_title(f'发射器 {eid} PSF')
            axes[i].set_xlabel('像素')
            axes[i].set_ylabel('像素')
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        output_file = output_dir / f"{prefix}psf_examples.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"PSF示例保存到: {output_file}")
    
    def visualize_frame_montage(self, stack: np.ndarray, output_dir: Path, 
                              max_frames: int = 16, prefix: str = ""):
        """可视化帧蒙太奇
        
        Args:
            stack: (T, H, W) 图像堆栈
            output_dir: 输出目录
            max_frames: 最大显示帧数
            prefix: 文件名前缀
        """
        n_frames = min(stack.shape[0], max_frames)
        cols = int(np.ceil(np.sqrt(n_frames)))
        rows = int(np.ceil(n_frames / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.ravel()
        else:
            axes = axes.ravel()
        
        # 计算全局强度范围
        vmin = stack.min()
        vmax = np.percentile(stack, 99.5)  # 使用99.5百分位数避免异常值
        
        for i in range(n_frames):
            frame = stack[i]
            im = axes[i].imshow(frame, cmap='gray', vmin=vmin, vmax=vmax)
            axes[i].set_title(f'帧 {i}')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        
        # 隐藏多余的子图
        for i in range(n_frames, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        output_file = output_dir / f"{prefix}frame_montage.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"帧蒙太奇保存到: {output_file}")
    
    def analyze_image_statistics(self, stack: np.ndarray) -> Dict[str, Any]:
        """分析图像统计信息
        
        Args:
            stack: (T, H, W) 图像堆栈
            
        Returns:
            stats: 统计信息字典
        """
        stats = {
            'shape': stack.shape,
            'dtype': str(stack.dtype),
            'total_photons': float(stack.sum()),
            'mean_intensity': float(stack.mean()),
            'std_intensity': float(stack.std()),
            'min_intensity': float(stack.min()),
            'max_intensity': float(stack.max()),
            'photons_per_frame': [float(stack[i].sum()) for i in range(stack.shape[0])]
        }
        
        return stats