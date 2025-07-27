#!/usr/bin/env python3
"""
内存优化的TIFF生成器

针对大批量生成进行内存优化：
1. 流式处理，避免一次性加载所有帧
2. 及时释放内存
3. 可配置的内存使用策略
4. 支持大文件的分块处理
"""

import os
import gc
import json
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Iterator

import h5py
import numpy as np
import tifffile as tiff
from scipy.fft import ifft2, ifftshift, fftshift
from skimage.transform import resize
from tqdm import tqdm

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入原始函数
from tiff_generator import (
    load_config, load_zernike_basis, build_pupil_mask,
    construct_pupil, apply_defocus, generate_psf, add_camera_noise
)


class MemoryOptimizedTiffGenerator:
    """内存优化的TIFF生成器"""
    
    def __init__(self, h5_path: str, output_path: str, config: Optional[Dict[str, Any]] = None):
        """初始化生成器
        
        Parameters
        ----------
        h5_path : str
            输入的HDF5文件路径
        output_path : str
            输出TIFF文件路径
        config : dict, optional
            配置参数
        """
        self.h5_path = h5_path
        self.output_path = output_path
        self.config = config or {}
        
        # 内存优化配置
        self.memory_config = self.config.get('memory_optimization', {})
        self.chunk_size = self.memory_config.get('chunk_size', 10)  # 每次处理的帧数
        self.enable_gc = self.memory_config.get('enable_gc', True)  # 是否启用垃圾回收
        self.gc_frequency = self.memory_config.get('gc_frequency', 5)  # 垃圾回收频率
        
        # 加载光学参数
        self.wavelength_nm, self.pix_x_nm, self.pix_y_nm, self.NA = load_config()
        self.wavelength_m = self.wavelength_nm * 1e-9
        self.pix_x_m = self.pix_x_nm * 1e-9
        self.pix_y_m = self.pix_y_nm * 1e-9
        
        # 加载Zernike基函数和瞳孔掩码
        self.basis = load_zernike_basis()
        N = self.basis.shape[1]
        self.pupil_mask = build_pupil_mask(N, self.pix_x_m, self.pix_y_m, self.NA, self.wavelength_m)
        
        # TIFF配置
        tiff_config = self.config.get('tiff', {})
        self.roi_size = tiff_config.get('roi_size', 1200)
        self.use_direct_rendering = tiff_config.get('use_direct_rendering', True)
        self.add_noise = tiff_config.get('add_noise', True)
        self.noise_params = tiff_config.get('noise_params', {
            'background': 100,
            'readout_noise': 10,
            'shot_noise': True
        })
        
        print(f"内存优化TIFF生成器初始化完成")
        print(f"块大小: {self.chunk_size} 帧")
        print(f"垃圾回收: {'启用' if self.enable_gc else '禁用'}")
    
    def load_emitters_data(self) -> Dict[str, np.ndarray]:
        """加载发射器数据"""
        print(f"从 {self.h5_path} 加载发射器数据...")
        
        with h5py.File(self.h5_path, 'r') as f:
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
        
        return {
            'frame_ix': frame_ix,
            'ids_rec': ids_rec,
            'xyz_rec': xyz_rec,
            'phot_rec': phot_rec,
            'coeff_mag_all': coeff_mag_all,
            'coeff_phase_all': coeff_phase_all
        }
    
    def get_frame_chunks(self, unique_frames: np.ndarray) -> Iterator[np.ndarray]:
        """生成帧块迭代器
        
        Parameters
        ----------
        unique_frames : ndarray
            所有唯一帧索引
            
        Yields
        ------
        chunk : ndarray
            帧块
        """
        for i in range(0, len(unique_frames), self.chunk_size):
            yield unique_frames[i:i + self.chunk_size]
    
    def simulate_frame_optimized(self, frame_idx: int, emitters_data: Dict[str, np.ndarray]) -> np.ndarray:
        """内存优化的单帧模拟
        
        Parameters
        ----------
        frame_idx : int
            帧索引
        emitters_data : dict
            发射器数据字典
            
        Returns
        -------
        frame : ndarray
            模拟的帧图像
        """
        # 获取该帧的活跃发射器
        frame_mask = emitters_data['frame_ix'] == frame_idx
        if not np.any(frame_mask):
            # 没有活跃发射器，返回空白帧
            frame = np.zeros((self.roi_size, self.roi_size), dtype=np.float32)
            if self.add_noise:
                frame = add_camera_noise(frame, **self.noise_params)
            return frame
        
        # 获取该帧的发射器数据
        active_ids = emitters_data['ids_rec'][frame_mask]
        active_xyz = emitters_data['xyz_rec'][frame_mask]
        active_phot = emitters_data['phot_rec'][frame_mask]
        
        # 获取对应的Zernike系数
        coeff_mag = emitters_data['coeff_mag_all'][active_ids]
        coeff_phase = emitters_data['coeff_phase_all'][active_ids]
        
        # 创建画布
        canvas = np.zeros((self.roi_size, self.roi_size), dtype=np.float32)
        half_psf = self.basis.shape[1] // 2
        
        # 为每个活跃发射器生成PSF并添加到画布
        for i in range(len(active_ids)):
            # 构建瞳函数
            pupil = construct_pupil(coeff_mag[i], coeff_phase[i], self.basis, self.pupil_mask)
            
            # 应用离焦
            z_nm = active_xyz[i, 2] * 1000  # 转换为纳米
            pupil_defocus = apply_defocus(pupil, z_nm, self.wavelength_m, self.pix_x_m, self.pix_y_m)
            
            # 生成PSF
            psf = generate_psf(pupil_defocus)
            
            # 缩放PSF强度
            psf_scaled = psf * active_phot[i]
            
            # 计算在目标画布上的亚像素位置
            cx_float = active_xyz[i, 0]
            cy_float = active_xyz[i, 1]
            
            # 计算整数像素位置和亚像素偏移
            cx_int = int(np.floor(cx_float))
            cy_int = int(np.floor(cy_float))
            dx = cx_float - cx_int
            dy = cy_float - cy_int
            
            # 使用双线性插值处理亚像素位置
            w00 = (1 - dx) * (1 - dy)
            w01 = (1 - dx) * dy
            w10 = dx * (1 - dy)
            w11 = dx * dy
            
            # 为四个相邻位置添加PSF
            positions = [
                (cy_int, cx_int, w00),
                (cy_int + 1, cx_int, w01),
                (cy_int, cx_int + 1, w10),
                (cy_int + 1, cx_int + 1, w11)
            ]
            
            for py, px, weight in positions:
                if weight < 1e-6:
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
                if r1 > self.roi_size:
                    r1 = self.roi_size
                if c1 > self.roi_size:
                    c1 = self.roi_size
                
                psf_r1 = psf_r0 + (r1 - r0)
                psf_c1 = psf_c0 + (c1 - c0)
                
                # 添加加权PSF到画布
                if r1 > r0 and c1 > c0:
                    canvas[r0:r1, c0:c1] += weight * psf_scaled[psf_r0:psf_r1, psf_c0:psf_c1]
            
            # 及时清理临时变量
            del pupil, pupil_defocus, psf, psf_scaled
        
        # 添加噪声
        if self.add_noise:
            canvas = add_camera_noise(canvas, **self.noise_params)
        
        return canvas.astype(np.float32)
    
    def generate_tiff_streaming(self) -> None:
        """流式生成TIFF文件"""
        print(f"开始流式生成TIFF文件: {self.output_path}")
        
        # 加载发射器数据
        emitters_data = self.load_emitters_data()
        
        # 获取所有帧
        unique_frames = np.unique(emitters_data['frame_ix'])
        num_frames = len(unique_frames)
        
        print(f"生成 {num_frames} 帧图像，大小: {self.roi_size}x{self.roi_size}")
        print(f"使用流式处理，块大小: {self.chunk_size} 帧")
        
        # 创建OME-XML元数据
        metadata = {
            'axes': 'TYX',
            'PhysicalSizeX': self.pix_x_nm / 1000,  # 转换为微米
            'PhysicalSizeY': self.pix_y_nm / 1000,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeYUnit': 'µm'
        }
        
        # 使用tifffile的写入器进行流式写入
        with tiff.TiffWriter(self.output_path, bigtiff=True) as writer:
            frame_count = 0
            
            # 分块处理帧
            for chunk_idx, frame_chunk in enumerate(self.get_frame_chunks(unique_frames)):
                print(f"\n处理块 {chunk_idx + 1}, 帧 {frame_chunk[0]}-{frame_chunk[-1]}")
                
                # 生成当前块的所有帧
                chunk_frames = []
                for frame_idx in tqdm(frame_chunk, desc=f"生成块 {chunk_idx + 1}"):
                    frame = self.simulate_frame_optimized(frame_idx, emitters_data)
                    chunk_frames.append(frame)
                    frame_count += 1
                    
                    # 定期垃圾回收
                    if self.enable_gc and frame_count % self.gc_frequency == 0:
                        gc.collect()
                
                # 将块写入TIFF文件
                chunk_array = np.stack(chunk_frames, axis=0)
                
                if chunk_idx == 0:
                    # 第一块，创建文件并写入元数据
                    writer.write(
                        chunk_array,
                        metadata=metadata,
                        compression='lzw'
                    )
                else:
                    # 后续块，追加写入
                    for frame in chunk_array:
                        writer.write(
                            frame[np.newaxis, ...],  # 添加时间维度
                            compression='lzw'
                        )
                
                # 清理内存
                del chunk_frames, chunk_array
                if self.enable_gc:
                    gc.collect()
                
                print(f"块 {chunk_idx + 1} 完成，已处理 {frame_count}/{num_frames} 帧")
        
        print(f"\nTIFF文件生成完成: {self.output_path}")
        print(f"总帧数: {frame_count}")
        
        # 最终垃圾回收
        if self.enable_gc:
            gc.collect()
    
    def generate_tiff_batch(self) -> None:
        """批量生成TIFF文件（传统方法，用于比较）"""
        print(f"开始批量生成TIFF文件: {self.output_path}")
        
        # 加载发射器数据
        emitters_data = self.load_emitters_data()
        
        # 获取所有帧
        unique_frames = np.unique(emitters_data['frame_ix'])
        num_frames = len(unique_frames)
        
        print(f"生成 {num_frames} 帧图像，大小: {self.roi_size}x{self.roi_size}")
        
        # 生成所有帧
        frames = []
        for frame_idx in tqdm(unique_frames, desc="生成帧"):
            frame = self.simulate_frame_optimized(frame_idx, emitters_data)
            frames.append(frame)
        
        # 转换为numpy数组
        frames_array = np.stack(frames, axis=0)
        
        print(f"最终图像堆栈形状: {frames_array.shape}")
        print(f"像素值范围: [{frames_array.min():.2f}, {frames_array.max():.2f}]")
        
        # 创建OME-XML元数据
        metadata = {
            'axes': 'TYX',
            'PhysicalSizeX': self.pix_x_nm / 1000,
            'PhysicalSizeY': self.pix_y_nm / 1000,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeYUnit': 'µm'
        }
        
        # 保存为OME-TIFF
        tiff.imwrite(
            self.output_path,
            frames_array,
            metadata=metadata,
            compression='lzw'
        )
        
        print(f"TIFF文件已保存: {self.output_path}")


def generate_tiff_memory_optimized(h5_path: str, output_path: str, 
                                  config: Optional[Dict[str, Any]] = None,
                                  use_streaming: bool = True) -> None:
    """内存优化的TIFF生成函数
    
    Parameters
    ----------
    h5_path : str
        输入的HDF5文件路径
    output_path : str
        输出TIFF文件路径
    config : dict, optional
        配置参数
    use_streaming : bool
        是否使用流式处理
    """
    generator = MemoryOptimizedTiffGenerator(h5_path, output_path, config)
    
    if use_streaming:
        generator.generate_tiff_streaming()
    else:
        generator.generate_tiff_batch()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='内存优化的TIFF图像生成器')
    parser.add_argument('--h5', type=str, required=True, help='输入HDF5文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出TIFF文件路径')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--no_streaming', action='store_true', help='不使用流式处理')
    parser.add_argument('--chunk_size', type=int, default=10, help='流式处理的块大小')
    
    args = parser.parse_args()
    
    config = None
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # 添加内存优化配置
    if config is None:
        config = {}
    
    if 'memory_optimization' not in config:
        config['memory_optimization'] = {}
    
    config['memory_optimization']['chunk_size'] = args.chunk_size
    
    generate_tiff_memory_optimized(
        args.h5, 
        args.output, 
        config, 
        use_streaming=not args.no_streaming
    )