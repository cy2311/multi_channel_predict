"""后处理模块
负责相机模型和噪声添加
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple
import tifffile as tiff
from tqdm import tqdm


class PostProcessor:
    """后处理器 - 应用相机模型"""
    
    def __init__(self, config):
        self.config = config
        self.camera_params = config.get_camera_params()
        self.rng = np.random.default_rng(config.get_simulation_params()['seed'])
        
        print(f"后处理器初始化完成")
        print(f"相机参数: {self.camera_params}")
    
    def apply_camera_model(self, photon_stack: np.ndarray) -> np.ndarray:
        """应用完整的相机模型
        
        Args:
            photon_stack: (T, H, W) 理想光子图像堆栈
            
        Returns:
            camera_stack: (T, H, W) 相机输出图像堆栈
        """
        print(f"应用相机模型到{photon_stack.shape[0]}帧图像")
        
        frames_out = []
        
        for idx in tqdm(range(photon_stack.shape[0]), desc="处理帧"):
            photons = photon_stack[idx].astype(np.float64)
            camera_frame = self._process_single_frame(photons)
            frames_out.append(camera_frame)
        
        camera_stack = np.stack(frames_out, axis=0)
        print(f"相机模型处理完成，输出形状: {camera_stack.shape}, 数据类型: {camera_stack.dtype}")
        
        return camera_stack
    
    def _process_single_frame(self, photons: np.ndarray) -> np.ndarray:
        """处理单帧图像
        
        Args:
            photons: (H, W) 光子图像
            
        Returns:
            camera_frame: (H, W) 相机输出图像
        """
        # 1. 量子效率 - 光子转换为电子
        qe = self.camera_params.get('QE', self.camera_params.get('qe', 0.9))
        electrons = self.rng.poisson(photons * qe)
        
        # 2. EM增益 (如果适用)
        emgain = self.camera_params.get('EMGain', self.camera_params.get('em_gain', 30.0))
        if emgain > 1.0:
            # 对每个电子应用伽马分布的EM增益
            flat_electrons = electrons.ravel()
            # 使用伽马分布模拟EM增益的随机性
            gained_electrons = self.rng.gamma(shape=flat_electrons, scale=emgain)
            em_electrons = gained_electrons.reshape(electrons.shape)
        else:
            em_electrons = electrons.astype(np.float64)
        
        # 3. 读取噪声 (高斯噪声)
        read_sigma = self.camera_params.get('read_noise_e', self.camera_params.get('read_noise', 1.0))
        read_noise = self.rng.normal(0.0, read_sigma, em_electrons.shape)
        electrons_with_noise = em_electrons + read_noise
        
        # 4. 基线偏移
        baseline = self.camera_params.get('offset', 0.0)
        
        # 5. A/D转换
        e_per_adu = self.camera_params.get('A2D', 1.0)  # 每ADU的电子数
        adu_values = electrons_with_noise / e_per_adu + baseline
        
        # 6. 量化和限制到ADU范围
        adu_quantized = np.round(adu_values)
        max_adu = self.camera_params.get('max_adu', 65535)
        adu_clipped = np.clip(adu_quantized, 0, max_adu).astype(np.uint16)
        
        return adu_clipped
    
    def add_background(self, stack: np.ndarray, background_level: float = None) -> np.ndarray:
        """添加背景噪声
        
        Args:
            stack: (T, H, W) 图像堆栈
            background_level: 背景光子水平，如果为None则自动估计
            
        Returns:
            stack_with_bg: 添加背景后的图像堆栈
        """
        if background_level is None:
            # 自动估计背景水平 (图像强度的1%)
            background_level = float(stack.mean() * 0.01)
        
        print(f"添加背景噪声，水平: {background_level:.2f} 光子/像素")
        
        # 为每个像素添加泊松背景噪声
        background_stack = self.rng.poisson(background_level, size=stack.shape).astype(stack.dtype)
        
        if stack.dtype == np.uint16:
            # 确保不会溢出
            stack_with_bg = np.clip(stack.astype(np.uint32) + background_stack.astype(np.uint32), 
                                  0, 65535).astype(np.uint16)
        else:
            stack_with_bg = stack + background_stack
        
        return stack_with_bg
    
    def analyze_noise_characteristics(self, photon_stack: np.ndarray, 
                                    camera_stack: np.ndarray) -> Dict[str, Any]:
        """分析噪声特性
        
        Args:
            photon_stack: 理想光子图像
            camera_stack: 相机输出图像
            
        Returns:
            noise_stats: 噪声统计信息
        """
        # 转换为相同的数据类型进行比较
        photon_float = photon_stack.astype(np.float64)
        camera_float = camera_stack.astype(np.float64)
        
        # 计算噪声
        noise = camera_float - photon_float
        
        # 计算SNR
        signal_power = np.mean(photon_float ** 2)
        noise_power = np.mean(noise ** 2)
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf
        
        stats = {
            'photon_mean': float(photon_stack.mean()),
            'photon_std': float(photon_stack.std()),
            'camera_mean': float(camera_stack.mean()),
            'camera_std': float(camera_stack.std()),
            'noise_mean': float(noise.mean()),
            'noise_std': float(noise.std()),
            'snr_db': float(snr_db),
            'dynamic_range': float(camera_stack.max() - camera_stack.min()),
            'camera_params': self.camera_params.copy()
        }
        
        return stats
    
    def visualize_noise_analysis(self, photon_stack: np.ndarray, 
                               camera_stack: np.ndarray, 
                               output_dir: Path, prefix: str = ""):
        """可视化噪声分析
        
        Args:
            photon_stack: 理想光子图像
            camera_stack: 相机输出图像
            output_dir: 输出目录
            prefix: 文件名前缀
        """
        # 选择中间帧进行分析
        mid_frame = photon_stack.shape[0] // 2
        photon_frame = photon_stack[mid_frame].astype(np.float64)
        camera_frame = camera_stack[mid_frame].astype(np.float64)
        noise_frame = camera_frame - photon_frame
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 第一行：图像比较
        # 理想光子图像
        im1 = axes[0, 0].imshow(photon_frame, cmap='gray')
        axes[0, 0].set_title('理想光子图像')
        axes[0, 0].set_xlabel('像素')
        axes[0, 0].set_ylabel('像素')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 相机输出图像
        im2 = axes[0, 1].imshow(camera_frame, cmap='gray')
        axes[0, 1].set_title('相机输出图像')
        axes[0, 1].set_xlabel('像素')
        axes[0, 1].set_ylabel('像素')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 噪声图像
        im3 = axes[0, 2].imshow(noise_frame, cmap='RdBu_r', 
                               vmin=-3*noise_frame.std(), vmax=3*noise_frame.std())
        axes[0, 2].set_title('噪声 (相机 - 理想)')
        axes[0, 2].set_xlabel('像素')
        axes[0, 2].set_ylabel('像素')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # 第二行：统计分析
        # 强度直方图比较
        axes[1, 0].hist(photon_frame.ravel(), bins=50, alpha=0.7, label='理想光子', 
                       density=True, color='blue')
        axes[1, 0].hist(camera_frame.ravel(), bins=50, alpha=0.7, label='相机输出', 
                       density=True, color='red')
        axes[1, 0].set_xlabel('强度')
        axes[1, 0].set_ylabel('概率密度')
        axes[1, 0].set_title('强度分布比较')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 噪声直方图
        axes[1, 1].hist(noise_frame.ravel(), bins=50, alpha=0.7, color='green', density=True)
        axes[1, 1].set_xlabel('噪声值')
        axes[1, 1].set_ylabel('概率密度')
        axes[1, 1].set_title(f'噪声分布\n(均值: {noise_frame.mean():.2f}, 标准差: {noise_frame.std():.2f})')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 信噪比分析
        # 计算局部SNR
        window_size = 32
        h, w = photon_frame.shape
        snr_map = np.zeros((h//window_size, w//window_size))
        
        for i in range(0, h-window_size, window_size):
            for j in range(0, w-window_size, window_size):
                signal_patch = photon_frame[i:i+window_size, j:j+window_size]
                noise_patch = noise_frame[i:i+window_size, j:j+window_size]
                
                signal_power = np.mean(signal_patch ** 2)
                noise_power = np.mean(noise_patch ** 2)
                
                if noise_power > 0:
                    snr_map[i//window_size, j//window_size] = 10 * np.log10(signal_power / noise_power)
                else:
                    snr_map[i//window_size, j//window_size] = 50  # 高SNR上限
        
        im4 = axes[1, 2].imshow(snr_map, cmap='viridis')
        axes[1, 2].set_title('局部信噪比 (dB)')
        axes[1, 2].set_xlabel('窗口位置')
        axes[1, 2].set_ylabel('窗口位置')
        plt.colorbar(im4, ax=axes[1, 2])
        
        plt.tight_layout()
        output_file = output_dir / f"{prefix}noise_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"噪声分析图保存到: {output_file}")
    
    def save_camera_stack(self, stack: np.ndarray, output_file: Path, 
                         metadata: Dict[str, Any] = None):
        """保存相机输出堆栈
        
        Args:
            stack: (T, H, W) 相机输出图像堆栈
            output_file: 输出文件路径
            metadata: 元数据
        """
        # 准备元数据
        camera_meta = {
            'Axes': 'TYX',
            'CameraModel': 'Simulated_EMCCD',
            'QE': self.camera_params.get('QE', 0.9),
            'EMGain': self.camera_params.get('EMGain', 30.0),
            'ReadNoise_e': self.camera_params.get('read_noise_e', 1.0),
            'Baseline': self.camera_params.get('offset', 0.0),
            'A2D': self.camera_params.get('A2D', 1.0),
            'MaxADU': self.camera_params.get('max_adu', 65535)
        }
        
        if metadata:
            camera_meta.update(metadata)
        
        # 确保输出目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存TIFF文件
        tiff.imwrite(output_file, stack, photometric='minisblack', metadata=camera_meta)
        print(f"相机输出堆栈保存到: {output_file}")
    
    def compare_stacks(self, photon_stack: np.ndarray, camera_stack: np.ndarray, 
                      output_dir: Path, prefix: str = ""):
        """比较理想和相机输出堆栈
        
        Args:
            photon_stack: 理想光子堆栈
            camera_stack: 相机输出堆栈
            output_dir: 输出目录
            prefix: 文件名前缀
        """
        # 计算统计信息
        stats = self.analyze_noise_characteristics(photon_stack, camera_stack)
        
        # 创建比较图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 每帧总强度比较
        photon_totals = [photon_stack[i].sum() for i in range(photon_stack.shape[0])]
        camera_totals = [camera_stack[i].sum() for i in range(camera_stack.shape[0])]
        
        axes[0, 0].plot(photon_totals, 'b-', label='理想光子', linewidth=2)
        axes[0, 0].plot(camera_totals, 'r-', label='相机输出', linewidth=2)
        axes[0, 0].set_xlabel('帧')
        axes[0, 0].set_ylabel('总强度')
        axes[0, 0].set_title('每帧总强度比较')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 强度分布比较
        axes[0, 1].hist(photon_stack.ravel(), bins=50, alpha=0.7, label='理想光子', 
                       density=True, color='blue')
        axes[0, 1].hist(camera_stack.ravel(), bins=50, alpha=0.7, label='相机输出', 
                       density=True, color='red')
        axes[0, 1].set_xlabel('强度')
        axes[0, 1].set_ylabel('概率密度')
        axes[0, 1].set_title('全局强度分布比较')
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # SNR随帧变化
        frame_snrs = []
        for i in range(photon_stack.shape[0]):
            signal_power = np.mean(photon_stack[i].astype(np.float64) ** 2)
            noise = camera_stack[i].astype(np.float64) - photon_stack[i].astype(np.float64)
            noise_power = np.mean(noise ** 2)
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
            else:
                snr = 50
            frame_snrs.append(snr)
        
        axes[1, 0].plot(frame_snrs, 'g-', linewidth=2, marker='o')
        axes[1, 0].set_xlabel('帧')
        axes[1, 0].set_ylabel('SNR (dB)')
        axes[1, 0].set_title('每帧信噪比')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 统计信息文本
        stats_text = f"""统计信息:
理想光子 - 均值: {stats['photon_mean']:.1f}, 标准差: {stats['photon_std']:.1f}
相机输出 - 均值: {stats['camera_mean']:.1f}, 标准差: {stats['camera_std']:.1f}
噪声 - 均值: {stats['noise_mean']:.2f}, 标准差: {stats['noise_std']:.2f}
平均SNR: {stats['snr_db']:.1f} dB
动态范围: {stats['dynamic_range']:.0f} ADU

相机参数:
QE: {stats['camera_params'].get('QE', 'N/A')}
EM增益: {stats['camera_params'].get('EMGain', 'N/A')}
读取噪声: {stats['camera_params'].get('read_noise_e', 'N/A')} e⁻"""
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        axes[1, 1].set_title('统计摘要')
        
        plt.tight_layout()
        output_file = output_dir / f"{prefix}stack_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"堆栈比较图保存到: {output_file}")
        
        return stats