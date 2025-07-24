"""发射器管理模块
整合发射器生成、闪烁时间管理和Zernike系数分配
"""

import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from typing import Dict, List, Tuple, Any
import random
from tqdm import tqdm


class EmitterManager:
    """发射器管理器"""
    
    def __init__(self, config):
        """初始化发射器管理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.sim_params = config.get_simulation_params()
        self.rng = np.random.default_rng(self.sim_params['seed'])
        
        # 发射器数据
        self.emitter_attrs = None
        self.frame_records = None
        self.zernike_coeffs = None
        
        print(f"发射器管理器初始化完成")
        print(f"模拟参数: {self.sim_params}")
    
    def generate_emitters(self, num_emitters: int = None, **kwargs) -> Dict[str, torch.Tensor]:
        """生成发射器属性
        
        Returns:
            dict with keys: xyz, intensity, t0, on_time, id
        """
        # 更新参数
        params = self.sim_params.copy()
        if num_emitters is not None:
            params['num_emitters'] = num_emitters
        params.update(kwargs)
        
        print(f"生成{params['num_emitters']}个发射器")
        
        # 空间位置: x,y在像素单位内，z在微米单位内
        image_size = params['image_size']
        xy = self.rng.uniform(0, [image_size[1], image_size[0]], size=(params['num_emitters'], 2))
        z_range = params.get('z_range_um', 1.0)  # 默认1微米范围
        z = self.rng.uniform(-z_range, z_range, size=(params['num_emitters'], 1))
        xyz = torch.tensor(np.concatenate([xy, z], axis=1), dtype=torch.float32)
        
        # 强度 (光子/帧) 从均匀分布采样，限制>=1e-8
        intensity_range = params.get('emitter_intensity_range', [1000, 5000])
        intensity = torch.tensor(
            self.rng.uniform(intensity_range[0], intensity_range[1], 
                           size=params['num_emitters']),
            dtype=torch.float32
        ).clamp_min(1e-8)
        
        # 首次开启时间t0，在帧范围两侧留出3倍生命周期的缓冲
        num_frames = params.get('num_frames', params.get('frames', 1000))
        lifetime_avg = params.get('lifetime_avg', 10.0)
        frame_range = (0, num_frames - 1)
        t0 = torch.tensor(
            self.rng.uniform(frame_range[0] - 3 * lifetime_avg,
                           frame_range[1] + 3 * lifetime_avg, 
                           size=params['num_emitters']), 
            dtype=torch.float32
        )
        
        # 开启持续时间从指数分布采样
        lifetime_avg = params.get('lifetime_avg', 10.0)
        on_time = torch.distributions.Exponential(1 / lifetime_avg).sample(
            (params['num_emitters'],)
        )
        
        # 发射器ID
        ids = torch.arange(params['num_emitters'], dtype=torch.long)
        
        self.emitter_attrs = {
            'xyz': xyz,
            'intensity': intensity,
            't0': t0,
            'on_time': on_time,
            'id': ids
        }
        
        return self.emitter_attrs
    
    def bin_emitters_to_frames(self, frame_range: Tuple[int, int] = None) -> Dict[str, torch.Tensor]:
        """将连续闪烁间隔转换为每帧记录
        
        Returns:
            records dict with xyz, phot, frame_ix, id tensors
        """
        if self.emitter_attrs is None:
            raise ValueError("请先生成发射器属性")
        
        if frame_range is None:
            num_frames = self.sim_params.get('num_frames', self.sim_params.get('frames', 1000))
            frame_range = (0, num_frames - 1)
        
        k_start, k_end = frame_range
        
        xyz_all = []
        phot_all = []
        frame_all = []
        id_all = []
        
        xyz = self.emitter_attrs['xyz']
        intensity = self.emitter_attrs['intensity']
        t0 = self.emitter_attrs['t0']
        on_time = self.emitter_attrs['on_time']
        ids = self.emitter_attrs['id']
        
        print(f"将发射器分配到帧{k_start}到{k_end}")
        
        for i in tqdm(range(len(ids)), desc="处理发射器"):
            te = float(t0[i] + on_time[i])
            for k in range(k_start, k_end + 1):
                dt = self._overlap(float(t0[i]), te, k)
                if dt > 0.0:
                    xyz_all.append(xyz[i])
                    phot_all.append(intensity[i] * dt)
                    frame_all.append(k)
                    id_all.append(ids[i])
        
        self.frame_records = {
            'xyz': torch.stack(xyz_all) if xyz_all else torch.zeros((0, 3), dtype=torch.float32),
            'phot': torch.tensor(phot_all, dtype=torch.float32),
            'frame_ix': torch.tensor(frame_all, dtype=torch.long),
            'id': torch.tensor(id_all, dtype=torch.long)
        }
        
        print(f"生成了{len(phot_all)}条发射器记录")
        return self.frame_records
    
    def _overlap(self, t0: float, te: float, k: int) -> float:
        """计算[t0, te)与整数帧[k, k+1)的重叠长度"""
        return max(0.0, min(te, k + 1) - max(t0, k))
    
    def assign_zernike_coefficients(self, zmap_processor):
        """分配Zernike系数"""
        if self.emitter_attrs is None:
            raise ValueError("请先生成发射器属性")
        
        print("为发射器分配Zernike系数")
        
        # 提取xy坐标
        emitter_xy = self.emitter_attrs['xyz'][:, :2].numpy()
        
        # 计算系数
        phase_coeffs, mag_coeffs = zmap_processor.compute_emitter_coefficients(emitter_xy)
        
        self.zernike_coeffs = {
            'phase': phase_coeffs,
            'mag': mag_coeffs
        }
        
        print(f"分配了{len(phase_coeffs)}个发射器的Zernike系数")
        return self.zernike_coeffs
    
    def visualize_emitters(self, output_dir: Path, num_plot: int = 20, prefix: str = ""):
        """可视化发射器分布和时间线"""
        if self.emitter_attrs is None:
            raise ValueError("请先生成发射器属性")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 空间分布图
        self._plot_spatial_distribution(output_dir, prefix)
        
        # 时间线图
        self._plot_timeline(output_dir, num_plot, prefix)
        
        # 生命周期统计
        self._plot_lifetime_statistics(output_dir, prefix)
        
        # 强度分布
        self._plot_intensity_distribution(output_dir, prefix)
        
        # 每帧发射器数量统计
        if self.frame_records is not None:
            self._plot_frame_statistics(output_dir, prefix)
    
    def _plot_spatial_distribution(self, output_dir: Path, prefix: str):
        """绘制空间分布"""
        xyz = self.emitter_attrs['xyz']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # XY分布
        axes[0].scatter(xyz[:, 0].numpy(), xyz[:, 1].numpy(), s=1, alpha=0.6, c='blue')
        axes[0].set_xlabel('X (像素)')
        axes[0].set_ylabel('Y (像素)')
        axes[0].set_title('发射器XY分布')
        axes[0].set_aspect('equal')
        axes[0].grid(True, alpha=0.3)
        
        # Z分布直方图
        axes[1].hist(xyz[:, 2].numpy(), bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1].set_xlabel('Z (μm)')
        axes[1].set_ylabel('数量')
        axes[1].set_title('发射器Z分布')
        axes[1].grid(True, alpha=0.3)
        
        # XZ投影
        axes[2].scatter(xyz[:, 0].numpy(), xyz[:, 2].numpy(), s=1, alpha=0.6, c='red')
        axes[2].set_xlabel('X (像素)')
        axes[2].set_ylabel('Z (μm)')
        axes[2].set_title('发射器XZ投影')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = output_dir / f"{prefix}emitter_spatial_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"空间分布图保存到: {output_file}")
    
    def _plot_timeline(self, output_dir: Path, num_plot: int, prefix: str):
        """绘制时间线"""
        total_emitters = len(self.emitter_attrs['id'])
        chosen = random.sample(range(total_emitters), k=min(num_plot, total_emitters))
        
        plt.figure(figsize=(12, 6))
        
        for idx, eid in enumerate(chosen):
            t0 = float(self.emitter_attrs['t0'][eid])
            te = float(self.emitter_attrs['t0'][eid] + self.emitter_attrs['on_time'][eid])
            color = cm.tab20(idx % 20)
            
            plt.hlines(y=idx, xmin=t0, xmax=te, color=color, linewidth=3, alpha=0.8)
            plt.scatter([t0, te], [idx, idx], color=color, s=30, zorder=5)
        
        plt.yticks([])
        plt.xlabel('帧', fontsize=12)
        plt.ylabel('发射器', fontsize=12)
        plt.title(f'{len(chosen)}个随机发射器的闪烁时间线', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = output_dir / f"{prefix}emitter_timeline.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"时间线图保存到: {output_file}")
    
    def _plot_lifetime_statistics(self, output_dir: Path, prefix: str):
        """绘制生命周期统计"""
        on_time = self.emitter_attrs['on_time'].numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 生命周期直方图
        axes[0].hist(on_time, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[0].axvline(on_time.mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'平均值: {on_time.mean():.2f}')
        axes[0].axvline(np.median(on_time), color='orange', linestyle='--', linewidth=2,
                       label=f'中位数: {np.median(on_time):.2f}')
        axes[0].set_xlabel('生命周期 (帧)')
        axes[0].set_ylabel('数量')
        axes[0].set_title('发射器生命周期分布')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 累积分布
        sorted_lifetime = np.sort(on_time)
        cumulative = np.arange(1, len(sorted_lifetime) + 1) / len(sorted_lifetime)
        axes[1].plot(sorted_lifetime, cumulative, linewidth=2, color='blue')
        axes[1].set_xlabel('生命周期 (帧)')
        axes[1].set_ylabel('累积概率')
        axes[1].set_title('生命周期累积分布函数')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = output_dir / f"{prefix}emitter_lifetime_statistics.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"生命周期统计图保存到: {output_file}")
    
    def _plot_intensity_distribution(self, output_dir: Path, prefix: str):
        """绘制强度分布"""
        intensity = self.emitter_attrs['intensity'].numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 强度直方图
        axes[0].hist(intensity, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[0].axvline(intensity.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'平均值: {intensity.mean():.1f}')
        axes[0].axvline(np.median(intensity), color='blue', linestyle='--', linewidth=2,
                       label=f'中位数: {np.median(intensity):.1f}')
        axes[0].set_xlabel('强度 (光子/帧)')
        axes[0].set_ylabel('数量')
        axes[0].set_title('发射器强度分布')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 对数尺度
        axes[1].hist(intensity, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1].set_xlabel('强度 (光子/帧)')
        axes[1].set_ylabel('数量')
        axes[1].set_title('发射器强度分布 (对数尺度)')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = output_dir / f"{prefix}emitter_intensity_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"强度分布图保存到: {output_file}")
    
    def _plot_frame_statistics(self, output_dir: Path, prefix: str):
        """绘制每帧统计"""
        frame_ix = self.frame_records['frame_ix'].numpy()
        phot = self.frame_records['phot'].numpy()
        
        unique_frames = np.unique(frame_ix)
        frame_counts = []
        frame_intensities = []
        
        for frame in unique_frames:
            mask = frame_ix == frame
            frame_counts.append(mask.sum())
            frame_intensities.append(phot[mask].sum())
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # 每帧发射器数量
        axes[0].bar(unique_frames, frame_counts, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_ylabel('发射器数量')
        axes[0].set_title('每帧活跃发射器数量')
        axes[0].grid(True, alpha=0.3)
        
        # 每帧总强度
        axes[1].bar(unique_frames, frame_intensities, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1].set_xlabel('帧')
        axes[1].set_ylabel('总强度 (光子)')
        axes[1].set_title('每帧总光子数')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = output_dir / f"{prefix}frame_statistics.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"帧统计图保存到: {output_file}")
    
    def save_to_h5(self, output_file: Path):
        """保存到HDF5文件"""
        if self.emitter_attrs is None:
            raise ValueError("请先生成发射器属性")
        
        print(f"保存发射器数据到: {output_file}")
        
        with h5py.File(output_file, 'w') as f:
            # 发射器属性
            grp_em = f.create_group('emitters')
            for key in ('xyz', 'intensity', 't0', 'on_time', 'id'):
                grp_em.create_dataset(key, data=self.emitter_attrs[key].cpu().numpy())
            
            # 帧记录
            if self.frame_records is not None:
                grp_rec = f.create_group('records')
                for key in ('xyz', 'phot', 'frame_ix', 'id'):
                    grp_rec.create_dataset(key, data=self.frame_records[key].cpu().numpy())
            
            # Zernike系数
            if self.zernike_coeffs is not None:
                grp_zern = f.create_group('zernike_coeffs')
                grp_zern.create_dataset('phase', data=self.zernike_coeffs['phase'])
                grp_zern.create_dataset('mag', data=self.zernike_coeffs['mag'])
            
            # 元数据
            f.attrs['num_emitters'] = len(self.emitter_attrs['id'])
            f.attrs['simulation_params'] = str(self.sim_params)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self.emitter_attrs is None:
            return {}
        
        stats = {
            'num_emitters': len(self.emitter_attrs['id']),
            'intensity_mean': float(self.emitter_attrs['intensity'].mean()),
            'intensity_std': float(self.emitter_attrs['intensity'].std()),
            'lifetime_mean': float(self.emitter_attrs['on_time'].mean()),
            'lifetime_std': float(self.emitter_attrs['on_time'].std()),
            'z_range': (float(self.emitter_attrs['xyz'][:, 2].min()), 
                       float(self.emitter_attrs['xyz'][:, 2].max()))
        }
        
        if self.frame_records is not None:
            frame_ix = self.frame_records['frame_ix'].numpy()
            stats.update({
                'num_records': len(frame_ix),
                'frames_with_emitters': len(np.unique(frame_ix)),
                'total_photons': float(self.frame_records['phot'].sum())
            })
        
        return stats