"""Zmap处理模块
直接从phase_retrieval_tiff2h5的result.h5文件中提取Zernike系数
"""

import h5py
import numpy as np
import scipy.interpolate as interp
import scipy.ndimage as ndi
from pathlib import Path
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
from matplotlib import cm


class ZmapProcessor:
    """Zmap数据处理器"""
    
    def __init__(self, config):
        """初始化Zmap处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        paths = config.get_paths()
        self.zmap_file = paths['zmap_file']
        self.phase_maps = None
        self.coords = None
        self.coeff_mag = None
        self.metadata = None
        
        if not self.zmap_file.exists():
            raise FileNotFoundError(f"Zmap文件不存在: {self.zmap_file}")
        
        print(f"Zmap处理器初始化完成: {self.zmap_file}")
    
    def load_zmap_data(self):
        """加载Zmap数据"""
        print(f"加载Zmap数据: {self.zmap_file}")
        
        with h5py.File(self.zmap_file, 'r') as f:
            # 检查文件结构
            print("HDF5文件结构:")
            self._print_h5_structure(f)
            
            # 加载相位图数据
            if 'z_maps/phase' in f:
                self.phase_maps = np.array(f['z_maps/phase'])
                print(f"相位图形状: {self.phase_maps.shape}")
            else:
                raise KeyError("未找到z_maps/phase数据集")
            
            # 加载坐标数据
            if 'coords' in f:
                self.coords = np.array(f['coords'], dtype=float)
                print(f"坐标形状: {self.coords.shape}")
            else:
                raise KeyError("未找到coords数据集")
            
            # 加载幅度系数
            if 'zernike/coeff_mag' in f:
                self.coeff_mag = np.array(f['zernike/coeff_mag'])
                print(f"幅度系数形状: {self.coeff_mag.shape}")
            else:
                raise KeyError("未找到zernike/coeff_mag数据集")
            
            # 加载元数据
            self.metadata = {}
            for key in f.attrs.keys():
                self.metadata[key] = f.attrs[key]
        
        # 返回数据字典
        return {
            'phase_maps': self.phase_maps,
            'coordinates': {
                'coords': self.coords,
                'x_range': [self.coords[:, 0].min(), self.coords[:, 0].max()],
                'y_range': [self.coords[:, 1].min(), self.coords[:, 1].max()]
            },
            'coefficients': self.coeff_mag
        }
    
    def _print_h5_structure(self, h5_group, prefix=""):
        """打印HDF5文件结构"""
        for key in h5_group.keys():
            item = h5_group[key]
            if isinstance(item, h5py.Group):
                print(f"{prefix}{key}/")
                self._print_h5_structure(item, prefix + "  ")
            else:
                print(f"{prefix}{key}: {item.shape} {item.dtype}")
    
    def compute_emitter_coefficients(self, emitter_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算发射器位置的Zernike系数
        
        Args:
            emitter_positions: (N, 2) 发射器xy坐标数组
            
        Returns:
            phase_coeffs: (N, n_coeff) 相位系数
            mag_coeffs: (N, n_coeff) 幅度系数
        """
        print(f"为{len(emitter_positions)}个发射器计算Zernike系数")
        
        # 计算相位系数 - 使用双三次插值
        phase_coeffs = self._compute_phase_coeffs(emitter_positions)
        
        # 计算幅度系数 - 使用griddata插值
        mag_coeffs = self._compute_mag_coeffs(emitter_positions)
        
        return phase_coeffs, mag_coeffs
    
    def _compute_phase_coeffs(self, emitter_positions: np.ndarray) -> np.ndarray:
        """计算相位系数"""
        n_coeff = self.phase_maps.shape[0]
        n_emitters = len(emitter_positions)
        phase_coeffs = np.zeros((n_emitters, n_coeff), dtype=np.float32)
        
        # 限制坐标到有效范围
        x = np.clip(emitter_positions[:, 0], 0, self.phase_maps.shape[2] - 1e-3)
        y = np.clip(emitter_positions[:, 1], 0, self.phase_maps.shape[1] - 1e-3)
        
        for idx in range(n_coeff):
            phase_coeffs[:, idx] = ndi.map_coordinates(
                self.phase_maps[idx], [y, x], order=3, mode='nearest'
            )
        
        return phase_coeffs
    
    def _compute_mag_coeffs(self, emitter_positions: np.ndarray) -> np.ndarray:
        """计算幅度系数"""
        n_coeff = self.coeff_mag.shape[1]
        n_emitters = len(emitter_positions)
        mag_coeffs = np.zeros((n_emitters, n_coeff), dtype=np.float32)
        
        for idx in range(n_coeff):
            # 使用griddata进行三次插值
            vals = interp.griddata(
                self.coords, self.coeff_mag[:, idx], 
                emitter_positions, method='cubic'
            )
            
            # 对于超出凸包的点使用最近邻插值
            mask = np.isnan(vals)
            if np.any(mask):
                vals[mask] = interp.griddata(
                    self.coords, self.coeff_mag[:, idx], 
                    emitter_positions[mask], method='nearest'
                )
            
            mag_coeffs[:, idx] = vals.astype(np.float32)
        
        return mag_coeffs
    
    def visualize_coefficients(self, phase_coeffs: np.ndarray, mag_coeffs: np.ndarray, 
                             output_dir: Path, num_plot: int = 10, prefix: str = ""):
        """可视化Zernike系数"""
        n_emitters, n_coeff = phase_coeffs.shape
        chosen = np.random.choice(n_emitters, size=min(num_plot, n_emitters), replace=False)
        coeff_idx = np.arange(n_coeff)
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        for i, eid in enumerate(chosen):
            color = cm.tab20(i % 20)
            axes[0].plot(coeff_idx, phase_coeffs[eid], color=color, linewidth=1.5, alpha=0.8)
            axes[1].plot(coeff_idx, mag_coeffs[eid], color=color, linewidth=1.5, alpha=0.8)
        
        axes[0].set_ylabel('相位系数', fontsize=12)
        axes[1].set_ylabel('幅度系数', fontsize=12)
        axes[1].set_xlabel('Zernike阶数索引', fontsize=12)
        axes[0].set_title(f'随机{len(chosen)}个发射器的相位系数', fontsize=14)
        axes[1].set_title(f'随机{len(chosen)}个发射器的幅度系数', fontsize=14)
        
        axes[0].grid(True, alpha=0.3)
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = output_dir / f"{prefix}zernike_coefficients.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Zernike系数可视化保存到: {output_file}")
    
    def get_data_info(self) -> Dict[str, Any]:
        """获取数据信息"""
        return {
            'phase_maps_shape': self.phase_maps.shape,
            'coords_shape': self.coords.shape,
            'coeff_mag_shape': self.coeff_mag.shape,
            'n_coefficients': self.phase_maps.shape[0],
            'field_size': (self.phase_maps.shape[1], self.phase_maps.shape[2]),
            'n_measurement_points': len(self.coords),
            'metadata': self.metadata
        }
    
    def save_interpolated_maps(self, output_dir: Path, grid_size: int = 512):
        """保存插值后的相位图"""
        print(f"保存插值相位图到: {output_dir}")
        
        # 创建规则网格
        x_min, x_max = self.coords[:, 0].min(), self.coords[:, 0].max()
        y_min, y_max = self.coords[:, 1].min(), self.coords[:, 1].max()
        
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])
        
        # 为前几个系数创建插值图
        n_plot = min(6, self.coeff_mag.shape[1])
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i in range(n_plot):
            # 插值到规则网格
            interp_vals = interp.griddata(
                self.coords, self.coeff_mag[:, i], 
                grid_points, method='cubic', fill_value=0
            )
            interp_map = interp_vals.reshape(grid_size, grid_size)
            
            # 绘制
            im = axes[i].imshow(interp_map, extent=[x_min, x_max, y_min, y_max], 
                              origin='lower', cmap='RdBu_r')
            axes[i].set_title(f'Zernike系数 {i+1}')
            axes[i].set_xlabel('X (像素)')
            axes[i].set_ylabel('Y (像素)')
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        output_file = output_dir / "interpolated_zernike_maps.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"插值相位图保存到: {output_file}")