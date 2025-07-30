#!/usr/bin/env python3
"""
批量TIFF生成器 - 优化版本

支持一次生成多个TIFF文件，包含以下优化：
1. 批量配置管理
2. 并行处理支持
3. 内存优化
4. 进度监控和错误处理
5. 可恢复的批量处理

使用方法:
    python batch_tiff_generator.py --batch_config batch_config.json
"""

import argparse
import json
import sys
import os
import time
import copy
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import traceback

# 添加父目录到路径以便导入其他模块
sys.path.append(str(Path(__file__).parent.parent))

from trainset_simulation.generate_emitters import (
    sample_emitters, bin_emitters_to_frames, save_to_h5, visualise
)
from trainset_simulation.compute_zernike_coeffs import (
    load_data, compute_phase_coeffs, compute_mag_coeffs, save_coeffs, visualise_coeffs
)
from tiff_generator import generate_tiff_stack


class BatchTiffGenerator:
    """批量TIFF生成器类"""
    
    def __init__(self, batch_config_path: str):
        """初始化批量生成器
        
        Parameters
        ----------
        batch_config_path : str
            批量配置文件路径
        """
        self.batch_config_path = batch_config_path
        self.batch_config = self.load_batch_config()
        self.base_output_dir = Path(self.batch_config.get('base_output_dir', 'batch_output'))
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建状态文件用于恢复
        self.status_file = self.base_output_dir / 'batch_status.json'
        self.load_or_create_status()
        
    def load_batch_config(self) -> Dict[str, Any]:
        """加载批量配置文件"""
        with open(self.batch_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    
    def load_or_create_status(self):
        """加载或创建状态文件"""
        if self.status_file.exists():
            with open(self.status_file, 'r', encoding='utf-8') as f:
                self.status = json.load(f)
        else:
            self.status = {
                'completed_jobs': [],
                'failed_jobs': [],
                'start_time': None,
                'last_update': None
            }
    
    def save_status(self):
        """保存状态文件"""
        self.status['last_update'] = time.time()
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(self.status, f, indent=2, ensure_ascii=False)
    
    def generate_job_configs(self) -> List[Dict[str, Any]]:
        """生成所有作业配置"""
        jobs = []
        base_config = self.batch_config.get('base_config', {})
        
        # 检查是否使用新的样本配置模式
        if 'sample_configs' in self.batch_config:
            return self._generate_sample_jobs(base_config)
        
        # 兼容旧的变量配置模式
        variables = self.batch_config.get('variables', {})
        
        # 如果没有变量，只生成一个作业
        if not variables:
            job_config = base_config.copy()
            job_config['job_id'] = 'single_job'
            job_config['output_subdir'] = 'single_job'
            jobs.append(job_config)
            return jobs
        
        # 生成所有变量组合
        import itertools
        
        # 获取所有变量名和值
        var_names = list(variables.keys())
        var_values = [variables[name] for name in var_names]
        
        # 生成笛卡尔积
        for i, combination in enumerate(itertools.product(*var_values)):
            job_config = base_config.copy()
            
            # 应用变量值
            job_id_parts = []
            for var_name, var_value in zip(var_names, combination):
                # 使用点记法设置嵌套配置
                self._set_nested_config(job_config, var_name, var_value)
                job_id_parts.append(f"{var_name.split('.')[-1]}_{var_value}")
            
            job_config['job_id'] = f"job_{i:04d}_{'_'.join(job_id_parts)}"
            job_config['output_subdir'] = job_config['job_id']
            jobs.append(job_config)
        
        return jobs
    
    def _generate_sample_jobs(self, base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成基于样本的作业配置"""
        sample_configs = self.batch_config['sample_configs']
        num_samples = sample_configs.get('num_samples', 5)
        frames_per_sample = sample_configs.get('frames_per_sample', 200)
        sample_naming = sample_configs.get('sample_naming', 'sample_{sample_id:03d}')
        
        jobs = []
        for sample_id in range(1, num_samples + 1):
            # 创建样本配置
            job_config = copy.deepcopy(base_config)
            
            # 设置帧数
            if 'emitters' not in job_config:
                job_config['emitters'] = {}
            job_config['emitters']['frames'] = frames_per_sample
            
            # 设置输出文件名
            sample_name = sample_naming.format(sample_id=sample_id)
            if 'tiff' not in job_config:
                job_config['tiff'] = {}
            job_config['tiff']['filename'] = f'{sample_name}.ome.tiff'
            
            # 设置随机种子以确保不同样本有不同的发射器分布
            job_config['emitters']['seed'] = sample_id * 1000
            
            # 设置作业信息
            job_config['job_id'] = sample_name
            job_config['output_subdir'] = sample_name
            job_config['base_output_dir'] = str(self.base_output_dir)
            job_config['_config_file_path'] = str(self.batch_config_path)  # 添加配置文件路径用于相对路径解析
            
            jobs.append(job_config)
        
        return jobs
    
    def _set_nested_config(self, config: Dict, key_path: str, value: Any):
        """设置嵌套配置值
        
        Parameters
        ----------
        config : dict
            配置字典
        key_path : str
            点分隔的键路径，如 'emitters.num_emitters'
        value : Any
            要设置的值
        """
        keys = key_path.split('.')
        current = config
        
        # 导航到最后一级
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # 设置最终值
        current[keys[-1]] = value
    
    def process_single_job(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个作业
        
        Parameters
        ----------
        job_config : dict
            作业配置
            
        Returns
        -------
        result : dict
            处理结果
        """
        job_id = job_config['job_id']
        
        try:
            print(f"\n=== 开始处理作业: {job_id} ===")
            
            # 创建作业输出目录
            job_output_dir = self.base_output_dir / job_config['output_subdir']
            job_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 获取zmap路径并转换为绝对路径
            zmap_path = job_config.get('zmap_path')
            if not zmap_path:
                raise ValueError(f"配置中未指定zmap_path")
            
            # 将相对路径转换为绝对路径（相对于配置文件所在目录）
            if not Path(zmap_path).is_absolute():
                config_dir = Path(self.batch_config_path).parent
                zmap_path = str(config_dir / zmap_path)
            
            if not Path(zmap_path).exists():
                raise ValueError(f"找不到zmap文件: {zmap_path}")
            
            # 步骤1: 生成发射器数据
            emitters_path = self._step1_generate_emitters(job_config, job_output_dir)
            
            # 步骤2: 计算Zernike系数
            self._step2_compute_zernike_coeffs(zmap_path, emitters_path, job_config, job_output_dir)
            
            # 步骤3: 生成TIFF
            tiff_path = self._step3_generate_tiff(emitters_path, job_config, job_output_dir)
            
            result = {
                'job_id': job_id,
                'status': 'success',
                'output_dir': str(job_output_dir),
                'tiff_path': str(tiff_path),
                'emitters_path': str(emitters_path)
            }
            
            print(f"作业 {job_id} 完成成功")
            return result
            
        except Exception as e:
            error_msg = f"作业 {job_id} 失败: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            
            return {
                'job_id': job_id,
                'status': 'failed',
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
    
    def _step1_generate_emitters(self, config: Dict[str, Any], output_dir: Path) -> Path:
        """步骤1: 生成发射器数据"""
        print("=== 步骤 1: 生成发射器数据 ===")
        
        emitters_config = config.get('emitters', {})
        
        # 参数设置
        num_emitters = emitters_config.get('num_emitters', 1000)
        frames = emitters_config.get('frames', 10)
        area_px = emitters_config.get('area_px', 1200.0)
        intensity_mu = emitters_config.get('intensity_mu', 2000.0)
        intensity_sigma = emitters_config.get('intensity_sigma', 400.0)
        lifetime_avg = emitters_config.get('lifetime_avg', 2.5)
        z_range_um = emitters_config.get('z_range_um', 1.0)
        seed = emitters_config.get('seed', 42)
        
        print(f"生成 {num_emitters} 个发射器，{frames} 帧，FOV: {area_px}x{area_px} 像素")
        
        # 生成发射器属性
        em_attrs = sample_emitters(
            num_emitters=num_emitters,
            frame_range=(0, frames - 1),
            area_px=area_px,
            intensity_mu=intensity_mu,
            intensity_sigma=intensity_sigma,
            lifetime_avg=lifetime_avg,
            z_range_um=z_range_um,
            seed=seed
        )
        
        # 转换为每帧记录
        records = bin_emitters_to_frames(em_attrs, (0, frames - 1))
        
        # 保存到HDF5文件
        emitters_path = output_dir / 'emitters.h5'
        save_to_h5(emitters_path, em_attrs, records)
        
        # 生成可视化（可选）
        if not emitters_config.get('no_plot', True):  # 默认不生成图像以节省时间
            visualise(em_attrs, records, num_plot=20, out_dir=output_dir)
        
        print(f"发射器数据已保存到: {emitters_path}")
        return emitters_path
    
    def _step2_compute_zernike_coeffs(self, zmap_path: str, emitters_path: Path, 
                                     config: Dict[str, Any], output_dir: Path) -> None:
        """步骤2: 计算Zernike系数"""
        print("\n=== 步骤 2: 计算Zernike系数 ===")
        
        zernike_config = config.get('zernike', {})
        
        print(f"从 {zmap_path} 加载相位图数据")
        print(f"处理发射器文件: {emitters_path}")
        
        # 获取裁剪参数
        crop_size = zernike_config.get('crop_size')
        crop_offset = zernike_config.get('crop_offset', (0, 0))
        
        if crop_size is not None:
            print(f"使用裁剪参数: 尺寸={crop_size}x{crop_size}, 偏移={crop_offset}")
        
        # 加载数据
        phase_maps, coords, coeff_mag_patch, em_xy = load_data(
            Path(zmap_path), emitters_path, crop_size, crop_offset
        )
        
        print(f"相位图形状: {phase_maps.shape}")
        print(f"发射器数量: {len(em_xy)}")
        
        # 计算系数
        print("计算相位系数...")
        phase_coeffs = compute_phase_coeffs(phase_maps, em_xy)
        
        print("计算幅度系数...")
        mag_coeffs = compute_mag_coeffs(coords, coeff_mag_patch, em_xy)
        
        # 生成可视化（可选）
        num_plot = zernike_config.get('num_plot', 10)
        if not zernike_config.get('no_plot', True):  # 默认不生成图像以节省时间
            visualise_coeffs(phase_coeffs, mag_coeffs, emitters_path, num_plot)
        
        # 保存系数到发射器文件
        save_coeffs(emitters_path, phase_coeffs, mag_coeffs)
        
        print(f"Zernike系数已计算并保存到: {emitters_path}")
    
    def _step3_generate_tiff(self, emitters_path: Path, config: Dict[str, Any], 
                           output_dir: Path) -> Path:
        """步骤3: 生成多帧TIFF图像"""
        print("\n=== 步骤 3: 生成多帧TIFF图像 ===")
        
        tiff_config = config.get('tiff', {})
        
        # 输出TIFF文件路径
        tiff_filename = tiff_config.get('filename', f"{config['job_id']}.ome.tiff")
        tiff_output = output_dir / tiff_filename
        
        print(f"TIFF图像将保存到: {tiff_output}")
        
        # 使用完整的TIFF生成模块
        generate_tiff_stack(str(emitters_path), str(tiff_output), config)
        
        return tiff_output
    
    def run_batch(self, max_workers: Optional[int] = None, resume: bool = True) -> Dict[str, Any]:
        """运行批量处理
        
        Parameters
        ----------
        max_workers : int, optional
            最大并行工作进程数，默认为CPU核心数
        resume : bool
            是否恢复之前的处理
            
        Returns
        -------
        summary : dict
            批量处理摘要
        """
        if max_workers is None:
            max_workers = min(cpu_count(), self.batch_config.get('max_workers', cpu_count()))
        
        print(f"\n=== 开始批量TIFF生成 ===")
        print(f"配置文件: {self.batch_config_path}")
        print(f"输出目录: {self.base_output_dir.absolute()}")
        print(f"最大并行数: {max_workers}")
        
        # 生成所有作业配置
        all_jobs = self.generate_job_configs()
        print(f"总作业数: {len(all_jobs)}")
        
        # 设置总作业数用于进度显示
        self._total_jobs = len(all_jobs)
        
        # 过滤已完成的作业（如果恢复）
        if resume:
            completed_job_ids = set(self.status['completed_jobs'])
            pending_jobs = [job for job in all_jobs if job['job_id'] not in completed_job_ids]
            print(f"已完成作业数: {len(completed_job_ids)}")
            print(f"待处理作业数: {len(pending_jobs)}")
        else:
            pending_jobs = all_jobs
            self.status = {
                'completed_jobs': [],
                'failed_jobs': [],
                'start_time': time.time(),
                'last_update': None
            }
        
        if not pending_jobs:
            print("所有作业已完成！")
            return self._generate_summary()
        
        # 设置开始时间
        if self.status['start_time'] is None:
            self.status['start_time'] = time.time()
        
        # 并行处理
        if max_workers == 1:
            # 串行处理
            for job_config in pending_jobs:
                result = self.process_single_job(job_config)
                self._update_status(result)
        else:
            # 并行处理
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有作业
                future_to_job = {executor.submit(process_job_wrapper, job_config): job_config 
                               for job_config in pending_jobs}
                
                # 处理完成的作业
                for future in as_completed(future_to_job):
                    result = future.result()
                    self._update_status(result)
        
        return self._generate_summary()
    
    def _update_status(self, result: Dict[str, Any]):
        """更新状态"""
        job_id = result['job_id']
        
        if result['status'] == 'success':
            self.status['completed_jobs'].append(job_id)
            print(f"✓ 作业 {job_id} 完成")
        else:
            self.status['failed_jobs'].append({
                'job_id': job_id,
                'error': result.get('error', '未知错误'),
                'timestamp': time.time()
            })
            print(f"✗ 作业 {job_id} 失败: {result.get('error', '未知错误')}")
        
        self.save_status()
        
        # 打印进度
        total_jobs = len(self.status['completed_jobs']) + len(self.status['failed_jobs'])
        if hasattr(self, '_total_jobs'):
            print(f"进度: {total_jobs}/{self._total_jobs} ({total_jobs/self._total_jobs*100:.1f}%)")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成处理摘要"""
        total_time = time.time() - self.status['start_time'] if self.status['start_time'] else 0
        
        summary = {
            'total_jobs': len(self.status['completed_jobs']) + len(self.status['failed_jobs']),
            'completed_jobs': len(self.status['completed_jobs']),
            'failed_jobs': len(self.status['failed_jobs']),
            'success_rate': len(self.status['completed_jobs']) / max(1, len(self.status['completed_jobs']) + len(self.status['failed_jobs'])) * 100,
            'total_time_seconds': total_time,
            'total_time_formatted': self._format_time(total_time),
            'output_directory': str(self.base_output_dir.absolute()),
            'failed_job_details': self.status['failed_jobs']
        }
        
        return summary
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            return f"{seconds/60:.1f}分钟"
        else:
            return f"{seconds/3600:.1f}小时"


def process_job_wrapper(job_config: Dict[str, Any]) -> Dict[str, Any]:
    """作业处理包装函数（用于多进程）"""
    try:
        # 直接处理作业，避免循环导入
        job_id = job_config['job_id']
        
        print(f"\n=== 开始处理作业: {job_id} ===")
        
        # 创建作业输出目录
        base_output_dir = Path(job_config.get('base_output_dir', 'batch_output'))
        job_output_dir = base_output_dir / job_config['output_subdir']
        job_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取zmap路径并转换为绝对路径
        zmap_path = job_config.get('zmap_path')
        if not zmap_path:
            raise ValueError(f"配置中未指定zmap_path")
        
        # 将相对路径转换为绝对路径（相对于配置文件所在目录）
        if not Path(zmap_path).is_absolute():
            # 从job_config中获取配置文件路径
            config_file_path = job_config.get('_config_file_path')
            if config_file_path:
                config_dir = Path(config_file_path).parent
                zmap_path = str(config_dir / zmap_path)
        
        if not Path(zmap_path).exists():
            raise ValueError(f"找不到zmap文件: {zmap_path}")
        
        # 步骤1: 生成发射器数据
        emitters_path = _step1_generate_emitters_standalone(job_config, job_output_dir)
        
        # 步骤2: 计算Zernike系数
        _step2_compute_zernike_coeffs_standalone(zmap_path, emitters_path, job_config, job_output_dir)
        
        # 步骤3: 生成TIFF
        tiff_path = _step3_generate_tiff_standalone(emitters_path, job_config, job_output_dir)
        
        result = {
            'job_id': job_id,
            'status': 'success',
            'output_dir': str(job_output_dir),
            'tiff_path': str(tiff_path),
            'emitters_path': str(emitters_path)
        }
        
        print(f"作业 {job_id} 完成成功")
        return result
        
    except Exception as e:
        error_msg = f"作业 {job_config.get('job_id', 'unknown')} 失败: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        return {
            'job_id': job_config.get('job_id', 'unknown'),
            'status': 'failed',
            'error': error_msg,
            'traceback': traceback.format_exc()
        }


def _step1_generate_emitters_standalone(config: Dict[str, Any], output_dir: Path) -> Path:
    """独立的步骤1: 生成发射器数据"""
    print("=== 步骤 1: 生成发射器数据 ===")
    
    emitters_config = config.get('emitters', {})
    
    # 参数设置
    num_emitters = emitters_config.get('num_emitters', 1000)
    frames = emitters_config.get('frames', 10)
    area_px = emitters_config.get('area_px', 1200.0)
    intensity_mu = emitters_config.get('intensity_mu', 2000.0)
    intensity_sigma = emitters_config.get('intensity_sigma', 400.0)
    lifetime_avg = emitters_config.get('lifetime_avg', 2.5)
    z_range_um = emitters_config.get('z_range_um', 1.0)
    seed = emitters_config.get('seed', 42)
    
    print(f"生成 {num_emitters} 个发射器，{frames} 帧，FOV: {area_px}x{area_px} 像素")
    
    # 生成发射器属性
    em_attrs = sample_emitters(
        num_emitters=num_emitters,
        frame_range=(0, frames - 1),
        area_px=area_px,
        intensity_mu=intensity_mu,
        intensity_sigma=intensity_sigma,
        lifetime_avg=lifetime_avg,
        z_range_um=z_range_um,
        seed=seed
    )
    
    # 转换为每帧记录
    records = bin_emitters_to_frames(em_attrs, (0, frames - 1))
    
    # 保存到HDF5文件
    emitters_path = output_dir / 'emitters.h5'
    save_to_h5(emitters_path, em_attrs, records)
    
    # 生成可视化（可选）
    if not emitters_config.get('no_plot', True):  # 默认不生成图像以节省时间
        visualise(em_attrs, records, num_plot=20, out_dir=output_dir)
    
    print(f"发射器数据已保存到: {emitters_path}")
    return emitters_path


def _step2_compute_zernike_coeffs_standalone(zmap_path: str, emitters_path: Path, 
                                            config: Dict[str, Any], output_dir: Path) -> None:
    """独立的步骤2: 计算Zernike系数"""
    print("\n=== 步骤 2: 计算Zernike系数 ===")
    
    zernike_config = config.get('zernike', {})
    
    print(f"从 {zmap_path} 加载相位图数据")
    print(f"处理发射器文件: {emitters_path}")
    
    # 获取裁剪参数
    crop_size = zernike_config.get('crop_size')
    crop_offset = zernike_config.get('crop_offset', (0, 0))
    
    if crop_size is not None:
        print(f"使用裁剪参数: 尺寸={crop_size}x{crop_size}, 偏移={crop_offset}")
    
    # 加载数据
    phase_maps, coords, coeff_mag_patch, em_xy = load_data(
        Path(zmap_path), emitters_path, crop_size, crop_offset
    )
    
    print(f"相位图形状: {phase_maps.shape}")
    print(f"发射器数量: {len(em_xy)}")
    
    # 计算系数
    print("计算相位系数...")
    phase_coeffs = compute_phase_coeffs(phase_maps, em_xy)
    
    print("计算幅度系数...")
    mag_coeffs = compute_mag_coeffs(coords, coeff_mag_patch, em_xy)
    
    # 生成可视化（可选）
    num_plot = zernike_config.get('num_plot', 10)
    if not zernike_config.get('no_plot', True):  # 默认不生成图像以节省时间
        visualise_coeffs(phase_coeffs, mag_coeffs, emitters_path, num_plot)
    
    # 保存系数到发射器文件
    save_coeffs(emitters_path, phase_coeffs, mag_coeffs)
    
    print(f"Zernike系数已计算并保存到: {emitters_path}")


def _step3_generate_tiff_standalone(emitters_path: Path, config: Dict[str, Any], 
                                   output_dir: Path) -> Path:
    """独立的步骤3: 生成多帧TIFF图像"""
    print("\n=== 步骤 3: 生成多帧TIFF图像 ===")
    
    tiff_config = config.get('tiff', {})
    
    # 输出TIFF文件路径
    tiff_filename = tiff_config.get('filename', f"{config['job_id']}.ome.tiff")
    tiff_output = output_dir / tiff_filename
    
    print(f"TIFF图像将保存到: {tiff_output}")
    
    # 使用完整的TIFF生成模块
    generate_tiff_stack(str(emitters_path), str(tiff_output), config)
    
    return tiff_output


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='批量TIFF生成器',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--batch_config',
        type=str,
        required=True,
        help='批量配置文件路径'
    )
    
    parser.add_argument(
        '--max_workers',
        type=int,
        help='最大并行工作进程数（默认为CPU核心数）'
    )
    
    parser.add_argument(
        '--no_resume',
        action='store_true',
        help='不恢复之前的处理，重新开始'
    )
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not Path(args.batch_config).exists():
        print(f"错误: 找不到批量配置文件: {args.batch_config}")
        sys.exit(1)
    
    try:
        # 创建批量生成器
        generator = BatchTiffGenerator(args.batch_config)
        
        # 运行批量处理
        summary = generator.run_batch(
            max_workers=args.max_workers,
            resume=not args.no_resume
        )
        
        # 打印摘要
        print("\n=== 批量处理完成 ===")
        print(f"总作业数: {summary['total_jobs']}")
        print(f"成功完成: {summary['completed_jobs']}")
        print(f"失败作业: {summary['failed_jobs']}")
        print(f"成功率: {summary['success_rate']:.1f}%")
        print(f"总耗时: {summary['total_time_formatted']}")
        print(f"输出目录: {summary['output_directory']}")
        
        if summary['failed_jobs'] > 0:
            print("\n失败作业详情:")
            for failed_job in summary['failed_job_details']:
                print(f"  - {failed_job['job_id']}: {failed_job['error']}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()