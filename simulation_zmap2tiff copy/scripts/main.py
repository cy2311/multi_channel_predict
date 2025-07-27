#!/usr/bin/env python3
"""
从Zmap到TIFF的完整处理流程整合脚本

这个脚本整合了以下步骤：
1. 生成发射器数据
2. 计算Zernike系数
3. 生成多帧TIFF图像

使用方法:
    python main.py --zmap path/to/patches.h5 --output_dir output/
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any

# 添加父目录到路径以便导入其他模块
sys.path.append(str(Path(__file__).parent.parent))

from trainset_simulation.generate_emitters import (
    sample_emitters, bin_emitters_to_frames, save_to_h5, visualise
)
from trainset_simulation.compute_zernike_coeffs import (
    load_data, compute_phase_coeffs, compute_mag_coeffs, save_coeffs, visualise_coeffs
)
from tiff_generator import generate_tiff_stack


def load_pipeline_config(config_path: str) -> Dict[str, Any]:
    """加载流水线配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def step1_generate_emitters(config: Dict[str, Any], output_dir: Path) -> Path:
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
    
    # 生成可视化
    if not emitters_config.get('no_plot', False):
        visualise(em_attrs, records, num_plot=20, out_dir=output_dir)
    
    print(f"发射器数据已保存到: {emitters_path}")
    return emitters_path


def step2_compute_zernike_coeffs(zmap_path: str, emitters_path: Path, 
                                 config: Dict[str, Any], output_dir: Path) -> None:
    """步骤2: 计算Zernike系数"""
    print("\n=== 步骤 2: 计算Zernike系数 ===")
    
    zernike_config = config.get('zernike', {})
    
    print(f"从 {zmap_path} 加载相位图数据")
    print(f"处理发射器文件: {emitters_path}")
    
    # 加载数据
    phase_maps, coords, coeff_mag_patch, em_xy = load_data(
        Path(zmap_path), emitters_path
    )
    
    print(f"相位图形状: {phase_maps.shape}")
    print(f"发射器数量: {len(em_xy)}")
    
    # 计算系数
    print("计算相位系数...")
    phase_coeffs = compute_phase_coeffs(phase_maps, em_xy)
    
    print("计算幅度系数...")
    mag_coeffs = compute_mag_coeffs(coords, coeff_mag_patch, em_xy)
    
    # 生成可视化
    num_plot = zernike_config.get('num_plot', 10)
    if not zernike_config.get('no_plot', False):
        visualise_coeffs(phase_coeffs, mag_coeffs, emitters_path, num_plot)
    
    # 保存系数到发射器文件
    save_coeffs(emitters_path, phase_coeffs, mag_coeffs)
    
    print(f"Zernike系数已计算并保存到: {emitters_path}")


def step3_generate_tiff(emitters_path: Path, config: Dict[str, Any], 
                       output_dir: Path) -> Path:
    """步骤3: 生成多帧TIFF图像"""
    print("\n=== 步骤 3: 生成多帧TIFF图像 ===")
    
    tiff_config = config.get('tiff', {})
    
    # 输出TIFF文件路径
    tiff_output = output_dir / tiff_config.get('filename', 'simulation.ome.tiff')
    
    print(f"TIFF图像将保存到: {tiff_output}")
    
    # 使用完整的TIFF生成模块
    generate_tiff_stack(str(emitters_path), str(tiff_output), config)
    
    return tiff_output


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='从Zmap到TIFF的完整处理流程',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--zmap', 
        type=str, 
        required=True, 
        help='输入的Zmap文件路径 (patches.h5)'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='output', 
        help='输出目录'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='pipeline_config.json', 
        help='流水线配置文件路径'
    )
    
    parser.add_argument(
        '--skip_emitters', 
        action='store_true', 
        help='跳过发射器生成步骤（使用现有的emitters.h5）'
    )
    
    parser.add_argument(
        '--skip_zernike', 
        action='store_true', 
        help='跳过Zernike系数计算步骤'
    )
    
    parser.add_argument(
        '--skip_tiff', 
        action='store_true', 
        help='跳过TIFF生成步骤'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.zmap).exists():
        print(f"错误: 找不到Zmap文件: {args.zmap}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    config_path = Path(args.config)
    if config_path.exists():
        config = load_pipeline_config(str(config_path))
        print(f"已加载配置文件: {config_path}")
    else:
        print(f"配置文件不存在，使用默认配置: {config_path}")
        config = {}
    
    print(f"\n开始处理流程...")
    print(f"输入Zmap: {args.zmap}")
    print(f"输出目录: {output_dir.absolute()}")
    
    try:
        # 步骤1: 生成发射器
        if not args.skip_emitters:
            emitters_path = step1_generate_emitters(config, output_dir)
        else:
            emitters_path = output_dir / 'emitters.h5'
            if not emitters_path.exists():
                print(f"错误: 找不到现有的发射器文件: {emitters_path}")
                sys.exit(1)
            print(f"跳过发射器生成，使用现有文件: {emitters_path}")
        
        # 步骤2: 计算Zernike系数
        if not args.skip_zernike:
            step2_compute_zernike_coeffs(args.zmap, emitters_path, config, output_dir)
        else:
            print("跳过Zernike系数计算")
        
        # 步骤3: 生成TIFF
        if not args.skip_tiff:
            tiff_path = step3_generate_tiff(emitters_path, config, output_dir)
        else:
            print("跳过TIFF生成")
        
        print("\n=== 处理流程完成! ===")
        print(f"所有输出文件保存在: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()