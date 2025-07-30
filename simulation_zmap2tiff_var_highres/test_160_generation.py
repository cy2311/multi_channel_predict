#!/usr/bin/env python3
"""
测试160x160分辨率TIFF生成

这个脚本用于测试新的160x160分辨率配置是否正确工作。
它会生成几个测试样本来验证：
1. 像素物理尺寸是否正确计算（保持与40x40相同的物理FOV）
2. 图像分辨率是否为160x160
3. 发射器位置是否正确缩放
"""

import json
import sys
from pathlib import Path
import numpy as np
import tifffile as tiff
import h5py

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from batch_tiff_generator import BatchTiffGenerator


def create_test_config():
    """创建测试配置文件"""
    test_config = {
        "description": "测试160x160分辨率TIFF生成",
        "version": "2.0",
        
        "base_output_dir": "test_outputs_160",
        "max_workers": 1,  # 使用单线程便于调试
        
        "base_config": {
            "zmap_path": "/home/guest/Others/DECODE_rewrite/phase_retrieval_tiff2h5/result/result.h5",
            
            "emitters": {
                "num_emitters": 100,  # 减少发射器数量用于测试
                "frames": 10,         # 减少帧数用于测试
                "area_px": 160.0,
                "intensity_mu": 7000.0,
                "intensity_sigma": 3000.0,
                "lifetime_avg": 1,
                "z_range_um": 0.8,
                "seed": 42,
                "no_plot": True
            },
            
            "zernike": {
                "num_plot": 5,
                "no_plot": True,
                "interpolation_method": "cubic",
                "crop_size": 160,
                "crop_offset": [0, 0]
            },
            
            "tiff": {
                "filename": "test_160.ome.tiff",
                "roi_size": 160,
                "crop_offset": [0, 0],
                "use_direct_rendering": True,
                "add_noise": True,
                "noise_params": {
                    "background": 100,
                    "readout_noise": 58.8,
                    "shot_noise": True
                }
            },
            
            "optical": {
                "use_default_config": False,
                "ignore_fixed_psf_coeffs": True,
                "wavelength_nm": 660,
                "pixel_size_nm_x": 25.2775,  # 101.11/4 = 25.2775 (4倍分辨率提升)
                "pixel_size_nm_y": 24.7075,  # 98.83/4 = 24.7075
                "NA": 1.4,
                "n_medium": 1.518,
                "psf_patch_size": 9,
                "max_noll_coeffs": 21,
                "defocus_noll_index": 4,
                "ao_strength_factor": 0.05
            },
            
            "output": {
                "save_intermediate": True,
                "save_emitters": True,
                "save_zernike": True,
                "save_tiff": True
            }
        },
        
        "sample_configs": {
            "num_samples": 3,  # 只生成3个测试样本
            "frames_per_sample": 10,
            "sample_naming": "test_160_{sample_id:03d}"
        }
    }
    
    return test_config


def analyze_results(output_dir: Path):
    """分析生成结果"""
    print("\n=== 分析生成结果 ===")
    
    # 查找生成的TIFF文件
    tiff_files = list(output_dir.glob("**/test_160_*.ome.tiff"))
    
    if not tiff_files:
        print("未找到生成的TIFF文件")
        return
    
    print(f"找到 {len(tiff_files)} 个TIFF文件")
    
    for tiff_file in tiff_files[:2]:  # 只分析前两个文件
        print(f"\n分析文件: {tiff_file.name}")
        
        # 读取TIFF文件
        with tiff.TiffFile(tiff_file) as tf:
            images = tf.asarray()
            metadata = tf.ome_metadata
            
            print(f"  图像形状: {images.shape}")
            print(f"  数据类型: {images.dtype}")
            print(f"  像素值范围: [{images.min():.2f}, {images.max():.2f}]")
            
            # 检查元数据中的像素尺寸
            if metadata:
                print(f"  OME元数据存在")
                # 这里可以解析OME-XML来获取像素尺寸信息
        
        # 分析对应的发射器数据
        emitters_file = tiff_file.parent / "emitters.h5"
        if emitters_file.exists():
            print(f"  分析发射器数据: {emitters_file.name}")
            
            with h5py.File(emitters_file, 'r') as f:
                xyz_rec = np.array(f['records/xyz'])
                frame_ix = np.array(f['records/frame_ix'])
                
                print(f"    发射器记录数: {len(xyz_rec)}")
                print(f"    X坐标范围: [{xyz_rec[:, 0].min():.2f}, {xyz_rec[:, 0].max():.2f}] 像素")
                print(f"    Y坐标范围: [{xyz_rec[:, 1].min():.2f}, {xyz_rec[:, 1].max():.2f}] 像素")
                print(f"    Z坐标范围: [{xyz_rec[:, 2].min():.4f}, {xyz_rec[:, 2].max():.4f}] μm")
                
                # 计算物理FOV
                pixel_size_x_nm = 25.2775
                pixel_size_y_nm = 24.7075
                fov_x_um = 160 * pixel_size_x_nm / 1000
                fov_y_um = 160 * pixel_size_y_nm / 1000
                
                print(f"    计算的物理FOV: {fov_x_um:.3f} x {fov_y_um:.3f} μm")
                
                # 对比40x40的物理FOV
                fov_40_x_um = 40 * 101.11 / 1000
                fov_40_y_um = 40 * 98.83 / 1000
                print(f"    40x40的物理FOV: {fov_40_x_um:.3f} x {fov_40_y_um:.3f} μm")
                print(f"    FOV差异: X={abs(fov_x_um - fov_40_x_um):.6f} μm, Y={abs(fov_y_um - fov_40_y_um):.6f} μm")


def main():
    """主函数"""
    print("=== 160x160分辨率TIFF生成测试 ===")
    
    # 创建测试配置
    test_config = create_test_config()
    
    # 保存测试配置文件
    config_path = Path("test_config_160.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(test_config, f, indent=2, ensure_ascii=False)
    
    print(f"测试配置已保存到: {config_path}")
    
    # 运行批量生成
    try:
        generator = BatchTiffGenerator(str(config_path))
        summary = generator.run_batch(max_workers=1, resume=False)
        
        print("\n=== 批量处理摘要 ===")
        print(f"成功: {summary['successful_jobs']}")
        print(f"失败: {summary['failed_jobs']}")
        print(f"总耗时: {summary['total_time']:.2f} 秒")
        
        if summary['successful_jobs'] > 0:
            # 分析结果
            output_dir = Path(test_config['base_output_dir'])
            analyze_results(output_dir)
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理测试配置文件
        if config_path.exists():
            config_path.unlink()
            print(f"\n已清理测试配置文件: {config_path}")


if __name__ == '__main__':
    main()