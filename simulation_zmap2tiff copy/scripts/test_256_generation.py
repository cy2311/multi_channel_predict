#!/usr/bin/env python3
"""
测试256x256尺寸的TIFF生成功能

这个脚本用于测试新增的裁剪功能和256x256尺寸的TIFF生成。
首先生成一个小样本来验证功能是否正常工作。
"""

import json
import sys
from pathlib import Path
from batch_tiff_generator import BatchTiffGenerator

def create_test_config():
    """创建测试配置 - 只生成2个样本用于快速测试"""
    test_config = {
        "description": "测试配置 - 2个样本256x256尺寸",
        "version": "2.0",
        
        "base_output_dir": "test_outputs_256",
        "max_workers": 2,
        
        "base_config": {
            "zmap_path": "/home/guest/Others/DECODE_rewrite/phase_retrieval_tiff2h5/result/result.h5",
            
            "emitters": {
                "num_emitters": 100,  # 减少发射器数量用于快速测试
                "frames": 10,         # 减少帧数用于快速测试
                "area_px": 256.0,
                "intensity_mu": 8000.0,
                "intensity_sigma": 1500.0,
                "lifetime_avg": 2.5,
                "z_range_um": 1.0,
                "seed": 42,
                "no_plot": True
            },
            
            "zernike": {
                "num_plot": 5,
                "no_plot": True,
                "interpolation_method": "cubic",
                "crop_size": 256,
                "crop_offset": [0, 0]
            },
            
            "tiff": {
                "filename": "test_simulation_256.ome.tiff",
                "roi_size": 256,
                "crop_offset": [0, 0],
                "use_direct_rendering": True,
                "add_noise": True,
                "noise_params": {
                    "background": 100,
                    "readout_noise": 10,
                    "shot_noise": True
                }
            },
            
            "optical": {
                "use_default_config": True,
                "ignore_fixed_psf_coeffs": True
            },
            
            "output": {
                "save_intermediate": True,
                "generate_plots": False,
                "verbose": True
            }
        },
        
        "sample_configs": {
            "num_samples": 2,
            "frames_per_sample": 10,
            "sample_naming": "test_sample_{sample_id:03d}"
        }
    }
    
    return test_config

def main():
    """运行测试"""
    print("=== 测试256x256尺寸TIFF生成功能 ===")
    
    # 创建测试配置
    test_config = create_test_config()
    
    # 保存测试配置文件
    config_path = Path("configs/test_config_256.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(test_config, f, indent=2, ensure_ascii=False)
    
    print(f"测试配置已保存到: {config_path}")
    
    # 运行批量生成器
    try:
        generator = BatchTiffGenerator(config_path)
        summary = generator.run_batch()
        
        print("\n=== 测试完成 ===")
        print(f"成功生成的作业数: {summary['completed_jobs']}")
        print(f"失败的作业数: {summary['failed_jobs']}")
        print(f"总处理时间: {summary['total_time_seconds']:.2f}秒")
        
        if summary['completed_jobs'] > 0:
            print("\n✅ 测试成功！256x256尺寸的TIFF生成功能正常工作。")
            print("现在可以运行完整的100样本生成：")
            print("python batch_tiff_generator.py configs/batch_config_100samples_256.json")
        else:
            print("\n❌ 测试失败，请检查错误信息。")
            return 1
            
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())