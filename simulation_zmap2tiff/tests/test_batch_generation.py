#!/usr/bin/env python3
"""
批量TIFF生成测试脚本

用于测试和验证批量生成功能
"""

import os
import json
import time
from pathlib import Path
from batch_tiff_generator import BatchTiffGenerator


def create_test_config():
    """创建测试配置"""
    config = {
        "base_output_dir": "./batch_output_test",
        "max_workers": 2,
        "base_config": {
            "emitters": {
                "num_emitters": 100,
                "density_mu_sig": [1.5, 0.3],
                "intensity_mu_sig": [8000, 1000],
                "xy_unit": "px",
                "z_range": [-750, 750],
                "lifetime": 1
            },
            "zernike": {
                "num_modes": 100,
                "z_range": [-750, 750],
                "mode_weights": {
                    "piston": 0.0,
                    "tip": 0.05,
                    "tilt": 0.05,
                    "defocus": 0.1,
                    "astigmatism": 0.08,
                    "coma": 0.06,
                    "spherical": 0.04,
                    "higher_order": 0.02
                }
            },
            "tiff": {
                "filename": "simulation.tiff",
                "roi_size": 800,
                "use_direct_rendering": True,
                "add_noise": True,
                "noise_params": {
                    "background": 100,
                    "readout_noise": 10,
                    "shot_noise": True
                }
            },
            "optical": {
                "description": "从Zmap插值获取Zernike系数"
            },
            "output": {
                "save_intermediate": False,
                "generate_plots": False,
                "verbose": True
            },
            "memory_optimization": {
                "chunk_size": 5,
                "enable_gc": True,
                "gc_frequency": 3
            }
        },
        "sample_configs": {
            "num_samples": 3,
            "frames_per_sample": 50,
            "sample_naming": "test_sample_{sample_id:03d}"
        }
    }
    
    return config


def test_basic_batch_generation():
    """测试基本批量生成功能"""
    print("=== 测试基本批量生成功能 ===")
    
    # 创建测试配置
    config = create_test_config()
    
    # 保存配置文件
    config_path = "test_batch_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"测试配置已保存到: {config_path}")
    
    # 创建批量生成器
    generator = BatchTiffGenerator(config_path)
    
    # 显示作业信息
    jobs = generator._generate_all_jobs()
    print(f"\n生成的作业数量: {len(jobs)}")
    for i, job in enumerate(jobs):
        if 'sample_info' in job:
            print(f"样本 {i+1}: {job['job_id']}, 帧数={job['sample_info']['frames']}, "
                  f"随机种子={job['sample_info']['random_seed']}")
        else:
            print(f"作业 {i+1}: {job['job_id']}")
    
    # 运行批量生成
    start_time = time.time()
    generator.run_batch()
    end_time = time.time()
    
    print(f"\n批量生成完成，耗时: {end_time - start_time:.2f} 秒")
    
    # 检查输出文件
    output_dir = Path(config['base_output_dir'])
    if output_dir.exists():
        print(f"\n输出目录: {output_dir}")
        for item in output_dir.rglob("*"):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  {item.relative_to(output_dir)}: {size_mb:.2f} MB")
    
    # 清理
    if os.path.exists(config_path):
        os.remove(config_path)
    
    return True


def test_memory_optimization():
    """测试内存优化功能"""
    print("\n=== 测试内存优化功能 ===")
    
    # 创建内存优化配置
    config = create_test_config()
    config['base_config']['memory_optimization'] = {
        'chunk_size': 3,
        'enable_gc': True,
        'gc_frequency': 2
    }
    config['variable_configs'] = {
        'num_emitters': [200],  # 更多发射器
        'roi_size': [1000]      # 更大图像
    }
    
    config_path = "test_memory_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"内存优化配置已保存到: {config_path}")
    
    # 运行测试
    generator = BatchTiffGenerator(config_path)
    
    start_time = time.time()
    generator.run_batch()
    end_time = time.time()
    
    print(f"内存优化测试完成，耗时: {end_time - start_time:.2f} 秒")
    
    # 清理
    if os.path.exists(config_path):
        os.remove(config_path)
    
    return True


def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    # 创建有问题的配置
    config = create_test_config()
    config['variable_configs'] = {
        'num_emitters': [0],  # 无效的发射器数量
        'roi_size': [100]     # 很小的ROI
    }
    
    config_path = "test_error_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"错误测试配置已保存到: {config_path}")
    
    try:
        generator = BatchTiffGenerator(config_path)
        generator.run_batch()
        print("错误处理测试完成")
    except Exception as e:
        print(f"捕获到预期错误: {e}")
    
    # 清理
    if os.path.exists(config_path):
        os.remove(config_path)
    
    return True


def main():
    """主测试函数"""
    print("开始批量TIFF生成测试")
    print("=" * 50)
    
    try:
        # 基本功能测试
        test_basic_batch_generation()
        
        # 内存优化测试
        test_memory_optimization()
        
        # 错误处理测试
        test_error_handling()
        
        print("\n=" * 50)
        print("所有测试完成！")
        
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)