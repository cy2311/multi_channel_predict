#!/usr/bin/env python3
"""
使用示例：从Zmap到TIFF的完整处理流程

这个脚本展示了如何使用整合的流水线处理数据
"""

import os
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from main import main as pipeline_main


def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 假设的输入文件路径
    zmap_path = "../simulated_data/patches.h5"
    output_dir = "example_output"
    
    # 构建命令行参数
    sys.argv = [
        "main.py",
        "--zmap", zmap_path,
        "--output_dir", output_dir,
        "--config", "pipeline_config.json"
    ]
    
    print(f"模拟命令: python main.py --zmap {zmap_path} --output_dir {output_dir}")
    print("注意: 这只是示例，需要实际的patches.h5文件才能运行")
    
    # 如果文件存在，可以取消注释下面的行来实际运行
    # pipeline_main()


def example_step_by_step():
    """分步骤运行示例"""
    print("\n=== 分步骤运行示例 ===")
    
    zmap_path = "../simulated_data/patches.h5"
    output_dir = "step_by_step_output"
    
    # 步骤1: 只生成发射器
    print("步骤1: 生成发射器")
    sys.argv = [
        "main.py",
        "--zmap", zmap_path,
        "--output_dir", output_dir,
        "--skip_zernike",
        "--skip_tiff"
    ]
    print(f"命令: python main.py --zmap {zmap_path} --output_dir {output_dir} --skip_zernike --skip_tiff")
    
    # 步骤2: 计算Zernike系数
    print("\n步骤2: 计算Zernike系数")
    sys.argv = [
        "main.py",
        "--zmap", zmap_path,
        "--output_dir", output_dir,
        "--skip_emitters",
        "--skip_tiff"
    ]
    print(f"命令: python main.py --zmap {zmap_path} --output_dir {output_dir} --skip_emitters --skip_tiff")
    
    # 步骤3: 生成TIFF
    print("\n步骤3: 生成TIFF")
    sys.argv = [
        "main.py",
        "--zmap", zmap_path,
        "--output_dir", output_dir,
        "--skip_emitters",
        "--skip_zernike"
    ]
    print(f"命令: python main.py --zmap {zmap_path} --output_dir {output_dir} --skip_emitters --skip_zernike")


def example_custom_config():
    """自定义配置示例"""
    print("\n=== 自定义配置示例 ===")
    
    # 创建自定义配置
    custom_config = {
        "emitters": {
            "num_emitters": 2000,  # 更多发射器
            "frames": 20,          # 更多帧
            "area_px": 1500.0,     # 更大的FOV
            "intensity_mu": 3000.0,
            "lifetime_avg": 3.0
        },
        "tiff": {
            "filename": "custom_simulation.ome.tiff",
            "roi_size": 1500,
            "add_noise": True,
            "noise_params": {
                "background_level": 150,
                "readout_noise": 15,
                "shot_noise": True
            }
        }
    }
    
    # 保存自定义配置
    import json
    custom_config_path = "custom_config.json"
    with open(custom_config_path, 'w') as f:
        json.dump(custom_config, f, indent=2)
    
    print(f"自定义配置已保存到: {custom_config_path}")
    
    # 使用自定义配置
    zmap_path = "../simulated_data/patches.h5"
    output_dir = "custom_output"
    
    sys.argv = [
        "main.py",
        "--zmap", zmap_path,
        "--output_dir", output_dir,
        "--config", custom_config_path
    ]
    
    print(f"命令: python main.py --zmap {zmap_path} --output_dir {output_dir} --config {custom_config_path}")


def show_help():
    """显示帮助信息"""
    print("\n=== 帮助信息 ===")
    
    sys.argv = ["main.py", "--help"]
    
    try:
        pipeline_main()
    except SystemExit:
        pass  # argparse会调用sys.exit()


def main():
    """运行所有示例"""
    print("从Zmap到TIFF的完整处理流程 - 使用示例")
    print("=" * 50)
    
    example_basic_usage()
    example_step_by_step()
    example_custom_config()
    show_help()
    
    print("\n=== 总结 ===")
    print("1. 基本用法: python main.py --zmap patches.h5 --output_dir output/")
    print("2. 分步运行: 使用 --skip_* 参数跳过特定步骤")
    print("3. 自定义配置: 修改 pipeline_config.json 或创建新的配置文件")
    print("4. 获取帮助: python main.py --help")
    
    print("\n注意事项:")
    print("- 确保输入的patches.h5文件存在")
    print("- 确保Zernike基函数文件在 ../simulated_data/zernike_polynomials/ 目录下")
    print("- 确保配置文件 ../configs/default_config.json 存在")
    print("- 输出目录会自动创建")


if __name__ == '__main__':
    main()