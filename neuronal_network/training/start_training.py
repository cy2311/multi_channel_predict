#!/usr/bin/env python3
"""
启动DECODE网络训练的便捷脚本
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from train_decode_network import main as train_main


def main():
    parser = argparse.ArgumentParser(description='启动DECODE网络训练')
    parser.add_argument('--samples', type=int, choices=[10, 20, 50, 80, 100], default=100,
                       help='使用的样本数量 (10, 20, 50, 80, 或 100)')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--output_suffix', type=str, default='', help='输出目录后缀')
    parser.add_argument('--data_dir', type=str, help='指定数据目录路径（可选）')
    
    args = parser.parse_args()
    
    # 设置路径
    project_root = Path(__file__).parent.parent.parent
    
    # 使用指定的数据目录或默认路径
    if args.data_dir:
        data_root = Path(args.data_dir)
    else:
        data_root = project_root / f"simulation_zmap2tiff/outputs_{args.samples}samples_256"
    
    config_path = Path(__file__).parent / "configs/train_config.json"
    
    # 检查数据目录
    if not data_root.exists():
        print(f"错误: 数据目录不存在: {data_root}")
        print("请先生成训练数据")
        return
    
    # 设置输出目录
    output_dir = project_root / f"neuronal_network/training/outputs/train_{args.samples}samples{args.output_suffix}"
    
    # 构建训练命令参数
    train_args = [
        '--config', str(config_path),
        '--data_root', str(data_root),
        '--output_dir', str(output_dir)
    ]
    
    if args.samples < 100:
        train_args.extend(['--sample_subset', str(args.samples)])
    
    print(f"开始训练DECODE网络:")
    print(f"  数据目录: {data_root}")
    print(f"  样本数量: {args.samples}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  输出目录: {output_dir}")
    print()
    
    # 修改sys.argv以传递给训练脚本
    original_argv = sys.argv.copy()
    sys.argv = ['train_decode_network.py'] + train_args
    
    try:
        # 调用训练主函数
        train_main()
    finally:
        # 恢复原始argv
        sys.argv = original_argv


if __name__ == '__main__':
    main()