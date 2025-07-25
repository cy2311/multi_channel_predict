#!/usr/bin/env python3
"""
运行相位恢复流程的简单脚本

这个脚本是tiff2h5_phase_retrieval.py的简单包装器，提供了命令行参数支持。

使用方法：
    python run.py [--tiff_path PATH] [--output_dir DIR] [--ncc_threshold THRESHOLD]
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 导入主脚本
from tiff2h5_phase_retrieval import main as phase_retrieval_main
import tiff2h5_phase_retrieval as pr

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行相位恢复流程')
    
    parser.add_argument('--tiff_path', type=str, 
                        default='../beads/spool_100mW_30ms_3D_1_2/spool_100mW_30ms_3D_1_2_MMStack_Default.ome.tif',
                        help='输入的OME-TIFF文件路径')
    
    parser.add_argument('--camera_params', type=str,
                        default='../beads/spool_100mW_30ms_3D_1_2/camera_parameters.json',
                        help='相机参数JSON文件路径')
    
    parser.add_argument('--config_path', type=str,
                        default='../configs/default_config.json',
                        help='配置JSON文件路径')
    
    parser.add_argument('--output_dir', type=str,
                        default='result',
                        help='输出目录')
    
    parser.add_argument('--ncc_threshold', type=float,
                        default=0.95,
                        help='相位恢复早停的NCC阈值')
    
    parser.add_argument('--max_iterations', type=int,
                        default=50,
                        help='相位恢复的最大迭代次数')
    
    parser.add_argument('--min_distance', type=int,
                        default=25,
                        help='发射体之间的最小距离（像素）')
    
    parser.add_argument('--n_emitters', type=int,
                        default=200,
                        help='要检测的最大发射体数量')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 验证输入文件存在
    if not os.path.exists(args.tiff_path):
        logger.error(f"TIFF文件不存在: {args.tiff_path}")
        return 1
    
    if not os.path.exists(args.camera_params):
        logger.error(f"相机参数文件不存在: {args.camera_params}")
        return 1
    
    if not os.path.exists(args.config_path):
        logger.error(f"配置文件不存在: {args.config_path}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 更新全局变量
    pr.TIFF_PATH = args.tiff_path
    pr.CAMERA_PARAMS_PATH = args.camera_params
    pr.CONFIG_PATH = args.config_path
    pr.OUTPUT_DIR = args.output_dir
    pr.OUTPUT_H5 = os.path.join(args.output_dir, 'result.h5')
    pr.NCC_THRESHOLD = args.ncc_threshold
    pr.ITER_MAX = args.max_iterations
    pr.MIN_DISTANCE = args.min_distance
    
    # 运行主流程
    try:
        logger.info("开始运行相位恢复流程...")
        phase_retrieval_main()
        logger.info("相位恢复流程完成！")
        return 0
    except Exception as e:
        logger.error(f"处理过程中出错: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())