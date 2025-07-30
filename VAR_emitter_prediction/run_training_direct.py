#!/usr/bin/env python3
"""
直接运行VAR训练的脚本
不使用SLURM，直接在当前环境中运行训练
"""

import os
import sys
import logging
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from train_true_var import VARTrainer

def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 配置文件路径
    config_path = 'configs/config_true_var_slurm.json'
    
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        return 1
    
    try:
        logger.info("开始VAR训练...")
        logger.info(f"使用配置文件: {config_path}")
        
        # 创建训练器并开始训练
        trainer = VARTrainer(config_path)
        trainer.train()
        
        logger.info("训练完成！")
        return 0
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)