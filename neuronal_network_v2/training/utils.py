"""训练工具模块

该模块提供训练过程中需要的工具函数，包括：
- 检查点保存和加载
- 模型状态管理
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   checkpoint_path: str,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   **kwargs) -> None:
    """保存训练检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮次
        loss: 当前损失
        checkpoint_path: 检查点保存路径
        scheduler: 学习率调度器（可选）
        **kwargs: 其他需要保存的信息
    """
    try:
        # 确保保存目录存在
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 准备检查点数据
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            **kwargs
        }
        
        # 如果有调度器，保存其状态
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # 保存检查点
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"检查点已保存到: {checkpoint_path}")
        
    except Exception as e:
        logger.error(f"保存检查点失败: {e}")
        raise


def load_checkpoint(checkpoint_path: str,
                   model: nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   device: Optional[torch.device] = None) -> Dict[str, Any]:
    """加载训练检查点
    
    Args:
        checkpoint_path: 检查点文件路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 设备（可选）
        
    Returns:
        Dict[str, Any]: 检查点信息
    """
    try:
        # 检查文件是否存在
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        # 加载检查点
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"模型状态已从检查点加载")
        
        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"优化器状态已从检查点加载")
        
        # 加载调度器状态
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info(f"调度器状态已从检查点加载")
        
        logger.info(f"检查点已从 {checkpoint_path} 加载")
        logger.info(f"轮次: {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"损失: {checkpoint.get('loss', 'unknown')}")
        
        return checkpoint
        
    except Exception as e:
        logger.error(f"加载检查点失败: {e}")
        raise


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """获取最新的检查点文件
    
    Args:
        checkpoint_dir: 检查点目录
        
    Returns:
        Optional[str]: 最新检查点文件路径，如果没有则返回None
    """
    try:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return None
        
        # 查找所有.pth文件
        checkpoint_files = list(checkpoint_dir.glob('*.pth'))
        if not checkpoint_files:
            return None
        
        # 按修改时间排序，返回最新的
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        return str(latest_checkpoint)
        
    except Exception as e:
        logger.error(f"获取最新检查点失败: {e}")
        return None


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last: int = 5) -> None:
    """清理旧的检查点文件
    
    Args:
        checkpoint_dir: 检查点目录
        keep_last: 保留最新的检查点数量
    """
    try:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return
        
        # 查找所有.pth文件
        checkpoint_files = list(checkpoint_dir.glob('*.pth'))
        if len(checkpoint_files) <= keep_last:
            return
        
        # 按修改时间排序
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # 删除多余的检查点
        for checkpoint_file in checkpoint_files[keep_last:]:
            checkpoint_file.unlink()
            logger.info(f"已删除旧检查点: {checkpoint_file}")
        
        logger.info(f"检查点清理完成，保留了最新的 {keep_last} 个文件")
        
    except Exception as e:
        logger.error(f"清理检查点失败: {e}")