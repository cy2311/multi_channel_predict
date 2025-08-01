"""训练回调函数实现

提供训练过程中的各种回调功能，包括早停、模型检查点保存、学习率记录等。
"""

import os
import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path


class TrainingCallback:
    """训练回调基类"""
    
    def __init__(self):
        self.trainer = None
    
    def set_trainer(self, trainer):
        """设置训练器引用"""
        self.trainer = trainer
    
    def on_train_begin(self):
        """训练开始时调用"""
        pass
    
    def on_train_end(self):
        """训练结束时调用"""
        pass
    
    def on_epoch_begin(self, epoch: int):
        """每个epoch开始时调用"""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """每个epoch结束时调用"""
        pass
    
    def on_batch_begin(self, batch_idx: int):
        """每个batch开始时调用"""
        pass
    
    def on_batch_end(self, batch_idx: int, metrics: Dict[str, Any]):
        """每个batch结束时调用"""
        pass


class EarlyStopping(TrainingCallback):
    """早停回调"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 monitor: str = 'val_loss', mode: str = 'min'):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = lambda a, b: a < b - min_delta
            self.best_score = float('inf')
        else:
            self.monitor_op = lambda a, b: a > b + min_delta
            self.best_score = float('-inf')
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        current_score = metrics.get(self.monitor)
        if current_score is None:
            logging.warning(f"早停监控指标 '{self.monitor}' 不存在于metrics中")
            return
        
        if self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            logging.info(f"早停触发：{self.patience}个epoch内{self.monitor}无改善")


class ModelCheckpoint(TrainingCallback):
    """模型检查点保存回调"""
    
    def __init__(self, filepath: str, monitor: str = 'val_loss', 
                 mode: str = 'min', save_best_only: bool = True,
                 save_freq: int = 1):
        super().__init__()
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.best_score = None
        
        # 创建保存目录
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if mode == 'min':
            self.monitor_op = lambda a, b: a < b
            self.best_score = float('inf')
        else:
            self.monitor_op = lambda a, b: a > b
            self.best_score = float('-inf')
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        if self.save_best_only:
            current_score = metrics.get(self.monitor)
            if current_score is None:
                logging.warning(f"检查点监控指标 '{self.monitor}' 不存在于metrics中")
                return
                
            if self.monitor_op(current_score, self.best_score):
                self.best_score = current_score
                self._save_checkpoint(epoch, metrics)
                logging.info(f"保存最佳模型检查点：{self.monitor}={current_score:.6f}")
        else:
            if epoch % self.save_freq == 0:
                self._save_checkpoint(epoch, metrics)
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, Any]):
        """保存检查点"""
        if self.trainer is None:
            return
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.trainer.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'metrics': metrics,
            'best_score': self.best_score
        }
        
        if hasattr(self.trainer, 'scheduler') and self.trainer.scheduler:
            checkpoint['scheduler_state_dict'] = self.trainer.scheduler.state_dict()
        
        torch.save(checkpoint, self.filepath)


class LearningRateLogger(TrainingCallback):
    """学习率记录回调"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        if self.trainer and hasattr(self.trainer, 'optimizer'):
            for i, param_group in enumerate(self.trainer.optimizer.param_groups):
                lr = param_group['lr']
                self.logger.info(f"Epoch {epoch}, 参数组 {i} 学习率: {lr:.2e}")