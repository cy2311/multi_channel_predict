"""训练器实现

包含标准训练器和分布式训练器，支持完整的训练流程。
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
import logging
from typing import Dict, List, Optional, Callable, Any, Union
from pathlib import Path

from ..models import SigmaMUNet, DoubleMUnet, SimpleSMLMNet
from ..loss import PPXYZBLoss, GaussianMMLoss, UnifiedLoss
from .callbacks import TrainingCallback
from ..utils.config import TrainingConfig
from .utils import save_checkpoint, load_checkpoint


class Trainer:
    """DECODE神经网络训练器
    
    支持完整的训练流程：
    - 模型训练和验证
    - 损失计算和优化
    - 检查点保存和恢复
    - TensorBoard日志
    - 回调系统
    - 早停和学习率调度
    
    Args:
        model: 神经网络模型
        loss_fn: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        config: 训练配置
        device: 训练设备
        logger: 日志记录器
    """
    
    def __init__(self,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 config: Optional[TrainingConfig] = None,
                 device: Optional[torch.device] = None,
                 logger: Optional[logging.Logger] = None):
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or TrainingConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger or self._setup_logger()
        
        # 移动模型到设备
        self.model.to(self.device)
        
        # 训练状态
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')
        self.best_metrics = {}
        
        # TensorBoard
        self.writer = None
        if hasattr(self.config, 'use_tensorboard') and self.config.use_tensorboard:
            log_dir = getattr(self.config, 'log_dir', 'runs')
            self.writer = SummaryWriter(log_dir=log_dir)
        
        # 回调系统
        self.callbacks: List[TrainingCallback] = []
        
        # 训练历史
        self.train_history = {
            'loss': [],
            'lr': [],
            'epoch_time': []
        }
        self.val_history = {
            'loss': [],
            'metrics': []
        }
    
    def add_callback(self, callback: TrainingCallback):
        """添加训练回调"""
        callback.set_trainer(self)
        self.callbacks.append(callback)
    
    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              num_epochs: Optional[int] = None,
              resume_from: Optional[str] = None) -> Dict[str, Any]:
        """开始训练
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            resume_from: 恢复训练的检查点路径
            
        Returns:
            训练历史字典
        """
        num_epochs = num_epochs or self.config.num_epochs
        
        # 恢复训练
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # 回调：训练开始
        for callback in self.callbacks:
            callback.on_train_begin()
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        try:
            for epoch in range(self.current_epoch, num_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # 回调：epoch开始
                for callback in self.callbacks:
                    callback.on_epoch_begin(epoch)
                
                # 训练一个epoch
                train_metrics = self._train_epoch(train_loader)
                
                # 验证
                val_metrics = {}
                if val_loader is not None:
                    val_metrics = self._validate_epoch(val_loader)
                
                # 学习率调度
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                    else:
                        self.scheduler.step()
                
                # 记录历史
                epoch_time = time.time() - epoch_start_time
                self.train_history['loss'].append(train_metrics['loss'])
                self.train_history['lr'].append(self.optimizer.param_groups[0]['lr'])
                self.train_history['epoch_time'].append(epoch_time)
                
                if val_metrics:
                    self.val_history['loss'].append(val_metrics['loss'])
                    self.val_history['metrics'].append(val_metrics)
                
                # TensorBoard日志
                if self.writer:
                    self._log_tensorboard(train_metrics, val_metrics, epoch)
                
                # 回调：epoch结束
                epoch_metrics = {**train_metrics, **val_metrics}
                for callback in self.callbacks:
                    callback.on_epoch_end(epoch, epoch_metrics)
                
                # 日志输出
                self._log_epoch_results(epoch, train_metrics, val_metrics, epoch_time)
                
                # 保存最佳模型
                current_loss = val_metrics.get('loss', train_metrics['loss'])
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.best_metrics = epoch_metrics
                    if getattr(self.config, 'save_best', True):
                        self.save_checkpoint('best_model.pth', is_best=True)
                
                # 定期保存检查点
                if (epoch + 1) % self.config.save_interval == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        finally:
            # 回调：训练结束
            for callback in self.callbacks:
                callback.on_train_end()
            
            # 关闭TensorBoard
            if self.writer:
                self.writer.close()
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_loss': self.best_loss,
            'best_metrics': self.best_metrics
        }
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        loss_components = {}
        
        for batch_idx, batch in enumerate(train_loader):
            # 回调：batch开始
            for callback in self.callbacks:
                callback.on_batch_begin(batch_idx)
            
            # 数据移动到设备
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                batch_kwargs = batch[2] if len(batch) > 2 else {}
            else:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                batch_kwargs = {k: v for k, v in batch.items() if k not in ['input', 'target']}
            
            batch_size = inputs.size(0)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # 计算损失（过滤掉不需要的参数）
            loss_kwargs = {k: v for k, v in batch_kwargs.items() 
                          if k in ['weight', 'mask', 'sample_weight']}
            try:
                loss_result = self.loss_fn(outputs, targets, **loss_kwargs)
            except Exception as e:
                print(f"Error computing ppxyzb loss: {e}")
                # 如果损失函数不接受额外参数，只传递基本参数
                loss_result = self.loss_fn(outputs, targets)
            
            if isinstance(loss_result, dict):
                loss = loss_result['total_loss']
                # 累积损失组件
                for key, value in loss_result.items():
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += value.item() * batch_size
            else:
                loss = loss_result
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            gradient_clip_val = getattr(self.config, 'gradient_clip_val', 0.0)
            if gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                             gradient_clip_val)
            
            # 优化器步骤
            self.optimizer.step()
            
            # 累积统计
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            self.current_step += 1
            
            # 回调：batch结束
            batch_metrics = {'loss': loss.item()}
            if isinstance(loss_result, dict):
                batch_metrics.update({k: v.item() for k, v in loss_result.items()})
            
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, batch_metrics)
            
            # 日志输出
            if (batch_idx + 1) % self.config.log_interval == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx+1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.6f}, LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )
        
        # 计算平均指标
        avg_loss = total_loss / total_samples
        metrics = {'loss': avg_loss}
        
        for key, value in loss_components.items():
            metrics[key] = value / total_samples
        
        return metrics
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        loss_components = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # 数据移动到设备
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                    batch_kwargs = batch[2] if len(batch) > 2 else {}
                else:
                    inputs = batch['input'].to(self.device)
                    targets = batch['target'].to(self.device)
                    batch_kwargs = {k: v for k, v in batch.items() if k not in ['input', 'target']}
                
                batch_size = inputs.size(0)
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失（过滤掉不需要的参数）
                loss_kwargs = {k: v for k, v in batch_kwargs.items() 
                              if k in ['weight', 'mask', 'sample_weight']}
                try:
                    loss_result = self.loss_fn(outputs, targets, **loss_kwargs)
                except Exception as e:
                    print(f"Error computing ppxyzb loss: {e}")
                    # 如果损失函数不接受额外参数，只传递基本参数
                    loss_result = self.loss_fn(outputs, targets)
                
                if isinstance(loss_result, dict):
                    loss = loss_result['total_loss']
                    # 累积损失组件
                    for key, value in loss_result.items():
                        if key not in loss_components:
                            loss_components[key] = 0.0
                        loss_components[key] += value.item() * batch_size
                else:
                    loss = loss_result
                
                # 累积统计
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        # 计算平均指标
        avg_loss = total_loss / total_samples
        metrics = {'loss': avg_loss}
        
        for key, value in loss_components.items():
            metrics[key] = value / total_samples
        
        return metrics
    
    def _log_tensorboard(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """记录TensorBoard日志"""
        # 训练指标
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        # 验证指标
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'Val/{key}', value, epoch)
        
        # 学习率
        self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        # 模型参数直方图
        if epoch % 10 == 0:  # 每10个epoch记录一次
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.writer.add_histogram(f'Parameters/{name}', param, epoch)
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
    
    def _log_epoch_results(self, epoch: int, train_metrics: Dict, val_metrics: Dict, epoch_time: float):
        """记录epoch结果"""
        num_epochs = getattr(self.config, 'num_epochs', 'N/A')
        log_msg = f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s - "
        log_msg += f"train_loss: {train_metrics['loss']:.6f}"
        
        if val_metrics:
            log_msg += f" - val_loss: {val_metrics['loss']:.6f}"
        
        log_msg += f" - lr: {self.optimizer.param_groups[0]['lr']:.2e}"
        
        self.logger.info(log_msg)
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """保存检查点"""
        checkpoint_dir = getattr(self.config, 'checkpoint_dir', 'checkpoints')
        checkpoint_path = Path(checkpoint_dir) / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'best_metrics': self.best_metrics,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            self.logger.info(f"Saved best model to {checkpoint_path}")
        else:
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.current_step = checkpoint['step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.best_metrics = checkpoint.get('best_metrics', {})
        self.train_history = checkpoint.get('train_history', {'loss': [], 'lr': [], 'epoch_time': []})
        self.val_history = checkpoint.get('val_history', {'loss': [], 'metrics': []})
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}, step {self.current_step}")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('DECODE_Trainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            
            # 文件处理器
            if self.config and hasattr(self.config, 'log_dir') and self.config.log_dir:
                log_dir = Path(self.config.log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(log_dir / 'training.log')
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        return logger


class DistributedTrainer(Trainer):
    """分布式训练器
    
    支持多GPU和多节点训练
    """
    
    def __init__(self,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 config: Optional[TrainingConfig] = None,
                 device: Optional[torch.device] = None,
                 logger: Optional[logging.Logger] = None,
                 local_rank: int = 0):
        
        super().__init__(model, loss_fn, optimizer, scheduler, config, device, logger)
        
        self.local_rank = local_rank
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # 包装模型为DDP
        if dist.is_initialized():
            self.model = DDP(self.model, device_ids=[local_rank])
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """分布式训练一个epoch"""
        # 设置sampler的epoch（用于数据打乱）
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(self.current_epoch)
        
        return super()._train_epoch(train_loader)
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """只在主进程保存检查点"""
        if self.local_rank == 0:
            super().save_checkpoint(filename, is_best)
    
    def _log_epoch_results(self, epoch: int, train_metrics: Dict, val_metrics: Dict, epoch_time: float):
        """只在主进程记录日志"""
        if self.local_rank == 0:
            super()._log_epoch_results(epoch, train_metrics, val_metrics, epoch_time)
    
    def _log_tensorboard(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """只在主进程记录TensorBoard"""
        if self.local_rank == 0:
            super()._log_tensorboard(train_metrics, val_metrics, epoch)