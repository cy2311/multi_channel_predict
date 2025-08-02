"""多通道训练器

实现多通道DECODE网络的三阶段训练策略：
1. 双通道独立训练
2. 比例网络训练
3. 联合微调

支持物理约束和不确定性量化。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import json
from tqdm import tqdm

from ..models.sigma_munet import SigmaMUNet
from ..models.ratio_net import RatioNet
from ..loss.ppxyzb_loss import PPXYZBLoss
from ..loss.gaussian_mm_loss import GaussianMMLoss
from ..loss.ratio_loss import RatioGaussianNLLLoss, MultiChannelLossWithGaussianRatio
from ..evaluation.multi_channel_evaluation import MultiChannelEvaluation
from ..training.trainer import Trainer


class MultiChannelTrainer:
    """多通道DECODE网络训练器
    
    实现三阶段训练策略：
    1. Stage 1: 双通道独立训练 - 分别训练两个通道的SigmaMUNet
    2. Stage 2: 比例网络训练 - 训练RatioNet预测光子数分配比例
    3. Stage 3: 联合微调 - 端到端联合优化所有组件
    
    Args:
        config: 训练配置字典
        device: 计算设备
        logger: 日志记录器
    """
    
    def __init__(self, 
                 config: Dict[str, Any], 
                 device: str = 'cuda',
                 logger: Optional[logging.Logger] = None):
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # 初始化模型
        self.channel1_net = None
        self.channel2_net = None
        self.ratio_net = None
        
        # 初始化损失函数
        self.channel_loss = None
        self.ratio_loss = None
        self.joint_loss = None
        
        # 初始化优化器
        self.optimizers = {}
        self.schedulers = {}
        
        # 训练状态
        self.current_stage = 1
        self.stage_epochs = {
            1: config.get('stage1_epochs', 100),
            2: config.get('stage2_epochs', 50),
            3: config.get('stage3_epochs', 30)
        }
        
        # 评估器
        self.evaluator = MultiChannelEvaluation(device=device)
        
        # 训练历史
        self.training_history = {
            'stage1': {'train_loss': [], 'val_loss': [], 'metrics': []},
            'stage2': {'train_loss': [], 'val_loss': [], 'metrics': []},
            'stage3': {'train_loss': [], 'val_loss': [], 'metrics': []}
        }
        
        self._setup_models()
        self._setup_losses()
    
    def _setup_models(self):
        """初始化模型"""
        model_config = self.config.get('model', {})
        
        # 双通道SigmaMUNet
        self.channel1_net = SigmaMUNet(
            n_inp=model_config.get('n_inp', 1),
            n_out=model_config.get('n_out', 10),
            **model_config.get('sigma_munet_params', {})
        ).to(self.device)
        
        self.channel2_net = SigmaMUNet(
            n_inp=model_config.get('n_inp', 1),
            n_out=model_config.get('n_out', 10),
            **model_config.get('sigma_munet_params', {})
        ).to(self.device)
        
        # 比例网络
        ratio_config = model_config.get('ratio_net', {})
        self.ratio_net = RatioNet(
            input_channels=ratio_config.get('input_channels', 20),  # 两个通道的特征
            hidden_dim=ratio_config.get('hidden_dim', 128),
            num_layers=ratio_config.get('num_layers', 3),
            dropout=ratio_config.get('dropout', 0.1)
        ).to(self.device)
        
        self.logger.info(f"Models initialized: Channel1 ({sum(p.numel() for p in self.channel1_net.parameters())} params), "
                        f"Channel2 ({sum(p.numel() for p in self.channel2_net.parameters())} params), "
                        f"RatioNet ({sum(p.numel() for p in self.ratio_net.parameters())} params)")
    
    def _setup_losses(self):
        """初始化损失函数"""
        loss_config = self.config.get('loss', {})
        
        # 单通道损失
        loss_type = loss_config.get('type', 'ppxyzb')
        if loss_type == 'ppxyzb':
            self.channel_loss = PPXYZBLoss(
                **loss_config.get('ppxyzb_params', {})
            )
        elif loss_type == 'gaussian_mm':
            self.channel_loss = GaussianMMLoss(
                **loss_config.get('gaussian_mm_params', {})
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        # 比例损失
        self.ratio_loss = RatioGaussianNLLLoss(
            **loss_config.get('ratio_params', {})
        )
        
        # 联合损失
        self.joint_loss = MultiChannelLossWithGaussianRatio(
            channel_loss=self.channel_loss,
            ratio_loss=self.ratio_loss,
            **loss_config.get('joint_params', {})
        )
    
    def _setup_optimizers(self, stage: int):
        """为指定阶段设置优化器"""
        opt_config = self.config.get('optimizer', {})
        lr = opt_config.get('lr', 1e-3)
        weight_decay = opt_config.get('weight_decay', 1e-4)
        
        if stage == 1:
            # 阶段1：独立训练两个通道
            self.optimizers['channel1'] = optim.Adam(
                self.channel1_net.parameters(), lr=lr, weight_decay=weight_decay
            )
            self.optimizers['channel2'] = optim.Adam(
                self.channel2_net.parameters(), lr=lr, weight_decay=weight_decay
            )
            
            # 学习率调度器
            self.schedulers['channel1'] = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizers['channel1'], patience=10, factor=0.5
            )
            self.schedulers['channel2'] = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizers['channel2'], patience=10, factor=0.5
            )
            
        elif stage == 2:
            # 阶段2：训练比例网络（冻结通道网络）
            for param in self.channel1_net.parameters():
                param.requires_grad = False
            for param in self.channel2_net.parameters():
                param.requires_grad = False
            
            self.optimizers['ratio'] = optim.Adam(
                self.ratio_net.parameters(), lr=lr * 0.1, weight_decay=weight_decay
            )
            
            self.schedulers['ratio'] = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizers['ratio'], patience=5, factor=0.7
            )
            
        elif stage == 3:
            # 阶段3：联合微调（解冻所有网络）
            for param in self.channel1_net.parameters():
                param.requires_grad = True
            for param in self.channel2_net.parameters():
                param.requires_grad = True
            
            # 使用不同的学习率
            self.optimizers['joint'] = optim.Adam([
                {'params': self.channel1_net.parameters(), 'lr': lr * 0.01},
                {'params': self.channel2_net.parameters(), 'lr': lr * 0.01},
                {'params': self.ratio_net.parameters(), 'lr': lr * 0.1}
            ], weight_decay=weight_decay)
            
            self.schedulers['joint'] = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizers['joint'], patience=5, factor=0.8
            )
    
    def train_stage1(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """阶段1：双通道独立训练"""
        self.logger.info("Starting Stage 1: Independent channel training")
        self.current_stage = 1
        self._setup_optimizers(1)
        
        history = {'train_loss': [], 'val_loss': [], 'metrics': []}
        
        for epoch in range(self.stage_epochs[1]):
            # 训练
            train_loss = self._train_epoch_stage1(train_loader)
            
            # 验证
            val_loss, val_metrics = self._validate_stage1(val_loader)
            
            # 记录
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['metrics'].append(val_metrics)
            
            # 学习率调度
            self.schedulers['channel1'].step(val_loss['channel1'])
            self.schedulers['channel2'].step(val_loss['channel2'])
            
            # 日志
            if epoch % 10 == 0:
                self.logger.info(
                    f"Stage 1 Epoch {epoch}: Train Loss = {train_loss['total']:.4f}, "
                    f"Val Loss = {val_loss['total']:.4f}"
                )
        
        self.training_history['stage1'] = history
        self.logger.info("Stage 1 completed")
        return history
    
    def train_stage2(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """阶段2：比例网络训练"""
        self.logger.info("Starting Stage 2: Ratio network training")
        self.current_stage = 2
        self._setup_optimizers(2)
        
        history = {'train_loss': [], 'val_loss': [], 'metrics': []}
        
        for epoch in range(self.stage_epochs[2]):
            # 训练
            train_loss = self._train_epoch_stage2(train_loader)
            
            # 验证
            val_loss, val_metrics = self._validate_stage2(val_loader)
            
            # 记录
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['metrics'].append(val_metrics)
            
            # 学习率调度
            self.schedulers['ratio'].step(val_loss['ratio'])
            
            # 日志
            if epoch % 5 == 0:
                self.logger.info(
                    f"Stage 2 Epoch {epoch}: Train Loss = {train_loss['ratio']:.4f}, "
                    f"Val Loss = {val_loss['ratio']:.4f}"
                )
        
        self.training_history['stage2'] = history
        self.logger.info("Stage 2 completed")
        return history
    
    def train_stage3(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """阶段3：联合微调"""
        self.logger.info("Starting Stage 3: Joint fine-tuning")
        self.current_stage = 3
        self._setup_optimizers(3)
        
        history = {'train_loss': [], 'val_loss': [], 'metrics': []}
        
        for epoch in range(self.stage_epochs[3]):
            # 训练
            train_loss = self._train_epoch_stage3(train_loader)
            
            # 验证
            val_loss, val_metrics = self._validate_stage3(val_loader)
            
            # 记录
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['metrics'].append(val_metrics)
            
            # 学习率调度
            self.schedulers['joint'].step(val_loss['total'])
            
            # 日志
            if epoch % 5 == 0:
                self.logger.info(
                    f"Stage 3 Epoch {epoch}: Train Loss = {train_loss['total']:.4f}, "
                    f"Val Loss = {val_loss['total']:.4f}"
                )
        
        self.training_history['stage3'] = history
        self.logger.info("Stage 3 completed")
        return history
    
    def _train_epoch_stage1(self, train_loader: DataLoader) -> Dict[str, float]:
        """阶段1训练一个epoch"""
        self.channel1_net.train()
        self.channel2_net.train()
        
        total_loss_ch1 = 0.0
        total_loss_ch2 = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Stage 1 Training"):
            # 数据准备
            input_ch1 = batch['channel1_input'].to(self.device)
            target_ch1 = batch['channel1_target'].to(self.device)
            input_ch2 = batch['channel2_input'].to(self.device)
            target_ch2 = batch['channel2_target'].to(self.device)
            
            # 通道1训练
            self.optimizers['channel1'].zero_grad()
            pred_ch1 = self.channel1_net(input_ch1)
            loss_ch1 = self.channel_loss(pred_ch1, target_ch1)
            loss_ch1.backward()
            self.optimizers['channel1'].step()
            
            # 通道2训练
            self.optimizers['channel2'].zero_grad()
            pred_ch2 = self.channel2_net(input_ch2)
            loss_ch2 = self.channel_loss(pred_ch2, target_ch2)
            loss_ch2.backward()
            self.optimizers['channel2'].step()
            
            total_loss_ch1 += loss_ch1.item()
            total_loss_ch2 += loss_ch2.item()
            num_batches += 1
        
        avg_loss_ch1 = total_loss_ch1 / num_batches
        avg_loss_ch2 = total_loss_ch2 / num_batches
        
        return {
            'channel1': avg_loss_ch1,
            'channel2': avg_loss_ch2,
            'total': avg_loss_ch1 + avg_loss_ch2
        }
    
    def _train_epoch_stage2(self, train_loader: DataLoader) -> Dict[str, float]:
        """阶段2训练一个epoch"""
        self.channel1_net.eval()
        self.channel2_net.eval()
        self.ratio_net.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Stage 2 Training"):
            # 数据准备
            input_ch1 = batch['channel1_input'].to(self.device)
            input_ch2 = batch['channel2_input'].to(self.device)
            true_ratio = batch['ratio'].to(self.device)
            
            # 提取特征（不计算梯度）
            with torch.no_grad():
                pred_ch1 = self.channel1_net(input_ch1)
                pred_ch2 = self.channel2_net(input_ch2)
            
            # 比例预测
            self.optimizers['ratio'].zero_grad()
            ratio_mean, ratio_std = self.ratio_net(pred_ch1, pred_ch2)
            loss = self.ratio_loss(ratio_mean, ratio_std, true_ratio)
            loss.backward()
            self.optimizers['ratio'].step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {'ratio': total_loss / num_batches}
    
    def _train_epoch_stage3(self, train_loader: DataLoader) -> Dict[str, float]:
        """阶段3训练一个epoch"""
        self.channel1_net.train()
        self.channel2_net.train()
        self.ratio_net.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Stage 3 Training"):
            # 数据准备
            input_ch1 = batch['channel1_input'].to(self.device)
            target_ch1 = batch['channel1_target'].to(self.device)
            input_ch2 = batch['channel2_input'].to(self.device)
            target_ch2 = batch['channel2_target'].to(self.device)
            true_ratio = batch['ratio'].to(self.device)
            
            # 前向传播
            self.optimizers['joint'].zero_grad()
            
            pred_ch1 = self.channel1_net(input_ch1)
            pred_ch2 = self.channel2_net(input_ch2)
            ratio_mean, ratio_std = self.ratio_net(pred_ch1, pred_ch2)
            
            # 联合损失计算
            loss = self.joint_loss(
                pred_ch1, target_ch1,
                pred_ch2, target_ch2,
                ratio_mean, ratio_std, true_ratio
            )
            
            loss.backward()
            self.optimizers['joint'].step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {'total': total_loss / num_batches}
    
    def _validate_stage1(self, val_loader: DataLoader) -> Tuple[Dict[str, float], Dict]:
        """阶段1验证"""
        self.channel1_net.eval()
        self.channel2_net.eval()
        
        total_loss_ch1 = 0.0
        total_loss_ch2 = 0.0
        num_batches = 0
        
        all_predictions = {'channel1': [], 'channel2': []}
        all_targets = {'channel1': [], 'channel2': []}
        
        with torch.no_grad():
            for batch in val_loader:
                input_ch1 = batch['channel1_input'].to(self.device)
                target_ch1 = batch['channel1_target'].to(self.device)
                input_ch2 = batch['channel2_input'].to(self.device)
                target_ch2 = batch['channel2_target'].to(self.device)
                
                pred_ch1 = self.channel1_net(input_ch1)
                pred_ch2 = self.channel2_net(input_ch2)
                
                loss_ch1 = self.channel_loss(pred_ch1, target_ch1)
                loss_ch2 = self.channel_loss(pred_ch2, target_ch2)
                
                total_loss_ch1 += loss_ch1.item()
                total_loss_ch2 += loss_ch2.item()
                num_batches += 1
                
                # 收集预测结果用于评估
                all_predictions['channel1'].append(pred_ch1.cpu())
                all_predictions['channel2'].append(pred_ch2.cpu())
                all_targets['channel1'].append(target_ch1.cpu())
                all_targets['channel2'].append(target_ch2.cpu())
        
        # 计算评估指标
        pred_results = {
            'channel1': torch.cat(all_predictions['channel1'], dim=0),
            'channel2': torch.cat(all_predictions['channel2'], dim=0)
        }
        ground_truth = {
            'channel1': torch.cat(all_targets['channel1'], dim=0),
            'channel2': torch.cat(all_targets['channel2'], dim=0)
        }
        
        metrics = self.evaluator.evaluate(pred_results, ground_truth)
        
        avg_loss_ch1 = total_loss_ch1 / num_batches
        avg_loss_ch2 = total_loss_ch2 / num_batches
        
        loss_dict = {
            'channel1': avg_loss_ch1,
            'channel2': avg_loss_ch2,
            'total': avg_loss_ch1 + avg_loss_ch2
        }
        
        return loss_dict, metrics
    
    def _validate_stage2(self, val_loader: DataLoader) -> Tuple[Dict[str, float], Dict]:
        """阶段2验证"""
        self.channel1_net.eval()
        self.channel2_net.eval()
        self.ratio_net.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        all_ratio_means = []
        all_ratio_stds = []
        all_true_ratios = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ch1 = batch['channel1_input'].to(self.device)
                input_ch2 = batch['channel2_input'].to(self.device)
                true_ratio = batch['ratio'].to(self.device)
                
                pred_ch1 = self.channel1_net(input_ch1)
                pred_ch2 = self.channel2_net(input_ch2)
                ratio_mean, ratio_std = self.ratio_net(pred_ch1, pred_ch2)
                
                loss = self.ratio_loss(ratio_mean, ratio_std, true_ratio)
                total_loss += loss.item()
                num_batches += 1
                
                # 收集预测结果
                all_ratio_means.append(ratio_mean.cpu())
                all_ratio_stds.append(ratio_std.cpu())
                all_true_ratios.append(true_ratio.cpu())
        
        # 计算评估指标
        pred_results = {
            'mean': torch.cat(all_ratio_means, dim=0),
            'std': torch.cat(all_ratio_stds, dim=0)
        }
        ground_truth = {
            'ratio': torch.cat(all_true_ratios, dim=0)
        }
        
        metrics = self.evaluator.evaluate(pred_results, ground_truth)
        
        return {'ratio': total_loss / num_batches}, metrics
    
    def _validate_stage3(self, val_loader: DataLoader) -> Tuple[Dict[str, float], Dict]:
        """阶段3验证"""
        self.channel1_net.eval()
        self.channel2_net.eval()
        self.ratio_net.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = {
            'channel1': [], 'channel2': [],
            'mean': [], 'std': []
        }
        all_targets = {
            'channel1': [], 'channel2': [], 'ratio': []
        }
        
        with torch.no_grad():
            for batch in val_loader:
                input_ch1 = batch['channel1_input'].to(self.device)
                target_ch1 = batch['channel1_target'].to(self.device)
                input_ch2 = batch['channel2_input'].to(self.device)
                target_ch2 = batch['channel2_target'].to(self.device)
                true_ratio = batch['ratio'].to(self.device)
                
                pred_ch1 = self.channel1_net(input_ch1)
                pred_ch2 = self.channel2_net(input_ch2)
                ratio_mean, ratio_std = self.ratio_net(pred_ch1, pred_ch2)
                
                loss = self.joint_loss(
                    pred_ch1, target_ch1,
                    pred_ch2, target_ch2,
                    ratio_mean, ratio_std, true_ratio
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                # 收集预测结果
                all_predictions['channel1'].append(pred_ch1.cpu())
                all_predictions['channel2'].append(pred_ch2.cpu())
                all_predictions['mean'].append(ratio_mean.cpu())
                all_predictions['std'].append(ratio_std.cpu())
                all_targets['channel1'].append(target_ch1.cpu())
                all_targets['channel2'].append(target_ch2.cpu())
                all_targets['ratio'].append(true_ratio.cpu())
        
        # 计算评估指标
        pred_results = {
            'channel1': torch.cat(all_predictions['channel1'], dim=0),
            'channel2': torch.cat(all_predictions['channel2'], dim=0),
            'mean': torch.cat(all_predictions['mean'], dim=0),
            'std': torch.cat(all_predictions['std'], dim=0)
        }
        ground_truth = {
            'channel1': torch.cat(all_targets['channel1'], dim=0),
            'channel2': torch.cat(all_targets['channel2'], dim=0),
            'ratio': torch.cat(all_targets['ratio'], dim=0)
        }
        
        metrics = self.evaluator.evaluate(pred_results, ground_truth)
        
        return {'total': total_loss / num_batches}, metrics
    
    def train_full_pipeline(self, 
                          train_loader: DataLoader, 
                          val_loader: DataLoader,
                          save_dir: Optional[str] = None) -> Dict[str, Any]:
        """执行完整的三阶段训练流程"""
        self.logger.info("Starting full multi-channel training pipeline")
        
        # 阶段1：双通道独立训练
        stage1_history = self.train_stage1(train_loader, val_loader)
        
        # 保存阶段1模型
        if save_dir:
            self.save_stage_models(save_dir, stage=1)
        
        # 阶段2：比例网络训练
        stage2_history = self.train_stage2(train_loader, val_loader)
        
        # 保存阶段2模型
        if save_dir:
            self.save_stage_models(save_dir, stage=2)
        
        # 阶段3：联合微调
        stage3_history = self.train_stage3(train_loader, val_loader)
        
        # 保存最终模型
        if save_dir:
            self.save_stage_models(save_dir, stage=3)
            self.save_training_history(save_dir)
        
        self.logger.info("Full training pipeline completed")
        
        return {
            'stage1': stage1_history,
            'stage2': stage2_history,
            'stage3': stage3_history,
            'training_history': self.training_history
        }
    
    def save_stage_models(self, save_dir: str, stage: int):
        """保存指定阶段的模型"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型状态
        torch.save({
            'channel1_net': self.channel1_net.state_dict(),
            'channel2_net': self.channel2_net.state_dict(),
            'ratio_net': self.ratio_net.state_dict(),
            'stage': stage,
            'config': self.config
        }, save_path / f'stage{stage}_models.pth')
        
        self.logger.info(f"Stage {stage} models saved to {save_path}")
    
    def save_training_history(self, save_dir: str):
        """保存训练历史"""
        save_path = Path(save_dir)
        with open(save_path / 'training_history.json', 'w') as f:
            # 转换tensor为list以便JSON序列化
            history_serializable = {}
            for stage, data in self.training_history.items():
                history_serializable[stage] = {
                    'train_loss': data['train_loss'],
                    'val_loss': data['val_loss'],
                    'metrics': [self._serialize_metrics(m) for m in data['metrics']]
                }
            json.dump(history_serializable, f, indent=2)
        
        self.logger.info(f"Training history saved to {save_path}")
    
    def _serialize_metrics(self, metrics: Dict) -> Dict:
        """序列化metrics字典中的tensor"""
        serialized = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                serialized[key] = self._serialize_metrics(value)
            elif isinstance(value, torch.Tensor):
                serialized[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                serialized[key] = [v.tolist() if isinstance(v, torch.Tensor) else v for v in value]
            else:
                serialized[key] = value
        return serialized
    
    def load_models(self, model_path: str, stage: Optional[int] = None):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.channel1_net.load_state_dict(checkpoint['channel1_net'])
        self.channel2_net.load_state_dict(checkpoint['channel2_net'])
        self.ratio_net.load_state_dict(checkpoint['ratio_net'])
        
        if stage is not None:
            self.current_stage = stage
        else:
            self.current_stage = checkpoint.get('stage', 3)
        
        self.logger.info(f"Models loaded from {model_path}, current stage: {self.current_stage}")