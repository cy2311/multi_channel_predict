#!/usr/bin/env python3
"""
DECODE双层神经网络训练脚本

支持使用TIFF图像和emitters.h5标注文件训练双层网络
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from models.first_level_unets import ThreeIndependentUNets
from models.second_level_network import SecondLevelNet
from loss.count_loss import CountLoss
from loss.loc_loss import LocLoss
from loss.background_loss import BackgroundLoss
from loss.improved_count_loss import ImprovedCountLoss, MultiLevelLoss
from utils.dataset import create_dataloaders


class DECODETrainer:
    """DECODE网络训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建模型
        self.first_level_net = ThreeIndependentUNets().to(self.device)
        self.second_level_net = SecondLevelNet().to(self.device)
        
        # 创建优化器
        self.optimizer_first = optim.Adam(
            self.first_level_net.parameters(),
            lr=config['training']['lr_first'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.optimizer_second = optim.Adam(
            self.second_level_net.parameters(),
            lr=config['training']['lr_second'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 创建损失函数
        self.count_loss = CountLoss()
        self.loc_loss = LocLoss()
        self.background_loss = BackgroundLoss()
        
        # 创建改进的损失函数
        self.improved_count_loss = ImprovedCountLoss(
            pos_weight=2.0,  # 前景权重，处理类别不平衡
            pixel_weight_strategy='adaptive',  # 自适应像素权重
            channel_weights=None  # 可根据需要设置通道权重
        )
        
        # 创建多层次损失函数
        self.multi_level_loss = MultiLevelLoss(
            count_pos_weight=2.0,
            pixel_weight_strategy='adaptive',
            loss_weights=config['training']['loss_weights']
        )
        
        # 损失权重
        self.loss_weights = config['training']['loss_weights']
        
        # 训练状态
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # 创建输出目录
        self.output_dir = Path(config['output']['save_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        print(f"模型参数数量:")
        print(f"  第一级网络: {sum(p.numel() for p in self.first_level_net.parameters()):,}")
        print(f"  第二级网络: {sum(p.numel() for p in self.second_level_net.parameters()):,}")
    
    def prepare_targets(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """准备训练目标
        
        将emitters数据转换为网络训练所需的目标格式
        """
        batch_size = batch['frames'].shape[0]
        image_size = self.config['data']['image_size']
        
        # 初始化目标张量
        targets = {
            'count_maps': torch.zeros(batch_size, 3, image_size, image_size, device=self.device),
            'loc_maps': torch.zeros(batch_size, 6, image_size, image_size, device=self.device),  # x,y for 3 frames
            'photon_maps': torch.zeros(batch_size, 3, image_size, image_size, device=self.device),
            'background_maps': torch.zeros(batch_size, 3, image_size, image_size, device=self.device)
        }
        
        # 为每个样本生成目标
        for b in range(batch_size):
            positions = batch['emitter_positions'][b]  # [N, 3]
            photons = batch['emitter_photons'][b]      # [N]
            frame_ids = batch['emitter_frame_ids'][b]  # [N]
            
            # 为每个发射器生成目标
            for i in range(len(positions)):
                if len(positions) == 0:
                    continue
                    
                x, y, z = positions[i]
                photon = photons[i]
                frame_id = frame_ids[i]
                
                # 转换为像素坐标
                px = int(round(x.item()))
                py = int(round(y.item()))
                
                # 检查边界
                if 0 <= px < image_size and 0 <= py < image_size and 0 <= frame_id < 3:
                    # 计数目标（高斯分布）
                    sigma = 1.5
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            nx, ny = px + dx, py + dy
                            if 0 <= nx < image_size and 0 <= ny < image_size:
                                weight = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                                targets['count_maps'][b, frame_id, ny, nx] = min(1.0, 
                                    targets['count_maps'][b, frame_id, ny, nx] + weight)
                    
                    # 定位目标
                    targets['loc_maps'][b, frame_id*2, py, px] = x - px      # x偏移
                    targets['loc_maps'][b, frame_id*2+1, py, px] = y - py    # y偏移
                    
                    # 光子数目标
                    targets['photon_maps'][b, frame_id, py, px] = photon
        
        return targets
    
    def forward_pass(self, frames: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播
        
        Parameters
        ----------
        frames : torch.Tensor
            输入帧 [batch_size, 3, H, W]
            
        Returns
        -------
        first_features : torch.Tensor
            第一级网络特征 [batch_size, 144, H, W]
        second_outputs : dict
            第二级网络输出
        """
        # 第一级网络
        first_features = self.first_level_net(frames)
        
        # 第二级网络
        second_outputs = self.second_level_net(first_features)
        
        return first_features, second_outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算损失"""
        losses = {}
        
        # 计数损失 - 使用改进的BCEWithLogitsLoss
        count_logits = outputs['prob']  # [B, 1, H, W] - 直接使用logits，不经过sigmoid
        count_target = targets['count_maps'][:, 0:1, :, :]  # 取第一帧 [B, 1, H, W]
        losses['count'] = self.improved_count_loss(count_logits, count_target)
        
        # 为后续损失计算生成概率图
        count_pred = torch.sigmoid(count_logits)  # [B, 1, H, W]
        
        # 定位损失 - 使用offset输出
        loc_pred = outputs['offset']            # [B, 3, H, W] (dx, dy, dz)
        loc_target = targets['loc_maps'][:, 0:3, :, :]  # 取前3个通道 [B, 3, H, W]
        # 只在有发射器的位置计算定位损失
        mask = count_target > 0.5  # [B, 1, H, W]
        mask = mask.expand(-1, 3, -1, -1)  # [B, 3, H, W]
        if mask.sum() > 0:
            losses['localization'] = nn.MSELoss()(loc_pred[mask], loc_target[mask])
        else:
            losses['localization'] = torch.tensor(0.0, device=loc_pred.device)
        
        # 光子数损失
        photon_pred = outputs['photon']         # [B, 1, H, W]
        photon_target = targets['photon_maps'][:, 0:1, :, :]  # 取第一帧 [B, 1, H, W]
        # 只在有发射器的位置计算光子损失
        mask_photon = count_target > 0.5  # [B, 1, H, W]
        if mask_photon.sum() > 0:
            losses['photon'] = nn.MSELoss()(photon_pred[mask_photon], photon_target[mask_photon])
        else:
            losses['photon'] = torch.tensor(0.0, device=photon_pred.device)
        
        # 背景损失
        bg_pred = outputs['background']         # [B, 1, H, W]
        bg_target = targets['background_maps'][:, 0:1, :, :]  # 取第一帧 [B, 1, H, W]
        losses['background'] = nn.MSELoss()(bg_pred, bg_target)
        
        # 总损失
        total_loss = (
            self.loss_weights['count'] * losses['count'] +
            self.loss_weights['localization'] * losses['localization'] +
            self.loss_weights['photon'] * losses['photon'] +
            self.loss_weights['background'] * losses['background']
        )
        
        losses['total'] = total_loss
        
        return losses
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """训练一个epoch"""
        self.first_level_net.train()
        self.second_level_net.train()
        
        epoch_losses = {'total': 0, 'count': 0, 'localization': 0, 'photon': 0, 'background': 0}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'训练 Epoch {self.epoch}')
        
        for batch in pbar:
            # 移动数据到设备
            frames = batch['frames'].to(self.device)  # [B, 3, H, W]
            
            # 准备目标
            targets = self.prepare_targets(batch)
            
            # 前向传播
            first_features, second_outputs = self.forward_pass(frames)
            
            # 计算损失
            losses = self.compute_loss(second_outputs, targets)
            
            # 反向传播
            self.optimizer_first.zero_grad()
            self.optimizer_second.zero_grad()
            
            losses['total'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.first_level_net.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.second_level_net.parameters(), max_norm=1.0)
            
            self.optimizer_first.step()
            self.optimizer_second.step()
            
            # 累积损失
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'count': f"{losses['count'].item():.4f}",
                'loc': f"{losses['localization'].item():.4f}"
            })
        
        # 计算平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """验证一个epoch"""
        self.first_level_net.eval()
        self.second_level_net.eval()
        
        epoch_losses = {'total': 0, 'count': 0, 'localization': 0, 'photon': 0, 'background': 0}
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'验证 Epoch {self.epoch}')
            
            for batch in pbar:
                # 移动数据到设备
                frames = batch['frames'].to(self.device)
                
                # 准备目标
                targets = self.prepare_targets(batch)
                
                # 前向传播
                first_features, second_outputs = self.forward_pass(frames)
                
                # 计算损失
                losses = self.compute_loss(second_outputs, targets)
                
                # 累积损失
                for key in epoch_losses:
                    epoch_losses[key] += losses[key].item()
                num_batches += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'val_loss': f"{losses['total'].item():.4f}"
                })
        
        # 计算平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'first_level_state_dict': self.first_level_net.state_dict(),
            'second_level_state_dict': self.second_level_net.state_dict(),
            'optimizer_first_state_dict': self.optimizer_first.state_dict(),
            'optimizer_second_state_dict': self.optimizer_second.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # 保存最新检查点
        torch.save(checkpoint, self.output_dir / 'latest_checkpoint.pth')
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_model.pth')
            print(f"保存最佳模型，验证损失: {self.best_val_loss:.6f}")
    
    def train(self, train_loader, val_loader):
        """主训练循环"""
        print(f"开始训练，共 {self.config['training']['epochs']} 个epoch")
        
        for epoch in range(self.config['training']['epochs']):
            self.epoch = epoch + 1
            
            # 训练
            train_losses = self.train_epoch(train_loader)
            
            # 验证
            val_losses = self.validate_epoch(val_loader)
            
            # 记录到TensorBoard
            for key in train_losses:
                self.writer.add_scalar(f'Train/{key}', train_losses[key], self.epoch)
                self.writer.add_scalar(f'Val/{key}', val_losses[key], self.epoch)
            
            # 打印结果
            print(f"Epoch {self.epoch}/{self.config['training']['epochs']}:")
            print(f"  训练损失: {train_losses['total']:.6f}")
            print(f"  验证损失: {val_losses['total']:.6f}")
            
            # 保存检查点
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
            
            self.save_checkpoint(is_best)
            
            # 学习率调度
            if self.epoch % 20 == 0:
                for param_group in self.optimizer_first.param_groups:
                    param_group['lr'] *= 0.8
                for param_group in self.optimizer_second.param_groups:
                    param_group['lr'] *= 0.8
                print(f"学习率调整为: {param_group['lr']:.6f}")
        
        self.writer.close()
        print("训练完成！")


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='训练DECODE双层神经网络')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--data_root', type=str, required=True, help='数据根目录')
    parser.add_argument('--sample_subset', type=int, help='使用的样本子集数量')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖命令行参数
    if args.output_dir:
        config['output']['save_dir'] = args.output_dir
    if args.sample_subset:
        config['data']['sample_subset'] = args.sample_subset
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        sample_subset=config['data'].get('sample_subset'),
        image_size=config['data']['image_size'],
        consecutive_frames=config['data']['consecutive_frames']
    )
    
    print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")
    
    # 创建训练器
    trainer = DECODETrainer(config)
    
    # 开始训练
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()