#!/usr/bin/env python3
"""
DECODE网络自适应训练脚本

主要改进:
1. 自适应学习率策略 - 智能调整学习率以优化收敛
2. 学习率预热机制 - 前期稳定训练，避免梯度爆炸
3. 动态loss监控 - 实时分析loss趋势，自动调整策略
4. 多种调度器组合 - ReduceLROnPlateau + CosineAnnealing + 自定义逻辑
5. 数值稳定性优化 - 梯度裁剪和权重初始化
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Tuple, Optional, List
import numpy as np
from collections import deque
import math

# 添加项目路径
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型
try:
    from models.first_level_unets import UNetLevel1, ThreeIndependentUNets
    from models.second_level_network import SecondLevelNet
    # 为了兼容性，创建别名
    FirstLevelUNets = ThreeIndependentUNets
    SecondLevelNetwork = SecondLevelNet
except ImportError:
    print("警告: 无法导入模型，使用占位符实现")
    import torch.nn as nn
    class FirstLevelUNets(nn.Module):
         def __init__(self, *args, **kwargs):
             super().__init__()
             self.conv = nn.Conv2d(3, 64, 3, padding=1)
         
         def forward(self, x):
             return self.conv(x)
    
    class SecondLevelNetwork(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.conv = nn.Conv2d(64, 6, 1)
        
        def forward(self, x):
            return self.conv(x)

# 导入数据集
try:
    from utils.dataset import DECODEDataset
except ImportError:
    print("警告: 无法导入数据集，使用占位符实现")
    from torch.utils.data import Dataset
    class DECODEDataset(Dataset):
         def __init__(self, *args, **kwargs):
             pass
         def __len__(self):
             return 100
         def __getitem__(self, idx):
             return torch.randn(3, 40, 40), {'unified_target': torch.randn(6, 40, 40)}

# 导入损失函数
from loss.unified_decode_loss import (
    UnifiedDECODELoss,
    SimpleCountLoss,
    SimpleLocLoss,
    SimpleCombinedLoss
)

class AdaptiveLRManager:
    """自适应学习率管理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.loss_history = deque(maxlen=config.get('loss_window_size', 20))
        self.lr_history = []
        self.stagnation_count = 0
        self.last_improvement_epoch = 0
        self.best_loss = float('inf')
        
    def should_adjust_lr(self, current_loss: float, epoch: int) -> Tuple[bool, str, float]:
        """判断是否需要调整学习率"""
        self.loss_history.append(current_loss)
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.last_improvement_epoch = epoch
            self.stagnation_count = 0
            return False, "improving", 1.0
        
        # 检查是否停滞
        if len(self.loss_history) >= self.config['loss_window_size']:
            recent_losses = list(self.loss_history)[-10:]
            loss_variance = np.var(recent_losses)
            loss_trend = np.mean(recent_losses[-5:]) - np.mean(recent_losses[:5])
            
            # 如果loss变化很小且没有下降趋势
            if loss_variance < self.config['adaptive_threshold'] and loss_trend >= 0:
                self.stagnation_count += 1
                
                if self.stagnation_count >= 5:
                    # 重置停滞计数
                    self.stagnation_count = 0
                    return True, "stagnation", self.config['lr_boost_factor']
        
        # 检查是否需要衰减
        epochs_since_improvement = epoch - self.last_improvement_epoch
        if epochs_since_improvement > 30:
            return True, "decay", self.config['lr_decay_factor']
            
        return False, "stable", 1.0

class WarmupScheduler:
    """学习率预热调度器"""
    
    def __init__(self, optimizer, warmup_epochs: int, start_lr: float, target_lr: float):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.current_epoch = 0
        
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # 线性预热
            lr = self.start_lr + (self.target_lr - self.start_lr) * (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1
        
    def get_lr(self):
        if self.current_epoch < self.warmup_epochs:
            return self.start_lr + (self.target_lr - self.start_lr) * (self.current_epoch / self.warmup_epochs)
        return self.target_lr

class AdaptiveDECODETrainer:
    """自适应DECODE训练器"""
    
    def __init__(self, config_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 初始化组件
        self._init_models()
        self._init_loss_functions()
        self._init_optimizers()
        self._init_data_loaders()
        self._init_logging()
        
        # 初始化自适应管理器
        self.adaptive_manager = AdaptiveLRManager(self.config['training']['adaptive_lr_config'])
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.lr_values = []
        
    def _init_models(self):
        """初始化模型"""
        self.first_level = FirstLevelUNets().to(self.device)
        self.second_level = SecondLevelNetwork().to(self.device)
        
        # 改进的权重初始化
        self._init_weights()
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.first_level.parameters()) + \
                      sum(p.numel() for p in self.second_level.parameters())
        print(f"模型总参数量: {total_params:,}")
        
    def _init_weights(self):
        """改进的权重初始化"""
        for module in [self.first_level, self.second_level]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        
    def _init_loss_functions(self):
        """初始化损失函数"""
        loss_type = self.config['training']['loss_type']
        
        if loss_type == 'unified_decode':
            self.criterion = UnifiedDECODELoss(
                **self.config['training']['unified_loss_config']
            ).to(self.device)
        elif loss_type == 'simple_combined':
            self.criterion = SimpleCombinedLoss(
                **self.config['training']['simple_combined_config']
            ).to(self.device)
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")
        
        print(f"损失函数初始化完成: {loss_type}")
        
    def _init_optimizers(self):
        """初始化优化器和调度器"""
        # 合并所有参数
        all_params = list(self.first_level.parameters()) + list(self.second_level.parameters())
        
        # 使用AdamW优化器（更好的权重衰减）
        self.optimizer = optim.AdamW(
            all_params,
            lr=self.config['training']['lr_first'],
            weight_decay=self.config['training']['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 初始化学习率预热
        adaptive_config = self.config['training']['adaptive_lr_config']
        if adaptive_config['enable_warmup']:
            self.warmup_scheduler = WarmupScheduler(
                self.optimizer,
                warmup_epochs=adaptive_config['warmup_epochs'],
                start_lr=adaptive_config['warmup_start_lr'],
                target_lr=self.config['training']['lr_first']
            )
        else:
            self.warmup_scheduler = None
            
        # 主学习率调度器
        self.main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=adaptive_config['plateau_factor'],
            patience=adaptive_config['plateau_patience'],
            min_lr=adaptive_config['plateau_min_lr']
        )
        
        # 余弦退火调度器（备用）
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=adaptive_config['cosine_restart_periods'][0],
            T_mult=2,
            eta_min=adaptive_config['plateau_min_lr']
        )
        
        print(f"优化器初始化完成，初始学习率: {self.config['training']['lr_first']}")
        
    def _init_data_loaders(self):
        """初始化数据加载器"""
        data_dir = "/home/guest/Others/DECODE_rewrite/simulation_zmap2tiff/outputs_100samples_40"
        
        try:
            from utils.dataset import create_train_val_split
            
            train_samples, val_samples = create_train_val_split(
                data_dir, 
                train_ratio=self.config['data']['train_val_split']
            )
            
            train_dataset = DECODEDataset(
                data_root=data_dir,
                sample_list=train_samples,
                consecutive_frames=self.config['data']['consecutive_frames'],
                image_size=self.config['data']['image_size']
            )
            
            val_dataset = DECODEDataset(
                data_root=data_dir,
                sample_list=val_samples,
                consecutive_frames=self.config['data']['consecutive_frames'],
                image_size=self.config['data']['image_size']
            )
            
            from utils.dataset import custom_collate_fn
            
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['data']['batch_size'],
                shuffle=True,
                num_workers=self.config['data']['num_workers'],
                pin_memory=True,
                collate_fn=custom_collate_fn
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['data']['batch_size'],
                shuffle=False,
                num_workers=self.config['data']['num_workers'],
                pin_memory=True,
                collate_fn=custom_collate_fn
            )
            
            print(f"数据加载器初始化完成")
            print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
            
        except Exception as e:
            print(f"数据加载器初始化失败: {e}")
            print("将使用模拟数据进行演示")
            self._create_dummy_data_loaders()
    
    def _create_dummy_data_loaders(self):
        """创建模拟数据加载器"""
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size=100):
                self.size = size
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                images = torch.randn(3, 40, 40)
                targets = {
                    'unified_target': torch.randn(6, 40, 40),
                    'count_maps': torch.sigmoid(torch.randn(1, 40, 40)),
                    'loc_maps': torch.randn(3, 40, 40),
                    'photon_maps': torch.abs(torch.randn(1, 40, 40)),
                    'background_maps': torch.abs(torch.randn(1, 40, 40))
                }
                return images, targets
        
        train_dataset = DummyDataset(80)
        val_dataset = DummyDataset(20)
        
        self.train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        print("使用模拟数据加载器")
        
    def _init_logging(self):
        """初始化日志记录"""
        log_dir = os.path.join(self.config['output']['save_dir'], 'tensorboard')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard日志目录: {log_dir}")
        
    def forward_pass(self, images: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.first_level(images)
        output = self.second_level(features)
        
        # 如果输出是字典（定位模式），转换为统一的6通道张量
        if isinstance(output, dict):
            # 将字典输出转换为6通道张量: [prob, x, y, z, photon, bg]
            prob = output['prob']  # (B, 1, H, W)
            offset = output['offset']  # (B, 3, H, W) - dx, dy, dz
            photon = output['photon']  # (B, 1, H, W)
            background = output.get('background', torch.zeros_like(prob))  # (B, 1, H, W)
            
            # 拼接成6通道输出
            unified_output = torch.cat([
                prob,           # 通道0: 概率
                offset,         # 通道1-3: x, y, z偏移
                photon,         # 通道4: 光子数
                background      # 通道5: 背景
            ], dim=1)  # (B, 6, H, W)
            
            return unified_output
        else:
            # 如果输出已经是张量，直接返回
            return output
        
    def create_unified_target(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """从数据集输出创建统一目标张量"""
        batch_size = batch['frames'].shape[0]
        height, width = batch['frames'].shape[-2:]
        
        # 创建6通道目标张量 [B, 6, H, W]
        unified_target = torch.zeros(batch_size, 6, height, width, device=self.device)
        
        # 这里需要根据实际的标注格式来填充目标张量
        # 暂时创建简单的占位符目标
        for b in range(batch_size):
            positions = batch['emitter_positions'][b]  # [N, 3]
            photons = batch['emitter_photons'][b]      # [N]
            frame_ids = batch['emitter_frame_ids'][b]  # [N]
            
            # 简单的目标生成（需要根据实际需求调整）
            for i, (pos, phot, fid) in enumerate(zip(positions, photons, frame_ids)):
                if pos.sum() > 0:  # 有效的发射器
                    x, y, z = pos
                    # 将坐标映射到图像空间
                    px = int(x.item() * width) if 0 <= x < 1 else width // 2
                    py = int(y.item() * height) if 0 <= y < 1 else height // 2
                    
                    if 0 <= px < width and 0 <= py < height:
                        unified_target[b, 0, py, px] = 1.0  # 概率
                        unified_target[b, 1, py, px] = x    # x坐标
                        unified_target[b, 2, py, px] = y    # y坐标
                        unified_target[b, 3, py, px] = z    # z坐标
                        unified_target[b, 4, py, px] = phot # 光子数
                        unified_target[b, 5, py, px] = 0.1  # 背景
        
        return unified_target
    
    def compute_loss(self, output: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算损失"""
        if self.config['training']['loss_type'] == 'unified_decode':
            # 创建统一目标
            unified_target = self.create_unified_target(batch)
            
            # 计算损失
            losses = self.criterion(output, unified_target)
            
            return {
                'total_loss': losses['total'],
                'prob_loss': losses['prob'],
                'x_loss': losses['x'],
                'y_loss': losses['y'],
                'z_loss': losses['z'],
                'photon_loss': losses['photon'],
                'bg_loss': losses['bg']
            }
        else:
            # 其他损失类型的实现
            unified_target = self.create_unified_target(batch)
            loss = self.criterion(output, unified_target)
            return {
                'total_loss': loss,
                'combined_loss': loss
            }
            
    def update_learning_rate(self, val_loss: float):
        """更新学习率"""
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # 预热阶段
        if self.warmup_scheduler and self.current_epoch < self.config['training']['adaptive_lr_config']['warmup_epochs']:
            self.warmup_scheduler.step()
            new_lr = self.warmup_scheduler.get_lr()
            print(f"Warmup阶段 - Epoch {self.current_epoch}: LR {current_lr:.2e} -> {new_lr:.2e}")
            return
        
        # 自适应调整
        should_adjust, reason, factor = self.adaptive_manager.should_adjust_lr(val_loss, self.current_epoch)
        
        if should_adjust:
            new_lr = current_lr * factor
            # 确保不超过最小学习率
            min_lr = self.config['training']['adaptive_lr_config']['plateau_min_lr']
            new_lr = max(new_lr, min_lr)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
                
            print(f"自适应调整 - {reason}: LR {current_lr:.2e} -> {new_lr:.2e} (factor: {factor})")
        else:
            # 使用主调度器
            self.main_scheduler.step(val_loss)
            
        # 记录学习率
        self.lr_values.append(self.optimizer.param_groups[0]['lr'])
        
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.first_level.train()
        self.second_level.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 提取数据
            if isinstance(batch, dict):
                images = batch['frames'].to(self.device)
                targets = {}
                for key in ['emitter_positions', 'emitter_photons', 'emitter_frame_ids', 'unified_target']:
                    if key in batch and isinstance(batch[key], torch.Tensor):
                        targets[key] = batch[key].to(self.device)
            else:
                images, targets = batch
                images = images.to(self.device)
                # 将targets移动到设备
                for key in targets:
                    if isinstance(targets[key], torch.Tensor):
                        targets[key] = targets[key].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            output = self.forward_pass(images)
            
            # 计算损失
            loss_dict = self.compute_loss(output, batch)
            loss = loss_dict['total_loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.first_level.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.second_level.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 每50个batch打印一次
            if batch_idx % 50 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.6f}, LR: {current_lr:.2e}")
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
        
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.first_level.eval()
        self.second_level.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 提取数据
                if isinstance(batch, dict):
                    images = batch['frames'].to(self.device)
                    targets = {}
                    for key in ['emitter_positions', 'emitter_photons', 'emitter_frame_ids', 'unified_target']:
                        if key in batch and isinstance(batch[key], torch.Tensor):
                            targets[key] = batch[key].to(self.device)
                else:
                    images, targets = batch
                    images = images.to(self.device)
                    
                    for key in targets:
                        if isinstance(targets[key], torch.Tensor):
                            targets[key] = targets[key].to(self.device)
                
                output = self.forward_pass(images)
                loss_dict = self.compute_loss(output, batch)
                loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
        
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'first_level_state_dict': self.first_level.state_dict(),
            'second_level_state_dict': self.second_level.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'lr_values': self.lr_values
        }
        
        save_dir = self.config['output']['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存最新检查点
        torch.save(checkpoint, os.path.join(save_dir, 'latest_checkpoint.pth'))
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            print(f"保存最佳模型，验证损失: {self.best_val_loss:.6f}")
            
    def train(self):
        """主训练循环"""
        print("开始自适应训练...")
        start_time = time.time()
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch()
            train_loss = train_metrics['train_loss']
            
            # 验证
            val_metrics = self.validate()
            val_loss = val_metrics['val_loss']
            
            # 更新学习率
            self.update_learning_rate(val_loss)
            
            # 记录指标
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 检查是否为最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # 保存检查点
            if epoch % 50 == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # 打印进度
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d}/{self.config['training']['epochs']}: "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"LR: {current_lr:.2e}, Best: {self.best_val_loss:.6f}")
            
            # 早停检查（可选）
            if current_lr < self.config['training']['adaptive_lr_config']['plateau_min_lr'] * 1.1:
                epochs_since_best = epoch - self.adaptive_manager.last_improvement_epoch
                if epochs_since_best > 100:
                    print(f"早停：学习率过小且{epochs_since_best}个epoch无改善")
                    break
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"\n训练完成！")
        print(f"总时间: {total_time:.2f}秒")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
        print(f"最终学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存最终检查点
        self.save_checkpoint()
        self.writer.close()
        
        return {
            'best_val_loss': self.best_val_loss,
            'total_time': total_time,
            'final_lr': self.optimizer.param_groups[0]['lr'],
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'lr_values': self.lr_values
        }

def main():
    """主函数"""
    config_path = "configs/train_config_adaptive.json"
    
    if not os.path.exists(config_path):
        print(f"错误: 配置文件 {config_path} 不存在")
        return
    
    print("=" * 60)
    print("DECODE网络自适应训练")
    print("=" * 60)
    
    trainer = AdaptiveDECODETrainer(config_path)
    results = trainer.train()
    
    print("\n训练结果:")
    for key, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value}")

if __name__ == '__main__':
    main()