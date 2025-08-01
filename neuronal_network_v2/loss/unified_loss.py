"""UnifiedLoss实现

统一损失函数，可以组合多种损失类型，支持动态权重调整和多阶段训练。
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Callable
from .ppxyzb_loss import PPXYZBLoss
from .gaussian_mm_loss import GaussianMMLoss


class UnifiedLoss(nn.Module):
    """统一损失函数
    
    支持多种损失函数的组合：
    - PPXYZBLoss: 用于基础的检测和回归
    - GaussianMMLoss: 用于概率建模
    - 自定义损失函数
    
    特性：
    - 动态权重调整
    - 多阶段训练支持
    - 损失平衡
    - 梯度裁剪
    
    Args:
        loss_configs: 损失配置列表
        dynamic_weighting: 是否启用动态权重
        gradient_clipping: 梯度裁剪阈值
        loss_balancing: 是否启用损失平衡
    """
    
    def __init__(self,
                 loss_configs: List[Dict],
                 dynamic_weighting: bool = True,
                 gradient_clipping: Optional[float] = None,
                 loss_balancing: bool = True,
                 warmup_epochs: int = 10):
        super().__init__()
        
        self.dynamic_weighting = dynamic_weighting
        self.gradient_clipping = gradient_clipping
        self.loss_balancing = loss_balancing
        self.warmup_epochs = warmup_epochs
        
        # 构建损失函数
        self.loss_functions = nn.ModuleDict()
        self.loss_weights = {}
        self.loss_schedules = {}
        
        for config in loss_configs:
            name = config['name']
            loss_type = config['type']
            weight = config.get('weight', 1.0)
            schedule = config.get('schedule', None)
            kwargs = config.get('kwargs', {})
            
            # 创建损失函数
            if loss_type == 'ppxyzb':
                loss_fn = PPXYZBLoss(**kwargs)
            elif loss_type == 'gaussian_mm':
                loss_fn = GaussianMMLoss(**kwargs)
            elif loss_type == 'mse':
                loss_fn = nn.MSELoss(**kwargs)
            elif loss_type == 'bce':
                loss_fn = nn.BCEWithLogitsLoss(**kwargs)
            elif loss_type == 'l1':
                loss_fn = nn.L1Loss(**kwargs)
            elif callable(loss_type):
                loss_fn = loss_type(**kwargs)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            self.loss_functions[name] = loss_fn
            self.loss_weights[name] = weight
            self.loss_schedules[name] = schedule
        
        # 动态权重历史
        if self.dynamic_weighting:
            self.register_buffer('loss_history', torch.zeros(len(loss_configs), 100))
            self.register_buffer('weight_history', torch.ones(len(loss_configs), 100))
            self.history_idx = 0
        
        # 损失平衡参数
        if self.loss_balancing:
            self.register_buffer('loss_scales', torch.ones(len(loss_configs)))
            self.register_buffer('initial_losses', torch.zeros(len(loss_configs)))
            self.initial_losses_set = False
    
    def forward(self,
                output: torch.Tensor,
                target: torch.Tensor,
                epoch: int = 0,
                step: int = 0,
                **kwargs) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            output: 模型输出
            target: 目标张量
            epoch: 当前epoch
            step: 当前step
            **kwargs: 其他参数
            
        Returns:
            损失字典
        """
        losses = {}
        weighted_losses = {}
        total_loss = 0.0
        
        # 计算各个损失
        for i, (name, loss_fn) in enumerate(self.loss_functions.items()):
            try:
                if isinstance(loss_fn, (PPXYZBLoss, GaussianMMLoss)):
                    loss_result = loss_fn(output, target, **kwargs)
                    if isinstance(loss_result, dict):
                        loss_value = loss_result['mean_total_loss']
                        losses.update({f"{name}_{k}": v for k, v in loss_result.items()})
                    else:
                        loss_value = loss_result.mean()
                else:
                    loss_value = loss_fn(output, target).mean()
                
                losses[f"{name}_loss"] = loss_value
                
                # 应用调度权重
                scheduled_weight = self._get_scheduled_weight(name, epoch, step)
                
                # 应用动态权重
                if self.dynamic_weighting:
                    dynamic_weight = self._get_dynamic_weight(name, loss_value, i)
                else:
                    dynamic_weight = 1.0
                
                # 应用损失平衡
                if self.loss_balancing:
                    balance_weight = self._get_balance_weight(loss_value, i)
                else:
                    balance_weight = 1.0
                
                # 总权重
                final_weight = scheduled_weight * dynamic_weight * balance_weight
                weighted_loss = final_weight * loss_value
                
                weighted_losses[f"{name}_weighted_loss"] = weighted_loss
                total_loss += weighted_loss
                
            except Exception as e:
                print(f"Error computing {name} loss: {e}")
                continue
        
        # 更新历史
        if self.dynamic_weighting:
            self._update_history(losses)
        
        # 梯度裁剪
        if self.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clipping)
        
        result = {
            'total_loss': total_loss,
            **losses,
            **weighted_losses
        }
        
        return result
    
    def _get_scheduled_weight(self, name: str, epoch: int, step: int) -> float:
        """获取调度权重"""
        base_weight = self.loss_weights[name]
        schedule = self.loss_schedules[name]
        
        if schedule is None:
            return base_weight
        
        if schedule['type'] == 'linear':
            start_epoch = schedule.get('start_epoch', 0)
            end_epoch = schedule.get('end_epoch', 100)
            start_weight = schedule.get('start_weight', base_weight)
            end_weight = schedule.get('end_weight', base_weight)
            
            if epoch < start_epoch:
                return start_weight
            elif epoch > end_epoch:
                return end_weight
            else:
                progress = (epoch - start_epoch) / (end_epoch - start_epoch)
                return start_weight + progress * (end_weight - start_weight)
        
        elif schedule['type'] == 'exponential':
            decay_rate = schedule.get('decay_rate', 0.95)
            return base_weight * (decay_rate ** epoch)
        
        elif schedule['type'] == 'cosine':
            max_epochs = schedule.get('max_epochs', 100)
            min_weight = schedule.get('min_weight', 0.1 * base_weight)
            return min_weight + 0.5 * (base_weight - min_weight) * \
                   (1 + torch.cos(torch.tensor(epoch * 3.14159 / max_epochs)))
        
        elif schedule['type'] == 'warmup':
            if epoch < self.warmup_epochs:
                return base_weight * (epoch + 1) / self.warmup_epochs
            else:
                return base_weight
        
        return base_weight
    
    def _get_dynamic_weight(self, name: str, loss_value: torch.Tensor, loss_idx: int) -> float:
        """获取动态权重"""
        if not self.dynamic_weighting:
            return 1.0
        
        # 记录当前损失
        self.loss_history[loss_idx, self.history_idx] = loss_value.detach()
        
        # 计算相对损失变化
        if self.history_idx > 10:  # 有足够历史数据
            recent_avg = self.loss_history[loss_idx, max(0, self.history_idx-10):self.history_idx].mean()
            old_avg = self.loss_history[loss_idx, max(0, self.history_idx-20):max(1, self.history_idx-10)].mean()
            
            if old_avg > 0:
                relative_change = (recent_avg - old_avg) / old_avg
                
                # 如果损失下降缓慢，增加权重
                if relative_change > -0.01:  # 下降小于1%
                    dynamic_weight = 1.2
                elif relative_change < -0.1:  # 下降大于10%
                    dynamic_weight = 0.8
                else:
                    dynamic_weight = 1.0
            else:
                dynamic_weight = 1.0
        else:
            dynamic_weight = 1.0
        
        # 记录权重历史
        self.weight_history[loss_idx, self.history_idx] = dynamic_weight
        
        return dynamic_weight
    
    def _get_balance_weight(self, loss_value: torch.Tensor, loss_idx: int) -> float:
        """获取平衡权重"""
        if not self.loss_balancing:
            return 1.0
        
        # 设置初始损失
        if not self.initial_losses_set:
            self.initial_losses[loss_idx] = loss_value.detach()
            if loss_idx == len(self.loss_functions) - 1:
                self.initial_losses_set = True
            return 1.0
        
        # 计算相对于初始损失的比例
        if self.initial_losses[loss_idx] > 0:
            relative_loss = loss_value / self.initial_losses[loss_idx]
            # 使用倒数作为平衡权重
            balance_weight = 1.0 / (relative_loss + 1e-8)
            # 限制权重范围
            balance_weight = torch.clamp(balance_weight, 0.1, 10.0).item()
        else:
            balance_weight = 1.0
        
        return balance_weight
    
    def _update_history(self, losses: Dict[str, torch.Tensor]):
        """更新历史记录"""
        self.history_idx = (self.history_idx + 1) % 100
    
    def get_loss_weights(self) -> Dict[str, float]:
        """获取当前损失权重"""
        weights = {}
        for name in self.loss_functions.keys():
            weights[name] = self.loss_weights[name]
        return weights
    
    def set_loss_weight(self, name: str, weight: float):
        """设置损失权重"""
        if name in self.loss_weights:
            self.loss_weights[name] = weight
        else:
            raise ValueError(f"Loss {name} not found")
    
    def get_loss_statistics(self) -> Dict[str, Dict[str, float]]:
        """获取损失统计信息"""
        if not self.dynamic_weighting:
            return {}
        
        stats = {}
        for i, name in enumerate(self.loss_functions.keys()):
            loss_hist = self.loss_history[i, :self.history_idx]
            weight_hist = self.weight_history[i, :self.history_idx]
            
            if len(loss_hist) > 0:
                stats[name] = {
                    'mean_loss': loss_hist.mean().item(),
                    'std_loss': loss_hist.std().item(),
                    'mean_weight': weight_hist.mean().item(),
                    'current_loss': loss_hist[-1].item() if len(loss_hist) > 0 else 0.0,
                    'current_weight': weight_hist[-1].item() if len(weight_hist) > 0 else 1.0
                }
        
        return stats


class MultiStageLoss(UnifiedLoss):
    """多阶段损失函数
    
    在不同训练阶段使用不同的损失配置
    """
    
    def __init__(self,
                 stage_configs: List[Dict],
                 **kwargs):
        """
        Args:
            stage_configs: 阶段配置列表，每个包含:
                - start_epoch: 开始epoch
                - end_epoch: 结束epoch  
                - loss_configs: 该阶段的损失配置
        """
        # 使用第一阶段的配置初始化
        super().__init__(stage_configs[0]['loss_configs'], **kwargs)
        
        self.stage_configs = stage_configs
        self.current_stage = 0
    
    def forward(self,
                output: torch.Tensor,
                target: torch.Tensor,
                epoch: int = 0,
                **kwargs) -> Dict[str, torch.Tensor]:
        """多阶段前向传播"""
        # 检查是否需要切换阶段
        self._update_stage(epoch)
        
        return super().forward(output, target, epoch, **kwargs)
    
    def _update_stage(self, epoch: int):
        """更新当前阶段"""
        for i, stage_config in enumerate(self.stage_configs):
            start_epoch = stage_config.get('start_epoch', 0)
            end_epoch = stage_config.get('end_epoch', float('inf'))
            
            if start_epoch <= epoch < end_epoch:
                if i != self.current_stage:
                    print(f"Switching to stage {i} at epoch {epoch}")
                    self.current_stage = i
                    self._rebuild_losses(stage_config['loss_configs'])
                break
    
    def _rebuild_losses(self, loss_configs: List[Dict]):
        """重建损失函数"""
        # 清除现有损失函数
        self.loss_functions.clear()
        self.loss_weights.clear()
        self.loss_schedules.clear()
        
        # 重新构建
        for config in loss_configs:
            name = config['name']
            loss_type = config['type']
            weight = config.get('weight', 1.0)
            schedule = config.get('schedule', None)
            kwargs = config.get('kwargs', {})
            
            # 创建损失函数
            if loss_type == 'ppxyzb':
                loss_fn = PPXYZBLoss(**kwargs)
            elif loss_type == 'gaussian_mm':
                loss_fn = GaussianMMLoss(**kwargs)
            elif loss_type == 'mse':
                loss_fn = nn.MSELoss(**kwargs)
            elif loss_type == 'bce':
                loss_fn = nn.BCEWithLogitsLoss(**kwargs)
            elif loss_type == 'l1':
                loss_fn = nn.L1Loss(**kwargs)
            elif callable(loss_type):
                loss_fn = loss_type(**kwargs)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            self.loss_functions[name] = loss_fn
            self.loss_weights[name] = weight
            self.loss_schedules[name] = schedule


# 预定义的损失配置
DEFAULT_PPXYZB_CONFIG = {
    'name': 'ppxyzb',
    'type': 'ppxyzb',
    'weight': 1.0,
    'kwargs': {
        'detection_weight': 1.0,
        'photon_weight': 1.0,
        'coord_weight': 1.0,
        'bg_weight': 0.1
    }
}

DEFAULT_GAUSSIAN_MM_CONFIG = {
    'name': 'gaussian_mm',
    'type': 'gaussian_mm',
    'weight': 1.0,
    'kwargs': {
        'eps': 1e-6,
        'bg_weight': 0.1,
        'gmm_weight': 1.0
    }
}

DEFAULT_UNIFIED_CONFIG = [
    DEFAULT_PPXYZB_CONFIG,
    {
        'name': 'regularization',
        'type': 'l1',
        'weight': 0.01,
        'schedule': {
            'type': 'warmup',
            'start_weight': 0.0,
            'end_weight': 0.01
        }
    }
]