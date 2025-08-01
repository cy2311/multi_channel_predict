"""工厂类模块

该模块提供了各种组件的工厂类，包括：
- 优化器工厂
- 学习率调度器工厂
- 损失函数工厂
- 模型工厂
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
    ReduceLROnPlateau, CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts
)
from typing import Dict, Any, Optional, Type, Union
import logging

logger = logging.getLogger(__name__)


class OptimizerFactory:
    """优化器工厂类"""
    
    _optimizers = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
        'adagrad': optim.Adagrad,
        'adadelta': optim.Adadelta,
        'adamax': optim.Adamax,
        'nadam': optim.NAdam,
        'radam': optim.RAdam,
        'lbfgs': optim.LBFGS
    }
    
    @classmethod
    def create(cls, 
               optimizer_name: str, 
               model_parameters,
               **kwargs) -> optim.Optimizer:
        """创建优化器
        
        Args:
            optimizer_name: 优化器名称
            model_parameters: 模型参数
            **kwargs: 优化器参数
            
        Returns:
            torch.optim.Optimizer: 优化器实例
        """
        optimizer_name = optimizer_name.lower()
        
        if optimizer_name not in cls._optimizers:
            raise ValueError(f"不支持的优化器: {optimizer_name}. "
                           f"支持的优化器: {list(cls._optimizers.keys())}")
        
        optimizer_class = cls._optimizers[optimizer_name]
        
        # 设置默认参数
        default_params = cls._get_default_params(optimizer_name)
        default_params.update(kwargs)
        
        try:
            optimizer = optimizer_class(model_parameters, **default_params)
            logger.info(f"创建优化器: {optimizer_name}, 参数: {default_params}")
            return optimizer
        except Exception as e:
            logger.error(f"创建优化器失败: {e}")
            raise
    
    @classmethod
    def _get_default_params(cls, optimizer_name: str) -> Dict[str, Any]:
        """获取优化器默认参数"""
        defaults = {
            'adam': {'lr': 1e-3, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 1e-4},
            'adamw': {'lr': 1e-3, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 1e-2},
            'sgd': {'lr': 1e-2, 'momentum': 0.9, 'weight_decay': 1e-4, 'nesterov': True},
            'rmsprop': {'lr': 1e-3, 'alpha': 0.99, 'eps': 1e-8, 'weight_decay': 1e-4},
            'adagrad': {'lr': 1e-2, 'lr_decay': 0, 'weight_decay': 1e-4, 'eps': 1e-10},
            'adadelta': {'lr': 1.0, 'rho': 0.9, 'eps': 1e-6, 'weight_decay': 1e-4},
            'adamax': {'lr': 2e-3, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 1e-4},
            'nadam': {'lr': 2e-3, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 1e-4},
            'radam': {'lr': 1e-3, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 1e-4},
            'lbfgs': {'lr': 1, 'max_iter': 20, 'max_eval': None, 'tolerance_grad': 1e-7}
        }
        return defaults.get(optimizer_name, {})
    
    @classmethod
    def get_available_optimizers(cls) -> list:
        """获取可用的优化器列表"""
        return list(cls._optimizers.keys())


class SchedulerFactory:
    """学习率调度器工厂类"""
    
    _schedulers = {
        'steplr': StepLR,
        'multisteplr': MultiStepLR,
        'exponentiallr': ExponentialLR,
        'cosineannealinglr': CosineAnnealingLR,
        'reducelronplateau': ReduceLROnPlateau,
        'cycliclr': CyclicLR,
        'onecyclelr': OneCycleLR,
        'cosineannealingwarmrestarts': CosineAnnealingWarmRestarts
    }
    
    @classmethod
    def create(cls, 
               scheduler_name: str, 
               optimizer: optim.Optimizer,
               **kwargs) -> Union[optim.lr_scheduler._LRScheduler, ReduceLROnPlateau]:
        """创建学习率调度器
        
        Args:
            scheduler_name: 调度器名称
            optimizer: 优化器
            **kwargs: 调度器参数
            
        Returns:
            学习率调度器实例
        """
        scheduler_name = scheduler_name.lower()
        
        if scheduler_name not in cls._schedulers:
            raise ValueError(f"不支持的调度器: {scheduler_name}. "
                           f"支持的调度器: {list(cls._schedulers.keys())}")
        
        scheduler_class = cls._schedulers[scheduler_name]
        
        # 设置默认参数
        default_params = cls._get_default_params(scheduler_name, optimizer)
        default_params.update(kwargs)
        
        try:
            scheduler = scheduler_class(optimizer, **default_params)
            logger.info(f"创建调度器: {scheduler_name}, 参数: {default_params}")
            return scheduler
        except Exception as e:
            logger.error(f"创建调度器失败: {e}")
            raise
    
    @classmethod
    def _get_default_params(cls, scheduler_name: str, optimizer: optim.Optimizer) -> Dict[str, Any]:
        """获取调度器默认参数"""
        defaults = {
            'steplr': {'step_size': 30, 'gamma': 0.1},
            'multisteplr': {'milestones': [30, 60, 90], 'gamma': 0.1},
            'exponentiallr': {'gamma': 0.95},
            'cosineannealinglr': {'T_max': 100, 'eta_min': 1e-6},
            'reducelronplateau': {'mode': 'min', 'factor': 0.5, 'patience': 10, 
                                'threshold': 1e-4, 'min_lr': 1e-6},
            'cycliclr': {'base_lr': 1e-5, 'max_lr': 1e-2, 'step_size_up': 2000},
            'onecyclelr': {'max_lr': 1e-2, 'total_steps': None, 'epochs': None},
            'cosineannealingwarmrestarts': {'T_0': 10, 'T_mult': 2, 'eta_min': 1e-6}
        }
        
        # 对于OneCycleLR，需要设置total_steps
        if scheduler_name == 'onecyclelr':
            if 'total_steps' not in defaults[scheduler_name] or defaults[scheduler_name]['total_steps'] is None:
                # 尝试从优化器获取参数组数量作为估计
                defaults[scheduler_name]['total_steps'] = 1000  # 默认值
        
        return defaults.get(scheduler_name, {})
    
    @classmethod
    def get_available_schedulers(cls) -> list:
        """获取可用的调度器列表"""
        return list(cls._schedulers.keys())


class LossFactory:
    """损失函数工厂类"""
    
    @classmethod
    def create(cls, loss_name: str, **kwargs) -> nn.Module:
        """创建损失函数
        
        Args:
            loss_name: 损失函数名称
            **kwargs: 损失函数参数
            
        Returns:
            nn.Module: 损失函数实例
        """
        loss_name = loss_name.lower()
        
        # 导入损失函数模块
        try:
            if loss_name == 'ppxyzbloss':
                from ..loss.ppxyzb_loss import PPXYZBLoss
                return PPXYZBLoss(**kwargs)
            elif loss_name == 'adaptiveppxyzbloss':
                from ..loss.ppxyzb_loss import AdaptivePPXYZBLoss
                return AdaptivePPXYZBLoss(**kwargs)
            elif loss_name == 'multiscaleppxyzbloss':
                from ..loss.ppxyzb_loss import MultiScalePPXYZBLoss
                return MultiScalePPXYZBLoss(**kwargs)
            elif loss_name == 'gaussianmmloss':
                from ..loss.gaussian_mm_loss import GaussianMMLoss
                return GaussianMMLoss(**kwargs)
            elif loss_name == 'adaptivegaussianmmloss':
                from ..loss.gaussian_mm_loss import AdaptiveGaussianMMLoss
                return AdaptiveGaussianMMLoss(**kwargs)
            elif loss_name == 'multimodalgaussianmmloss':
                from ..loss.gaussian_mm_loss import MultiModalGaussianMMLoss
                return MultiModalGaussianMMLoss(**kwargs)
            elif loss_name == 'unifiedloss':
                from ..loss.unified_loss import UnifiedLoss
                return UnifiedLoss(**kwargs)
            elif loss_name == 'multistageloss':
                from ..loss.unified_loss import MultiStageLoss
                return MultiStageLoss(**kwargs)
            elif loss_name == 'mse' or loss_name == 'mseloss':
                return nn.MSELoss(**kwargs)
            elif loss_name == 'mae' or loss_name == 'l1loss':
                return nn.L1Loss(**kwargs)
            elif loss_name == 'bce' or loss_name == 'bceloss':
                return nn.BCELoss(**kwargs)
            elif loss_name == 'bcewithlogits' or loss_name == 'bcewithlogitsloss':
                return nn.BCEWithLogitsLoss(**kwargs)
            elif loss_name == 'crossentropy' or loss_name == 'crossentropyloss':
                return nn.CrossEntropyLoss(**kwargs)
            elif loss_name == 'smoothl1' or loss_name == 'smoothl1loss':
                return nn.SmoothL1Loss(**kwargs)
            elif loss_name == 'huber' or loss_name == 'huberloss':
                return nn.HuberLoss(**kwargs)
            else:
                raise ValueError(f"不支持的损失函数: {loss_name}")
        except ImportError as e:
            logger.error(f"导入损失函数失败: {e}")
            raise
        except Exception as e:
            logger.error(f"创建损失函数失败: {e}")
            raise
    
    @classmethod
    def get_available_losses(cls) -> list:
        """获取可用的损失函数列表"""
        return [
            'ppxyzbloss', 'adaptiveppxyzbloss', 'multiscaleppxyzbloss',
            'gaussianmmloss', 'adaptivegaussianmmloss', 'multimodalgaussianmmloss',
            'unifiedloss', 'multistageloss',
            'mse', 'mseloss', 'mae', 'l1loss', 'bce', 'bceloss',
            'bcewithlogits', 'bcewithlogitsloss', 'crossentropy', 'crossentropyloss',
            'smoothl1', 'smoothl1loss', 'huber', 'huberloss'
        ]


class ModelFactory:
    """模型工厂类"""
    
    @classmethod
    def create(cls, model_name: str, **kwargs) -> nn.Module:
        """创建模型
        
        Args:
            model_name: 模型名称
            **kwargs: 模型参数
            
        Returns:
            nn.Module: 模型实例
        """
        model_name = model_name.lower()
        
        try:
            if model_name == 'simplesmlmnet':
                from ..models.simple_smlm_net import SimpleSMLMNet
                return SimpleSMLMNet(**kwargs)
            elif model_name == 'enhancedsimplesmlmnet':
                from ..models.simple_smlm_net import EnhancedSimpleSMLMNet
                return EnhancedSimpleSMLMNet(**kwargs)
            elif model_name == 'adaptivesmlmnet':
                from ..models.simple_smlm_net import AdaptiveSMLMNet
                return AdaptiveSMLMNet(**kwargs)
            elif model_name == 'unet2d':
                from ..models.unet2d import UNet2d
                return UNet2d(**kwargs)
            elif model_name == 'double_munet' or model_name == 'doublemunet':
                from ..models.double_munet import DoubleMUnet
                # 转换参数名称以匹配DoubleMUnet的期望
                double_munet_kwargs = kwargs.copy()
                if 'depth' in double_munet_kwargs:
                    depth = double_munet_kwargs.pop('depth')
                    double_munet_kwargs['depth_shared'] = depth
                    double_munet_kwargs['depth_union'] = depth
                return DoubleMUnet(**double_munet_kwargs)
            elif model_name == 'sigma_munet' or model_name == 'sigmamunet':
                from ..models.sigma_munet import SigmaMUnet
                return SigmaMUnet(**kwargs)
            elif model_name == 'sigmaunet':
                from ..models.sigma_mu_net import SigmaMUNet
                return SigmaMUNet(**kwargs)
            else:
                raise ValueError(f"不支持的模型: {model_name}")
        except ImportError as e:
            logger.error(f"导入模型失败: {e}")
            raise
        except Exception as e:
            logger.error(f"创建模型失败: {e}")
            raise
    
    @classmethod
    def get_available_models(cls) -> list:
        """获取可用的模型列表"""
        return [
            'simplesmlmnet', 'enhancedsimplesmlmnet', 'adaptivesmlmnet',
            'unet2d', 'double_munet', 'doublemunet', 'sigma_munet', 'sigmamunet', 'sigmaunet'
        ]


# 便捷函数
def create_optimizer(optimizer_name: str, model_parameters, **kwargs) -> optim.Optimizer:
    """创建优化器的便捷函数"""
    return OptimizerFactory.create(optimizer_name, model_parameters, **kwargs)


def create_scheduler(scheduler_name: str, optimizer: optim.Optimizer, **kwargs):
    """创建调度器的便捷函数"""
    return SchedulerFactory.create(scheduler_name, optimizer, **kwargs)


def create_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """创建损失函数的便捷函数"""
    return LossFactory.create(loss_name, **kwargs)


def create_model(model_name: str, **kwargs) -> nn.Module:
    """创建模型的便捷函数"""
    return ModelFactory.create(model_name, **kwargs)


def get_all_available_components() -> Dict[str, list]:
    """获取所有可用组件的字典"""
    return {
        'optimizers': OptimizerFactory.get_available_optimizers(),
        'schedulers': SchedulerFactory.get_available_schedulers(),
        'losses': LossFactory.get_available_losses(),
        'models': ModelFactory.get_available_models()
    }


def print_available_components():
    """打印所有可用组件"""
    components = get_all_available_components()
    
    print("=== 可用组件 ===")
    for category, items in components.items():
        print(f"\n{category.upper()}:")
        for item in items:
            print(f"  - {item}")


if __name__ == "__main__":
    # 测试工厂类
    print_available_components()