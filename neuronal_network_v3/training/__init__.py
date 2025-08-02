"""训练模块

提供模型训练相关的功能，包括：
- 训练器 (Trainer)
- 数据集 (Dataset)
- 回调函数 (Callbacks)
- 配置类 (从utils.config导入)
"""

from .trainer import Trainer, DistributedTrainer
from .dataset import SMLMStaticDataset
from .target_generator import UnifiedEmbeddingTarget
from .callbacks import TrainingCallback, EarlyStopping, ModelCheckpoint, LearningRateLogger
from ..utils.config import TrainingConfig, DataConfig, OptimizationConfig

__all__ = [
    # 训练器
    'Trainer',
    'DistributedTrainer',
    
    # 数据集
    'SMLMDataset',
    
    # 目标生成器
    'TargetGenerator',
    
    # 回调函数
    'TrainingCallback',
    'EarlyStopping', 
    'ModelCheckpoint',
    'LearningRateLogger',
    
    # 配置类
    'TrainingConfig',
    'DataConfig',
    'OptimizationConfig',
]