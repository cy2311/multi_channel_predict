"""DECODE神经网络v2 - 基于深度学习的单分子定位显微镜框架

这个模块实现了DECODE论文中描述的完整神经网络架构，包括：
- SigmaMUNet: 主要的双重U-Net架构
- DoubleMUnet: 双重U-Net基础架构
- SimpleSMLMNet: 简化版SMLM网络
- 完整的损失函数系统
- 训练和推理流水线
- 评估系统
"""

__version__ = "2.0.0"
__author__ = "DECODE Team"

from .models import (
    SigmaMUNet,
    DoubleMUnet,
    SimpleSMLMNet,
    UNet2d
)

from .loss import (
    PPXYZBLoss,
    GaussianMMLoss,
    UnifiedLoss
)

from .training import (
    Trainer,
    TrainingConfig
)

from .inference import (
    Infer,
    BatchInfer
)

from .evaluation import (
    ModelEvaluator,
    BenchmarkEvaluator,
    MetricsVisualizer
)

from .utils import (
    TrainingConfig,
    DataConfig,
    ModelConfig
)

__all__ = [
    # Models
    'SigmaMUNet',
    'DoubleMUnet', 
    'SimpleSMLMNet',
    'UNet2d',
    
    # Loss functions
    'PPXYZBLoss',
    'GaussianMMLoss',
    'UnifiedLoss',
    
    # Training
    'DECODETrainer',
    'TrainingConfig',
    
    # Inference
    'DECODEInfer',
    'LiveInfer',
    
    # Evaluation
    'SegmentationEvaluation',
    'DistanceEvaluation', 
    'WeightedErrors',
    
    # Utils
    'UnifiedEmbeddingTarget',
    'SimpleWeight',
    'DatasetFactory'
]