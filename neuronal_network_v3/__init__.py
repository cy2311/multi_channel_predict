"""DECODE神经网络v3 - 基于深度学习的单分子定位显微镜框架

这个模块实现了DECODE论文中描述的完整神经网络架构，包括：
- SigmaMUNet: 主要的双重U-Net架构
- DoubleMUnet: 双重U-Net基础架构
- SimpleSMLMNet: 简化版SMLM网络
- RatioNet: 多通道比率网络
- 完整的损失函数系统
- 训练和推理流水线
- 评估系统
- 多通道支持
"""

__version__ = "3.0.0"
__author__ = "DECODE Team"
__description__ = "Neural network components for DECODE - Deep learning for single molecule localization microscopy with multi-channel support"

from . import models
from . import loss
from . import trainer
from . import inference
from . import evaluation
from . import data

# Multi-channel components
from .models.ratio_net import RatioNet, FeatureExtractor
from .loss.ratio_loss import RatioGaussianNLLLoss, MultiChannelLossWithGaussianRatio
from .inference.multi_channel_infer import MultiChannelInfer, MultiChannelBatchInfer
from .evaluation.multi_channel_evaluation import MultiChannelEvaluation, RatioEvaluationMetrics
from .trainer.multi_channel_trainer import MultiChannelTrainer
from .data.multi_channel_dataset import MultiChannelSMLMDataset, MultiChannelDataModule

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

# from .training import (
#     Trainer,
#     TrainingConfig
# )

# from .inference import (
#     Infer,
#     BatchInfer
# )

# from .evaluation import (
#     ModelEvaluator,
#     BenchmarkEvaluator,
#     MetricsVisualizer
# )

# from .utils import (
#     TrainingConfig,
#     DataConfig,
#     ModelConfig
# )

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