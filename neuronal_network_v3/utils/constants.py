"""常量定义模块

该模块定义了DECODE神经网络中使用的各种常量，包括：
- 数据格式常量
- 模型参数常量
- 训练配置常量
- 文件路径常量
- 数学常量
"""

import numpy as np
from enum import Enum, IntEnum
from typing import Dict, List, Tuple

# =============================================================================
# 版本信息
# =============================================================================
VERSION = "2.0.0"
AUTHOR = "DECODE Team"
LICENSE = "MIT"

# =============================================================================
# 数据格式常量
# =============================================================================

class DataFormat(Enum):
    """数据格式枚举"""
    CHW = "CHW"  # Channel, Height, Width
    HWC = "HWC"  # Height, Width, Channel
    NCHW = "NCHW"  # Batch, Channel, Height, Width
    NHWC = "NHWC"  # Batch, Height, Width, Channel
    NCDHW = "NCDHW"  # Batch, Channel, Depth, Height, Width
    NDHWC = "NDHWC"  # Batch, Depth, Height, Width, Channel


class OutputFormat(Enum):
    """输出格式枚举"""
    PPXYZB = "ppxyzb"  # Probability, Position X, Y, Z, Background
    SIGMA_MU = "sigma_mu"  # Sigma, Mu
    SIMPLE = "simple"  # Simple format
    UNIFIED = "unified"  # Unified format


class CoordinateSystem(Enum):
    """坐标系统枚举"""
    PIXEL = "pixel"  # 像素坐标
    NANOMETER = "nanometer"  # 纳米坐标
    MICROMETER = "micrometer"  # 微米坐标


# =============================================================================
# 文件格式常量
# =============================================================================

class FileFormat(Enum):
    """文件格式枚举"""
    HDF5 = ".h5"
    CSV = ".csv"
    JSON = ".json"
    YAML = ".yaml"
    PICKLE = ".pkl"
    NUMPY = ".npy"
    TIFF = ".tiff"
    PNG = ".png"
    MATLAB = ".mat"


# 支持的图像格式
SUPPORTED_IMAGE_FORMATS = [
    ".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp", ".gif"
]

# 支持的数据格式
SUPPORTED_DATA_FORMATS = [
    ".h5", ".hdf5", ".csv", ".json", ".yaml", ".yml", ".pkl", ".npy", ".mat"
]

# =============================================================================
# 模型架构常量
# =============================================================================

class ModelType(Enum):
    """模型类型枚举"""
    SIMPLE_SMLM = "SimpleSMLMNet"
    UNET_2D = "UNet2d"
    SIGMA_MU = "SigmaMUNet"
    CUSTOM = "Custom"


class ActivationFunction(Enum):
    """激活函数枚举"""
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    GELU = "gelu"
    SWISH = "swish"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    LOG_SOFTMAX = "log_softmax"


class NormalizationType(Enum):
    """归一化类型枚举"""
    BATCH_NORM = "batch_norm"
    INSTANCE_NORM = "instance_norm"
    LAYER_NORM = "layer_norm"
    GROUP_NORM = "group_norm"
    NONE = "none"


# 默认模型参数
DEFAULT_MODEL_PARAMS = {
    "input_channels": 1,
    "output_channels": 6,  # ppxyzb format
    "base_channels": 64,
    "depth": 4,
    "activation": ActivationFunction.RELU.value,
    "normalization": NormalizationType.BATCH_NORM.value,
    "dropout_rate": 0.1,
    "use_attention": False,
    "use_residual": True
}

# =============================================================================
# 训练配置常量
# =============================================================================

class OptimizerType(Enum):
    """优化器类型枚举"""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"


class SchedulerType(Enum):
    """学习率调度器类型枚举"""
    STEP_LR = "step_lr"
    EXPONENTIAL_LR = "exponential_lr"
    COSINE_ANNEALING_LR = "cosine_annealing_lr"
    REDUCE_LR_ON_PLATEAU = "reduce_lr_on_plateau"
    CYCLIC_LR = "cyclic_lr"
    ONE_CYCLE_LR = "one_cycle_lr"
    NONE = "none"


class LossFunction(Enum):
    """损失函数类型枚举"""
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    BCE = "bce"
    BCE_WITH_LOGITS = "bce_with_logits"
    CROSS_ENTROPY = "cross_entropy"
    FOCAL = "focal"
    DICE = "dice"
    PPXYZB = "ppxyzb"
    GAUSSIAN_MM = "gaussian_mm"
    UNIFIED = "unified"


# 默认训练参数
DEFAULT_TRAINING_PARAMS = {
    "batch_size": 32,
    "learning_rate": 1e-3,
    "num_epochs": 100,
    "optimizer": OptimizerType.ADAM.value,
    "scheduler": SchedulerType.REDUCE_LR_ON_PLATEAU.value,
    "loss_function": LossFunction.PPXYZB.value,
    "weight_decay": 1e-4,
    "gradient_clip_norm": 1.0,
    "early_stopping_patience": 10,
    "validation_frequency": 1,
    "checkpoint_frequency": 5
}

# =============================================================================
# 数据处理常量
# =============================================================================

class AugmentationType(Enum):
    """数据增强类型枚举"""
    FLIP = "flip"
    ROTATION = "rotation"
    SCALE = "scale"
    TRANSLATION = "translation"
    NOISE = "noise"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    ELASTIC = "elastic"
    AFFINE = "affine"


class NormalizationMethod(Enum):
    """归一化方法枚举"""
    ZSCORE = "zscore"
    MINMAX = "minmax"
    PERCENTILE = "percentile"
    ROBUST = "robust"
    NONE = "none"


# 默认数据处理参数
DEFAULT_DATA_PARAMS = {
    "image_size": (64, 64),
    "pixel_size": 100.0,  # nm
    "normalization": NormalizationMethod.ZSCORE.value,
    "augmentation_probability": 0.5,
    "num_workers": 4,
    "pin_memory": True,
    "prefetch_factor": 2
}

# =============================================================================
# 物理常量
# =============================================================================

# 光学常量
PLANCK_CONSTANT = 6.62607015e-34  # J⋅s
SPEED_OF_LIGHT = 299792458  # m/s
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K

# 显微镜常量
TYPICAL_WAVELENGTH = 680e-9  # m (典型荧光波长)
TYPICAL_NA = 1.4  # 典型数值孔径
TYPICAL_PIXEL_SIZE = 100e-9  # m (典型像素大小)

# 分辨率限制
DIFFRACTION_LIMIT = TYPICAL_WAVELENGTH / (2 * TYPICAL_NA)  # m
NYQUIIST_FREQUENCY = 1 / (2 * TYPICAL_PIXEL_SIZE)  # 1/m

# =============================================================================
# 数学常量
# =============================================================================

# 数值精度
EPS = 1e-8  # 数值稳定性常量
INF = float('inf')  # 无穷大
NAN = float('nan')  # 非数值

# 统计常量
CONFIDENCE_LEVELS = {
    "90%": 1.645,
    "95%": 1.96,
    "99%": 2.576,
    "99.9%": 3.291
}

# 高斯分布常量
GAUSSIAN_FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2))  # FWHM = σ * GAUSSIAN_FWHM_FACTOR
GAUSSIAN_SIGMA_TO_FWHM = GAUSSIAN_FWHM_FACTOR
GAUSSIAN_FWHM_TO_SIGMA = 1 / GAUSSIAN_FWHM_FACTOR

# =============================================================================
# 评估指标常量
# =============================================================================

class MetricType(Enum):
    """评估指标类型枚举"""
    # 检测指标
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    JACCARD = "jaccard"
    
    # 定位指标
    RMSE = "rmse"
    MAE = "mae"
    BIAS = "bias"
    STD = "std"
    
    # 光子数指标
    PHOTON_RMSE = "photon_rmse"
    PHOTON_BIAS = "photon_bias"
    
    # 综合指标
    EFFICIENCY = "efficiency"
    ACCURACY = "accuracy"
    CRLB_RATIO = "crlb_ratio"


# 默认评估参数
DEFAULT_EVALUATION_PARAMS = {
    "tolerance_xy": 250.0,  # nm
    "tolerance_z": 500.0,   # nm
    "min_photons": 50,
    "max_photons": 10000,
    "confidence_level": 0.95,
    "bootstrap_samples": 1000
}

# =============================================================================
# 硬件配置常量
# =============================================================================

class DeviceType(Enum):
    """设备类型枚举"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal Performance Shaders
    AUTO = "auto"


class PrecisionType(Enum):
    """精度类型枚举"""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    MIXED = "mixed"


# 默认硬件配置
DEFAULT_DEVICE_PARAMS = {
    "device": DeviceType.AUTO.value,
    "precision": PrecisionType.FLOAT32.value,
    "num_workers": 4,
    "pin_memory": True,
    "non_blocking": True
}

# =============================================================================
# 文件路径常量
# =============================================================================

# 默认目录结构
DEFAULT_DIRS = {
    "data": "data",
    "models": "models",
    "checkpoints": "checkpoints",
    "logs": "logs",
    "results": "results",
    "cache": "cache",
    "temp": "temp"
}

# 配置文件名
CONFIG_FILES = {
    "model": "model_config.yaml",
    "training": "training_config.yaml",
    "data": "data_config.yaml",
    "inference": "inference_config.yaml",
    "evaluation": "evaluation_config.yaml"
}

# 日志文件名模式
LOG_FILE_PATTERNS = {
    "training": "training_{timestamp}.log",
    "inference": "inference_{timestamp}.log",
    "evaluation": "evaluation_{timestamp}.log",
    "system": "system_{timestamp}.log"
}

# =============================================================================
# 状态码常量
# =============================================================================

class StatusCode(IntEnum):
    """状态码枚举"""
    SUCCESS = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    
    # 训练状态
    TRAINING_STARTED = 100
    TRAINING_COMPLETED = 101
    TRAINING_STOPPED = 102
    TRAINING_FAILED = 103
    
    # 推理状态
    INFERENCE_STARTED = 200
    INFERENCE_COMPLETED = 201
    INFERENCE_FAILED = 202
    
    # 评估状态
    EVALUATION_STARTED = 300
    EVALUATION_COMPLETED = 301
    EVALUATION_FAILED = 302
    
    # 数据状态
    DATA_LOADED = 400
    DATA_PROCESSED = 401
    DATA_ERROR = 402
    
    # 模型状态
    MODEL_LOADED = 500
    MODEL_SAVED = 501
    MODEL_ERROR = 502


# =============================================================================
# 错误消息常量
# =============================================================================

ERROR_MESSAGES = {
    "file_not_found": "文件未找到: {path}",
    "invalid_format": "无效的文件格式: {format}",
    "dimension_mismatch": "维度不匹配: 期望 {expected}，实际 {actual}",
    "out_of_memory": "内存不足，请减少批量大小或使用更小的模型",
    "cuda_not_available": "CUDA不可用，将使用CPU进行计算",
    "invalid_parameter": "无效参数: {parameter} = {value}",
    "model_not_trained": "模型尚未训练，请先进行训练",
    "checkpoint_not_found": "检查点文件未找到: {path}",
    "data_empty": "数据集为空或无效",
    "convergence_failed": "训练未收敛，请检查学习率和损失函数"
}

# =============================================================================
# 警告消息常量
# =============================================================================

WARNING_MESSAGES = {
    "deprecated_function": "函数 {function} 已弃用，请使用 {alternative}",
    "low_memory": "可用内存较低: {available}MB，建议减少批量大小",
    "slow_convergence": "训练收敛较慢，考虑调整学习率",
    "data_imbalance": "数据不平衡，考虑使用数据增强或重采样",
    "overfitting": "可能存在过拟合，验证损失开始上升",
    "underfitting": "可能存在欠拟合，训练损失较高",
    "gpu_utilization_low": "GPU利用率较低: {utilization}%",
    "large_gradient": "梯度较大: {norm}，可能需要梯度裁剪"
}

# =============================================================================
# 信息消息常量
# =============================================================================

INFO_MESSAGES = {
    "training_started": "开始训练，共 {epochs} 个epoch",
    "epoch_completed": "Epoch {epoch}/{total_epochs} 完成，损失: {loss:.6f}",
    "model_saved": "模型已保存到: {path}",
    "checkpoint_saved": "检查点已保存到: {path}",
    "inference_completed": "推理完成，处理了 {num_samples} 个样本",
    "evaluation_completed": "评估完成，准确率: {accuracy:.4f}",
    "data_loaded": "数据加载完成，共 {num_samples} 个样本",
    "gpu_detected": "检测到 {num_gpus} 个GPU: {gpu_names}",
    "using_device": "使用设备: {device}",
    "memory_usage": "内存使用: {used}MB / {total}MB ({percentage:.1f}%)"
}

# =============================================================================
# 默认配置字典
# =============================================================================

DEFAULT_CONFIG = {
    "model": DEFAULT_MODEL_PARAMS,
    "training": DEFAULT_TRAINING_PARAMS,
    "data": DEFAULT_DATA_PARAMS,
    "evaluation": DEFAULT_EVALUATION_PARAMS,
    "device": DEFAULT_DEVICE_PARAMS,
    "dirs": DEFAULT_DIRS
}

# =============================================================================
# 实用函数
# =============================================================================

def get_default_config(config_type: str) -> Dict:
    """获取默认配置
    
    Args:
        config_type: 配置类型
        
    Returns:
        Dict: 默认配置字典
    """
    return DEFAULT_CONFIG.get(config_type, {}).copy()


def get_supported_formats(format_type: str) -> List[str]:
    """获取支持的文件格式
    
    Args:
        format_type: 格式类型 ('image' 或 'data')
        
    Returns:
        List[str]: 支持的格式列表
    """
    if format_type == "image":
        return SUPPORTED_IMAGE_FORMATS.copy()
    elif format_type == "data":
        return SUPPORTED_DATA_FORMATS.copy()
    else:
        return []


def get_error_message(error_type: str, **kwargs) -> str:
    """获取错误消息
    
    Args:
        error_type: 错误类型
        **kwargs: 格式化参数
        
    Returns:
        str: 格式化的错误消息
    """
    template = ERROR_MESSAGES.get(error_type, "未知错误: {error_type}")
    return template.format(error_type=error_type, **kwargs)


def get_warning_message(warning_type: str, **kwargs) -> str:
    """获取警告消息
    
    Args:
        warning_type: 警告类型
        **kwargs: 格式化参数
        
    Returns:
        str: 格式化的警告消息
    """
    template = WARNING_MESSAGES.get(warning_type, "未知警告: {warning_type}")
    return template.format(warning_type=warning_type, **kwargs)


def get_info_message(info_type: str, **kwargs) -> str:
    """获取信息消息
    
    Args:
        info_type: 信息类型
        **kwargs: 格式化参数
        
    Returns:
        str: 格式化的信息消息
    """
    template = INFO_MESSAGES.get(info_type, "信息: {info_type}")
    return template.format(info_type=info_type, **kwargs)