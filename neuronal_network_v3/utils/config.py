"""配置管理模块

该模块提供了DECODE神经网络的配置管理功能，包括：
- 各种配置类的定义
- 配置文件的加载和保存
- 配置验证和合并
- 默认配置的管理
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置类"""
    architecture: str = "SimpleSMLMNet"  # 模型架构
    input_channels: int = 1  # 输入通道数
    output_channels: int = 5  # 输出通道数 (p, x, y, z, photons)
    output_mode: str = "ppxyzb"  # 输出模式: ppxyzb, sigma_mu, simple
    
    # UNet参数
    depth: int = 4  # UNet深度
    initial_features: int = 64  # 初始特征数
    growth_factor: int = 2  # 特征增长因子
    
    # 激活函数和归一化
    activation: str = "ReLU"  # 激活函数
    normalization: str = "BatchNorm2d"  # 归一化方法
    dropout_rate: float = 0.1  # Dropout率
    
    # 高级特性
    use_attention: bool = False  # 是否使用注意力机制
    use_multiscale: bool = False  # 是否使用多尺度特征
    use_residual: bool = True  # 是否使用残差连接
    
    # 模型特定参数
    sigma_range: tuple = (0.5, 3.0)  # sigma范围（用于GaussianMM）
    num_gaussians: int = 3  # 高斯分量数量
    
    def __post_init__(self):
        """后处理验证"""
        if self.output_mode == "ppxyzb" and self.output_channels != 6:
            self.output_channels = 6
        elif self.output_mode == "simple" and self.output_channels != 5:
            self.output_channels = 5
        elif self.output_mode == "sigma_mu" and self.output_channels != 10:
            self.output_channels = 10


@dataclass
class DataConfig:
    """数据配置类"""
    # 数据路径
    train_data_path: str = ""
    val_data_path: str = ""
    test_data_path: str = ""
    
    # 数据格式
    data_format: str = "hdf5"  # 数据格式: hdf5, tiff, numpy
    input_key: str = "frames"  # 输入数据键名
    target_key: str = "emitters"  # 目标数据键名
    
    # 数据预处理
    normalize: bool = True  # 是否归一化
    normalization_method: str = "zscore"  # 归一化方法: zscore, minmax, percentile
    clip_percentile: tuple = (1, 99)  # 裁剪百分位数
    
    # 数据增强
    augmentation: bool = True  # 是否使用数据增强
    flip_probability: float = 0.5  # 翻转概率
    rotation_probability: float = 0.3  # 旋转概率
    noise_probability: float = 0.2  # 噪声概率
    noise_std: float = 0.1  # 噪声标准差
    
    # 数据加载
    batch_size: int = 16  # 批量大小
    num_workers: int = 4  # 工作进程数
    pin_memory: bool = True  # 是否固定内存
    prefetch_factor: int = 2  # 预取因子
    
    # 数据分割
    train_split: float = 0.8  # 训练集比例
    val_split: float = 0.1  # 验证集比例
    test_split: float = 0.1  # 测试集比例
    
    # 像素和物理参数
    pixel_size: float = 100.0  # 像素大小（nm）
    frame_size: tuple = (64, 64)  # 帧大小
    z_range: tuple = (-500, 500)  # Z轴范围（nm）
    
    # 缓存设置
    use_cache: bool = True  # 是否使用缓存
    cache_size: int = 1000  # 缓存大小
    preload_data: bool = False  # 是否预加载数据


@dataclass
class OptimizationConfig:
    """优化配置类"""
    # 优化器
    optimizer: str = "Adam"  # 优化器类型
    learning_rate: float = 1e-3  # 学习率
    weight_decay: float = 1e-4  # 权重衰减
    
    # Adam参数
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # SGD参数
    momentum: float = 0.9
    nesterov: bool = True
    
    # 学习率调度
    scheduler: str = "StepLR"  # 调度器类型
    step_size: int = 30  # 步长
    gamma: float = 0.1  # 衰减因子
    
    # CosineAnnealingLR参数
    T_max: int = 100
    eta_min: float = 1e-6
    
    # ReduceLROnPlateau参数
    patience: int = 10
    factor: float = 0.5
    threshold: float = 1e-4
    
    # 梯度裁剪
    gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    
    # 混合精度
    use_amp: bool = True  # 是否使用自动混合精度
    amp_opt_level: str = "O1"  # AMP优化级别


@dataclass
class TrainingConfig:
    """训练配置类"""
    # 基本训练参数
    epochs: int = 100  # 训练轮数
    save_interval: int = 10  # 保存间隔
    eval_interval: int = 5  # 评估间隔
    log_interval: int = 100  # 日志间隔
    
    # 早停
    early_stopping: bool = True  # 是否使用早停
    patience: int = 20  # 早停耐心值
    min_delta: float = 1e-4  # 最小改进
    
    # 检查点
    save_best_only: bool = True  # 是否只保存最佳模型
    save_last: bool = True  # 是否保存最后一个模型
    monitor_metric: str = "val_loss"  # 监控指标
    monitor_mode: str = "min"  # 监控模式: min, max
    
    # 输出路径
    output_dir: str = "./outputs"  # 输出目录
    experiment_name: str = "decode_experiment"  # 实验名称
    
    # 日志
    use_tensorboard: bool = True  # 是否使用TensorBoard
    use_wandb: bool = False  # 是否使用Weights & Biases
    log_level: str = "INFO"  # 日志级别
    
    # 分布式训练
    distributed: bool = False  # 是否使用分布式训练
    world_size: int = 1  # 世界大小
    rank: int = 0  # 进程排名
    local_rank: int = 0  # 本地排名
    
    # 设备设置
    device: str = "auto"  # 设备: auto, cpu, cuda, cuda:0
    num_gpus: int = 1  # GPU数量
    
    # 随机种子
    seed: int = 42  # 随机种子
    deterministic: bool = True  # 是否确定性训练


@dataclass
class InferenceConfig:
    """推理配置类"""
    # 模型路径
    model_path: str = ""  # 模型权重路径
    config_path: str = ""  # 配置文件路径
    
    # 推理参数
    batch_size: int = 32  # 批量大小
    device: str = "auto"  # 设备
    use_amp: bool = True  # 是否使用混合精度
    
    # 后处理参数
    detection_threshold: float = 0.5  # 检测阈值
    nms_threshold: float = 0.3  # NMS阈值
    max_detections: int = 1000  # 最大检测数
    
    # 峰值检测
    peak_method: str = "local_maxima"  # 峰值检测方法
    min_distance: int = 3  # 最小距离
    exclude_border: bool = True  # 是否排除边界
    
    # 输出设置
    output_format: str = "csv"  # 输出格式: csv, hdf5, json
    output_path: str = "./results"  # 输出路径
    save_predictions: bool = True  # 是否保存预测结果
    
    # 内存优化
    memory_efficient: bool = True  # 是否内存高效
    max_memory_gb: float = 8.0  # 最大内存使用（GB）


@dataclass
class EvaluationConfig:
    """评估配置类"""
    # 评估指标
    metrics: List[str] = field(default_factory=lambda: [
        "precision", "recall", "f1", "jaccard", "rmse", "mae"
    ])
    
    # 匹配参数
    tolerance_xy: float = 100.0  # XY容差（nm）
    tolerance_z: float = 200.0  # Z容差（nm）
    tolerance_photons: float = 0.2  # 光子数容差（相对）
    
    # 评估设置
    confidence_threshold: float = 0.5  # 置信度阈值
    iou_threshold: float = 0.5  # IoU阈值
    
    # 可视化
    create_plots: bool = True  # 是否创建图表
    plot_format: str = "png"  # 图表格式
    dpi: int = 300  # 图表DPI
    
    # 基准测试
    benchmark_datasets: List[str] = field(default_factory=list)  # 基准数据集
    compare_methods: List[str] = field(default_factory=list)  # 比较方法


@dataclass
class Config:
    """主配置类"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    version: str = "1.0.0"  # 配置版本
    description: str = "DECODE神经网络配置"  # 配置描述


def load_config(config_path: Union[str, Path]) -> Config:
    """加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Config: 配置对象
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    # 根据文件扩展名选择加载方法
    if config_path.suffix.lower() == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
    elif config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    else:
        raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
    
    # 创建配置对象
    config = Config()
    
    # 更新配置
    if 'model' in config_dict:
        config.model = ModelConfig(**config_dict['model'])
    if 'data' in config_dict:
        config.data = DataConfig(**config_dict['data'])
    if 'optimization' in config_dict:
        config.optimization = OptimizationConfig(**config_dict['optimization'])
    if 'training' in config_dict:
        config.training = TrainingConfig(**config_dict['training'])
    if 'inference' in config_dict:
        config.inference = InferenceConfig(**config_dict['inference'])
    if 'evaluation' in config_dict:
        config.evaluation = EvaluationConfig(**config_dict['evaluation'])
    
    # 更新顶级属性
    for key in ['version', 'description']:
        if key in config_dict:
            setattr(config, key, config_dict[key])
    
    logger.info(f"配置文件加载成功: {config_path}")
    return config


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """保存配置文件
    
    Args:
        config: 配置对象
        config_path: 配置文件路径
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换为字典
    config_dict = asdict(config)
    
    # 根据文件扩展名选择保存方法
    if config_path.suffix.lower() == '.json':
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    elif config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, 
                     allow_unicode=True, indent=2)
    else:
        raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
    
    logger.info(f"配置文件保存成功: {config_path}")


def merge_configs(base_config: Config, override_config: Dict[str, Any]) -> Config:
    """合并配置
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置字典
        
    Returns:
        Config: 合并后的配置
    """
    # 转换为字典
    base_dict = asdict(base_config)
    
    # 递归合并
    def _merge_dict(base: Dict, override: Dict) -> Dict:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = _merge_dict(base[key], value)
            else:
                base[key] = value
        return base
    
    merged_dict = _merge_dict(base_dict, override_config)
    
    # 重新创建配置对象
    config = Config()
    if 'model' in merged_dict:
        config.model = ModelConfig(**merged_dict['model'])
    if 'data' in merged_dict:
        config.data = DataConfig(**merged_dict['data'])
    if 'optimization' in merged_dict:
        config.optimization = OptimizationConfig(**merged_dict['optimization'])
    if 'training' in merged_dict:
        config.training = TrainingConfig(**merged_dict['training'])
    if 'inference' in merged_dict:
        config.inference = InferenceConfig(**merged_dict['inference'])
    if 'evaluation' in merged_dict:
        config.evaluation = EvaluationConfig(**merged_dict['evaluation'])
    
    # 更新顶级属性
    for key in ['version', 'description']:
        if key in merged_dict:
            setattr(config, key, merged_dict[key])
    
    return config


def validate_config(config: Config) -> List[str]:
    """验证配置
    
    Args:
        config: 配置对象
        
    Returns:
        List[str]: 验证错误列表
    """
    errors = []
    
    # 验证模型配置
    if config.model.output_channels <= 0:
        errors.append("模型输出通道数必须大于0")
    
    if config.model.depth <= 0:
        errors.append("UNet深度必须大于0")
    
    if config.model.initial_features <= 0:
        errors.append("初始特征数必须大于0")
    
    # 验证数据配置
    if not config.data.train_data_path:
        errors.append("训练数据路径不能为空")
    
    if config.data.batch_size <= 0:
        errors.append("批量大小必须大于0")
    
    if config.data.num_workers < 0:
        errors.append("工作进程数不能为负数")
    
    # 验证优化配置
    if config.optimization.learning_rate <= 0:
        errors.append("学习率必须大于0")
    
    if config.optimization.weight_decay < 0:
        errors.append("权重衰减不能为负数")
    
    # 验证训练配置
    if config.training.epochs <= 0:
        errors.append("训练轮数必须大于0")
    
    if config.training.patience <= 0:
        errors.append("早停耐心值必须大于0")
    
    # 验证推理配置
    if config.inference.detection_threshold < 0 or config.inference.detection_threshold > 1:
        errors.append("检测阈值必须在0-1之间")
    
    if config.inference.batch_size <= 0:
        errors.append("推理批量大小必须大于0")
    
    # 验证评估配置
    if config.evaluation.tolerance_xy <= 0:
        errors.append("XY容差必须大于0")
    
    if config.evaluation.tolerance_z <= 0:
        errors.append("Z容差必须大于0")
    
    return errors


def create_default_config() -> Config:
    """创建默认配置
    
    Returns:
        Config: 默认配置对象
    """
    return Config()


def get_config_template() -> Dict[str, Any]:
    """获取配置模板
    
    Returns:
        Dict[str, Any]: 配置模板字典
    """
    return asdict(create_default_config())