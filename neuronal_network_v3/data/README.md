# Data 数据处理模块

本模块包含DECODE神经网络v3的数据处理、加载和变换功能，支持单通道和多通道SMLM数据的高效处理。

## 📋 模块概览

### 核心组件

#### 🔹 MultiChannelSMLMDataset (`multi_channel_dataset.py`)
- **功能**: 多通道SMLM数据集处理
- **特点**:
  - 支持HDF5格式数据
  - 双通道数据同步加载
  - 自动数据验证和清理
  - 灵活的数据分割策略
- **用途**: 多通道训练和推理的主要数据接口

#### 🔹 SMLMDataset (`dataset.py`)
- **功能**: 基础SMLM数据集类
- **特点**:
  - 标准化的数据接口
  - 内存高效的数据加载
  - 支持多种数据格式
  - 可配置的预处理流水线
- **用途**: 单通道数据处理和基础数据操作

#### 🔹 数据变换 (`transforms.py`)
- **功能**: 数据增强和预处理变换
- **特点**:
  - 专为SMLM数据设计的变换
  - 支持几何和强度变换
  - 保持物理一致性
  - 可组合的变换流水线
- **用途**: 数据增强、归一化和预处理

## 🚀 使用示例

### 多通道数据集使用

```python
from neuronal_network_v3.data import MultiChannelSMLMDataset, create_multi_channel_dataloader
from neuronal_network_v3.data.transforms import get_default_transforms

# 创建数据集
dataset = MultiChannelSMLMDataset(
    data_path='data/multi_channel_train.h5',
    mode='train',
    patch_size=64,
    transform=get_default_transforms(mode='train')
)

# 创建数据加载器
dataloader = create_multi_channel_dataloader(
    data_path='data/multi_channel_train.h5',
    config={
        'batch_size': 16,
        'num_workers': 4,
        'patch_size': 64
    },
    mode='train'
)

# 数据迭代
for batch in dataloader:
    ch1_images = batch['channel1']['images']    # [B, 1, H, W]
    ch2_images = batch['channel2']['images']    # [B, 1, H, W]
    ch1_targets = batch['channel1']['targets']  # [B, C, H, W]
    ch2_targets = batch['channel2']['targets']  # [B, C, H, W]
    metadata = batch['metadata']                # 元数据信息
```

### 单通道数据集使用

```python
from neuronal_network_v3.data import SMLMDataset
from torch.utils.data import DataLoader

# 创建数据集
dataset = SMLMDataset(
    data_path='data/single_channel_train.h5',
    mode='train',
    patch_size=64,
    augment=True
)

# 创建数据加载器
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# 数据迭代
for images, targets in dataloader:
    # images: [B, 1, H, W]
    # targets: [B, C, H, W]
    pass
```

### 数据变换使用

```python
from neuronal_network_v3.data.transforms import (
    RandomRotation, RandomFlip, GaussianNoise, 
    Normalize, Compose
)

# 创建变换流水线
transforms = Compose([
    RandomRotation(angle_range=(-180, 180)),  # 随机旋转
    RandomFlip(p=0.5),                        # 随机翻转
    GaussianNoise(noise_std=0.1),             # 高斯噪声
    Normalize(mean=0.5, std=0.2)              # 归一化
])

# 应用变换
transformed_data = transforms({
    'image': image,
    'target': target,
    'metadata': metadata
})
```

## 📊 数据格式规范

### HDF5数据结构

```
data_file.h5
├── train/
│   ├── channel1/
│   │   ├── images          # [N, H, W] 输入图像
│   │   ├── targets         # [N, C, H, W] 目标数据
│   │   ├── photons         # [N] 光子数信息
│   │   └── positions       # [N, M, 3] 发射体位置 (x, y, z)
│   ├── channel2/
│   │   ├── images          # [N, H, W] 输入图像
│   │   ├── targets         # [N, C, H, W] 目标数据
│   │   ├── photons         # [N] 光子数信息
│   │   └── positions       # [N, M, 3] 发射体位置
│   └── metadata
│       ├── pixel_size      # 像素尺寸 (nm)
│       ├── wavelength      # 波长信息
│       ├── na              # 数值孔径
│       └── acquisition_params  # 采集参数
├── val/
│   └── ... (同train结构)
└── test/
    └── ... (同train结构)
```

### 目标数据格式

目标数据通常包含以下通道：
- **通道0**: 检测概率图
- **通道1-2**: X, Y位置偏移
- **通道3**: Z位置信息
- **通道4**: 亮度信息
- **通道5-6**: X, Y位置不确定性
- **通道7**: Z位置不确定性
- **通道8**: 亮度不确定性
- **通道9**: 背景信息

## ⚙️ 数据配置

### 数据集参数

```python
# MultiChannelSMLMDataset配置
dataset_config = {
    'data_path': 'path/to/data.h5',
    'mode': 'train',           # 'train', 'val', 'test'
    'patch_size': 64,          # 图像块大小
    'overlap': 0.1,            # 图像块重叠比例
    'min_photons': 100,        # 最小光子数阈值
    'max_photons': 10000,      # 最大光子数阈值
    'normalize': True,         # 是否归一化
    'augment': True,           # 是否数据增强
    'cache_size': 1000         # 缓存大小
}
```

### 数据加载器参数

```python
# DataLoader配置
loader_config = {
    'batch_size': 16,          # 批大小
    'num_workers': 4,          # 工作进程数
    'pin_memory': True,        # 是否固定内存
    'drop_last': True,         # 是否丢弃最后不完整批次
    'persistent_workers': True, # 是否保持工作进程
    'prefetch_factor': 2       # 预取因子
}
```

## 🔧 数据变换详解

### 几何变换

#### RandomRotation
```python
# 随机旋转变换
rotation = RandomRotation(
    angle_range=(-180, 180),   # 旋转角度范围
    p=0.8,                     # 应用概率
    interpolation='bilinear'   # 插值方法
)
```

#### RandomFlip
```python
# 随机翻转变换
flip = RandomFlip(
    horizontal=True,           # 水平翻转
    vertical=True,             # 垂直翻转
    p=0.5                      # 应用概率
)
```

#### RandomCrop
```python
# 随机裁剪变换
crop = RandomCrop(
    size=(64, 64),             # 裁剪尺寸
    padding=4,                 # 填充大小
    padding_mode='reflect'     # 填充模式
)
```

### 强度变换

#### GaussianNoise
```python
# 高斯噪声变换
noise = GaussianNoise(
    noise_std=0.1,             # 噪声标准差
    p=0.7                      # 应用概率
)
```

#### RandomBrightness
```python
# 随机亮度变换
brightness = RandomBrightness(
    factor_range=(0.8, 1.2),   # 亮度因子范围
    p=0.6                      # 应用概率
)
```

#### PoissonNoise
```python
# 泊松噪声变换（模拟光子噪声）
poisson = PoissonNoise(
    scale_factor=1.0,          # 缩放因子
    p=0.5                      # 应用概率
)
```

### 归一化变换

#### Normalize
```python
# 标准化变换
normalize = Normalize(
    mean=0.5,                  # 均值
    std=0.2,                   # 标准差
    per_channel=True           # 是否按通道归一化
)
```

#### ZScoreNormalize
```python
# Z-score归一化
zscore = ZScoreNormalize(
    eps=1e-8                   # 数值稳定性参数
)
```

## 🚀 高级功能

### 数据缓存

```python
# 启用数据缓存以提高训练速度
dataset = MultiChannelSMLMDataset(
    data_path='data/large_dataset.h5',
    cache_size=5000,           # 缓存样本数
    cache_mode='memory',       # 'memory' 或 'disk'
    cache_dir='/tmp/cache'     # 磁盘缓存目录
)
```

### 数据验证

```python
# 数据完整性验证
from neuronal_network_v3.data.validation import DataValidator

validator = DataValidator()
validation_report = validator.validate_dataset(
    data_path='data/dataset.h5',
    check_consistency=True,    # 检查数据一致性
    check_completeness=True,   # 检查数据完整性
    check_quality=True         # 检查数据质量
)

print(validation_report)
```

### 数据统计

```python
# 获取数据集统计信息
stats = dataset.get_statistics()
print(f"数据集大小: {stats['size']}")
print(f"图像尺寸: {stats['image_shape']}")
print(f"光子数范围: {stats['photon_range']}")
print(f"发射体密度: {stats['emitter_density']}")
```

### 自定义数据变换

```python
from neuronal_network_v3.data.transforms import BaseTransform

class CustomTransform(BaseTransform):
    def __init__(self, param1, param2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def __call__(self, data):
        image = data['image']
        target = data['target']
        
        # 自定义变换逻辑
        transformed_image = self.apply_transform(image)
        transformed_target = self.apply_transform(target)
        
        return {
            'image': transformed_image,
            'target': transformed_target,
            'metadata': data['metadata']
        }
    
    def apply_transform(self, tensor):
        # 实现具体的变换
        return tensor
```

## 📊 性能优化

### 数据加载优化

```python
# 优化数据加载性能
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,             # 增加工作进程
    pin_memory=True,           # GPU训练时启用
    persistent_workers=True,   # 保持工作进程
    prefetch_factor=4          # 增加预取
)
```

### 内存优化

```python
# 内存高效的数据处理
dataset = MultiChannelSMLMDataset(
    data_path='data/large_dataset.h5',
    lazy_loading=True,         # 延迟加载
    memory_map=True,           # 内存映射
    chunk_size=1000            # 分块处理
)
```

## 🐛 常见问题

### Q: 数据加载速度慢怎么办？
A: 优化建议：
- 增加`num_workers`
- 启用`pin_memory`
- 使用SSD存储
- 启用数据缓存
- 优化HDF5文件结构

### Q: 内存不足怎么办？
A: 解决方案：
- 减少`batch_size`
- 启用`lazy_loading`
- 使用内存映射
- 减少缓存大小
- 使用数据流式处理

### Q: 如何处理不平衡数据？
A: 策略：
- 使用加权采样
- 数据增强
- 损失函数加权
- 分层采样

### Q: 数据变换后目标不匹配？
A: 检查：
- 变换是否同时应用于图像和目标
- 坐标变换是否正确
- 插值方法是否合适
- 边界处理是否正确

## 📚 相关文档

- [模型文档](../models/README.md)
- [训练模块文档](../training/README.md)
- [推理模块文档](../inference/README.md)
- [多通道训练指南](../README_MultiChannel.md)
- [数据格式规范](./DATA_FORMAT.md)