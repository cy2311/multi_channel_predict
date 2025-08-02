# Models 模块

本模块包含DECODE神经网络v3的所有模型定义，支持单通道和多通道SMLM数据处理。

## 📋 模块概览

### 核心模型

#### 🔹 SigmaMUNet (`sigma_munet.py`)
- **功能**: 带有不确定性量化的MUNet变体
- **特点**: 
  - 预测均值和方差
  - 支持多尺度特征提取
  - 集成批归一化和dropout
- **用途**: 单通道SMLM数据的主要处理网络

#### 🔹 RatioNet (`ratio_net.py`)
- **功能**: 预测双通道间光子数分配比例
- **特点**:
  - 基于特征提取的比例预测
  - 输出比例的均值和不确定性
  - 支持物理约束集成
- **用途**: 多通道系统中的比例预测

#### 🔹 DoubleMUNet (`double_munet.py`)
- **功能**: 双通道并行处理网络
- **特点**:
  - 两个独立的MUNet分支
  - 共享或独立的编码器
  - 支持特征融合
- **用途**: 双通道数据的联合处理

### 基础模型

#### 🔹 UNet2D (`unet2d.py`)
- **功能**: 标准2D UNet实现
- **特点**:
  - 经典的编码器-解码器架构
  - 跳跃连接
  - 可配置的深度和特征数
- **用途**: 基础的图像分割和回归任务

#### 🔹 SimpleSMLMNet (`simple_smlm_net.py`)
- **功能**: 简化的SMLM处理网络
- **特点**:
  - 轻量级架构
  - 快速推理
  - 适合资源受限环境
- **用途**: 快速原型和轻量级应用

## 🚀 使用示例

### 单通道模型使用

```python
from neuronal_network_v3.models import SigmaMUNet

# 初始化模型
model = SigmaMUNet(
    n_inp=1,        # 输入通道数
    n_out=10,       # 输出通道数
    depth=3,        # 网络深度
    initial_features=64  # 初始特征数
)

# 前向传播
output_mean, output_var = model(input_tensor)
```

### 多通道模型使用

```python
from neuronal_network_v3.models import RatioNet, SigmaMUNet

# 初始化通道网络
channel1_net = SigmaMUNet(n_inp=1, n_out=10)
channel2_net = SigmaMUNet(n_inp=1, n_out=10)

# 初始化比例网络
ratio_net = RatioNet(
    input_channels=20,  # 两个通道的特征总数
    hidden_dim=128,     # 隐藏层维度
    num_layers=3        # 网络层数
)

# 联合推理
ch1_output = channel1_net(ch1_input)
ch2_output = channel2_net(ch2_input)

# 提取特征用于比例预测
features = torch.cat([ch1_output[0], ch2_output[0]], dim=1)
ratio_mean, ratio_var = ratio_net(features)
```

## ⚙️ 模型配置

### 通用参数
- `n_inp`: 输入通道数
- `n_out`: 输出通道数
- `depth`: 网络深度（编码器层数）
- `initial_features`: 第一层的特征数
- `norm_layer`: 归一化层类型（BatchNorm2d, InstanceNorm2d等）
- `activation`: 激活函数类型

### SigmaMUNet特有参数
- `predict_variance`: 是否预测方差
- `dropout_rate`: Dropout比率
- `use_attention`: 是否使用注意力机制

### RatioNet特有参数
- `input_channels`: 输入特征通道数
- `hidden_dim`: 隐藏层维度
- `num_layers`: 全连接层数
- `output_activation`: 输出激活函数

## 🔧 自定义模型

### 继承基类创建新模型

```python
import torch.nn as nn
from neuronal_network_v3.models.unet2d import UNet2D

class CustomSMLMNet(UNet2D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 添加自定义层
        self.custom_head = nn.Conv2d(
            self.n_out, self.n_out * 2, 1
        )
    
    def forward(self, x):
        # 基础UNet前向传播
        features = super().forward(x)
        # 自定义处理
        output = self.custom_head(features)
        return output
```

## 📊 模型性能

| 模型 | 参数量 | 推理速度 | 内存占用 | 适用场景 |
|------|--------|----------|----------|----------|
| SimpleSMLMNet | ~1M | 快 | 低 | 快速原型 |
| UNet2D | ~5M | 中等 | 中等 | 基础任务 |
| SigmaMUNet | ~8M | 中等 | 中等 | 不确定性量化 |
| DoubleMUNet | ~16M | 慢 | 高 | 双通道处理 |
| RatioNet | ~0.5M | 快 | 低 | 比例预测 |

## 🐛 常见问题

### Q: 如何选择合适的网络深度？
A: 一般建议：
- 小图像（<128x128）：depth=2-3
- 中等图像（128-512）：depth=3-4
- 大图像（>512）：depth=4-5

### Q: 内存不足怎么办？
A: 可以尝试：
- 减少`initial_features`
- 降低`depth`
- 使用梯度检查点
- 减小批大小

### Q: 如何提高推理速度？
A: 建议：
- 使用SimpleSMLMNet
- 启用模型编译
- 使用半精度推理
- 批量处理

## 📚 相关文档

- [训练模块文档](../training/README.md)
- [损失函数文档](../loss/README.md)
- [推理模块文档](../inference/README.md)
- [多通道训练指南](../README_MultiChannel.md)