# Loss 损失函数模块

本模块包含DECODE神经网络v3的所有损失函数定义，支持单通道、多通道以及不确定性量化的训练。

## 📋 模块概览

### 核心损失函数

#### 🔹 RatioGaussianNLLLoss (`ratio_loss.py`)
- **功能**: 基于高斯负对数似然的比例预测损失
- **特点**:
  - 支持不确定性量化
  - 集成物理约束（光子数守恒、比例一致性）
  - 可配置的正则化权重
- **用途**: 多通道系统中的比例预测训练

#### 🔹 GaussianMMLoss (`gaussian_mm_loss.py`)
- **功能**: 高斯混合模型损失函数
- **特点**:
  - 支持多峰分布建模
  - 自适应权重学习
  - 处理复杂的空间分布
- **用途**: 复杂发射体分布的建模

#### 🔹 PPXYZBLoss (`ppxyzb_loss.py`)
- **功能**: 针对PPXYZB参数的专用损失
- **特点**:
  - 分别处理位置(XY)、深度(Z)、亮度(B)和概率(P)
  - 加权损失组合
  - 支持不同参数的不同损失类型
- **用途**: 精确的发射体参数估计

#### 🔹 UnifiedLoss (`unified_loss.py`)
- **功能**: 统一的多任务损失函数
- **特点**:
  - 集成多种损失类型
  - 自适应权重平衡
  - 支持任务特定的损失配置
- **用途**: 多任务学习和复杂训练场景

## 🚀 使用示例

### 比例预测损失

```python
from neuronal_network_v3.loss import RatioGaussianNLLLoss

# 初始化损失函数
ratio_loss = RatioGaussianNLLLoss(
    photon_conservation_weight=1.0,    # 光子数守恒权重
    ratio_consistency_weight=0.5,      # 比例一致性权重
    uncertainty_regularization=0.1     # 不确定性正则化权重
)

# 计算损失
loss = ratio_loss(
    ratio_mean=pred_ratio_mean,        # 预测比例均值
    ratio_std=pred_ratio_std,          # 预测比例标准差
    target_ratio=true_ratio,           # 真实比例
    ch1_photons=pred_ch1_photons,      # 通道1预测光子数
    ch2_photons=pred_ch2_photons,      # 通道2预测光子数
    total_photons=true_total_photons   # 真实总光子数
)
```

### 高斯混合模型损失

```python
from neuronal_network_v3.loss import GaussianMMLoss

# 初始化损失函数
gmm_loss = GaussianMMLoss(
    num_components=3,          # 高斯组件数量
    spatial_weight=1.0,       # 空间损失权重
    intensity_weight=0.8      # 强度损失权重
)

# 计算损失
loss = gmm_loss(
    pred_means=pred_means,     # 预测的高斯均值
    pred_stds=pred_stds,       # 预测的高斯标准差
    pred_weights=pred_weights, # 预测的组件权重
    targets=ground_truth       # 真实目标
)
```

### PPXYZB损失

```python
from neuronal_network_v3.loss import PPXYZBLoss

# 初始化损失函数
ppxyzb_loss = PPXYZBLoss(
    position_weight=2.0,       # 位置损失权重
    depth_weight=1.5,          # 深度损失权重
    brightness_weight=1.0,     # 亮度损失权重
    probability_weight=0.8     # 概率损失权重
)

# 计算损失
loss = ppxyzb_loss(
    pred_params=predictions,   # 预测的PPXYZB参数
    target_params=targets      # 真实的PPXYZB参数
)
```

### 统一损失函数

```python
from neuronal_network_v3.loss import UnifiedLoss

# 配置损失组件
loss_config = {
    'spatial_loss': {
        'type': 'mse',
        'weight': 1.0
    },
    'intensity_loss': {
        'type': 'gaussian_nll',
        'weight': 0.8
    },
    'regularization': {
        'type': 'l2',
        'weight': 0.01
    }
}

# 初始化统一损失
unified_loss = UnifiedLoss(loss_config)

# 计算损失
loss = unified_loss(
    predictions=model_output,
    targets=ground_truth,
    model=model  # 用于正则化
)
```

## ⚙️ 损失函数配置

### RatioGaussianNLLLoss参数
- `photon_conservation_weight`: 光子数守恒约束权重
- `ratio_consistency_weight`: 比例一致性约束权重
- `uncertainty_regularization`: 不确定性正则化权重
- `min_std`: 最小标准差阈值
- `reduction`: 损失归约方式（'mean', 'sum', 'none'）

### GaussianMMLoss参数
- `num_components`: 高斯混合组件数量
- `spatial_weight`: 空间位置损失权重
- `intensity_weight`: 强度损失权重
- `regularization_weight`: 正则化权重
- `temperature`: 软分配温度参数

### PPXYZBLoss参数
- `position_weight`: XY位置损失权重
- `depth_weight`: Z深度损失权重
- `brightness_weight`: 亮度损失权重
- `probability_weight`: 检测概率损失权重
- `loss_types`: 各参数的损失函数类型

## 🔧 自定义损失函数

### 创建新的损失函数

```python
import torch
import torch.nn as nn
from neuronal_network_v3.loss.base_loss import BaseLoss

class CustomSMLMLoss(BaseLoss):
    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, predictions, targets, **kwargs):
        # 主要损失
        main_loss = self.mse_loss(predictions, targets)
        
        # 正则化损失
        reg_loss = self.l1_loss(predictions, targets)
        
        # 组合损失
        total_loss = self.alpha * main_loss + self.beta * reg_loss
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'reg_loss': reg_loss
        }
```

### 集成物理约束

```python
class PhysicsConstrainedLoss(nn.Module):
    def __init__(self, base_loss, constraint_weight=1.0):
        super().__init__()
        self.base_loss = base_loss
        self.constraint_weight = constraint_weight
    
    def forward(self, predictions, targets, **kwargs):
        # 基础损失
        base_loss = self.base_loss(predictions, targets)
        
        # 物理约束
        constraint_loss = self.compute_physics_constraints(
            predictions, **kwargs
        )
        
        total_loss = base_loss + self.constraint_weight * constraint_loss
        return total_loss
    
    def compute_physics_constraints(self, predictions, **kwargs):
        # 实现具体的物理约束
        # 例如：能量守恒、质量守恒等
        pass
```

## 📊 损失函数性能对比

| 损失函数 | 收敛速度 | 稳定性 | 内存占用 | 适用场景 |
|----------|----------|--------|----------|----------|
| RatioGaussianNLL | 中等 | 高 | 低 | 比例预测 |
| GaussianMM | 慢 | 中等 | 高 | 复杂分布 |
| PPXYZB | 快 | 高 | 中等 | 参数估计 |
| Unified | 中等 | 高 | 中等 | 多任务学习 |

## 🎯 训练技巧

### 损失权重调优

```python
# 动态权重调整
class AdaptiveWeightScheduler:
    def __init__(self, initial_weights, decay_rate=0.95):
        self.weights = initial_weights
        self.decay_rate = decay_rate
    
    def step(self, epoch, loss_values):
        # 根据损失值动态调整权重
        for key, loss_val in loss_values.items():
            if loss_val < threshold:
                self.weights[key] *= self.decay_rate
        return self.weights
```

### 梯度裁剪

```python
# 在训练循环中使用
loss = loss_function(predictions, targets)
loss.backward()

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

### 损失监控

```python
# 损失组件监控
class LossMonitor:
    def __init__(self):
        self.loss_history = {}
    
    def log_loss(self, epoch, loss_dict):
        for key, value in loss_dict.items():
            if key not in self.loss_history:
                self.loss_history[key] = []
            self.loss_history[key].append(value)
    
    def plot_losses(self):
        # 绘制损失曲线
        import matplotlib.pyplot as plt
        for key, values in self.loss_history.items():
            plt.plot(values, label=key)
        plt.legend()
        plt.show()
```

## 🐛 常见问题

### Q: 损失函数不收敛怎么办？
A: 检查以下几点：
- 学习率是否过大
- 损失权重是否合理
- 数据是否归一化
- 梯度是否爆炸或消失

### Q: 如何平衡多个损失组件？
A: 建议策略：
- 从单一损失开始
- 逐步添加其他组件
- 使用验证集调优权重
- 监控各组件的贡献

### Q: 内存不足怎么办？
A: 优化方法：
- 减少批大小
- 使用梯度累积
- 简化损失计算
- 使用混合精度训练

## 📚 相关文档

- [模型文档](../models/README.md)
- [训练模块文档](../training/README.md)
- [评估模块文档](../evaluation/README.md)
- [多通道训练指南](../README_MultiChannel.md)