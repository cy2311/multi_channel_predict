# Stable VAR Emitter Prediction

这是一个经过数值稳定性优化的VAR（Vector-quantized Autoregressive）发射器预测系统，基于DECODE标准实现，解决了原始实现中的训练不稳定问题。

## 主要改进

### 1. 数值稳定性优化
- **使用PyTorch内置损失函数**：替换自定义损失函数，使用`BCEWithLogitsLoss`和`MSELoss`
- **Log-sum-exp技巧**：避免概率计算中的数值溢出
- **更大的eps值**：提高数值稳定性（eps=1e-4）
- **梯度裁剪**：防止梯度爆炸（max_grad_norm=1.0）
- **改进的边界检查**：避免越界访问和无效计算

### 2. DECODE标准兼容
- **6通道统一输出**：[概率, 光子数, x偏移, y偏移, z偏移, 背景]
- **Z维度预测**：支持3D定位
- **背景建模**：显式建模背景信号
- **子像素精度**：亚像素级定位精度

### 3. 架构优化
- **统一输出头**：`UnifiedOutputHead`整合所有预测通道
- **多尺度渐进训练**：从低分辨率到高分辨率的渐进学习
- **自适应权重**：基于训练阶段的动态损失权重

## 文件结构

```
VAR_emitter_prediction/
├── var_emitter_model_unified.py      # 统一6通道模型
├── var_emitter_loss_stable.py        # 稳定损失函数
├── train_var_emitter_stable.py       # 稳定训练器
├── data_loader.py                     # 数据加载器
├── main.py                           # 主训练脚本
├── evaluate.py                       # 评估脚本
├── configs/config_stable.json        # 稳定训练配置
└── README_STABLE.md                  # 本文档
```

## 快速开始

### 1. 环境要求

```bash
python >= 3.8
torch >= 1.9.0
torchvision
numpy
scipy
scikit-learn
matplotlib
seaborn
h5py
tensorboard
```

### 2. 数据准备

数据应该以HDF5格式存储，包含以下字段：
- `image`: 输入图像 [H, W]
- `positions`: 发射器位置 [N, 3] (x, y, z)
- `photons`: 光子数 [N]
- `background`: 背景信号 [N] (可选)

### 3. 训练模型

```bash
# 使用默认配置训练
python main.py --config configs/config_stable.json

# 从检查点恢复训练
python main.py --config configs/config_stable.json --resume models/checkpoint_epoch_50.pth

# 调试模式
python main.py --config configs/config_stable.json --debug

# 干运行（测试配置）
python main.py --config configs/config_stable.json --dry-run
```

### 4. 评估模型

```bash
# 基本评估
python evaluate.py --config configs/config_stable.json \
                   --checkpoint models/best_model.pth \
                   --data-path data/test

# 包含可视化
python evaluate.py --config configs/config_stable.json \
                   --checkpoint models/best_model.pth \
                   --data-path data/test \
                   --visualize \
                   --output-dir results
```

## 配置说明

### 模型配置

```json
{
  "model": {
    "input_size": 40,              // 输入图像尺寸
    "target_sizes": [40, 80, 160, 320],  // 多尺度目标尺寸
    "base_channels": 64,           // 基础通道数
    "embed_dim": 512,              // 嵌入维度
    "num_heads": 8,                // 注意力头数
    "num_layers": 12,              // Transformer层数
    "codebook_size": 1024,         // 码本大小
    "embedding_dim": 256           // 嵌入维度
  }
}
```

### 损失函数配置

```json
{
  "loss": {
    "use_unified_loss": true,      // 使用统一6通道损失
    "channel_weights": [1.5, 1.0, 1.2, 1.2, 0.8, 0.8],  // 通道权重
    "pos_weight": 2.0,             // 正样本权重
    "scale_weights": [0.1, 0.3, 0.6, 1.0],  // 尺度权重
    "eps": 1e-4,                   // 数值稳定性参数
    "warmup_epochs": 20            // 预热轮数
  }
}
```

### 优化器配置

```json
{
  "optimizer": {
    "type": "adamw",               // 优化器类型
    "lr": 1e-4,                   // 学习率
    "weight_decay": 1e-2,         // 权重衰减
    "betas": [0.9, 0.999],        // Adam参数
    "eps": 1e-8                   // 优化器eps
  }
}
```

## 核心组件

### 1. UnifiedEmitterPredictor

统一的6通道发射器预测模型：

```python
from var_emitter_model_unified import UnifiedEmitterPredictor

model = UnifiedEmitterPredictor(
    input_size=40,
    target_sizes=[40, 80, 160, 320],
    base_channels=64,
    embed_dim=512
)

# 前向传播
outputs = model(images)  # 返回多尺度6通道预测
```

### 2. StableVAREmitterLoss

数值稳定的损失函数：

```python
from var_emitter_loss_stable import StableVAREmitterLoss

loss_fn = StableVAREmitterLoss(
    scale_weights=[0.1, 0.3, 0.6, 1.0],
    eps=1e-4,
    warmup_epochs=20
)

loss = loss_fn(predictions, targets, epoch)
```

### 3. UnifiedPPXYZBLoss

6通道统一损失函数：

```python
from var_emitter_loss_stable import UnifiedPPXYZBLoss

loss_fn = UnifiedPPXYZBLoss(
    channel_weights=[1.5, 1.0, 1.2, 1.2, 0.8, 0.8],
    pos_weight=2.0,
    eps=1e-4
)

loss = loss_fn(predictions, targets)
```

## 训练监控

### TensorBoard日志

```bash
tensorboard --logdir logs
```

监控指标：
- 总损失和各组件损失
- 学习率变化
- 梯度范数
- 验证指标（精度、召回率、F1分数）

### 检查点管理

- 自动保存最佳模型（基于验证损失）
- 定期保存检查点（可配置频率）
- 支持训练恢复

## 性能优化建议

### 1. 内存优化
- 使用梯度累积处理大批次
- 启用混合精度训练
- 适当的批次大小（建议8-16）

### 2. 训练稳定性
- 使用预热学习率调度
- 监控梯度范数
- 检查损失值的有限性

### 3. 数据增强
- 随机旋转和翻转
- 噪声注入
- 亮度和对比度调整

## 故障排除

### 常见问题

1. **训练损失不收敛**
   - 检查学习率设置
   - 验证数据格式
   - 调整损失权重

2. **内存不足**
   - 减小批次大小
   - 使用梯度累积
   - 启用混合精度

3. **数值不稳定**
   - 增大eps值
   - 检查梯度裁剪
   - 验证输入数据范围

### 调试模式

```bash
python main.py --config configs/config_stable.json --debug
```

调试模式会输出详细的错误信息和堆栈跟踪。

## 引用

如果您使用此代码，请引用相关论文：

```bibtex
@article{decode2021,
  title={DECODE: Deep learning for comprehensive characterization of single-molecule localization microscopy data},
  author={..},
  journal={Nature Methods},
  year={2021}
}
```

## 许可证

本项目遵循MIT许可证。详见LICENSE文件。