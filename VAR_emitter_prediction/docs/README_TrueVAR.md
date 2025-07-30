# True VAR Emitter Predictor

基于VAR (Visual AutoRegressive) 核心思想的高分辨率结构化emitter预测模型。本实现充分利用VAR的"下一尺度预测"机制，实现从低分辨率输入到高分辨率结构化信息的渐进式预测。

## 🎯 核心特性

### 1. 真正的VAR架构实现
- **渐进式残差累积**: 实现VAR的核心Phi残差网络和多尺度特征融合
- **下一尺度预测**: 采用"next-scale prediction"而非传统的"next-token prediction"
- **自回归尺度生成**: 从低分辨率逐步生成高分辨率结构化信息

### 2. 多尺度渐进式预测
- **尺度序列**: 10×10 → 20×20 → 40×40 → 80×80
- **无强制下采样**: 保持输入分辨率，避免信息丢失
- **残差累积机制**: 通过Phi网络实现跨尺度特征累积

### 3. 高分辨率结构化输出
- **训练分辨率**: 160×160
- **推理能力**: 40×40 → 80×80 (可扩展到更高分辨率)
- **结构化预测**: 同时预测emitter概率图和精确位置

## 🏗️ 架构设计

### 核心组件

#### 1. EmitterVectorQuantizer
```python
class EmitterVectorQuantizer(nn.Module):
    def get_next_autoregressive_input(self, si, SN, f_hat, h_BChw):
        """VAR的核心残差累积机制"""
        # 实现渐进式残差累积和下一尺度输入生成
```

#### 2. AdaLNSelfAttn (VAR Transformer Block)
```python
class AdaLNSelfAttn(nn.Module):
    """VAR的自适应层归一化自注意力机制"""
    # 支持尺度条件的Transformer块
```

#### 3. TrueVAREmitterPredictor
```python
class TrueVAREmitterPredictor(nn.Module):
    """真正的VAR emitter预测器"""
    def autoregressive_inference(self, x, target_resolution):
        """VAR风格的自回归推理"""
```

### 关键设计原则

1. **无下采样编码**: 保持输入分辨率，避免信息丢失
2. **渐进式特征融合**: 通过残差累积实现多尺度信息整合
3. **尺度条件生成**: 每个尺度都有专门的预测头和条件嵌入
4. **VAR自回归机制**: 实现真正的"下一尺度预测"

## 📊 与原始实现的对比

| 特性 | 原始实现 | TrueVAR实现 |
|------|----------|-------------|
| 编码器 | 4倍下采样 | 无下采样 |
| 特征融合 | 简单上采样 | VAR残差累积 |
| 预测机制 | 独立尺度预测 | 自回归尺度预测 |
| 分辨率能力 | 受限于编码器 | 真正高分辨率 |
| VAR核心思想 | 未充分利用 | 完全实现 |

## 🚀 使用方法

### 1. 训练

```bash
# 使用配置文件训练
python train_true_var.py --config config_true_var.json
```

训练配置:
- **输入分辨率**: 160×160
- **目标尺度**: [10, 20, 40, 80]
- **批次大小**: 8
- **学习率**: 1e-4 (带warmup)

### 2. 推理

```bash
# 渐进式多尺度预测
python inference_true_var.py --model best_model.pt --mode progressive

# 超分辨率预测 (40x40 -> 80x80)
python inference_true_var.py --model best_model.pt --mode super_resolution --target_res 80

# 两种模式都运行
python inference_true_var.py --model best_model.pt --mode both
```

### 3. 代码示例

```python
from var_emitter_model_true import TrueVAREmitterPredictor

# 初始化模型
model = TrueVAREmitterPredictor(
    patch_nums=(10, 20, 40, 80),
    embed_dim=768,
    num_heads=12,
    num_layers=24
)

# 渐进式预测
input_image = torch.randn(1, 1, 160, 160)
outputs = model(input_image)

# 超分辨率推理
low_res_input = torch.randn(1, 1, 40, 40)
high_res_output = model.autoregressive_inference(low_res_input, target_resolution=80)
```

## 🔬 技术细节

### VAR残差累积机制

```python
def get_next_autoregressive_input(self, si, SN, f_hat, h_BChw):
    """VAR的核心机制"""
    max_HW = self.patch_nums[-1]
    
    if si != SN - 1:
        # 应用Phi残差网络
        h = self.quant_resi[si](F.interpolate(h_BChw, size=(max_HW, max_HW), mode='bicubic'))
        f_hat.add_(h)  # 残差累积
        # 准备下一尺度输入
        next_input = F.interpolate(f_hat, size=(self.patch_nums[si+1], self.patch_nums[si+1]), mode='area')
        return f_hat, next_input
```

### 多尺度损失函数

```python
class VAREmitterLoss(nn.Module):
    def __init__(self, scale_weights=[0.5, 0.7, 0.9, 1.0]):
        # 不同尺度使用不同权重，高分辨率权重更大
```

### 自适应位置编码

```python
# 每个尺度都有独立的位置编码
self.pos_embeds = nn.ParameterList([
    nn.Parameter(torch.zeros(1, pn * pn, embed_dim))
    for pn in patch_nums
])
```

## 📈 预期效果

### 1. 渐进式分辨率提升
- 10×10: 粗略结构检测
- 20×20: 中等精度定位
- 40×40: 精细结构识别
- 80×80: 高精度emitter预测

### 2. 超分辨率能力
- 输入: 40×40 低分辨率图像
- 输出: 80×80 高分辨率结构化预测
- 保持结构一致性和细节增强

### 3. VAR优势体现
- **结构化生成**: 比传统方法更好的结构保持
- **渐进式细化**: 从粗到细的自然生成过程
- **可扩展性**: 易于扩展到更高分辨率

## 🔧 配置说明

### 模型配置 (config_true_var.json)

```json
{
  "model": {
    "patch_nums": [10, 20, 40, 80],  // VAR尺度序列
    "embed_dim": 768,               // 嵌入维度
    "num_heads": 12,                // 注意力头数
    "num_layers": 24,               // Transformer层数
    "vocab_size": 8192              // 量化词汇表大小
  },
  "training": {
    "input_resolution": [160, 160],  // 训练输入分辨率
    "target_resolutions": {          // 多尺度目标
      "target_10": [10, 10],
      "target_20": [20, 20],
      "target_40": [40, 40],
      "target_80": [80, 80]
    }
  }
}
```

## 🎯 核心创新点

1. **真正的VAR实现**: 完全遵循VAR的"下一尺度预测"范式
2. **无信息丢失**: 移除强制下采样，保持完整输入信息
3. **渐进式残差累积**: 实现VAR的核心Phi网络机制
4. **多尺度自回归**: 真正的尺度条件生成
5. **高分辨率能力**: 支持从低分辨率到高分辨率的结构化预测

## 📝 总结

本实现充分利用了VAR的核心思想，通过"下一尺度预测"机制实现了真正的高分辨率结构化信息预测。相比原始实现，新架构:

- ✅ 实现了真正的VAR残差累积机制
- ✅ 支持渐进式多尺度预测
- ✅ 具备超分辨率生成能力
- ✅ 保持了结构化信息的一致性
- ✅ 可扩展到更高分辨率

这是一个真正意义上的VAR emitter预测器，能够从40×40的低分辨率输入生成80×80甚至更高分辨率的结构化emitter概率图。