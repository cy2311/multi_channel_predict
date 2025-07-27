# DECODE神经网络训练

这个目录包含了训练DECODE双层神经网络的完整代码和配置。

## 文件结构

```
training/
├── train_decode_network.py    # 主训练脚本
├── start_training.py          # 便捷启动脚本
├── configs/
│   └── train_config.json      # 训练配置文件
└── README.md                  # 本文件
```

## 快速开始

### 1. 使用便捷启动脚本（推荐）

```bash
# 使用100个样本训练
python start_training.py --samples 100

# 使用50个样本训练
python start_training.py --samples 50

# 使用10个样本训练（快速测试）
python start_training.py --samples 10

# 自定义参数
python start_training.py --samples 100 --epochs 50 --batch_size 8 --lr 2e-4
```

### 2. 直接使用训练脚本

```bash
python train_decode_network.py \
    --config configs/train_config.json \
    --data_root ../../simulation_zmap2tiff/outputs_100samples_256 \
    --output_dir outputs/my_training
```

## 配置说明

### 训练配置 (train_config.json)

```json
{
  "data": {
    "batch_size": 4,           // 批次大小
    "num_workers": 4,          // 数据加载器工作进程数
    "image_size": 256,         // 图像尺寸
    "consecutive_frames": 3,   // 连续帧数
    "sample_subset": null,     // 使用的样本子集数量
    "train_val_split": 0.8     // 训练/验证集分割比例
  },
  "training": {
    "epochs": 100,             // 训练轮数
    "lr_first": 1e-4,          // 第一级网络学习率
    "lr_second": 1e-4,         // 第二级网络学习率
    "weight_decay": 1e-5,      // 权重衰减
    "loss_weights": {          // 损失函数权重
      "count": 1.0,
      "localization": 1.0,
      "photon": 0.5,
      "background": 0.1
    }
  },
  "output": {
    "save_dir": "outputs/training_results"  // 输出目录
  }
}
```

## 输出文件

训练过程中会生成以下文件：

- `latest_checkpoint.pth` - 最新的模型检查点
- `best_model.pth` - 验证损失最低的最佳模型
- `tensorboard/` - TensorBoard日志文件

## 监控训练

使用TensorBoard监控训练过程：

```bash
tensorboard --logdir outputs/train_100samples/tensorboard
```

## 数据要求

训练数据应该位于以下结构中：

```
data_root/
├── sample_001/
│   ├── sample_001.ome.tiff
│   └── emitters.h5
├── sample_002/
│   ├── sample_002.ome.tiff
│   └── emitters.h5
└── ...
```

每个样本包含：
- `.ome.tiff` 文件：多帧TIFF图像
- `emitters.h5` 文件：发射器标注数据

## 训练策略

1. **双层网络架构**：
   - 第一级：三个独立的U-Net网络
   - 第二级：特征融合和最终预测网络

2. **多任务学习**：
   - 发射器计数
   - 精确定位
   - 光子数估计
   - 背景估计

3. **损失函数**：
   - 计数损失：用于发射器检测
   - 定位损失：用于精确位置回归
   - 光子损失：用于光子数估计
   - 背景损失：用于背景建模

## 性能优化

- 使用梯度裁剪防止梯度爆炸
- 学习率调度：每20个epoch衰减0.8倍
- 权重衰减正则化
- 批量归一化和Dropout

## 故障排除

1. **内存不足**：减小batch_size或num_workers
2. **训练太慢**：增加num_workers或使用更小的样本子集
3. **损失不收敛**：调整学习率或损失权重
4. **数据加载错误**：检查数据路径和文件格式