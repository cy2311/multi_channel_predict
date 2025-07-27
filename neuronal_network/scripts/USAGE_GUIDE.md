# DECODE神经网络批量训练系统使用指南

## 系统概述

本系统为DECODE神经网络提供了完整的批量训练解决方案，支持在多GPU环境下并行训练不同样本数量的模型。

### 系统特性
- ✅ **多GPU并行训练**: 自动检测并利用所有可用GPU
- ✅ **批量任务管理**: 同时运行多个不同配置的训练任务
- ✅ **智能资源分配**: 自动分配GPU资源，避免冲突
- ✅ **实时监控**: TensorBoard集成，实时查看训练进度
- ✅ **任务状态管理**: 启动、停止、清理训练任务
- ✅ **交互式界面**: 友好的命令行界面

## 快速开始

### 1. 检查系统状态
```bash
# 查看GPU状态和训练数据
./quick_start.sh --status
```

### 2. 启动批量训练
```bash
# 使用默认配置启动批量训练
./quick_start.sh --start

# 或者进入交互模式
./quick_start.sh
```

### 3. 监控训练进度
```bash
# 查看任务状态
python batch_train_manager.py --status

# 监控GPU使用情况
watch -n 1 nvidia-smi
```

## 默认训练配置

系统预设了4个训练配置，充分利用4块RTX 3090 GPU：

| 配置 | 样本数 | 训练轮数 | GPU | 用途 |
|------|--------|----------|-----|------|
| 配置1 | 10样本 | 2轮 | GPU 0 | 快速测试 |
| 配置2 | 50样本 | 5轮 | GPU 1 | 小规模训练 |
| 配置3 | 100样本 | 10轮 | GPU 2 | 中等规模训练 |
| 配置4 | 200样本 | 15轮 | GPU 3 | 大规模训练 |

## 核心脚本说明

### 1. quick_start.sh - 快速启动脚本
主要的用户界面，提供所有功能的统一入口。

```bash
# 显示帮助
./quick_start.sh --help

# 显示GPU状态
./quick_start.sh --gpu

# 显示训练配置
./quick_start.sh --config

# 启动批量训练
./quick_start.sh --start

# 停止所有任务
./quick_start.sh --stop

# 运行演示
./quick_start.sh --demo
```

### 2. batch_train_manager.py - 批量训练管理器
核心的训练任务管理器，支持任务的启动、监控和管理。

```bash
# 显示配置
python batch_train_manager.py --config

# 显示任务状态
python batch_train_manager.py --status

# 启动批量训练
python batch_train_manager.py

# 停止所有任务
python batch_train_manager.py --stop-all

# 清理完成的任务
python batch_train_manager.py --cleanup
```

### 3. demo_batch_training.py - 交互式演示
提供交互式演示，展示系统的各种功能。

```bash
python demo_batch_training.py
```

## 高级功能

### 自定义训练配置

可以通过修改 `batch_train_manager.py` 中的配置来自定义训练参数：

```python
# 在 get_default_configs() 函数中修改
configs = [
    {
        'name': '自定义配置',
        'samples': 300,  # 样本数
        'epochs': 20,    # 训练轮数
        'gpu_id': 0,     # GPU ID
        'batch_size': 8, # 批大小
        'lr': 0.001      # 学习率
    }
]
```

### TensorBoard监控

每个训练任务都会自动启动TensorBoard服务：

```bash
# 查看TensorBoard端口
python batch_train_manager.py --status

# 在浏览器中访问
# http://localhost:6106  (端口号根据任务而定)
```

### 日志文件

训练日志保存在输出目录中：
```
training/outputs/train_XXXsamples_YYYYMMDD_HHMMSS/
├── training.log          # 训练日志
├── tensorboard_logs/     # TensorBoard日志
├── latest_checkpoint.pth # 最新检查点
└── best_model.pth       # 最佳模型
```

## 故障排除

### 常见问题

1. **GPU内存不足**
   ```bash
   # 检查GPU使用情况
   nvidia-smi
   
   # 停止占用GPU的任务
   ./quick_start.sh --stop
   ```

2. **训练数据不存在**
   ```bash
   # 生成训练数据
   ./quick_start.sh --generate-data
   ```

3. **端口冲突**
   ```bash
   # 清理完成的任务
   python batch_train_manager.py --cleanup
   ```

### 系统要求

- Python 3.7+
- PyTorch 1.8+
- NVIDIA GPU (推荐RTX 3090或更高)
- CUDA 11.0+
- 足够的磁盘空间用于训练数据和模型输出

## 性能优化建议

1. **数据预处理**: 确保训练数据已预处理并缓存
2. **批大小调整**: 根据GPU内存调整批大小
3. **混合精度训练**: 启用FP16以提高训练速度
4. **数据加载**: 使用多进程数据加载器

## 输出文件结构

```
training/outputs/
├── train_10samples_YYYYMMDD_HHMMSS/     # 10样本训练输出
├── train_50samples_YYYYMMDD_HHMMSS/     # 50样本训练输出
├── train_100samples_YYYYMMDD_HHMMSS/    # 100样本训练输出
└── train_200samples_YYYYMMDD_HHMMSS/    # 200样本训练输出
```

每个训练目录包含：
- `training.log`: 详细的训练日志
- `tensorboard_logs/`: TensorBoard可视化数据
- `*.pth`: 模型检查点文件
- `config.json`: 训练配置信息

## 联系支持

如果遇到问题，请检查：
1. 系统日志文件
2. GPU状态和内存使用
3. 训练数据完整性
4. 依赖包版本兼容性

---

**祝您训练愉快！** 🚀