# DECODE 神经网络训练指南

本文档提供了关于如何训练和使用DECODE神经网络模型的详细说明。该模型用于从显微镜图像中检测和定位发射体（emitters）。

## 项目结构

```
/home/guest/Others/DECODE_rewrite/
├── neuronal_network/               # 神经网络相关代码
│   ├── first_level_unets.py        # 第一层UNet网络定义
│   ├── second_level_network.py     # 第二层网络定义
│   ├── loss/                       # 损失函数
│   │   ├── count_loss.py           # 计数损失函数
│   ├── train_network.py            # 训练脚本
│   └── inference.py                # 推理脚本
├── simulated_data_multi_frames/    # 模拟数据
│   ├── emitter_sets/               # 发射体信息
│   │   ├── emitters_set0.h5        # 发射体位置和属性
│   │   └── ...
│   └── simulated_multi_frames/     # 模拟图像
│       ├── frames_set0.ome.tiff    # 模拟帧序列
│       └── ...
├── nn_train/                       # 训练输出目录
│   ├── models/                     # 保存的模型
│   └── tensorboard/               # TensorBoard日志
├── train_network.sh                # 训练SLURM提交脚本
└── inference.sh                    # 推理SLURM提交脚本
```

## 网络架构

该模型采用两级网络架构：

1. **第一级网络**：三个共享参数的UNet网络，每个处理一帧图像。每个UNet有3个下采样和上采样层，输出48通道特征图。

2. **第二级网络**：将三个连续帧的特征（共144通道）作为输入，通过另一个UNet网络处理，最终输出每个像素存在发射体的概率图。

## 训练流程

### 数据准备

训练数据包括：
- TIFF格式的图像序列（`simulated_data_multi_frames/simulated_multi_frames/`）
- H5格式的发射体信息（`simulated_data_multi_frames/emitter_sets/`）

### 训练参数

主要训练参数包括：
- `patch_size`：图像块大小（默认600）
- `stride`：图像块滑动步长（默认300）
- `batch_size`：批次大小（默认4）
- `learning_rate`：学习率（默认1e-4）
- `num_epochs`：训练轮数（默认100）

### 启动训练

使用SLURM提交训练任务：

```bash
sbatch train_network.sh
```

训练脚本会：
1. 启动TensorBoard服务（端口6006）
2. 加载数据并进行分块处理
3. 训练模型并保存检查点
4. 记录训练过程中的损失和内存使用情况

### 监控训练

通过TensorBoard监控训练进度：

```bash
# 在本地机器上
ssh -L 6006:localhost:6006 your_username@your_server
# 然后在浏览器中访问 http://localhost:6006
```

## 推理流程

### 运行推理

使用SLURM提交推理任务：

```bash
sbatch inference.sh
```

或者直接运行：

```bash
python neuronal_network/inference.py \
    --tiff_path=/path/to/your/data.ome.tiff \
    --output_dir=/path/to/output \
    --checkpoint=/path/to/model.pth \
    --threshold=0.7
```

### 推理参数

- `tiff_path`：输入TIFF文件路径
- `output_dir`：输出目录
- `checkpoint`：模型检查点路径
- `threshold`：概率阈值（默认0.7），高于此值的像素被视为发射体

### 推理输出

推理脚本会生成：
- 概率图（`.npy`格式）
- 发射体计数（`.npy`格式）
- 可视化结果（`.png`格式）

## 内存优化

为了处理大型图像，该实现采用了以下内存优化策略：

1. **图像分块处理**：将大图像分成较小的块进行处理，然后合并结果。
2. **自动混合精度（AMP）**：使用半精度（FP16）计算以减少内存使用并加速训练。
3. **内存跟踪**：监控训练过程中的内存使用情况。

## 自定义训练

如果需要自定义训练过程，可以修改以下文件：

- `train_network.py`：调整训练参数、数据加载和模型架构
- `train_network.sh`：修改SLURM配置和训练命令行参数

## 故障排除

### 常见问题

1. **内存不足**：
   - 减小`patch_size`
   - 减小`batch_size`
   - 确保启用`--use_amp`

2. **训练不收敛**：
   - 调整`learning_rate`
   - 增加`num_epochs`
   - 检查数据预处理和归一化

3. **推理结果不准确**：
   - 调整`threshold`值
   - 使用更好的训练模型
   - 检查输入数据的质量和预处理

## 引用

DECODE项目基于以下论文：

Artur Speiser, Lucas-Raphael Müller, Philipp Hoess, Ulf Matti, Christopher J. Obara, Wesley R. Legant, Anna Kreshuk, Jakob H. Macke, Jonas Ries, and Srinivas C. Turaga. "Deep learning enables fast and dense single-molecule localization with high accuracy." bioRxiv, 2020.