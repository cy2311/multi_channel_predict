# 从Checkpoint恢复训练指南

本文档说明如何使用修改后的代码从之前的checkpoint恢复训练。

## 功能说明

我们对`train_with_count_loss.py`文件进行了修改，添加了从checkpoint恢复训练的功能。主要修改包括：

1. 添加了`checkpoint_path`参数，用于指定要恢复的checkpoint文件路径
2. 在训练开始前加载checkpoint中的模型状态、优化器状态和训练epoch
3. 从上次中断的epoch继续训练，而不是从头开始

## 使用方法

### 方法1：使用resume_training.py脚本（推荐）

我们提供了一个简单的脚本`resume_training.py`，可以直接用来恢复训练：

```bash
# 使用默认的checkpoint（nn_train/checkpoint_epoch_250.pt）
./resume_training.py

# 或者指定其他checkpoint文件
./resume_training.py nn_train/checkpoint_epoch_100.pt
```

### 方法2：直接运行train_with_count_loss.py

您也可以直接运行`train_with_count_loss.py`并传递checkpoint路径作为命令行参数：

```bash
python neuronal_network/train_with_count_loss.py nn_train/checkpoint_epoch_250.pt
```

## 注意事项

1. 确保checkpoint文件存在且路径正确
2. 恢复训练时会使用checkpoint中保存的模型参数、优化器状态
3. 训练会从checkpoint保存的epoch之后开始，而不是从头开始
4. 如果提供的checkpoint文件不存在，程序会给出警告并从头开始训练

## 故障排除

如果遇到问题，请检查：

1. checkpoint文件是否存在
2. checkpoint文件格式是否正确（应包含模型状态字典、优化器状态字典和epoch信息）
3. 训练参数（如batch_size、learning_rate等）是否与生成checkpoint时一致