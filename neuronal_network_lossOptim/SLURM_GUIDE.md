# DECODE自适应训练 SLURM使用指南

本指南介绍如何在SLURM集群上运行DECODE自适应训练。

## 📁 文件说明

### 脚本文件
- `submit_adaptive_training.sh` - 基础SLURM提交脚本
- `submit_adaptive_training_flexible.sh` - 灵活配置的SLURM脚本
- `quick_submit.sh` - 快速提交工具

### 配置文件
- `training/configs/train_config_adaptive.json` - 自适应训练配置

## 🚀 快速开始

### 方法1: 使用快速提交脚本（推荐）

```bash
# 基本使用
./quick_submit.sh

# 自定义配置
./quick_submit.sh --gpu 2 --time 48 --mem 64

# 恢复训练
./quick_submit.sh --resume

# 查看帮助
./quick_submit.sh --help
```

### 方法2: 直接使用sbatch

```bash
# 基础提交
sbatch submit_adaptive_training.sh

# 灵活配置提交
sbatch submit_adaptive_training_flexible.sh

# 自定义资源
sbatch --time=12:00:00 --mem=64G --gres=gpu:2 submit_adaptive_training_flexible.sh
```

## ⚙️ 配置选项

### 资源配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gres=gpu:N` | 1 | GPU数量 |
| `--time=HH:MM:SS` | 24:00:00 | 时间限制 |
| `--mem=XG` | 32G | 内存大小 |
| `--cpus-per-task=N` | 16 | CPU核心数 |
| `--partition=NAME` | gpu | 分区名称 |

### 环境变量

```bash
# 设置配置文件
export CONFIG_FILE="training/configs/train_config_adaptive.json"

# 恢复训练
export RESUME="true"

# 仅监控模式
export MONITOR_ONLY="true"

# 指定设备
export DEVICE="cuda"
```

## 📊 监控和管理

### 查看作业状态

```bash
# 查看所有作业
squeue -u $USER

# 查看特定作业
squeue -j JOB_ID

# 查看作业详情
scontrol show job JOB_ID
```

### 查看日志

```bash
# 实时查看输出日志
tail -f logs/adaptive_JOB_ID.out

# 查看错误日志
tail -f logs/adaptive_JOB_ID.err

# 查看TensorBoard日志
tensorboard --logdir=outputs/training_results_adaptive/tensorboard
```

### 管理作业

```bash
# 取消作业
scancel JOB_ID

# 取消所有作业
scancel -u $USER

# 暂停作业
scontrol suspend JOB_ID

# 恢复作业
scontrol resume JOB_ID
```

## 🔧 环境配置

### Conda环境

在脚本中取消注释并修改：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate decode_env
```

### 模块加载

```bash
module load cuda/11.8
module load python/3.9
module load pytorch/1.13
```

### 虚拟环境

```bash
source ~/venv/decode/bin/activate
```

## 📈 性能优化

### GPU配置

```bash
# 单GPU训练
./quick_submit.sh --gpu 1

# 多GPU训练
./quick_submit.sh --gpu 2 --cpu 32

# 高内存配置
./quick_submit.sh --gpu 4 --mem 128 --cpu 64
```

### 时间管理

```bash
# 短时间测试
./quick_submit.sh --time 2

# 长时间训练
./quick_submit.sh --time 72

# 使用检查点恢复
./quick_submit.sh --resume --time 24
```

## 🐛 故障排除

### 常见问题

1. **作业排队时间过长**
   ```bash
   # 查看分区状态
   sinfo
   
   # 使用不同分区
   ./quick_submit.sh --partition=cpu
   ```

2. **内存不足**
   ```bash
   # 增加内存
   ./quick_submit.sh --mem 64
   
   # 减少batch size（修改配置文件）
   ```

3. **GPU内存不足**
   ```bash
   # 使用更多GPU
   ./quick_submit.sh --gpu 2
   
   # 减少模型大小或batch size
   ```

4. **训练中断**
   ```bash
   # 使用检查点恢复
   ./quick_submit.sh --resume
   ```

### 日志分析

```bash
# 查看GPU使用情况
grep "GPU" logs/adaptive_*.out

# 查看训练进度
grep "Epoch" logs/adaptive_*.out

# 查看错误信息
grep -i "error\|exception\|failed" logs/adaptive_*.err
```

## 📋 最佳实践

### 1. 资源规划

- 根据数据集大小选择合适的内存
- 使用多GPU时确保CPU核心数充足
- 预估训练时间，设置合理的时间限制

### 2. 检查点管理

- 定期保存检查点
- 使用`--resume`恢复中断的训练
- 备份重要的检查点文件

### 3. 监控策略

- 使用TensorBoard监控训练进度
- 定期检查日志文件
- 监控GPU和内存使用情况

### 4. 批量提交

```bash
# 提交多个不同配置的作业
for lr in 0.001 0.0001 0.00001; do
    CONFIG_FILE="configs/config_lr_${lr}.json"
    ./quick_submit.sh --config $CONFIG_FILE --name "lr_${lr}"
done
```

## 📞 支持

如果遇到问题，请：

1. 检查日志文件中的错误信息
2. 确认环境配置正确
3. 验证配置文件格式
4. 查看集群状态和资源可用性

## 📚 相关文档

- [DECODE训练指南](README_ADAPTIVE_TRAINING.md)
- [配置文件说明](training/configs/README.md)
- [SLURM官方文档](https://slurm.schedmd.com/documentation.html)