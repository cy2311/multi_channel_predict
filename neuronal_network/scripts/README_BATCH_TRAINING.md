# DECODE网络批量训练脚本使用指南

本目录包含了多个批量训练脚本，可以帮助您充分利用多GPU资源，同时训练不同样本数的DECODE模型。

## 📁 脚本文件说明

### 1. `batch_train_manager.py` (推荐)
**Python版本的智能批量训练管理器**
- ✅ 功能最完整，支持任务管理和监控
- ✅ 自动GPU检测和分配
- ✅ 实时状态监控和日志管理
- ✅ 支持任务停止、重启和清理
- ✅ JSON格式的配置和状态记录

### 2. `batch_train_local.sh`
**本地环境批量训练脚本**
- ✅ 适用于本地多GPU环境
- ✅ 后台并行运行多个训练任务
- ✅ 简单的任务状态管理
- ✅ 自动TensorBoard启动

### 3. `batch_train_multi_gpu.sh`
**SLURM集群批量训练脚本**
- ✅ 适用于SLURM管理的GPU集群
- ✅ 自动任务队列提交
- ✅ 完整的SLURM作业管理
- ✅ 适合大规模计算环境

## 🚀 快速开始

### 方法一：使用Python管理器（推荐）

```bash
# 1. 查看默认配置
python batch_train_manager.py --config

# 2. 启动批量训练（使用默认配置）
python batch_train_manager.py

# 3. 查看任务状态
python batch_train_manager.py --status

# 4. 停止所有任务
python batch_train_manager.py --stop-all
```

### 方法二：使用Shell脚本

```bash
# 本地环境
./batch_train_local.sh

# 查看运行状态
./batch_train_local.sh --status

# 停止所有任务
./batch_train_local.sh --stop
```

## ⚙️ 配置说明

### 默认训练配置

| 配置 | 样本数 | 训练轮数 | GPU ID | 描述 |
|------|--------|----------|--------|---------|
| 1 | 10 | 2 | 0 | 快速测试 |
| 2 | 50 | 5 | 1 | 小规模训练 |
| 3 | 100 | 10 | 2 | 中等规模训练 |
| 4 | 200 | 15 | 3 | 大规模训练 |

### 自定义配置

#### Python管理器配置
编辑 `batch_train_manager.py` 中的 `default_configs` 列表：

```python
self.default_configs = [
    TrainingConfig(samples=10, epochs=2, gpu_id=0, description="快速测试"),
    TrainingConfig(samples=50, epochs=5, gpu_id=1, description="小规模训练"),
    TrainingConfig(samples=100, epochs=10, gpu_id=2, description="中等规模训练"),
    TrainingConfig(samples=200, epochs=15, gpu_id=3, description="大规模训练"),
    # 添加更多配置...
]
```

#### Shell脚本配置
编辑脚本中的 `SAMPLE_CONFIGS` 数组：

```bash
SAMPLE_CONFIGS=(
    "10:2:0"     # 样本数:epoch:GPU_ID
    "50:5:1"     # 样本数:epoch:GPU_ID  
    "100:10:2"   # 样本数:epoch:GPU_ID
    "200:15:3"   # 样本数:epoch:GPU_ID
)
```

## 📊 监控和管理

### 1. 实时状态监控

```bash
# Python管理器
python batch_train_manager.py --status

# Shell脚本
./batch_train_local.sh --status
```

### 2. GPU使用监控

```bash
# 实时GPU状态
watch -n 1 nvidia-smi

# GPU内存使用情况
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```

### 3. 训练日志查看

```bash
# 查看特定任务日志
tail -f /path/to/output/training.log

# 查看所有任务日志
tail -f outputs/train_*samples_*/training.log
```

### 4. TensorBoard监控

每个训练任务会自动启动TensorBoard服务：
- 端口规则：`6006 + 样本数`
- 例如：10样本训练 → 端口6016，100样本训练 → 端口6106

```bash
# 访问TensorBoard
# http://localhost:6016  (10样本训练)
# http://localhost:6106  (100样本训练)
```

## 🔧 高级功能

### 1. 任务优先级设置

在Python管理器中，可以设置任务优先级：

```python
TrainingConfig(samples=100, epochs=10, gpu_id=2, priority=1, description="高优先级任务")
```

### 2. 动态GPU分配

Python管理器会自动检测可用GPU并智能分配：

```python
# 自动检测GPU数量和状态
gpus = manager.check_gpu_availability()
```

### 3. 任务恢复和清理

```bash
# 清理已完成的任务记录
python batch_train_manager.py --cleanup

# 停止特定任务
python batch_train_manager.py --stop JOB_ID
```

## 📋 使用前检查清单

### 必要条件
- [ ] 已安装NVIDIA驱动和CUDA
- [ ] 已生成对应样本数的训练数据
- [ ] 确认GPU内存足够（建议每个任务至少8GB）
- [ ] 确认磁盘空间充足（每个任务约需要1-5GB）

### 数据准备

确保以下数据目录存在：
```
simulation_zmap2tiff/
├── outputs_10samples_256/
├── outputs_50samples_256/
├── outputs_100samples_256/
└── outputs_200samples_256/
```

如果数据不存在，请先生成：
```bash
cd simulation_zmap2tiff
python batch_tiff_generator.py --samples 100 --size 256
```

## 🚨 故障排除

### 常见问题

1. **GPU内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   - 解决：减少batch_size或使用更少的GPU
   - 修改配置中的batch_size参数

2. **数据目录不存在**
   ```
   错误: 数据目录不存在
   ```
   - 解决：先生成对应样本数的训练数据

3. **端口冲突**
   ```
   TensorBoard端口被占用
   ```
   - 解决：检查并关闭占用端口的进程
   ```bash
   lsof -i :6006
   kill -9 PID
   ```

4. **任务卡死**
   - 使用管理器停止任务：
   ```bash
   python batch_train_manager.py --stop-all
   ```

### 性能优化建议

1. **GPU分配策略**
   - 根据GPU内存大小分配不同样本数的任务
   - 避免在同一GPU上运行多个大型任务

2. **批处理大小调优**
   - 8GB GPU：batch_size = 2-4
   - 16GB GPU：batch_size = 4-8
   - 24GB GPU：batch_size = 8-16

3. **并行任务数量**
   - 建议同时运行的任务数 ≤ GPU数量
   - 监控系统资源使用情况

## 📈 输出文件结构

每个训练任务会在以下位置生成输出：

```
training/outputs/train_XXXsamples_TIMESTAMP/
├── tensorboard/           # TensorBoard日志
├── checkpoints/          # 模型检查点
├── training.log          # 训练日志
├── training.pid          # 进程ID
├── training_summary.json # 训练总结
└── run_training.sh       # 训练脚本
```

## 🔗 相关文档

- [训练系统总体说明](../training/README.md)
- [数据生成指南](../../simulation_zmap2tiff/README_256_generation.md)
- [DECODE网络架构说明](../README_TRAINING.md)

---

**提示**: 建议首次使用时先运行小样本数的测试任务，确认环境配置正确后再启动大规模批量训练。