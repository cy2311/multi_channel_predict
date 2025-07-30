# VAR Emitter训练 SLURM使用指南

本指南介绍如何在SLURM集群上提交和监控VAR Emitter预测模型的训练任务。

## 📁 文件说明

- `submit_var_training.slurm` - SLURM作业脚本
- `submit_training.sh` - 训练提交脚本
- `check_training_status.sh` - 状态检查脚本

## 🚀 快速开始

### 1. 提交训练作业

```bash
cd /home/guest/Others/DECODE_rewrite/VAR_emitter_prediction
./submit_training.sh
```

### 2. 检查作业状态

```bash
# 查看所有作业
./check_training_status.sh

# 查看特定作业详情
./check_training_status.sh <JOB_ID>
```

## 📊 SLURM作业配置

### 资源配置
- **作业名称**: `var_emitter_training`
- **运行时间**: 24小时
- **节点数量**: 1
- **CPU核心**: 8
- **内存**: 64GB
- **GPU**: 1张
- **分区**: gpu

### 输出文件
- 标准输出: `logs/var_training_<JOB_ID>.out`
- 错误输出: `logs/var_training_<JOB_ID>.err`
- 训练日志: `logs/training_<JOB_ID>.log`

## 🔧 自定义配置

### 修改资源需求

编辑 `submit_var_training.slurm` 文件中的SBATCH参数：

```bash
#SBATCH --time=24:00:00        # 运行时间
#SBATCH --cpus-per-task=8      # CPU核心数
#SBATCH --mem=64G              # 内存大小
#SBATCH --gres=gpu:1           # GPU数量
#SBATCH --partition=gpu        # 分区名称
```

### 修改训练参数

编辑 `configs/config_true_var.json` 配置文件：

```json
{
  "batch_size": 2,
  "learning_rate": 1e-4,
  "num_epochs": 100,
  "input_resolution": 160
}
```

## 📋 监控命令

### 基本SLURM命令

```bash
# 查看作业队列
squeue -u $USER

# 查看特定作业
squeue -j <JOB_ID>

# 取消作业
scancel <JOB_ID>

# 查看节点状态
sinfo

# 查看作业详情
scontrol show job <JOB_ID>
```

### 日志监控

```bash
# 实时查看输出日志
tail -f logs/var_training_<JOB_ID>.out

# 实时查看错误日志
tail -f logs/var_training_<JOB_ID>.err

# 实时查看训练日志
tail -f logs/training_<JOB_ID>.log
```

### TensorBoard监控

```bash
# 启动TensorBoard
tensorboard --logdir=logs/tensorboard --port=6006

# SSH端口转发（在本地机器上运行）
ssh -L 6006:localhost:6006 user@server

# 浏览器访问
http://localhost:6006
```

## 🔍 故障排除

### 常见问题

1. **作业提交失败**
   ```bash
   # 检查SLURM状态
   sinfo
   
   # 检查资源配额
   sshare -u $USER
   ```

2. **GPU不可用**
   ```bash
   # 检查GPU节点
   sinfo -p gpu
   
   # 修改分区为CPU
   # 在submit_var_training.slurm中注释掉GPU相关行
   ```

3. **内存不足**
   ```bash
   # 减少批次大小
   # 编辑configs/config_true_var.json中的batch_size
   ```

4. **训练中断**
   ```bash
   # 检查检查点
   ls -la outputs/
   
   # 从检查点恢复训练
   # 训练脚本会自动检测并恢复
   ```

### 日志分析

```bash
# 查看训练进度
grep -i "epoch" logs/training_<JOB_ID>.log

# 查看损失变化
grep -i "loss" logs/training_<JOB_ID>.log

# 查看Count Loss效果
grep -i "count" logs/training_<JOB_ID>.log

# 查看错误信息
grep -i "error\|exception" logs/var_training_<JOB_ID>.err
```

## 📈 训练监控指标

### 主要损失函数
- `Total loss` - 总损失
- `scale_0_count` - 10x10分辨率计数损失
- `scale_1_count` - 20x20分辨率计数损失
- `scale_2_count` - 40x40分辨率计数损失
- `scale_3_count` - 80x80分辨率计数损失
- `Count loss` - 总计数损失

### 性能指标
- `Prob sum` - 概率图总和（应接近真实emitter数量）
- 训练时间每epoch
- GPU利用率
- 内存使用情况

## 💡 最佳实践

1. **资源规划**
   - 根据数据集大小调整内存需求
   - 长时间训练建议使用检查点保存
   - 监控GPU利用率优化批次大小

2. **日志管理**
   - 定期清理旧日志文件
   - 使用TensorBoard可视化训练过程
   - 保存重要的训练配置和结果

3. **故障恢复**
   - 启用自动检查点保存
   - 设置合理的重启策略
   - 监控磁盘空间使用

## 📞 支持

如遇到问题，请检查：
1. SLURM集群状态
2. 配置文件格式
3. 数据路径是否正确
4. 依赖包是否安装完整

更多详细信息请参考 `docs/README_SLURM.md`。