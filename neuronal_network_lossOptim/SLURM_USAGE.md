# DECODE网络训练 SLURM使用指南

本指南介绍如何在SLURM集群上提交和监控DECODE网络的训练任务。

## 📁 文件说明

- `submit_decode_training.slurm` - SLURM作业脚本
- `submit_training.sh` - 训练提交脚本
- `check_training_status.sh` - 状态检查脚本

## 🚀 快速开始

### 1. 提交训练作业

```bash
cd /home/guest/Others/DECODE_rewrite/neuronal_network_lossOptim
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
- **作业名称**: `decode_training`
- **运行时间**: 24小时
- **节点数量**: 1
- **CPU核心**: 8
- **内存**: 64GB
- **GPU**: 1张
- **分区**: gpu

### 输出文件
- 标准输出: `logs/decode_training_<JOB_ID>.out`
- 错误输出: `logs/decode_training_<JOB_ID>.err`
- 训练日志: `logs/training_<JOB_ID>.log`

## 🔧 自定义配置

### 修改资源需求

编辑 `submit_decode_training.slurm` 文件中的SBATCH参数：

```bash
#SBATCH --time=24:00:00        # 运行时间
#SBATCH --cpus-per-task=8      # CPU核心数
#SBATCH --mem=64G              # 内存大小
#SBATCH --gres=gpu:1           # GPU数量
#SBATCH --partition=gpu        # 分区名称
```

### 修改训练参数

编辑 `training/configs/train_config_fixed.json` 配置文件：

```json
{
  "data": {
    "batch_size": 16,
    "image_size": 256,
    "consecutive_frames": 1,
    "train_val_split": 0.8,
    "num_workers": 4
  },
  "training": {
    "epochs": 1000,
    "lr_first": 0.0001,
    "lr_second": 0.0001
  }
}
```

### 修改数据路径

编辑 `training/train_decode_network_fixed.py` 文件中的数据路径：

```python
data_dir = "/path/to/your/dataset"  # 修改为实际数据路径
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

# 查看历史作业
sacct -u $USER --starttime=now-1day
```

### 日志监控

```bash
# 实时查看输出日志
tail -f logs/decode_training_<JOB_ID>.out

# 实时查看错误日志
tail -f logs/decode_training_<JOB_ID>.err

# 实时查看训练日志
tail -f logs/training_<JOB_ID>.log

# 查看训练进度
grep -E "Epoch [0-9]+/[0-9]+ 完成" logs/training_<JOB_ID>.log | tail -10
```

### TensorBoard监控

训练开始后，脚本会自动启动TensorBoard并显示访问地址：

```
✅ TensorBoard已启动 (PID: 12345, 端口: 6006)
🌐 访问地址: http://节点名:6006
🔗 远程访问: ssh -L 6006:节点名:6006 user@server
```

**远程访问步骤：**
1. 在本地机器上建立SSH隧道：`ssh -L 6006:节点名:6006 user@server_ip`
2. 在本地浏览器访问：`http://localhost:6006`

## 🔍 故障排除

### 常见问题

1. **作业提交失败**
   ```bash
   # 检查SLURM状态
   sinfo
   
   # 检查资源配额
   sshare -u $USER
   
   # 检查分区可用性
   sinfo -p gpu
   ```

2. **GPU不可用**
   ```bash
   # 检查GPU节点
   sinfo -p gpu
   
   # 修改为CPU分区（如果需要）
   # 编辑submit_decode_training.slurm，将partition改为cpu
   ```

3. **内存不足**
   ```bash
   # 减少批次大小
   # 编辑training/configs/train_config_fixed.json中的batch_size
   ```

4. **训练中断**
   ```bash
   # 检查检查点
   ls -la training/outputs/
   
   # 训练脚本会自动检测并从最新检查点恢复
   ```

5. **数据加载失败**
   ```bash
   # 检查数据目录
   ls -la /home/guest/Others/DECODE_rewrite/simulation_zmap2tiff/outputs_100samples_40/
   
   # 如果数据不存在，训练会自动使用模拟数据
   ```

### 日志分析

```bash
# 查看错误信息
grep -i error logs/decode_training_<JOB_ID>.err

# 查看GPU使用情况
grep -i gpu logs/decode_training_<JOB_ID>.out

# 查看内存使用情况
grep -i memory logs/decode_training_<JOB_ID>.out

# 查看训练损失趋势
grep "训练损失" logs/training_<JOB_ID>.log | tail -20
```

## 📈 性能优化

### 1. 批次大小调优
- GPU内存充足时可增加batch_size
- 内存不足时减少batch_size

### 2. 数据加载优化
- 增加num_workers可提高数据加载速度
- 但不要超过CPU核心数

### 3. 混合精度训练
- 脚本已启用自动混合精度训练
- 可显著减少GPU内存使用

## 📊 训练监控指标

### 关键指标
- **训练损失**: 应该逐渐下降
- **验证损失**: 应该下降且不应远高于训练损失
- **GPU利用率**: 应该保持在80%以上
- **内存使用**: 不应超过分配的内存

### 正常训练的标志
- 损失值在合理范围内（通常4-6之间）
- 训练和验证损失都在下降
- 没有内存溢出错误
- GPU利用率稳定

## 💡 最佳实践

1. **提交前检查**
   - 确认数据路径正确
   - 检查配置文件语法
   - 验证资源需求合理

2. **监控策略**
   - 定期检查作业状态
   - 监控TensorBoard指标
   - 关注日志文件大小

3. **资源管理**
   - 不要请求过多资源
   - 及时取消不需要的作业
   - 合理设置运行时间限制

4. **数据管理**
   - 确保数据目录可访问
   - 定期清理临时文件
   - 备份重要的模型检查点

## 🆘 获取帮助

如果遇到问题，可以：

1. 查看本指南的故障排除部分
2. 检查SLURM官方文档
3. 联系集群管理员
4. 查看相关日志文件获取详细错误信息

---

**注意**: 请根据你的具体SLURM集群配置调整分区名称、资源限制等参数。