# VAR模型训练 - Slurm提交指南

## 🚀 快速开始

### 1. 提交训练任务

```bash
# 进入VAR目录
cd /home/guest/Others/DECODE_rewrite/VAR_emitter_prediction

# 提交训练任务
sbatch submit_var_training.slurm
```

### 2. 查看任务状态

```bash
# 查看任务队列
squeue -u $USER

# 查看训练日志
tail -f logs/var_training_<job_id>.out

# 查看错误日志
tail -f logs/var_training_<job_id>.err
```

### 3. 访问TensorBoard

训练开始后，脚本会自动启动TensorBoard并显示访问地址：

```
✅ 找到可用端口: 6008
📊 TensorBoard已启动 (PID: 12345)
🌐 访问地址: http://localhost:6008
🔗 远程访问: ssh -L 6008:localhost:6008 user@server
```

**远程访问步骤：**
1. 在本地机器上建立SSH隧道：`ssh -L 6008:localhost:6008 user@server_ip`
2. 在本地浏览器访问：`http://localhost:6008`

## ⚙️ 配置说明

### 修改数据路径

编辑 `submit_var_training.slurm` 文件中的数据路径：

```bash
# 设置数据路径（根据实际情况修改）
TIFF_DIR="/path/to/your/tiff/data"
EMITTER_DIR="/path/to/your/emitter/data"
OUTPUT_DIR="./training_outputs"
```

### 调整资源配置

根据需要修改Slurm资源配置：

```bash
#SBATCH --time=12:00:00        # 运行时间
#SBATCH --cpus-per-task=8      # CPU核心数
#SBATCH --mem=64G              # 内存大小
#SBATCH --gres=gpu:1           # GPU数量
```

## 📊 TensorBoard监控

### 自动功能
- **智能端口检测**: 自动从6006开始检测可用端口
- **自动启动**: 训练开始时自动启动TensorBoard
- **自动清理**: 训练结束时自动停止TensorBoard

### 监控内容
- 训练损失 (总损失、计数损失、定位损失等)
- 验证损失
- 学习率变化
- 训练进度

### 手动控制

```bash
# 查看TensorBoard进程
ps aux | grep tensorboard

# 手动停止TensorBoard
kill <tensorboard_pid>

# 手动启动TensorBoard (如果需要)
tensorboard --logdir=training_outputs/tensorboard --port=6006
```

## 🔧 故障排除

### 常见问题

1. **端口被占用**
   - 脚本会自动检测6006-6020范围内的可用端口
   - 如果全部被占用，会显示错误信息

2. **训练失败**
   - 检查错误日志：`cat logs/var_training_<job_id>.err`
   - 检查数据路径是否正确
   - 确认GPU资源是否可用

3. **TensorBoard无法访问**
   - 确认端口转发设置正确
   - 检查防火墙设置
   - 确认TensorBoard进程正在运行

### 有用的命令

```bash
# 取消任务
scancel <job_id>

# 查看任务详情
scontrol show job <job_id>

# 查看GPU使用情况
nvidia-smi

# 查看磁盘使用
du -sh training_outputs/
```

## 📁 输出结构

```
training_outputs/
├── tensorboard/          # TensorBoard日志
├── checkpoint_*.pth      # 训练检查点
├── best_model.pth        # 最佳模型
└── logs/                 # 训练日志
```

## 💡 提示

- 训练过程中可以随时通过TensorBoard监控进度
- 建议定期检查日志文件确保训练正常
- 长时间训练建议增加时间限制
- 可以通过修改配置文件调整训练参数