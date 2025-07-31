#!/bin/bash
#SBATCH --job-name=decode_adaptive
#SBATCH --output=logs/adaptive_%j.out
#SBATCH --error=logs/adaptive_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# 使用方法:
# sbatch submit_adaptive_training_flexible.sh
# sbatch --time=12:00:00 --mem=32G submit_adaptive_training_flexible.sh
# sbatch --gres=gpu:2 --cpus-per-task=32 submit_adaptive_training_flexible.sh

# 可配置参数
CONFIG_FILE="${CONFIG_FILE:-training/configs/train_config_adaptive.json}"
DEVICE="${DEVICE:-cuda}"
RESUME="${RESUME:-false}"
MONITOR_ONLY="${MONITOR_ONLY:-false}"

# 打印作业信息
echo "==================== 作业信息 ===================="
echo "作业ID: $SLURM_JOB_ID"
echo "作业名称: $SLURM_JOB_NAME"
echo "节点: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "CPU核心数: $SLURM_CPUS_PER_TASK"
echo "内存: $SLURM_MEM_PER_NODE MB"
echo "开始时间: $(date)"
echo "工作目录: $(pwd)"
echo "配置文件: $CONFIG_FILE"
echo "设备: $DEVICE"
echo "恢复训练: $RESUME"
echo "仅监控模式: $MONITOR_ONLY"
echo "================================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# 环境设置（根据集群配置修改）
# 选项1: Conda环境
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate decode_env

# 选项2: 模块加载
# module load cuda/11.8
# module load python/3.9
# module load pytorch/1.13

# 选项3: 虚拟环境
# source ~/venv/decode/bin/activate

# 创建必要目录
mkdir -p logs
mkdir -p outputs/training_results_adaptive

# 系统检查
echo "\n==================== 系统检查 ===================="
echo "主机名: $(hostname)"
echo "操作系统: $(uname -a)"
echo "磁盘空间:"
df -h .
echo "\nGPU状态:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
echo "\nPython环境:"
echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未安装')"
echo "CUDA可用性: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo '检查失败')"
echo "GPU数量: $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '0')"
echo "================================================="

# 进入项目目录
cd /home/guest/Others/DECODE_rewrite/neuronal_network_lossOptim || {
    echo "错误: 无法进入项目目录"
    exit 1
}

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 构建训练命令
TRAIN_CMD="python start_adaptive_training.py --config $CONFIG_FILE"

if [ "$RESUME" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume"
fi

if [ "$MONITOR_ONLY" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --monitor-only"
fi

# 运行训练
echo "\n==================== 开始训练 ===================="
echo "执行命令: $TRAIN_CMD"
echo "================================================="

# 使用timeout防止作业超时 (默认47小时，比SLURM时间限制少1小时)
# 如果需要自定义超时时间，可以设置TIMEOUT_HOURS环境变量
TIMEOUT_HOURS=${TIMEOUT_HOURS:-47}
timeout ${TIMEOUT_HOURS}h $TRAIN_CMD
EXIT_CODE=$?

# 检查结果
echo "\n==================== 训练结果 ===================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 训练成功完成！"
    echo "📁 结果保存在: outputs/training_results_adaptive/"
    echo "📊 TensorBoard日志: outputs/training_results_adaptive/tensorboard/"
    echo "💾 检查点文件: outputs/training_results_adaptive/checkpoints/"
elif [ $EXIT_CODE -eq 124 ]; then
    echo "⏰ 训练因超时而终止"
    echo "💡 建议: 增加时间限制或使用检查点恢复"
elif [ $EXIT_CODE -eq 130 ]; then
    echo "🛑 训练被用户中断"
else
    echo "❌ 训练失败，退出码: $EXIT_CODE"
    echo "📋 请检查错误日志: logs/adaptive_${SLURM_JOB_ID}.err"
fi

# 显示输出文件信息
if [ -d "outputs/training_results_adaptive" ]; then
    echo "\n📂 输出文件列表:"
    ls -la outputs/training_results_adaptive/
fi

echo "\n结束时间: $(date)"
echo "总运行时间: $SECONDS 秒"
echo "================================================="

exit $EXIT_CODE