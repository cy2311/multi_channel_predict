#!/bin/bash
#SBATCH --job-name=var_emitter_training
#SBATCH --output=var_training_%j.out
#SBATCH --error=var_training_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1

# 设置工作目录
WORK_DIR="/home/guest/Others/DECODE_rewrite/VAR_emitter_prediction"
cd $WORK_DIR

# 创建日志目录
mkdir -p logs

# 激活conda环境（如果需要）
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate your_env_name

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:/home/guest/Others/DECODE_rewrite"

# 记录开始时间和系统信息
echo "作业开始时间: $(date)"
echo "节点信息: $(hostname)"
echo "GPU信息:"
nvidia-smi
echo "内存信息:"
free -h
echo "CPU信息:"
lscpu | grep "Model name"
echo "========================================"

# 设置数据路径（根据实际情况修改）
TIFF_DIR="/home/guest/Others/DECODE_rewrite/simulation_zmap2tiff_var_highres/outputs_100samples_160"
EMITTER_DIR="/home/guest/Others/DECODE_rewrite/simulation_zmap2tiff_var_highres/outputs_100samples_160"
OUTPUT_DIR="./training_outputs"

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/tensorboard

# 智能端口检测函数
find_available_port() {
    local start_port=6006
    local max_port=6020
    
    for ((port=start_port; port<=max_port; port++)); do
        if ! netstat -tuln 2>/dev/null | grep -q ":$port " && ! ss -tuln 2>/dev/null | grep -q ":$port "; then
            echo $port
            return 0
        fi
    done
    return 1
}

# 启动TensorBoard监控
echo "🚀 启动TensorBoard监控..."
AVAILABLE_PORT=$(find_available_port)
if [ $? -eq 0 ] && [ -n "$AVAILABLE_PORT" ]; then
    echo "✅ 找到可用端口: $AVAILABLE_PORT"
    tensorboard --logdir=$OUTPUT_DIR/tensorboard --host=0.0.0.0 --port=$AVAILABLE_PORT --reload_interval=10 &
    TENSORBOARD_PID=$!
    echo "📊 TensorBoard已启动 (PID: $TENSORBOARD_PID)"
    echo "🌐 访问地址: http://localhost:$AVAILABLE_PORT"
    echo "🔗 远程访问: ssh -L $AVAILABLE_PORT:localhost:$AVAILABLE_PORT user@server"
else
    echo "❌ 端口6006-6020均被占用，TensorBoard启动失败"
    TENSORBOARD_PID=""
fi

# 运行VAR模型训练
echo "开始训练VAR模型..."
python train_true_var.py \
    --config config_true_var_slurm.json

# 检查退出状态
TRAIN_EXIT_CODE=$?
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "========================================"
    echo "✅ VAR模型训练成功完成!"
    echo "完成时间: $(date)"
    
    # 显示输出目录统计
    echo "输出目录统计:"
    if [ -d "$OUTPUT_DIR" ]; then
        echo "模型文件: $(ls -1 $OUTPUT_DIR/*.pth 2>/dev/null | wc -l)"
        echo "日志文件: $(ls -1 $OUTPUT_DIR/logs 2>/dev/null | wc -l)"
        echo "总大小: $(du -sh $OUTPUT_DIR 2>/dev/null | cut -f1)"
    fi
    
    # 显示最终模型信息
    echo "最新检查点:"
    ls -la $OUTPUT_DIR/checkpoint_*.pth 2>/dev/null | tail -1
    
    # TensorBoard信息
    if [ -n "$TENSORBOARD_PID" ] && kill -0 $TENSORBOARD_PID 2>/dev/null; then
        echo "📊 TensorBoard仍在运行 (PID: $TENSORBOARD_PID, 端口: $AVAILABLE_PORT)"
        echo "🌐 查看训练结果: http://localhost:$AVAILABLE_PORT"
        echo "💡 停止TensorBoard: kill $TENSORBOARD_PID"
    fi
else
    echo "❌ 训练失败，退出码: $TRAIN_EXIT_CODE"
    echo "请检查错误日志: logs/var_training_${SLURM_JOB_ID}.err"
fi

# 清理函数
cleanup() {
    echo "🧹 清理资源..."
    if [ -n "$TENSORBOARD_PID" ] && kill -0 $TENSORBOARD_PID 2>/dev/null; then
        echo "🛑 停止TensorBoard (PID: $TENSORBOARD_PID)"
        kill $TENSORBOARD_PID 2>/dev/null
        sleep 2
        if kill -0 $TENSORBOARD_PID 2>/dev/null; then
            kill -9 $TENSORBOARD_PID 2>/dev/null
        fi
    fi
}

# 设置信号处理
trap cleanup EXIT INT TERM

echo "作业结束时间: $(date)"
echo "========================================"
echo "📋 任务摘要:"
echo "   - 训练状态: $([ $TRAIN_EXIT_CODE -eq 0 ] && echo '成功' || echo '失败')"
echo "   - 输出目录: $OUTPUT_DIR"
echo "   - TensorBoard: $([ -n '$TENSORBOARD_PID' ] && echo '已启动' || echo '未启动')"
echo "   - 日志文件: logs/var_training_${SLURM_JOB_ID}.out"
echo "========================================"