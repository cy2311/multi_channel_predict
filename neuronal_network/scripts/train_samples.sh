#!/bin/bash
#SBATCH --job-name=decode_train_samples
#SBATCH --output=../training/outputs/train_samples_%j.out
#SBATCH --error=../training/outputs/train_samples_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1
#SBATCH --exclusive=user

# ============================================
# 训练参数配置区域 - 根据需要修改以下参数
# ============================================
SAMPLES=100          # 样本数量: 20, 50, 80, 100 等
EPOCHS=1000           # 训练轮数
BATCH_SIZE=64        # 批处理大小
LEARNING_RATE=1e-4  # 学习率
# GPU_ID 将通过自动检测获得，不再硬编码
# ============================================

# 自动检测Slurm分配的GPU
echo "检测GPU分配..."
if [ -n "$SLURM_LOCALID" ] && [ -n "$SLURM_STEP_GPUS" ]; then
    # 使用Slurm分配的GPU
    ALLOCATED_GPU=$SLURM_LOCALID
    export CUDA_VISIBLE_DEVICES=$SLURM_STEP_GPUS
    echo "Slurm分配的GPU: $SLURM_STEP_GPUS (本地ID: $ALLOCATED_GPU)"
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # 如果环境中已设置CUDA_VISIBLE_DEVICES
    ALLOCATED_GPU=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1)
    echo "使用环境变量中的GPU: $CUDA_VISIBLE_DEVICES (主GPU: $ALLOCATED_GPU)"
else
    # 回退到自动检测可用GPU
    echo "未检测到Slurm GPU分配，自动检测可用GPU..."
    # 检测可用GPU
    AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 > 1000 {print $1}' | head -1)
    if [ -n "$AVAILABLE_GPUS" ]; then
        ALLOCATED_GPU=$AVAILABLE_GPUS
        export CUDA_VISIBLE_DEVICES=$ALLOCATED_GPU
        echo "自动选择GPU: $ALLOCATED_GPU (内存充足)"
    else
        ALLOCATED_GPU=0
        export CUDA_VISIBLE_DEVICES=0
        echo "警告: 未找到可用GPU，使用默认GPU 0"
    fi
fi

# 基础路径配置
BASE_DIR="/home/guest/Others/DECODE_rewrite"
TRAINING_DIR="$BASE_DIR/neuronal_network/training"
DATA_DIR="$BASE_DIR/simulation_zmap2tiff/outputs_100samples_40"  # 使用100样本40x40数据集

# 创建输出目录
OUTPUT_DIR="$TRAINING_DIR/outputs/train_${SAMPLES}samples_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "DECODE网络训练 - $SAMPLES 样本"
echo "开始时间: $(date)"
echo "样本数: $SAMPLES"
echo "训练轮数: $EPOCHS"
echo "批处理大小: $BATCH_SIZE"
echo "学习率: $LEARNING_RATE"
echo "GPU设备: $CUDA_VISIBLE_DEVICES"
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "=========================================="

# 检查数据目录
if [ ! -d "$DATA_DIR" ]; then
    echo "错误: 数据目录不存在: $DATA_DIR"
    exit 1
fi

# 切换到训练目录
cd "$TRAINING_DIR"

# 动态端口分配
echo "分配TensorBoard端口..."
# 使用Job ID和GPU ID确保端口唯一性
if [ -n "$SLURM_JOB_ID" ]; then
    TB_PORT=$((6006 + ($SLURM_JOB_ID % 1000) + ($ALLOCATED_GPU * 10)))
else
    TB_PORT=$((6006 + $SAMPLES + ($ALLOCATED_GPU * 10) + ($(date +%s) % 100)))
fi

# 检查端口是否被占用，如果被占用则递增
while netstat -ln | grep -q ":$TB_PORT "; do
    echo "端口 $TB_PORT 已被占用，尝试下一个端口..."
    TB_PORT=$((TB_PORT + 1))
done

# 启动TensorBoard (后台运行)
echo "启动TensorBoard在端口: $TB_PORT"
tensorboard --logdir="$OUTPUT_DIR/tensorboard" --port=$TB_PORT --host=0.0.0.0 &
TB_PID=$!
echo "TensorBoard PID: $TB_PID"
echo $TB_PID > "$OUTPUT_DIR/tensorboard.pid"

# 运行训练
echo "开始训练..."
echo "使用GPU: $ALLOCATED_GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "当前GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | grep "^$ALLOCATED_GPU," || echo "GPU $ALLOCATED_GPU 信息获取失败"
python start_training.py \
    --data_dir "$DATA_DIR" \
    --samples $SAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --output_suffix "_$(date +%Y%m%d_%H%M%S)" 2>&1

TRAIN_EXIT_CODE=$?

# 训练完成后的清理
echo "=========================================="
echo "训练完成时间: $(date)"
echo "训练退出码: $TRAIN_EXIT_CODE"
echo "正在清理TensorBoard进程..."
kill $TB_PID 2>/dev/null || true
rm -f "$OUTPUT_DIR/tensorboard.pid"

# 生成训练总结
echo "训练总结:" > "$OUTPUT_DIR/training_summary.txt"
echo "样本数: $SAMPLES" >> "$OUTPUT_DIR/training_summary.txt"
echo "训练轮数: $EPOCHS" >> "$OUTPUT_DIR/training_summary.txt"
echo "批处理大小: $BATCH_SIZE" >> "$OUTPUT_DIR/training_summary.txt"
echo "学习率: $LEARNING_RATE" >> "$OUTPUT_DIR/training_summary.txt"
echo "开始时间: $(date)" >> "$OUTPUT_DIR/training_summary.txt"
echo "退出码: $TRAIN_EXIT_CODE" >> "$OUTPUT_DIR/training_summary.txt"
echo "输出目录: $OUTPUT_DIR" >> "$OUTPUT_DIR/training_summary.txt"
echo "TensorBoard端口: $TB_PORT" >> "$OUTPUT_DIR/training_summary.txt"

echo "=========================================="
echo "训练脚本执行完成"
echo "TensorBoard访问地址: http://your_server:$TB_PORT"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

exit $TRAIN_EXIT_CODE