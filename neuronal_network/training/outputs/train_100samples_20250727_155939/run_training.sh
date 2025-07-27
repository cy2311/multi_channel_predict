#!/bin/bash

# 设置GPU
export CUDA_VISIBLE_DEVICES=2

# 记录开始信息
echo "==========================================="
echo "训练开始时间: $(date)"
echo "任务ID: decode_100s_5e_gpu2_20250727_155939"
echo "样本数: 100"
echo "训练轮数: 5"
echo "批处理大小: 4"
echo "学习率: 0.0001"
echo "GPU ID: 2"
echo "输出目录: /home/guest/Others/DECODE_rewrite/neuronal_network/training/outputs/train_100samples_20250727_155939"
echo "PID: $$"
echo "==========================================="

# 保存PID
echo $$ > /home/guest/Others/DECODE_rewrite/neuronal_network/training/outputs/train_100samples_20250727_155939/training.pid

# 切换到训练目录
cd /home/guest/Others/DECODE_rewrite/neuronal_network/training

# 启动TensorBoard (后台运行)
TB_PORT=6106
echo "启动TensorBoard在端口: $TB_PORT"
tensorboard --logdir=/home/guest/Others/DECODE_rewrite/neuronal_network/training/outputs/train_100samples_20250727_155939/tensorboard --port=$TB_PORT --host=0.0.0.0 &
TB_PID=$!
echo "TensorBoard PID: $TB_PID"
echo $TB_PID > /home/guest/Others/DECODE_rewrite/neuronal_network/training/outputs/train_100samples_20250727_155939/tensorboard.pid

# 运行训练
echo "开始训练..."
python start_training.py \
    --samples 100 \
    --epochs 5 \
    --batch_size 4 \
    --lr 0.0001 \
    --output_suffix "_20250727_155939" 2>&1

TRAIN_EXIT_CODE=$?

# 训练完成后的清理
echo "==========================================="
echo "训练完成时间: $(date)"
echo "训练退出码: $TRAIN_EXIT_CODE"
echo "正在清理TensorBoard进程..."
kill $TB_PID 2>/dev/null || true
rm -f /home/guest/Others/DECODE_rewrite/neuronal_network/training/outputs/train_100samples_20250727_155939/tensorboard.pid

# 生成训练总结
cat > /home/guest/Others/DECODE_rewrite/neuronal_network/training/outputs/train_100samples_20250727_155939/training_summary.json << EOF
{
    "job_id": "decode_100s_5e_gpu2_20250727_155939",
    "samples": 100,
    "epochs": 5,
    "batch_size": 4,
    "learning_rate": 0.0001,
    "gpu_id": 2,
    "start_time": "$(date -Iseconds)",
    "end_time": "$(date -Iseconds)",
    "exit_code": $TRAIN_EXIT_CODE,
    "output_dir": "/home/guest/Others/DECODE_rewrite/neuronal_network/training/outputs/train_100samples_20250727_155939",
    "tensorboard_port": 6106
}
EOF

# 清理PID文件
rm -f /home/guest/Others/DECODE_rewrite/neuronal_network/training/outputs/train_100samples_20250727_155939/training.pid

echo "任务完成!"
exit $TRAIN_EXIT_CODE
