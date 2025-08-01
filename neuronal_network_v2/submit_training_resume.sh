#!/bin/bash
#SBATCH --job-name=decode_training_resume
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# 设置工作目录
cd /home/guest/Others/DECODE_rewrite/neuronal_network_v2

# 创建必要的目录
mkdir -p logs
mkdir -p models
mkdir -p results
mkdir -p checkpoints

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/guest/Others/DECODE_rewrite:$PYTHONPATH"

# 查找最新的检查点
LATEST_CHECKPOINT=$(find checkpoints -name "*.pth" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# 检查是否有检查点可以恢复
if [ -n "$LATEST_CHECKPOINT" ] && [ -f "$LATEST_CHECKPOINT" ]; then
    echo "Resuming training from checkpoint: $LATEST_CHECKPOINT"
    python train.py --config training_config.yaml --resume "$LATEST_CHECKPOINT"
else
    echo "No checkpoint found, starting fresh training..."
    python train.py --config training_config.yaml
fi

echo "Job finished at: $(date)"