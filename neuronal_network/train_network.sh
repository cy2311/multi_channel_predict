#!/bin/bash
#SBATCH --job-name=decode_train
#SBATCH --output=/home/guest/Others/DECODE_rewrite/nn_train/slurm_output_%j.log
#SBATCH --error=/home/guest/Others/DECODE_rewrite/nn_train/slurm_error_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# 激活conda环境（如果需要）
# source activate your_env_name

# 设置工作目录
cd /home/guest/Others/DECODE_rewrite

# 创建输出目录
mkdir -p /home/guest/Others/DECODE_rewrite/nn_train/tensorboard

# 启动TensorBoard（后台运行）
tensorboard --logdir=/home/guest/Others/DECODE_rewrite/nn_train/tensorboard --port=6006 &

# 运行训练脚本
python -u /home/guest/Others/DECODE_rewrite/neuronal_network/train_network.py \
    --tiff_dir=/home/guest/Others/DECODE_rewrite/simulated_data_multi_frames/simulated_multi_frames \
    --emitter_dir=/home/guest/Others/DECODE_rewrite/simulated_data_multi_frames/emitter_sets \
    --output_dir=/home/guest/Others/DECODE_rewrite/nn_train \
    --patch_size=600 \
    --stride=300 \
    --batch_size=4 \
    --num_workers=4 \
    --learning_rate=1e-4 \
    --num_epochs=100 \
    --use_amp \
    --memory_tracking

# 训练完成后，杀掉TensorBoard进程
pkill -f "tensorboard --logdir=/home/guest/Others/DECODE_rewrite/nn_train/tensorboard"