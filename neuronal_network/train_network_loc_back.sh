#!/bin/bash
#SBATCH --job-name=decode_train_loc
#SBATCH --output=/home/guest/Others/DECODE_rewrite/nn_train_loc_back/slurm_output_%j.log
#SBATCH --error=/home/guest/Others/DECODE_rewrite/nn_train_loc_back/slurm_error_%j.log
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
mkdir -p /home/guest/Others/DECODE_rewrite/nn_train_loc_back/tensorboard

# 启动TensorBoard
tensorboard --logdir=/home/guest/Others/DECODE_rewrite/nn_train_loc_back/tensorboard --port=6007 &

# 运行训练脚本
python -u /home/guest/Others/DECODE_rewrite/neuronal_network/train_network_loc_back.py \
    --tiff_dir=/home/guest/Others/DECODE_rewrite/simulated_data_multi_frames/simulated_multi_frames \
    --emitter_dir=/home/guest/Others/DECODE_rewrite/simulated_data_multi_frames/emitter_sets \
    --output_dir=/home/guest/Others/DECODE_rewrite/nn_train_loc_back \
    --patch_size=600 \
    --stride=300 \
    --batch_size=4 \
    --num_workers=4 \
    --learning_rate=1e-4 \
    --num_epochs=100 \
    --count_loss_weight=1.0 \
    --loc_loss_weight=1.0 \
    --background_loss_weight=1.0 \
    --use_amp \
    --memory_tracking

# 训练结束后终止TensorBoard
pkill -f "tensorboard --logdir=/home/guest/Others/DECODE_rewrite/nn_train_loc_back/tensorboard"