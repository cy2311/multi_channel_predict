#!/bin/bash
#SBATCH --job-name=decode_inference_loc
#SBATCH --output=/home/guest/Others/DECODE_rewrite/nn_train/slurm_inference_loc_%j.log
#SBATCH --error=/home/guest/Others/DECODE_rewrite/nn_train/slurm_inference_loc_error_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00

# 激活conda环境（如果需要）
# source activate your_env_name

# 设置工作目录
cd /home/guest/Others/DECODE_rewrite

# 创建输出目录
mkdir -p /home/guest/Others/DECODE_rewrite/nn_train/predictions_loc

# 运行推理脚本（使用带定位功能的推理脚本）
python -u /home/guest/Others/DECODE_rewrite/neuronal_network/inference_loc.py \
    --tiff_path=/home/guest/Others/DECODE_rewrite/simulated_data_multi_frames/simulated_multi_frames/frames_set0.ome.tiff \
    --output_dir=/home/guest/Others/DECODE_rewrite/nn_train/predictions_loc \
    --checkpoint=/home/guest/Others/DECODE_rewrite/nn_train/models/best_model.pth \
    --patch_size=600 \
    --stride=300 \
    --threshold=0.7 \
    --use_amp

# 如果需要处理多个文件，可以添加更多的推理命令
# python -u /home/guest/Others/DECODE_rewrite/neuronal_network/inference_loc.py \
#     --tiff_path=/home/guest/Others/DECODE_rewrite/simulated_data_multi_frames/simulated_multi_frames/frames_set1.ome.tiff \
#     --output_dir=/home/guest/Others/DECODE_rewrite/nn_train/predictions_loc \
#     --checkpoint=/home/guest/Others/DECODE_rewrite/nn_train/models/best_model.pth \
#     --patch_size=600 \
#     --stride=300 \
#     --threshold=0.7 \
#     --use_amp