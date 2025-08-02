#!/bin/bash
#SBATCH --job-name=decode_training
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

# 加载必要的模块（根据你的集群环境调整）
# module load python/3.8
# module load cuda/11.8
# module load cudnn/8.6

# 激活虚拟环境（如果使用）
# source /path/to/your/venv/bin/activate

# 设置CUDA环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/guest/Others/DECODE_rewrite:$PYTHONPATH"

# 打印环境信息
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Python path: $PYTHONPATH"
echo "Working directory: $(pwd)"

# 检查GPU可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else "CPU"}')"

# 运行训练脚本
echo "Starting training..."
python train.py --config training_config.yaml

# 检查训练结果
if [ $? -eq 0 ]; then
    echo "Training completed successfully at: $(date)"
else
    echo "Training failed at: $(date)"
    exit 1
fi

echo "Job finished at: $(date)"