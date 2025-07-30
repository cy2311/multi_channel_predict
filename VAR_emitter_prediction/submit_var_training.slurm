#!/bin/bash
#SBATCH --job-name=var_emitter_training
#SBATCH --output=logs/var_training_%j.out
#SBATCH --error=logs/var_training_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1

# VAR Emitter预测模型训练SLURM脚本
# 提交方式: sbatch submit_var_training.slurm

echo "======================================"
echo "🚀 VAR Emitter训练作业开始"
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "开始时间: $(date)"
echo "======================================"

# 设置工作目录
cd /home/guest/Others/DECODE_rewrite/VAR_emitter_prediction

# 创建必要的目录
mkdir -p logs
mkdir -p outputs
mkdir -p models

# 激活conda环境（根据实际环境调整）
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate pytorch_env

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:/home/guest/Others/DECODE_rewrite"

# 设置CUDA可见设备（如果有多GPU）
export CUDA_VISIBLE_DEVICES=0

# 记录系统信息
echo "📊 系统信息:"
echo "节点信息: $(hostname)"
echo "GPU信息:"
nvidia-smi
echo "内存信息:"
free -h
echo "CPU信息:"
lscpu | grep "Model name"
echo "Python版本:"
python --version
echo "PyTorch版本:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else "N/A"}')"
echo "======================================"

# 检查配置文件
if [ ! -f "configs/config_true_var.json" ]; then
    echo "❌ 错误: 配置文件 configs/config_true_var.json 不存在"
    exit 1
fi

# 检查训练脚本
if [ ! -f "train_true_var.py" ]; then
    echo "❌ 错误: 训练脚本 train_true_var.py 不存在"
    exit 1
fi

echo "✅ 所有必要文件检查完成"
echo "======================================"

# 开始训练
echo "🎯 开始VAR Emitter模型训练..."
echo "配置文件: configs/config_true_var.json"
echo "训练脚本: train_true_var.py"
echo "======================================"

# 运行训练（添加详细日志）
python train_true_var.py --config configs/config_true_var.json 2>&1 | tee logs/training_${SLURM_JOB_ID}.log

# 检查训练结果
TRAIN_EXIT_CODE=$?
echo "======================================"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ 训练成功完成!"
    echo "完成时间: $(date)"
    
    # 显示输出统计
    echo "📊 训练结果统计:"
    if [ -d "outputs" ]; then
        echo "输出目录大小: $(du -sh outputs)"
        echo "模型文件数量: $(find outputs -name "*.pth" | wc -l)"
    fi
    
    if [ -d "logs/tensorboard" ]; then
        echo "TensorBoard日志: $(du -sh logs/tensorboard)"
    fi
    
else
    echo "❌ 训练失败，退出码: $TRAIN_EXIT_CODE"
    echo "请检查错误日志: logs/var_training_${SLURM_JOB_ID}.err"
fi

echo "======================================"
echo "🏁 作业结束时间: $(date)"
echo "总运行时间: $SECONDS 秒"
echo "======================================"