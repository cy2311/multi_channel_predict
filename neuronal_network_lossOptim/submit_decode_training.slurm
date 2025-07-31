#!/bin/bash
#SBATCH --job-name=decode_training
#SBATCH --output=logs/decode_training_%j.out
#SBATCH --error=logs/decode_training_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=cpu1
#SBATCH --gres=gpu:1

# DECODE网络训练SLURM脚本
# 提交方式: sbatch submit_decode_training.slurm

echo "======================================"
echo "🚀 DECODE训练作业开始"
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "开始时间: $(date)"
echo "======================================"

# 设置工作目录
cd /home/guest/Others/DECODE_rewrite/neuronal_network_lossOptim

# 创建必要的目录
mkdir -p logs
mkdir -p outputs
mkdir -p training/outputs

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
python -c "import torch; print(torch.__version__)"
echo "CUDA版本:"
python -c "import torch; print(torch.version.cuda)"
echo "======================================"

# 启动TensorBoard（后台运行）
echo "📊 启动TensorBoard..."
TENSORBOARD_PORT=6006
for port in {6006..6020}; do
    if ! netstat -tuln | grep -q ":$port "; then
        TENSORBOARD_PORT=$port
        break
    fi
done

tensorboard --logdir=training/outputs --port=$TENSORBOARD_PORT --host=0.0.0.0 &
TENSORBOARD_PID=$!
echo "✅ TensorBoard已启动 (PID: $TENSORBOARD_PID, 端口: $TENSORBOARD_PORT)"
echo "🌐 访问地址: http://$(hostname):$TENSORBOARD_PORT"
echo "🔗 远程访问: ssh -L $TENSORBOARD_PORT:$(hostname):$TENSORBOARD_PORT user@server"
echo "======================================"

# 开始训练
echo "🎯 开始DECODE网络训练..."
echo "配置文件: training/configs/train_config_fixed.json"
echo "数据路径: /home/guest/Others/DECODE_rewrite/simulation_zmap2tiff/outputs_100samples_40"
echo "======================================"

# 记录训练开始时间
TRAIN_START_TIME=$(date +%s)

# 运行训练脚本
python training/train_decode_network_fixed.py --config training/configs/train_config_fixed.json 2>&1 | tee logs/training_${SLURM_JOB_ID}.log

# 获取训练退出码
TRAIN_EXIT_CODE=$?

# 记录训练结束时间
TRAIN_END_TIME=$(date +%s)
TRAIN_DURATION=$((TRAIN_END_TIME - TRAIN_START_TIME))

echo "======================================"
echo "📊 训练完成统计:"
echo "训练时长: $((TRAIN_DURATION / 3600))小时 $((TRAIN_DURATION % 3600 / 60))分钟 $((TRAIN_DURATION % 60))秒"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ 训练成功完成!"
    
    # 显示输出文件信息
    echo "📁 输出文件:"
    if [ -d "training/outputs" ]; then
        echo "模型文件: $(find training/outputs -name '*.pth' | wc -l) 个"
        echo "输出目录大小: $(du -sh training/outputs)"
    fi
    
    if [ -d "logs" ]; then
        echo "日志文件: $(ls logs/ | wc -l) 个"
        echo "日志目录大小: $(du -sh logs)"
    fi
    
else
    echo "❌ 训练失败，退出码: $TRAIN_EXIT_CODE"
    echo "请检查错误日志: logs/decode_training_${SLURM_JOB_ID}.err"
fi

# 停止TensorBoard
if [ ! -z "$TENSORBOARD_PID" ]; then
    kill $TENSORBOARD_PID 2>/dev/null
    echo "📊 TensorBoard已停止"
fi

echo "======================================"
echo "🏁 作业结束时间: $(date)"
echo "总运行时间: $SECONDS 秒"
echo "======================================"