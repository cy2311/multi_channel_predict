#!/bin/bash

# VAR Emitter训练SLURM提交脚本
# 使用方法: ./submit_training.sh

echo "🚀 提交VAR Emitter训练作业到SLURM..."
echo "======================================"

# 检查当前目录
if [ ! -f "train_true_var.py" ]; then
    echo "❌ 错误: 请在VAR_emitter_prediction目录下运行此脚本"
    echo "当前目录: $(pwd)"
    exit 1
fi

# 检查配置文件
if [ ! -f "configs/config_true_var.json" ]; then
    echo "❌ 错误: 配置文件 configs/config_true_var.json 不存在"
    exit 1
fi

# 检查SLURM脚本
if [ ! -f "submit_var_training.slurm" ]; then
    echo "❌ 错误: SLURM脚本 submit_var_training.slurm 不存在"
    exit 1
fi

# 创建日志目录
mkdir -p logs

echo "📋 作业配置:"
echo "   训练脚本: train_true_var.py"
echo "   配置文件: configs/config_true_var.json"
echo "   SLURM脚本: submit_var_training.slurm"
echo "   日志目录: logs/"
echo "======================================"

# 提交作业
echo "📤 提交作业到SLURM..."
JOB_ID=$(sbatch submit_var_training.slurm 2>&1 | grep -o '[0-9]\+' | tail -1)

if [ $? -eq 0 ] && [ -n "$JOB_ID" ]; then
    echo "✅ 作业提交成功!"
    echo "======================================"
    echo "📊 作业信息:"
    echo "   作业ID: $JOB_ID"
    echo "   作业名称: var_emitter_training"
    echo "   预计运行时间: 24小时"
    echo "======================================"
    echo "📋 监控命令:"
    echo "   查看作业状态: squeue -j $JOB_ID"
    echo "   查看所有作业: squeue -u \$USER"
    echo "   查看输出日志: tail -f logs/var_training_${JOB_ID}.out"
    echo "   查看错误日志: tail -f logs/var_training_${JOB_ID}.err"
    echo "   查看训练日志: tail -f logs/training_${JOB_ID}.log"
    echo "   取消作业: scancel $JOB_ID"
    echo "======================================"
    echo "📊 TensorBoard监控:"
    echo "   日志目录: logs/tensorboard/"
    echo "   启动命令: tensorboard --logdir=logs/tensorboard --port=6006"
    echo "   访问方式: ssh -L 6006:localhost:6006 user@server"
    echo "======================================"
    echo "💡 提示:"
    echo "   - 训练过程中可以通过TensorBoard监控损失和指标"
    echo "   - 模型检查点将保存在 outputs/ 目录"
    echo "   - Count Loss集成已启用，将约束emitter数量预测"
    echo "======================================"
else
    echo "❌ 作业提交失败"
    echo "请检查:"
    echo "   1. SLURM是否正常运行: sinfo"
    echo "   2. 是否有足够的资源配额"
    echo "   3. 脚本权限是否正确"
    exit 1
fi