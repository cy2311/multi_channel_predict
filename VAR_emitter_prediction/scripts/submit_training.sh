#!/bin/bash

# VAR模型训练提交脚本
# 使用方法: ./submit_training.sh

echo "🚀 提交VAR模型训练作业到SLURM..."
echo "配置文件: config_true_var_slurm.json"
echo "输入分辨率: 160x160"
echo "批次大小: 2"
echo "======================================"

# 检查配置文件是否存在
if [ ! -f "config_true_var_slurm.json" ]; then
    echo "❌ 错误: 配置文件 config_true_var_slurm.json 不存在"
    exit 1
fi

# 检查训练脚本是否存在
if [ ! -f "train_true_var.py" ]; then
    echo "❌ 错误: 训练脚本 train_true_var.py 不存在"
    exit 1
fi

# 检查SLURM脚本是否存在
if [ ! -f "submit_var_training.slurm" ]; then
    echo "❌ 错误: SLURM脚本 submit_var_training.slurm 不存在"
    exit 1
fi

# 提交作业
echo "📤 提交作业..."
JOB_ID=$(sbatch submit_var_training.slurm | awk '{print $4}')

if [ $? -eq 0 ] && [ -n "$JOB_ID" ]; then
    echo "✅ 作业提交成功!"
    echo "作业ID: $JOB_ID"
    echo "======================================"
    echo "📋 监控命令:"
    echo "   查看作业状态: squeue -j $JOB_ID"
    echo "   查看输出日志: tail -f var_training_${JOB_ID}.out"
    echo "   查看错误日志: tail -f var_training_${JOB_ID}.err"
    echo "   取消作业: scancel $JOB_ID"
    echo "======================================"
    echo "📊 TensorBoard将在作业开始后启动"
    echo "   端口范围: 6006-6020"
    echo "   访问方式: ssh -L <port>:localhost:<port> user@server"
    echo "======================================"
else
    echo "❌ 作业提交失败"
    exit 1
fi

echo "🎯 训练配置摘要:"
echo "   - 输入分辨率: 160x160"
echo "   - 批次大小: 2"
echo "   - 学习率: 1e-4"
echo "   - 训练轮数: 100"
echo "   - 使用GPU: 是"
echo "   - 混合精度: 是"
echo "======================================"