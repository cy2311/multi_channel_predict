#!/bin/bash

# VAR Emitter训练状态检查脚本
# 使用方法: ./check_training_status.sh [JOB_ID]

JOB_ID=$1

echo "📊 VAR Emitter训练状态检查"
echo "======================================"

if [ -z "$JOB_ID" ]; then
    echo "📋 显示所有训练作业:"
    echo "当前用户的所有作业:"
    squeue -u $USER
    echo ""
    echo "VAR训练相关作业:"
    squeue -u $USER --name=var_emitter_training
    echo ""
    echo "💡 使用方法: $0 <JOB_ID> 查看特定作业详情"
else
    echo "🔍 检查作业 ID: $JOB_ID"
    echo "======================================"
    
    # 检查作业状态
    echo "📊 作业状态:"
    squeue -j $JOB_ID
    echo ""
    
    # 检查作业详细信息
    echo "📋 作业详细信息:"
    scontrol show job $JOB_ID
    echo ""
    
    # 检查日志文件
    echo "📄 日志文件状态:"
    LOG_OUT="logs/var_training_${JOB_ID}.out"
    LOG_ERR="logs/var_training_${JOB_ID}.err"
    TRAIN_LOG="logs/training_${JOB_ID}.log"
    
    if [ -f "$LOG_OUT" ]; then
        echo "✅ 输出日志: $LOG_OUT ($(wc -l < $LOG_OUT) 行)"
        echo "   最新内容:"
        tail -5 "$LOG_OUT" | sed 's/^/   /'
    else
        echo "❌ 输出日志不存在: $LOG_OUT"
    fi
    
    if [ -f "$LOG_ERR" ]; then
        ERR_SIZE=$(wc -l < "$LOG_ERR")
        if [ $ERR_SIZE -gt 0 ]; then
            echo "⚠️  错误日志: $LOG_ERR ($ERR_SIZE 行)"
            echo "   最新错误:"
            tail -5 "$LOG_ERR" | sed 's/^/   /'
        else
            echo "✅ 错误日志: $LOG_ERR (无错误)"
        fi
    else
        echo "❌ 错误日志不存在: $LOG_ERR"
    fi
    
    if [ -f "$TRAIN_LOG" ]; then
        echo "✅ 训练日志: $TRAIN_LOG ($(wc -l < $TRAIN_LOG) 行)"
        echo "   最新训练信息:"
        tail -5 "$TRAIN_LOG" | sed 's/^/   /'
    else
        echo "❌ 训练日志不存在: $TRAIN_LOG"
    fi
    
    echo ""
    
    # 检查输出目录
    echo "📁 输出目录状态:"
    if [ -d "outputs" ]; then
        echo "✅ 输出目录存在: outputs/"
        MODEL_COUNT=$(find outputs -name "*.pth" 2>/dev/null | wc -l)
        echo "   模型文件数量: $MODEL_COUNT"
        if [ $MODEL_COUNT -gt 0 ]; then
            echo "   最新模型文件:"
            find outputs -name "*.pth" -printf "   %T@ %p\n" | sort -n | tail -3 | cut -d' ' -f2-
        fi
    else
        echo "❌ 输出目录不存在: outputs/"
    fi
    
    # 检查TensorBoard日志
    if [ -d "logs/tensorboard" ]; then
        echo "✅ TensorBoard日志存在: logs/tensorboard/"
        TB_SIZE=$(du -sh logs/tensorboard 2>/dev/null | cut -f1)
        echo "   日志大小: $TB_SIZE"
    else
        echo "❌ TensorBoard日志不存在: logs/tensorboard/"
    fi
fi

echo "======================================"
echo "💡 常用命令:"
echo "   取消作业: scancel <JOB_ID>"
echo "   查看队列: squeue -u \$USER"
echo "   查看节点: sinfo"
echo "   实时日志: tail -f logs/var_training_<JOB_ID>.out"
echo "   TensorBoard: tensorboard --logdir=logs/tensorboard --port=6006"
echo "======================================"