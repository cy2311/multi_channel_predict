#!/bin/bash

# DECODE自适应训练快速提交脚本
# 使用方法:
#   ./quick_submit.sh                    # 默认配置
#   ./quick_submit.sh --gpu 2            # 使用2个GPU
#   ./quick_submit.sh --time 12          # 12小时时间限制
#   ./quick_submit.sh --mem 64           # 64GB内存
#   ./quick_submit.sh --resume           # 恢复训练
#   ./quick_submit.sh --monitor          # 仅监控模式

# 默认参数
GPU_COUNT=1
TIME_LIMIT=24
MEMORY=32
CPU_COUNT=16
PARTITION="cpu1"
CONFIG_FILE="training/configs/train_config_adaptive.json"
RESUME=false
MONITOR_ONLY=false
JOB_NAME="decode_adaptive"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_COUNT="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --mem)
            MEMORY="$2"
            shift 2
            ;;
        --cpu)
            CPU_COUNT="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --name)
            JOB_NAME="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --monitor)
            MONITOR_ONLY=true
            shift
            ;;
        --help|-h)
            echo "DECODE自适应训练快速提交脚本"
            echo ""
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --gpu NUM        GPU数量 (默认: 1)"
            echo "  --time HOURS     时间限制(小时) (默认: 24)"
            echo "  --mem GB         内存大小(GB) (默认: 32)"
            echo "  --cpu NUM        CPU核心数 (默认: 16)"
            echo "  --partition NAME 分区名称 (默认: gpu)"
            echo "  --config FILE    配置文件路径 (默认: training/configs/train_config_adaptive.json)"
            echo "  --name NAME      作业名称 (默认: decode_adaptive)"
            echo "  --resume         恢复训练"
            echo "  --monitor        仅监控模式"
            echo "  --help, -h       显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                           # 默认配置"
            echo "  $0 --gpu 2 --time 48        # 2个GPU，48小时"
            echo "  $0 --mem 64 --cpu 32        # 64GB内存，32核CPU"
            echo "  $0 --resume                 # 恢复训练"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 创建日志目录
mkdir -p logs

# 生成作业名称后缀
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ "$RESUME" = "true" ]; then
    JOB_NAME="${JOB_NAME}_resume_${TIMESTAMP}"
elif [ "$MONITOR_ONLY" = "true" ]; then
    JOB_NAME="${JOB_NAME}_monitor_${TIMESTAMP}"
else
    JOB_NAME="${JOB_NAME}_${TIMESTAMP}"
fi

# 构建SLURM参数
SLURM_ARGS=""
SLURM_ARGS="$SLURM_ARGS --job-name=$JOB_NAME"
SLURM_ARGS="$SLURM_ARGS --output=logs/${JOB_NAME}_%j.out"
SLURM_ARGS="$SLURM_ARGS --error=logs/${JOB_NAME}_%j.err"
SLURM_ARGS="$SLURM_ARGS --time=${TIME_LIMIT}:00:00"
SLURM_ARGS="$SLURM_ARGS --partition=$PARTITION"
SLURM_ARGS="$SLURM_ARGS --gres=gpu:$GPU_COUNT"
SLURM_ARGS="$SLURM_ARGS --cpus-per-task=$CPU_COUNT"
SLURM_ARGS="$SLURM_ARGS --mem=${MEMORY}G"
SLURM_ARGS="$SLURM_ARGS --nodes=1"
SLURM_ARGS="$SLURM_ARGS --ntasks-per-node=1"

# 设置环境变量
export CONFIG_FILE="$CONFIG_FILE"
export RESUME="$RESUME"
export MONITOR_ONLY="$MONITOR_ONLY"

# 显示提交信息
echo "🚀 提交DECODE自适应训练作业"
echo "=============================="
echo "作业名称: $JOB_NAME"
echo "GPU数量: $GPU_COUNT"
echo "时间限制: ${TIME_LIMIT}小时"
echo "内存: ${MEMORY}GB"
echo "CPU核心: $CPU_COUNT"
echo "分区: $PARTITION"
echo "配置文件: $CONFIG_FILE"
echo "恢复训练: $RESUME"
echo "仅监控: $MONITOR_ONLY"
echo "=============================="

# 提交作业
echo "📤 提交作业..."
JOB_ID=$(sbatch $SLURM_ARGS submit_adaptive_training_flexible.sh | grep -o '[0-9]*')

if [ $? -eq 0 ] && [ -n "$JOB_ID" ]; then
    echo "✅ 作业提交成功！"
    echo "📋 作业ID: $JOB_ID"
    echo "📁 输出日志: logs/${JOB_NAME}_${JOB_ID}.out"
    echo "📁 错误日志: logs/${JOB_NAME}_${JOB_ID}.err"
    echo ""
    echo "💡 有用的命令:"
    echo "   查看作业状态: squeue -j $JOB_ID"
    echo "   查看作业详情: scontrol show job $JOB_ID"
    echo "   取消作业: scancel $JOB_ID"
    echo "   实时查看日志: tail -f logs/${JOB_NAME}_${JOB_ID}.out"
    echo "   查看GPU使用: ssh \$(squeue -j $JOB_ID -h -o %N) nvidia-smi"
else
    echo "❌ 作业提交失败！"
    exit 1
fi