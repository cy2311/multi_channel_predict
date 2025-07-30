#!/bin/bash
#SBATCH --job-name=tiff_gen_cpu
#SBATCH --output=../logs/tiff_generation_cpu_%j.out
#SBATCH --error=../logs/tiff_generation_cpu_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu1

# 设置工作目录
cd /home/guest/Others/DECODE_rewrite/simulation_zmap2tiff\ copy

# 创建日志目录
mkdir -p ../logs

# 激活conda环境（根据实际情况修改）
# module load anaconda3
# source activate base
# conda activate your_env_name

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:/home/guest/Others/DECODE_rewrite"

# 设置多线程优化
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 记录系统信息
echo "=== 作业信息 ==="
echo "作业ID: $SLURM_JOB_ID"
echo "开始时间: $(date)"
echo "节点: $SLURM_NODELIST"
echo "CPU核心数: $SLURM_CPUS_PER_TASK"
echo "内存: ${SLURM_MEM_PER_NODE}MB"
echo "工作目录: $(pwd)"
echo "Python版本: $(python --version)"
echo "========================================"

# 运行任务
echo "开始生成100样本256x256尺寸TIFF数据集..."
echo "预计运行时间: 4-6小时"
echo "========================================"

# 使用timeout确保不超时
timeout 7h python ../scripts/run_100_samples.py --auto-confirm
exit_code=$?

# 处理结果
echo "========================================"
if [ $exit_code -eq 0 ]; then
    echo "✅ 任务成功完成!"
elif [ $exit_code -eq 124 ]; then
    echo "⚠️ 任务超时，但可能部分完成"
else
    echo "❌ 任务失败，退出码: $exit_code"
fi

echo "完成时间: $(date)"

# 显示结果统计
if [ -d "../outputs_100samples_256" ]; then
    echo "\n=== 输出统计 ==="
    sample_count=$(ls -1d ../outputs_100samples_256/sample_* 2>/dev/null | wc -l)
    tiff_count=$(find ../outputs_100samples_256 -name "*.tiff" 2>/dev/null | wc -l)
    total_size=$(du -sh ../outputs_100samples_256 2>/dev/null | cut -f1)
    
    echo "生成样本数: $sample_count / 100"
    echo "TIFF文件数: $tiff_count"
    echo "总数据大小: $total_size"
    
    if [ $sample_count -ge 90 ]; then
        echo "✅ 数据集生成基本完成 (≥90%)"
    elif [ $sample_count -ge 50 ]; then
        echo "⚠️ 数据集部分完成 (≥50%)"
    else
        echo "❌ 数据集生成不完整 (<50%)"
    fi
fi

echo "作业结束: $(date)"