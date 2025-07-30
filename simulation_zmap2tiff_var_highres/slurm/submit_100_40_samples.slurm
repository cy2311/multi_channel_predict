#!/bin/bash
#SBATCH --job-name=tiff_generation_40
#SBATCH --output=../logs/tiff_generation_%j.out
#SBATCH --error=../logs/tiff_generation_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu1
# #SBATCH --gres=gpu:1  # 当前集群无GPU，注释掉

# 设置工作目录
cd /home/guest/Others/DECODE_rewrite/simulation_zmap2tiff

# 创建日志目录
mkdir -p ../logs

# 激活conda环境（如果需要）
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate your_env_name

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:/home/guest/Others/DECODE_rewrite"

# 记录开始时间
echo "作业开始时间: $(date)"
echo "节点信息: $(hostname)"
echo "GPU信息:"
nvidia-smi
echo "内存信息:"
free -h
echo "CPU信息:"
lscpu | grep "Model name"
echo "========================================"

# 运行40x40 TIFF生成任务
echo "开始生成100样本40x40尺寸TIFF数据集..."
python scripts/run_100_40_samples.py --auto-confirm

# 检查退出状态
if [ $? -eq 0 ]; then
    echo "========================================"
    echo "✅ 任务成功完成!"
    echo "完成时间: $(date)"
    
    # 显示输出目录统计
    echo "输出目录统计:"
    if [ -d "../outputs_100samples_40" ]; then
        echo "样本数量: $(ls -1 ../outputs_100samples_40/sample_* 2>/dev/null | wc -l)"
        echo "总文件数: $(find ../outputs_100samples_40 -name "*.tiff" 2>/dev/null | wc -l)"
        echo "总大小: $(du -sh ../outputs_100samples_40 2>/dev/null | cut -f1)"
    fi
else
    echo "========================================"
    echo "❌ 任务执行失败，退出码: $?"
    echo "失败时间: $(date)"
fi

echo "作业结束时间: $(date)"