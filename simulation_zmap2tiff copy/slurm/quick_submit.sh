#!/bin/bash

# 256x256 TIFF生成任务快速提交脚本
# 使用方法: ./quick_submit.sh [gpu|cpu]

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查SLURM是否可用
check_slurm() {
    if ! command -v sbatch &> /dev/null; then
        print_error "SLURM未安装或不可用"
        print_info "请确保在SLURM集群环境中运行此脚本"
        exit 1
    fi
}

# 检查必要文件
check_files() {
    local files=(
        "../configs/batch_config_100samples_256.json"
        "../scripts/run_100_samples.py"
        "../batch_tiff_generator.py"
    )
    
    for file in "${files[@]}"; do
        if [[ ! -f "$file" ]]; then
            print_error "必要文件不存在: $file"
            exit 1
        fi
    done
    
    # 检查Zernike图文件
    local zmap_path="/home/guest/Others/DECODE_rewrite/phase_retrieval_tiff2h5/result/result.h5"
    if [[ ! -f "$zmap_path" ]]; then
        print_error "Zernike图文件不存在: $zmap_path"
        print_info "请先运行相位检索生成Zernike图"
        exit 1
    fi
}

# 创建必要目录
setup_directories() {
    mkdir -p ../logs
    mkdir -p ../outputs_100samples_256
    print_info "已创建必要目录"
}

# 显示系统信息
show_system_info() {
    print_info "=== 系统信息 ==="
    echo "当前用户: $(whoami)"
    echo "工作目录: $(pwd)"
    echo "可用分区: $(sinfo -h -o '%P' | tr '\n' ' ')"
    echo "队列状态: $(squeue -u $(whoami) | wc -l) 个任务在队列中"
    echo ""
}

# 提交任务
submit_job() {
    local job_type="$1"
    local script_file
    
    case "$job_type" in
        "gpu")
            script_file="submit_100_samples.slurm"
            print_info "准备提交GPU任务..."
            ;;
        "cpu")
            script_file="submit_100_samples_cpu.slurm"
            print_info "准备提交CPU任务..."
            ;;
        *)
            print_error "无效的任务类型: $job_type"
            print_info "使用方法: $0 [gpu|cpu]"
            exit 1
            ;;
    esac
    
    if [[ ! -f "$script_file" ]]; then
        print_error "SLURM脚本不存在: $script_file"
        exit 1
    fi
    
    # 显示脚本信息
    print_info "=== 任务配置 ==="
    echo "脚本文件: $script_file"
    echo "任务类型: $job_type"
    
    # 从脚本中提取关键信息
    local cpus=$(grep "#SBATCH --cpus-per-task" "$script_file" | awk -F'=' '{print $2}' | awk '{print $1}' || echo "未指定")
    local mem=$(grep "#SBATCH --mem" "$script_file" | awk -F'=' '{print $2}' | awk '{print $1}' || echo "未指定")
    local time=$(grep "#SBATCH --time" "$script_file" | awk -F'=' '{print $2}' | awk '{print $1}' || echo "未指定")
    local partition=$(grep "#SBATCH --partition" "$script_file" | awk -F'=' '{print $2}' | awk '{print $1}' || echo "未指定")
    
    echo "CPU核心: $cpus"
    echo "内存: $mem"
    echo "最大时间: $time"
    echo "分区: $partition"
    echo ""
    
    # 确认提交
    print_warning "即将提交100样本生成任务 (预计生成20,000帧数据)"
    read -p "确认提交? [y/N]: " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "已取消提交"
        exit 0
    fi
    
    # 提交任务
    print_info "正在提交任务..."
    local job_id
    job_id=$(sbatch "$script_file" | awk '{print $4}')
    
    if [[ $? -eq 0 && -n "$job_id" ]]; then
        print_success "任务提交成功!"
        echo ""
        print_info "=== 任务信息 ==="
        echo "任务ID: $job_id"
        echo "脚本: $script_file"
        echo "输出日志: ../logs/tiff_generation_${job_id}.out"
        echo "错误日志: ../logs/tiff_generation_${job_id}.err"
        echo ""
        
        print_info "=== 监控命令 ==="
        echo "查看任务状态: squeue -j $job_id"
        echo "查看实时日志: tail -f ../logs/tiff_generation_${job_id}.out"
        echo "取消任务: scancel $job_id"
        echo "查看进度: ls -1d ../outputs_100samples_256/sample_* | wc -l"
        echo ""
        
        print_success "任务已在后台运行，您可以关闭IDE了!"
    else
        print_error "任务提交失败"
        exit 1
    fi
}

# 主函数
main() {
    echo "=== 256x256 TIFF生成任务快速提交工具 ==="
    echo ""
    
    # 检查参数
    if [[ $# -eq 0 ]]; then
        print_info "使用方法: $0 [gpu|cpu]"
        print_info "  gpu - 使用GPU节点 (推荐，速度更快)"
        print_info "  cpu - 使用CPU节点 (兼容性更好)"
        echo ""
        read -p "请选择任务类型 [gpu/cpu]: " job_type
    else
        job_type="$1"
    fi
    
    # 执行检查和提交
    check_slurm
    check_files
    setup_directories
    show_system_info
    submit_job "$job_type"
}

# 运行主函数
main "$@"