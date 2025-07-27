#!/bin/bash

# 256x256 TIFF生成进度检查脚本
# 使用方法: ./check_progress.sh

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

# 检查SLURM任务状态
check_slurm_jobs() {
    if command -v squeue &> /dev/null; then
        print_info "=== SLURM任务状态 ==="
        local jobs=$(squeue -u $(whoami) --format="%.10i %.20j %.8T %.10M %.6D %R" --noheader)
        
        if [[ -z "$jobs" ]]; then
            echo "当前没有运行中的SLURM任务"
        else
            echo "任务ID     任务名称              状态    运行时间  节点数 原因"
            echo "$jobs"
        fi
        echo ""
    else
        print_warning "SLURM不可用，跳过任务状态检查"
    fi
}

# 检查输出目录
check_output_directory() {
    local output_dir="outputs_100samples_256"
    
    print_info "=== 输出目录状态 ==="
    
    if [[ ! -d "$output_dir" ]]; then
        print_warning "输出目录不存在: $output_dir"
        return
    fi
    
    # 统计样本数量
    local sample_count=$(ls -1d $output_dir/sample_* 2>/dev/null | wc -l)
    local total_samples=100
    local progress_percent=$((sample_count * 100 / total_samples))
    
    echo "样本进度: $sample_count / $total_samples ($progress_percent%)"
    
    # 进度条
    local bar_length=50
    local filled_length=$((progress_percent * bar_length / 100))
    local bar=""
    
    for ((i=0; i<filled_length; i++)); do
        bar+="█"
    done
    
    for ((i=filled_length; i<bar_length; i++)); do
        bar+="░"
    done
    
    echo "[$bar] $progress_percent%"
    
    # 统计TIFF文件
    local tiff_count=$(find $output_dir -name "*.tiff" 2>/dev/null | wc -l)
    local expected_tiff=$((sample_count * 1))  # 每个样本1个TIFF文件
    echo "TIFF文件: $tiff_count / $expected_tiff"
    
    # 统计总大小
    if [[ $sample_count -gt 0 ]]; then
        local total_size=$(du -sh $output_dir 2>/dev/null | cut -f1)
        echo "总大小: $total_size"
    fi
    
    # 状态评估
    if [[ $sample_count -eq $total_samples ]]; then
        print_success "✅ 数据集生成完成!"
    elif [[ $sample_count -ge 90 ]]; then
        print_success "✅ 数据集基本完成 (≥90%)"
    elif [[ $sample_count -ge 50 ]]; then
        print_warning "⚠️ 数据集部分完成 (≥50%)"
    elif [[ $sample_count -gt 0 ]]; then
        print_info "🔄 数据集生成中..."
    else
        print_warning "⚠️ 尚未开始生成或生成失败"
    fi
    
    echo ""
}

# 检查批量状态文件
check_batch_status() {
    local status_file="outputs_100samples_256/batch_status.json"
    
    print_info "=== 批量处理状态 ==="
    
    if [[ ! -f "$status_file" ]]; then
        print_warning "批量状态文件不存在: $status_file"
        return
    fi
    
    # 检查是否有jq命令
    if command -v jq &> /dev/null; then
        echo "批量处理摘要:"
        jq -r '.summary | to_entries[] | "\(.key): \(.value)"' "$status_file" 2>/dev/null || {
            print_warning "无法解析JSON状态文件"
        }
    else
        print_warning "jq未安装，显示原始状态文件:"
        tail -20 "$status_file"
    fi
    
    echo ""
}

# 检查最近的日志
check_recent_logs() {
    print_info "=== 最近日志 ==="
    
    local log_dir="logs"
    
    if [[ ! -d "$log_dir" ]]; then
        print_warning "日志目录不存在: $log_dir"
        return
    fi
    
    # 查找最新的输出日志
    local latest_out=$(ls -t $log_dir/tiff_generation_*.out 2>/dev/null | head -1)
    local latest_err=$(ls -t $log_dir/tiff_generation_*.err 2>/dev/null | head -1)
    
    if [[ -n "$latest_out" ]]; then
        echo "最新输出日志: $latest_out"
        echo "最后10行:"
        tail -10 "$latest_out" | sed 's/^/  /'
        echo ""
    fi
    
    if [[ -n "$latest_err" && -s "$latest_err" ]]; then
        echo "最新错误日志: $latest_err"
        echo "最后5行:"
        tail -5 "$latest_err" | sed 's/^/  /'
        echo ""
    fi
}

# 显示有用的命令
show_useful_commands() {
    print_info "=== 有用的命令 ==="
    
    echo "监控命令:"
    echo "  实时查看样本数: watch 'ls -1d outputs_100samples_256/sample_* | wc -l'"
    echo "  查看最新日志: tail -f logs/tiff_generation_*.out"
    echo "  查看SLURM队列: squeue -u $(whoami)"
    echo ""
    
    echo "管理命令:"
    echo "  取消所有任务: scancel -u $(whoami)"
    echo "  重新提交任务: ./quick_submit.sh gpu"
    echo "  清理输出: rm -rf outputs_100samples_256/"
    echo ""
}

# 主函数
main() {
    echo "=== 256x256 TIFF生成进度检查 ==="
    echo "检查时间: $(date)"
    echo ""
    
    check_slurm_jobs
    check_output_directory
    check_batch_status
    check_recent_logs
    show_useful_commands
    
    print_info "检查完成。使用 './check_progress.sh' 重新检查进度。"
}

# 运行主函数
main "$@"