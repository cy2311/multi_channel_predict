#!/usr/bin/env python3
"""
DECODE网络批量训练管理器
支持多GPU并行训练不同样本数的模型，提供完整的任务管理和监控功能

作者: AI Assistant
日期: 2024
"""

import os
import sys
import json
import time
import argparse
import subprocess
import signal
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import threading
import queue

class Colors:
    """终端颜色常量"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    NC = '\033[0m'  # No Color

class TrainingConfig:
    """训练配置类"""
    def __init__(self, samples: int, epochs: int, gpu_id: int, 
                 batch_size: int = 4, lr: float = 1e-4, 
                 priority: int = 0, description: str = ""):
        self.samples = samples
        self.epochs = epochs
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.lr = lr
        self.priority = priority
        self.description = description
        
    def __str__(self):
        return f"{self.samples}样本_{self.epochs}epochs_GPU{self.gpu_id}"
        
    def to_dict(self):
        return {
            'samples': self.samples,
            'epochs': self.epochs,
            'gpu_id': self.gpu_id,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'priority': self.priority,
            'description': self.description
        }

class TrainingJob:
    """训练任务类"""
    def __init__(self, config: TrainingConfig, job_id: str):
        self.config = config
        self.job_id = job_id
        self.pid: Optional[int] = None
        self.tb_pid: Optional[int] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.status = "pending"  # pending, running, completed, failed, stopped
        self.output_dir: Optional[Path] = None
        self.log_file: Optional[Path] = None
        self.tb_port: Optional[int] = None
        self.exit_code: Optional[int] = None
        
    def is_running(self) -> bool:
        """检查任务是否正在运行"""
        if self.pid is None:
            return False
        try:
            os.kill(self.pid, 0)
            return True
        except OSError:
            return False
            
    def get_duration(self) -> str:
        """获取运行时长"""
        if self.start_time is None:
            return "未开始"
        
        end_time = self.end_time or datetime.now()
        duration = end_time - self.start_time
        
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
    def to_dict(self):
        return {
            'job_id': self.job_id,
            'config': self.config.to_dict(),
            'pid': self.pid,
            'tb_pid': self.tb_pid,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status,
            'output_dir': str(self.output_dir) if self.output_dir else None,
            'log_file': str(self.log_file) if self.log_file else None,
            'tb_port': self.tb_port,
            'exit_code': self.exit_code
        }

class BatchTrainingManager:
    """批量训练管理器"""
    
    def __init__(self, base_dir: str = "/home/guest/Others/DECODE_rewrite"):
        self.base_dir = Path(base_dir)
        self.training_dir = self.base_dir / "neuronal_network" / "training"
        self.data_base_dir = self.base_dir / "simulation_zmap2tiff"
        self.jobs: Dict[str, TrainingJob] = {}
        self.job_file = Path("/tmp/decode_training_jobs.json")
        
        # 加载样本配置
        self.load_sample_config()
        
        # 默认训练配置
        self.default_configs = [
            TrainingConfig(10, 2, 0, description="快速测试"),
            TrainingConfig(50, 5, 1, description="小规模训练"),
            TrainingConfig(100, 10, 2, description="中等规模训练"),
            TrainingConfig(200, 15, 3, description="大规模训练"),
        ]
        
        # 加载已有任务
        self.load_jobs()
        
    def load_sample_config(self):
        """加载样本配置文件"""
        config_file = self.base_dir / "neuronal_network" / "scripts" / "sample_config.json"
        try:
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                self.available_samples = config['sample_sizes']['available_samples']
                self.default_demo_samples = config['sample_sizes']['default_demo_samples']
                self.status_display_samples = config['sample_sizes']['status_display_samples']
                self.gpu_requirements = config['gpu_requirements']
                
                self.print_colored(f"✓ 已加载样本配置: {config_file}", Colors.GREEN)
            else:
                # 使用默认配置
                self.available_samples = [10, 50, 100, 200, 500, 1000]
                self.default_demo_samples = [10, 50, 100, 200]
                self.status_display_samples = [10, 50, 100, 200]
                self.gpu_requirements = {'10': 1, '50': 2, '100': 3, '200': 4, '500': 4, '1000': 4}
                self.print_colored(f"⚠ 配置文件不存在，使用默认配置: {config_file}", Colors.YELLOW)
                
        except Exception as e:
            self.print_colored(f"错误: 加载配置文件失败: {e}", Colors.RED)
            # 使用默认配置
            self.available_samples = [10, 50, 100, 200, 500, 1000]
            self.default_demo_samples = [10, 50, 100, 200]
            self.status_display_samples = [10, 50, 100, 200]
            self.gpu_requirements = {'10': 1, '50': 2, '100': 3, '200': 4, '500': 4, '1000': 4}
        
    def print_colored(self, text: str, color: str = Colors.NC):
        """打印彩色文本"""
        print(f"{color}{text}{Colors.NC}")
        
    def check_gpu_availability(self) -> List[Dict]:
        """检查GPU可用性"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            )
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'memory_used': int(parts[2]),
                        'memory_total': int(parts[3]),
                        'utilization': int(parts[4])
                    })
            
            return gpus
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.print_colored("错误: 无法获取GPU信息，请确保安装了NVIDIA驱动", Colors.RED)
            return []
            
    def check_data_dir(self, samples: int) -> bool:
        """检查数据目录是否存在，支持从更大的数据集中选择前N个样本"""
        # 首先检查精确匹配的数据目录
        exact_data_dir = self.data_base_dir / f"outputs_{samples}samples_256"
        if exact_data_dir.exists():
            return True
            
        # 如果精确匹配不存在，查找可用的更大数据集
        available_datasets = []
        for possible_samples in self.available_samples:
            possible_dir = self.data_base_dir / f"outputs_{possible_samples}samples_256"
            if possible_dir.exists():
                # 检查实际样本数量
                sample_dirs = list(possible_dir.glob("sample_*"))
                actual_count = len([d for d in sample_dirs if d.is_dir()])
                if actual_count >= samples:
                    available_datasets.append((possible_samples, actual_count, possible_dir))
                    
        if available_datasets:
            # 选择最小的但足够的数据集
            best_dataset = min(available_datasets, key=lambda x: x[0])
            dataset_samples, actual_count, dataset_dir = best_dataset
            
            self.print_colored(f"✓ 将使用 {dataset_samples} 样本数据集中的前 {samples} 个样本", Colors.GREEN)
            self.print_colored(f"  数据目录: {dataset_dir}", Colors.BLUE)
            self.print_colored(f"  可用样本: {actual_count} 个", Colors.BLUE)
            return True
            
        # 如果都没有找到合适的数据集
        self.print_colored(f"错误: 没有找到包含至少 {samples} 个样本的数据集", Colors.RED)
        self.print_colored("可用选项:", Colors.YELLOW)
        for possible_samples in self.status_display_samples:
            possible_dir = self.data_base_dir / f"outputs_{possible_samples}samples_256"
            if possible_dir.exists():
                sample_count = len(list(possible_dir.glob("sample_*")))
                self.print_colored(f"  ✓ {possible_samples} 样本数据集 ({sample_count} 个样本)", Colors.GREEN)
            else:
                self.print_colored(f"  ✗ {possible_samples} 样本数据集 (不存在)", Colors.RED)
        return False
        
    def find_suitable_data_dir(self, samples: int) -> Optional[Path]:
        """找到合适的数据目录"""
        # 首先检查精确匹配的数据目录
        exact_data_dir = self.data_base_dir / f"outputs_{samples}samples_256"
        if exact_data_dir.exists():
            return exact_data_dir
            
        # 查找可用的更大数据集
        for possible_samples in self.available_samples:
            possible_dir = self.data_base_dir / f"outputs_{possible_samples}samples_256"
            if possible_dir.exists():
                sample_dirs = list(possible_dir.glob("sample_*"))
                actual_count = len([d for d in sample_dirs if d.is_dir()])
                if actual_count >= samples:
                    return possible_dir
        return None
        
    def generate_job_id(self, config: TrainingConfig) -> str:
        """生成任务ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"decode_{config.samples}s_{config.epochs}e_gpu{config.gpu_id}_{timestamp}"
        
    def create_training_script(self, job: TrainingJob) -> Path:
        """创建训练脚本"""
        # 找到合适的数据目录
        data_dir = self.find_suitable_data_dir(job.config.samples)
        if not data_dir:
            raise ValueError(f"无法找到包含 {job.config.samples} 个样本的数据目录")
            
        script_content = f'''#!/bin/bash

# 设置GPU
export CUDA_VISIBLE_DEVICES={job.config.gpu_id}

# 记录开始信息
echo "==========================================="
echo "训练开始时间: $(date)"
echo "任务ID: {job.job_id}"
echo "样本数: {job.config.samples}"
echo "训练轮数: {job.config.epochs}"
echo "批处理大小: {job.config.batch_size}"
echo "学习率: {job.config.lr}"
echo "GPU ID: {job.config.gpu_id}"
echo "数据目录: {data_dir}"
echo "输出目录: {job.output_dir}"
echo "PID: $$"
echo "==========================================="

# 保存PID
echo $$ > {job.output_dir}/training.pid

# 切换到训练目录
cd {self.training_dir}

# 启动TensorBoard (后台运行)
TB_PORT={job.tb_port}
echo "启动TensorBoard在端口: $TB_PORT"
tensorboard --logdir={job.output_dir}/tensorboard --port=$TB_PORT --host=0.0.0.0 &
TB_PID=$!
echo "TensorBoard PID: $TB_PID"
echo $TB_PID > {job.output_dir}/tensorboard.pid

# 运行训练
echo "开始训练..."
python start_training.py \\
    --data_dir "{data_dir}" \\
    --samples {job.config.samples} \\
    --epochs {job.config.epochs} \\
    --batch_size {job.config.batch_size} \\
    --lr {job.config.lr} \\
    --output_suffix "_{datetime.now().strftime('%Y%m%d_%H%M%S')}" 2>&1

TRAIN_EXIT_CODE=$?

# 训练完成后的清理
echo "==========================================="
echo "训练完成时间: $(date)"
echo "训练退出码: $TRAIN_EXIT_CODE"
echo "正在清理TensorBoard进程..."
kill $TB_PID 2>/dev/null || true
rm -f {job.output_dir}/tensorboard.pid

# 生成训练总结
cat > {job.output_dir}/training_summary.json << EOF
{{
    "job_id": "{job.job_id}",
    "samples": {job.config.samples},
    "epochs": {job.config.epochs},
    "batch_size": {job.config.batch_size},
    "learning_rate": {job.config.lr},
    "gpu_id": {job.config.gpu_id},
    "data_dir": "{data_dir}",
    "start_time": "$(date -Iseconds)",
    "end_time": "$(date -Iseconds)",
    "exit_code": $TRAIN_EXIT_CODE,
    "output_dir": "{job.output_dir}",
    "tensorboard_port": {job.tb_port}
}}
EOF

# 清理PID文件
rm -f {job.output_dir}/training.pid

echo "任务完成!"
exit $TRAIN_EXIT_CODE
'''
        
        script_path = job.output_dir / "run_training.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        
        return script_path
        
    def start_job(self, config: TrainingConfig) -> Optional[TrainingJob]:
        """启动训练任务"""
        # 检查数据目录
        if not self.check_data_dir(config.samples):
            return None
            
        # 创建任务
        job_id = self.generate_job_id(config)
        job = TrainingJob(config, job_id)
        
        # 设置输出目录和端口
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job.output_dir = self.training_dir / "outputs" / f"train_{config.samples}samples_{timestamp}"
        job.output_dir.mkdir(parents=True, exist_ok=True)
        
        job.log_file = job.output_dir / "training.log"
        job.tb_port = 6006 + config.samples
        
        # 创建训练脚本
        script_path = self.create_training_script(job)
        
        # 启动任务
        try:
            process = subprocess.Popen(
                ['bash', str(script_path)],
                stdout=open(job.log_file, 'w'),
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid
            )
            
            job.pid = process.pid
            job.start_time = datetime.now()
            job.status = "running"
            
            self.jobs[job_id] = job
            self.save_jobs()
            
            self.print_colored(f"✓ 任务启动成功! Job ID: {job_id}", Colors.GREEN)
            self.print_colored(f"  PID: {job.pid}", Colors.BLUE)
            self.print_colored(f"  输出目录: {job.output_dir}", Colors.BLUE)
            self.print_colored(f"  日志文件: {job.log_file}", Colors.BLUE)
            self.print_colored(f"  TensorBoard端口: {job.tb_port}", Colors.BLUE)
            
            return job
            
        except Exception as e:
            self.print_colored(f"✗ 任务启动失败: {e}", Colors.RED)
            return None
            
    def stop_job(self, job_id: str) -> bool:
        """停止指定任务"""
        if job_id not in self.jobs:
            self.print_colored(f"任务 {job_id} 不存在", Colors.RED)
            return False
            
        job = self.jobs[job_id]
        
        if not job.is_running():
            self.print_colored(f"任务 {job_id} 未在运行", Colors.YELLOW)
            return False
            
        try:
            # 停止训练进程组
            os.killpg(os.getpgid(job.pid), signal.SIGTERM)
            
            # 停止TensorBoard
            if job.tb_pid and job.output_dir:
                tb_pid_file = job.output_dir / "tensorboard.pid"
                if tb_pid_file.exists():
                    try:
                        tb_pid = int(tb_pid_file.read_text().strip())
                        os.kill(tb_pid, signal.SIGTERM)
                        tb_pid_file.unlink()
                    except (ValueError, OSError):
                        pass
                        
            job.status = "stopped"
            job.end_time = datetime.now()
            self.save_jobs()
            
            self.print_colored(f"✓ 任务 {job_id} 已停止", Colors.GREEN)
            return True
            
        except Exception as e:
            self.print_colored(f"✗ 停止任务失败: {e}", Colors.RED)
            return False
            
    def stop_all_jobs(self) -> int:
        """停止所有运行中的任务"""
        running_jobs = [job for job in self.jobs.values() if job.is_running()]
        
        if not running_jobs:
            self.print_colored("无运行中的任务", Colors.YELLOW)
            return 0
            
        stopped_count = 0
        for job in running_jobs:
            if self.stop_job(job.job_id):
                stopped_count += 1
                
        self.print_colored(f"已停止 {stopped_count}/{len(running_jobs)} 个任务", Colors.GREEN)
        return stopped_count
        
    def update_job_status(self):
        """更新任务状态"""
        for job in self.jobs.values():
            if job.status == "running" and not job.is_running():
                # 检查退出码
                if job.output_dir:
                    summary_file = job.output_dir / "training_summary.json"
                    if summary_file.exists():
                        try:
                            summary = json.loads(summary_file.read_text())
                            job.exit_code = summary.get('exit_code', -1)
                        except:
                            job.exit_code = -1
                    else:
                        job.exit_code = -1
                        
                job.status = "completed" if job.exit_code == 0 else "failed"
                job.end_time = datetime.now()
                
        self.save_jobs()
        
    def show_status(self):
        """显示任务状态"""
        self.update_job_status()
        
        if not self.jobs:
            self.print_colored("无任务记录", Colors.YELLOW)
            return
            
        self.print_colored("\n=== DECODE训练任务状态 ===", Colors.BLUE)
        
        # 按状态分组显示
        status_groups = {
            "running": [],
            "pending": [],
            "completed": [],
            "failed": [],
            "stopped": []
        }
        
        for job in self.jobs.values():
            status_groups[job.status].append(job)
            
        for status, jobs in status_groups.items():
            if not jobs:
                continue
                
            status_colors = {
                "running": Colors.GREEN,
                "pending": Colors.YELLOW,
                "completed": Colors.BLUE,
                "failed": Colors.RED,
                "stopped": Colors.PURPLE
            }
            
            self.print_colored(f"\n{status.upper()} ({len(jobs)}):", status_colors[status])
            
            for job in sorted(jobs, key=lambda x: x.start_time or datetime.min):
                duration = job.get_duration()
                config_str = f"{job.config.samples}样本/{job.config.epochs}epochs/GPU{job.config.gpu_id}"
                
                if job.status == "running":
                    self.print_colored(
                        f"  {job.job_id}: {config_str} | 运行时长: {duration} | TensorBoard: {job.tb_port}",
                        Colors.WHITE
                    )
                else:
                    self.print_colored(
                        f"  {job.job_id}: {config_str} | 时长: {duration}",
                        Colors.WHITE
                    )
                    
                if job.output_dir:
                    self.print_colored(f"    输出: {job.output_dir}", Colors.CYAN)
                    
        # 显示GPU状态
        gpus = self.check_gpu_availability()
        if gpus:
            self.print_colored("\n=== GPU状态 ===", Colors.BLUE)
            for gpu in gpus:
                usage_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
                self.print_colored(
                    f"  GPU {gpu['index']}: {gpu['name']} | "
                    f"内存: {gpu['memory_used']}/{gpu['memory_total']}MB ({usage_percent:.1f}%) | "
                    f"利用率: {gpu['utilization']}%",
                    Colors.WHITE
                )
                
    def save_jobs(self):
        """保存任务信息到文件"""
        jobs_data = {job_id: job.to_dict() for job_id, job in self.jobs.items()}
        self.job_file.write_text(json.dumps(jobs_data, indent=2, ensure_ascii=False))
        
    def load_jobs(self):
        """从文件加载任务信息"""
        if not self.job_file.exists():
            return
            
        try:
            jobs_data = json.loads(self.job_file.read_text())
            
            for job_id, data in jobs_data.items():
                config = TrainingConfig(**data['config'])
                job = TrainingJob(config, job_id)
                
                job.pid = data.get('pid')
                job.tb_pid = data.get('tb_pid')
                job.start_time = datetime.fromisoformat(data['start_time']) if data.get('start_time') else None
                job.end_time = datetime.fromisoformat(data['end_time']) if data.get('end_time') else None
                job.status = data.get('status', 'pending')
                job.output_dir = Path(data['output_dir']) if data.get('output_dir') else None
                job.log_file = Path(data['log_file']) if data.get('log_file') else None
                job.tb_port = data.get('tb_port')
                job.exit_code = data.get('exit_code')
                
                self.jobs[job_id] = job
                
        except Exception as e:
            self.print_colored(f"加载任务信息失败: {e}", Colors.RED)
            
    def cleanup_finished_jobs(self):
        """清理已完成的任务记录"""
        self.update_job_status()
        
        finished_jobs = [
            job_id for job_id, job in self.jobs.items() 
            if job.status in ['completed', 'failed', 'stopped']
        ]
        
        for job_id in finished_jobs:
            del self.jobs[job_id]
            
        self.save_jobs()
        self.print_colored(f"已清理 {len(finished_jobs)} 个已完成的任务记录", Colors.GREEN)
        
    def show_data_status(self):
        """显示数据状态（智能检查）"""
        samples_list = self.status_display_samples
        
        for samples in samples_list:
            # 首先检查精确匹配的数据目录
            exact_data_dir = self.data_base_dir / f"outputs_{samples}samples_256"
            if exact_data_dir.exists():
                sample_dirs = list(exact_data_dir.glob("sample_*"))
                actual_count = len([d for d in sample_dirs if d.is_dir()])
                self.print_colored(f"  ✓ {samples} 样本数据: {actual_count} 个样本", Colors.GREEN)
                continue
                
            # 查找可用的更大数据集
            found_alternative = False
            for possible_samples in self.available_samples:
                if possible_samples <= samples:
                    continue
                    
                possible_dir = self.data_base_dir / f"outputs_{possible_samples}samples_256"
                if possible_dir.exists():
                    sample_dirs = list(possible_dir.glob("sample_*"))
                    actual_count = len([d for d in sample_dirs if d.is_dir()])
                    if actual_count >= samples:
                        self.print_colored(
                            f"  ✓ {samples} 样本数据: 可从 {possible_samples} 样本数据集中选择前 {samples} 个 ({actual_count} 个可用)", 
                            Colors.YELLOW
                        )
                        found_alternative = True
                        break
                        
            if not found_alternative:
                self.print_colored(f"  ✗ {samples} 样本数据: 不存在", Colors.RED)
    
    def start_batch_training(self, configs: List[TrainingConfig] = None):
        """启动批量训练"""
        if configs is None:
            configs = self.default_configs
            
        self.print_colored(f"\n=== 准备启动 {len(configs)} 个训练任务 ===", Colors.BLUE)
        
        # 检查GPU
        gpus = self.check_gpu_availability()
        if not gpus:
            return
            
        self.print_colored(f"检测到 {len(gpus)} 块GPU", Colors.GREEN)
        
        # 按优先级排序
        configs = sorted(configs, key=lambda x: x.priority, reverse=True)
        
        success_count = 0
        for i, config in enumerate(configs, 1):
            self.print_colored(f"\n[{i}/{len(configs)}] 启动任务: {config}", Colors.YELLOW)
            
            if self.start_job(config):
                success_count += 1
                time.sleep(2)  # 避免同时启动过多任务
            else:
                self.print_colored(f"跳过任务: {config}", Colors.RED)
                
        self.print_colored(f"\n=== 批量启动完成 ===", Colors.GREEN)
        self.print_colored(f"成功启动: {success_count}/{len(configs)} 个任务", Colors.GREEN)
        
        if success_count > 0:
            self.print_colored("\n监控建议:", Colors.BLUE)
            self.print_colored("1. 查看任务状态: python batch_train_manager.py --status", Colors.BLUE)
            self.print_colored("2. 停止所有任务: python batch_train_manager.py --stop-all", Colors.BLUE)
            self.print_colored("3. 监控GPU使用: watch -n 1 nvidia-smi", Colors.BLUE)

def main():
    parser = argparse.ArgumentParser(description="DECODE网络批量训练管理器")
    parser.add_argument('--base-dir', default='/home/guest/Others/DECODE_rewrite',
                       help='项目根目录')
    parser.add_argument('--status', '-s', action='store_true',
                       help='显示任务状态')
    parser.add_argument('--stop', help='停止指定任务')
    parser.add_argument('--stop-all', action='store_true',
                       help='停止所有任务')
    parser.add_argument('--cleanup', action='store_true',
                       help='清理已完成任务记录')
    parser.add_argument('--config', '-c', action='store_true',
                       help='显示默认配置')
    parser.add_argument('--data-status', action='store_true',
                       help='显示数据状态')
    
    args = parser.parse_args()
    
    manager = BatchTrainingManager(args.base_dir)
    
    if args.status:
        manager.show_status()
    elif args.stop:
        manager.stop_job(args.stop)
    elif args.stop_all:
        manager.stop_all_jobs()
    elif args.cleanup:
        manager.cleanup_finished_jobs()
    elif args.config:
        print("\n=== 默认训练配置 ===")
        for i, config in enumerate(manager.default_configs, 1):
            print(f"{i}. {config} - {config.description}")
    elif args.data_status:
        manager.show_data_status()
    else:
        # 启动批量训练
        manager.start_batch_training()

if __name__ == "__main__":
    main()