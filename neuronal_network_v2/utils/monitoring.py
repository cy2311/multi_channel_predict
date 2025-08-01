"""监控工具模块

该模块提供了系统监控和性能分析功能，包括：
- 系统资源监控
- GPU监控
- 训练过程监控
- 性能分析
- 内存监控
"""

import time
import psutil
import threading
import queue
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    nvml.nvmlInit()
    NVML_AVAILABLE = True
except (ImportError, Exception):
    NVML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used: float  # MB
    memory_available: float  # MB
    disk_usage: float  # percent
    network_sent: float  # MB
    network_recv: float  # MB
    load_average: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used': self.memory_used,
            'memory_available': self.memory_available,
            'disk_usage': self.disk_usage,
            'network_sent': self.network_sent,
            'network_recv': self.network_recv,
            'load_average': self.load_average
        }


@dataclass
class GPUMetrics:
    """GPU指标"""
    timestamp: datetime
    gpu_id: int
    gpu_name: str
    gpu_utilization: float  # percent
    memory_utilization: float  # percent
    memory_used: float  # MB
    memory_total: float  # MB
    temperature: float  # Celsius
    power_usage: float  # Watts
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'gpu_id': self.gpu_id,
            'gpu_name': self.gpu_name,
            'gpu_utilization': self.gpu_utilization,
            'memory_utilization': self.memory_utilization,
            'memory_used': self.memory_used,
            'memory_total': self.memory_total,
            'temperature': self.temperature,
            'power_usage': self.power_usage
        }


@dataclass
class TrainingMetrics:
    """训练指标"""
    timestamp: datetime
    epoch: int
    step: int
    loss: float
    learning_rate: float
    batch_size: int
    samples_per_second: float
    gpu_memory_used: float = 0.0
    gradient_norm: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'epoch': self.epoch,
            'step': self.step,
            'loss': self.loss,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'samples_per_second': self.samples_per_second,
            'gpu_memory_used': self.gpu_memory_used,
            'gradient_norm': self.gradient_norm
        }


class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.running = False
        self.thread = None
        self.metrics_queue = queue.Queue()
        self.callbacks = []
        
        # 网络统计基线
        self.network_baseline = psutil.net_io_counters()
    
    def add_callback(self, callback: Callable[[SystemMetrics], None]):
        """添加回调函数"""
        self.callbacks.append(callback)
    
    def get_current_metrics(self) -> SystemMetrics:
        """获取当前系统指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # 内存使用情况
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used / 1024 / 1024  # MB
        memory_available = memory.available / 1024 / 1024  # MB
        
        # 磁盘使用情况
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        
        # 网络使用情况
        network_current = psutil.net_io_counters()
        network_sent = (network_current.bytes_sent - self.network_baseline.bytes_sent) / 1024 / 1024
        network_recv = (network_current.bytes_recv - self.network_baseline.bytes_recv) / 1024 / 1024
        
        # 负载平均值
        try:
            load_average = list(psutil.getloadavg())
        except AttributeError:
            # Windows不支持getloadavg
            load_average = []
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used=memory_used,
            memory_available=memory_available,
            disk_usage=disk_usage,
            network_sent=network_sent,
            network_recv=network_recv,
            load_average=load_average
        )
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                metrics = self.get_current_metrics()
                self.metrics_queue.put(metrics)
                
                # 调用回调函数
                for callback in self.callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"监控回调函数执行失败: {e}")
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"系统监控出错: {e}")
                time.sleep(self.interval)
    
    def start(self):
        """开始监控"""
        if self.running:
            logger.warning("系统监控已在运行")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("系统监控已启动")
    
    def stop(self):
        """停止监控"""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("系统监控已停止")
    
    def get_metrics_history(self, max_items: int = 100) -> List[SystemMetrics]:
        """获取指标历史"""
        metrics = []
        count = 0
        
        while not self.metrics_queue.empty() and count < max_items:
            try:
                metrics.append(self.metrics_queue.get_nowait())
                count += 1
            except queue.Empty:
                break
        
        return metrics


class GPUMonitor:
    """GPU监控器"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.running = False
        self.thread = None
        self.metrics_queue = queue.Queue()
        self.callbacks = []
        
        # 检查GPU可用性
        self.gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
        self.nvml_available = NVML_AVAILABLE
        
        if self.gpu_available:
            self.device_count = torch.cuda.device_count()
        else:
            self.device_count = 0
    
    def add_callback(self, callback: Callable[[List[GPUMetrics]], None]):
        """添加回调函数"""
        self.callbacks.append(callback)
    
    def get_current_metrics(self) -> List[GPUMetrics]:
        """获取当前GPU指标"""
        if not self.gpu_available:
            return []
        
        metrics = []
        timestamp = datetime.now()
        
        for gpu_id in range(self.device_count):
            try:
                # PyTorch GPU信息
                torch.cuda.set_device(gpu_id)
                gpu_name = torch.cuda.get_device_name(gpu_id)
                memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024 / 1024  # MB
                memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024 / 1024  # MB
                
                # NVML GPU信息
                if self.nvml_available:
                    try:
                        handle = nvml.nvmlDeviceGetHandleByIndex(gpu_id)
                        
                        # GPU使用率
                        utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_utilization = utilization.gpu
                        memory_utilization = utilization.memory
                        
                        # 内存信息
                        memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                        memory_total = memory_info.total / 1024 / 1024  # MB
                        memory_used = memory_info.used / 1024 / 1024  # MB
                        
                        # 温度
                        temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                        
                        # 功耗
                        try:
                            power_usage = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
                        except nvml.NVMLError:
                            power_usage = 0.0
                        
                    except nvml.NVMLError as e:
                        logger.warning(f"NVML获取GPU {gpu_id}信息失败: {e}")
                        gpu_utilization = 0.0
                        memory_utilization = 0.0
                        memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024 / 1024
                        memory_used = memory_allocated
                        temperature = 0.0
                        power_usage = 0.0
                else:
                    # 仅使用PyTorch信息
                    gpu_utilization = 0.0
                    memory_utilization = 0.0
                    memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024 / 1024
                    memory_used = memory_allocated
                    temperature = 0.0
                    power_usage = 0.0
                
                metrics.append(GPUMetrics(
                    timestamp=timestamp,
                    gpu_id=gpu_id,
                    gpu_name=gpu_name,
                    gpu_utilization=gpu_utilization,
                    memory_utilization=memory_utilization,
                    memory_used=memory_used,
                    memory_total=memory_total,
                    temperature=temperature,
                    power_usage=power_usage
                ))
                
            except Exception as e:
                logger.error(f"获取GPU {gpu_id}指标失败: {e}")
        
        return metrics
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                metrics = self.get_current_metrics()
                if metrics:
                    self.metrics_queue.put(metrics)
                    
                    # 调用回调函数
                    for callback in self.callbacks:
                        try:
                            callback(metrics)
                        except Exception as e:
                            logger.error(f"GPU监控回调函数执行失败: {e}")
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"GPU监控出错: {e}")
                time.sleep(self.interval)
    
    def start(self):
        """开始监控"""
        if not self.gpu_available:
            logger.warning("GPU不可用，无法启动GPU监控")
            return
        
        if self.running:
            logger.warning("GPU监控已在运行")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("GPU监控已启动")
    
    def stop(self):
        """停止监控"""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("GPU监控已停止")
    
    def get_metrics_history(self, max_items: int = 100) -> List[List[GPUMetrics]]:
        """获取指标历史"""
        metrics = []
        count = 0
        
        while not self.metrics_queue.empty() and count < max_items:
            try:
                metrics.append(self.metrics_queue.get_nowait())
                count += 1
            except queue.Empty:
                break
        
        return metrics


class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.metrics_history = []
        self.log_file = log_file
        self.callbacks = []
        
        if self.log_file:
            # 确保日志目录存在
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def add_callback(self, callback: Callable[[TrainingMetrics], None]):
        """添加回调函数"""
        self.callbacks.append(callback)
    
    def log_metrics(self, metrics: TrainingMetrics):
        """记录训练指标"""
        self.metrics_history.append(metrics)
        
        # 写入日志文件
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(metrics.to_dict(), ensure_ascii=False) + '\n')
            except Exception as e:
                logger.error(f"写入训练日志失败: {e}")
        
        # 调用回调函数
        for callback in self.callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"训练监控回调函数执行失败: {e}")
    
    def get_metrics_history(self, last_n: Optional[int] = None) -> List[TrainingMetrics]:
        """获取指标历史"""
        if last_n is None:
            return self.metrics_history.copy()
        else:
            return self.metrics_history[-last_n:]
    
    def get_average_metrics(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """获取平均指标"""
        history = self.get_metrics_history(last_n)
        
        if not history:
            return {}
        
        avg_metrics = {
            'loss': sum(m.loss for m in history) / len(history),
            'learning_rate': sum(m.learning_rate for m in history) / len(history),
            'samples_per_second': sum(m.samples_per_second for m in history) / len(history),
            'gpu_memory_used': sum(m.gpu_memory_used for m in history) / len(history),
            'gradient_norm': sum(m.gradient_norm for m in history) / len(history)
        }
        
        return avg_metrics
    
    def clear_history(self):
        """清除历史记录"""
        self.metrics_history.clear()
        logger.info("训练监控历史已清除")


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.timers = {}
        self.counters = {}
        self.memory_snapshots = {}
    
    def start_timer(self, name: str):
        """开始计时"""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """结束计时"""
        if name not in self.timers:
            logger.warning(f"计时器 {name} 不存在")
            return 0.0
        
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        return elapsed
    
    def increment_counter(self, name: str, value: int = 1):
        """增加计数器"""
        self.counters[name] = self.counters.get(name, 0) + value
    
    def get_counter(self, name: str) -> int:
        """获取计数器值"""
        return self.counters.get(name, 0)
    
    def snapshot_memory(self, name: str):
        """内存快照"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            gpu_memory = 0.0
        
        cpu_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        self.memory_snapshots[name] = {
            'timestamp': datetime.now(),
            'cpu_memory': cpu_memory,
            'gpu_memory': gpu_memory
        }
    
    def get_memory_diff(self, start_name: str, end_name: str) -> Dict[str, float]:
        """获取内存差异"""
        if start_name not in self.memory_snapshots or end_name not in self.memory_snapshots:
            return {'cpu_memory_diff': 0.0, 'gpu_memory_diff': 0.0}
        
        start = self.memory_snapshots[start_name]
        end = self.memory_snapshots[end_name]
        
        return {
            'cpu_memory_diff': end['cpu_memory'] - start['cpu_memory'],
            'gpu_memory_diff': end['gpu_memory'] - start['gpu_memory']
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            'counters': self.counters.copy(),
            'memory_snapshots': {k: v.copy() for k, v in self.memory_snapshots.items()},
            'active_timers': list(self.timers.keys())
        }
    
    def reset(self):
        """重置分析器"""
        self.timers.clear()
        self.counters.clear()
        self.memory_snapshots.clear()


class MonitoringManager:
    """监控管理器"""
    
    def __init__(self, 
                 system_interval: float = 5.0,
                 gpu_interval: float = 2.0,
                 log_dir: Optional[str] = None):
        self.system_monitor = SystemMonitor(interval=system_interval)
        self.gpu_monitor = GPUMonitor(interval=gpu_interval)
        self.training_monitor = TrainingMonitor(
            log_file=str(Path(log_dir) / "training_metrics.jsonl") if log_dir else None
        )
        self.profiler = PerformanceProfiler()
        
        self.log_dir = log_dir
        if self.log_dir:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
    
    def start_monitoring(self):
        """开始所有监控"""
        self.system_monitor.start()
        self.gpu_monitor.start()
        logger.info("监控管理器已启动")
    
    def stop_monitoring(self):
        """停止所有监控"""
        self.system_monitor.stop()
        self.gpu_monitor.stop()
        logger.info("监控管理器已停止")
    
    def log_training_step(self, 
                         epoch: int,
                         step: int,
                         loss: float,
                         learning_rate: float,
                         batch_size: int,
                         samples_per_second: float,
                         **kwargs):
        """记录训练步骤"""
        # 获取GPU内存使用
        gpu_memory_used = 0.0
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        metrics = TrainingMetrics(
            timestamp=datetime.now(),
            epoch=epoch,
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            batch_size=batch_size,
            samples_per_second=samples_per_second,
            gpu_memory_used=gpu_memory_used,
            gradient_norm=kwargs.get('gradient_norm', 0.0)
        )
        
        self.training_monitor.log_metrics(metrics)
    
    def export_metrics(self, output_file: str):
        """导出所有指标"""
        try:
            # 收集所有指标
            system_metrics = self.system_monitor.get_metrics_history()
            gpu_metrics = self.gpu_monitor.get_metrics_history()
            training_metrics = self.training_monitor.get_metrics_history()
            
            # 转换为字典格式
            data = {
                'export_time': datetime.now().isoformat(),
                'system_metrics': [m.to_dict() for m in system_metrics],
                'gpu_metrics': [[gpu.to_dict() for gpu in gpu_list] for gpu_list in gpu_metrics],
                'training_metrics': [m.to_dict() for m in training_metrics],
                'profiler_summary': self.profiler.get_summary()
            }
            
            # 写入文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"指标已导出到: {output_file}")
            
        except Exception as e:
            logger.error(f"导出指标失败: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'system_monitoring': self.system_monitor.running,
            'gpu_monitoring': self.gpu_monitor.running,
            'gpu_available': self.gpu_monitor.gpu_available,
            'device_count': self.gpu_monitor.device_count
        }
        
        # 当前系统指标
        try:
            current_system = self.system_monitor.get_current_metrics()
            status['current_system'] = current_system.to_dict()
        except Exception as e:
            logger.error(f"获取系统指标失败: {e}")
            status['current_system'] = None
        
        # 当前GPU指标
        try:
            current_gpu = self.gpu_monitor.get_current_metrics()
            status['current_gpu'] = [gpu.to_dict() for gpu in current_gpu]
        except Exception as e:
            logger.error(f"获取GPU指标失败: {e}")
            status['current_gpu'] = []
        
        # 训练统计
        training_history = self.training_monitor.get_metrics_history()
        if training_history:
            status['training_stats'] = {
                'total_steps': len(training_history),
                'latest_epoch': training_history[-1].epoch,
                'latest_loss': training_history[-1].loss,
                'average_metrics': self.training_monitor.get_average_metrics(last_n=100)
            }
        else:
            status['training_stats'] = None
        
        return status