"""推理工具模块

包含自动批量大小调整、内存高效推理等实用功能。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
import time
import logging
import gc
import psutil
import h5py
from tqdm import tqdm


def auto_batch_size(model: nn.Module,
                   sample_input: torch.Tensor,
                   max_memory_gb: float = 4.0,
                   start_batch_size: int = 1,
                   max_batch_size: int = 128,
                   safety_factor: float = 0.8) -> int:
    """自动确定最佳批量大小
    
    通过二分搜索找到在给定内存限制下的最大批量大小。
    
    Args:
        model: 神经网络模型
        sample_input: 样本输入张量
        max_memory_gb: 最大内存使用量（GB）
        start_batch_size: 起始批量大小
        max_batch_size: 最大批量大小
        safety_factor: 安全系数
        
    Returns:
        最佳批量大小
    """
    device = next(model.parameters()).device
    model.eval()
    
    # 清理内存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    def test_batch_size(batch_size: int) -> bool:
        """测试指定批量大小是否可行"""
        try:
            # 创建测试批量
            batch_input = sample_input.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            
            # 记录初始内存
            if device.type == 'cuda':
                initial_memory = torch.cuda.memory_allocated(device)
            else:
                initial_memory = psutil.virtual_memory().used
            
            # 前向传播
            with torch.no_grad():
                output = model(batch_input)
            
            # 检查内存使用
            if device.type == 'cuda':
                current_memory = torch.cuda.memory_allocated(device)
                memory_used_gb = (current_memory - initial_memory) / 1024**3
            else:
                current_memory = psutil.virtual_memory().used
                memory_used_gb = (current_memory - initial_memory) / 1024**3
            
            # 清理
            del batch_input, output
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            return memory_used_gb <= max_memory_gb * safety_factor
            
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            # 内存不足
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            return False
        except Exception:
            return False
    
    # 二分搜索
    left, right = start_batch_size, max_batch_size
    best_batch_size = start_batch_size
    
    while left <= right:
        mid = (left + right) // 2
        
        if test_batch_size(mid):
            best_batch_size = mid
            left = mid + 1
        else:
            right = mid - 1
    
    return best_batch_size


def memory_efficient_inference(model: nn.Module,
                              data_path: Union[str, Path],
                              device: torch.device,
                              chunk_size: int = 1000,
                              batch_size: Optional[int] = None,
                              post_processor: Optional[Callable] = None,
                              result_parser: Optional[Callable] = None,
                              enable_amp: bool = True,
                              show_progress: bool = True,
                              **kwargs) -> List[Dict[str, Any]]:
    """内存高效的大数据集推理
    
    Args:
        model: 神经网络模型
        data_path: 数据文件路径
        device: 推理设备
        chunk_size: 块大小
        batch_size: 批量大小
        post_processor: 后处理器
        result_parser: 结果解析器
        enable_amp: 是否启用混合精度
        show_progress: 是否显示进度
        
    Returns:
        推理结果列表
    """
    data_path = Path(data_path)
    model.eval()
    
    # 自动确定批量大小
    if batch_size is None:
        # 创建样本输入用于测试
        with h5py.File(data_path, 'r') as f:
            sample_data = f['data'][0:1]  # 获取第一个样本
            sample_input = torch.from_numpy(sample_data).float().to(device)
            batch_size = auto_batch_size(model, sample_input)
    
    logger = logging.getLogger('MemoryEfficientInference')
    logger.info(f"Using batch size: {batch_size}")
    
    results = []
    
    # 分块处理数据
    with h5py.File(data_path, 'r') as f:
        total_samples = f['data'].shape[0]
        num_chunks = (total_samples + chunk_size - 1) // chunk_size
        
        chunk_iterator = range(num_chunks)
        if show_progress:
            chunk_iterator = tqdm(chunk_iterator, desc="Processing chunks")
        
        for chunk_idx in chunk_iterator:
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_samples)
            
            # 加载数据块
            chunk_data = f['data'][start_idx:end_idx]
            chunk_tensor = torch.from_numpy(chunk_data).float().to(device)
            
            # 批量处理数据块
            chunk_results = _process_chunk(
                model, chunk_tensor, batch_size,
                post_processor, result_parser,
                enable_amp, start_idx
            )
            
            results.extend(chunk_results)
            
            # 内存清理
            del chunk_tensor
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
    
    return results


def _process_chunk(model: nn.Module,
                  chunk_tensor: torch.Tensor,
                  batch_size: int,
                  post_processor: Optional[Callable],
                  result_parser: Optional[Callable],
                  enable_amp: bool,
                  start_idx: int) -> List[Dict[str, Any]]:
    """处理数据块"""
    chunk_size = chunk_tensor.size(0)
    results = []
    
    for i in range(0, chunk_size, batch_size):
        end_idx = min(i + batch_size, chunk_size)
        batch_input = chunk_tensor[i:end_idx]
        
        # 推理
        with torch.no_grad():
            if enable_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    batch_output = model(batch_input)
            else:
                batch_output = model(batch_input)
        
        # 处理批量输出
        for j in range(batch_output.size(0)):
            sample_output = batch_output[j]
            
            # 后处理
            if post_processor is not None:
                processed_output = post_processor(sample_output)
            else:
                processed_output = {'raw_output': sample_output}
            
            # 结果解析
            if result_parser is not None:
                parsed_result = result_parser(processed_output)
            else:
                parsed_result = processed_output
            
            # 添加索引信息
            parsed_result['global_index'] = start_idx + i + j
            
            results.append(parsed_result)
    
    return results


def estimate_memory_usage(model: nn.Module,
                         input_shape: Tuple[int, ...],
                         batch_size: int = 1,
                         device: Optional[torch.device] = None) -> Dict[str, float]:
    """估计内存使用量
    
    Args:
        model: 神经网络模型
        input_shape: 输入形状（不包括批量维度）
        batch_size: 批量大小
        device: 设备
        
    Returns:
        内存使用量估计（GB）
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # 清理内存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    # 记录初始内存
    if device.type == 'cuda':
        initial_memory = torch.cuda.memory_allocated(device)
    else:
        initial_memory = psutil.virtual_memory().used
    
    try:
        # 创建测试输入
        test_input = torch.randn(batch_size, *input_shape, device=device)
        
        # 记录输入后内存
        if device.type == 'cuda':
            input_memory = torch.cuda.memory_allocated(device)
        else:
            input_memory = psutil.virtual_memory().used
        
        # 前向传播
        with torch.no_grad():
            output = model(test_input)
        
        # 记录输出后内存
        if device.type == 'cuda':
            output_memory = torch.cuda.memory_allocated(device)
        else:
            output_memory = psutil.virtual_memory().used
        
        # 计算内存使用量
        input_size_gb = (input_memory - initial_memory) / 1024**3
        forward_size_gb = (output_memory - input_memory) / 1024**3
        total_size_gb = (output_memory - initial_memory) / 1024**3
        
        # 清理
        del test_input, output
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'input_memory_gb': input_size_gb,
            'forward_memory_gb': forward_size_gb,
            'total_memory_gb': total_size_gb,
            'per_sample_gb': total_size_gb / batch_size
        }
        
    except Exception as e:
        # 清理
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'error': str(e),
            'input_memory_gb': 0.0,
            'forward_memory_gb': 0.0,
            'total_memory_gb': 0.0,
            'per_sample_gb': 0.0
        }


def benchmark_inference(model: nn.Module,
                       input_shape: Tuple[int, ...],
                       batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
                       num_iterations: int = 10,
                       warmup_iterations: int = 3,
                       device: Optional[torch.device] = None) -> Dict[int, Dict[str, float]]:
    """推理性能基准测试
    
    Args:
        model: 神经网络模型
        input_shape: 输入形状（不包括批量维度）
        batch_sizes: 要测试的批量大小列表
        num_iterations: 测试迭代次数
        warmup_iterations: 预热迭代次数
        device: 设备
        
    Returns:
        性能基准结果
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    results = {}
    
    for batch_size in batch_sizes:
        try:
            # 创建测试输入
            test_input = torch.randn(batch_size, *input_shape, device=device)
            
            # 预热
            for _ in range(warmup_iterations):
                with torch.no_grad():
                    _ = model(test_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # 基准测试
            times = []
            for _ in range(num_iterations):
                start_time = time.time()
                
                with torch.no_grad():
                    output = model(test_input)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            # 计算统计信息
            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = batch_size / avg_time
            
            # 内存使用量
            memory_info = estimate_memory_usage(model, input_shape, batch_size, device)
            
            results[batch_size] = {
                'avg_time_s': avg_time,
                'std_time_s': std_time,
                'throughput_samples_per_s': throughput,
                'memory_gb': memory_info['total_memory_gb'],
                'per_sample_time_ms': (avg_time / batch_size) * 1000
            }
            
            # 清理
            del test_input, output
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            results[batch_size] = {
                'error': str(e),
                'avg_time_s': float('inf'),
                'std_time_s': 0.0,
                'throughput_samples_per_s': 0.0,
                'memory_gb': float('inf'),
                'per_sample_time_ms': float('inf')
            }
            
            # 清理
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
    
    return results


def optimize_inference_config(model: nn.Module,
                             input_shape: Tuple[int, ...],
                             max_memory_gb: float = 4.0,
                             target_throughput: Optional[float] = None,
                             device: Optional[torch.device] = None) -> Dict[str, Any]:
    """优化推理配置
    
    Args:
        model: 神经网络模型
        input_shape: 输入形状
        max_memory_gb: 最大内存限制
        target_throughput: 目标吞吐量（samples/s）
        device: 设备
        
    Returns:
        优化的配置
    """
    if device is None:
        device = next(model.parameters()).device
    
    # 基准测试
    benchmark_results = benchmark_inference(model, input_shape, device=device)
    
    # 过滤可行的配置
    valid_configs = {}
    for batch_size, result in benchmark_results.items():
        if ('error' not in result and 
            result['memory_gb'] <= max_memory_gb and
            result['avg_time_s'] < float('inf')):
            valid_configs[batch_size] = result
    
    if not valid_configs:
        return {
            'optimal_batch_size': 1,
            'expected_throughput': 0.0,
            'expected_memory_gb': 0.0,
            'error': 'No valid configuration found'
        }
    
    # 选择最优配置
    if target_throughput is not None:
        # 选择满足目标吞吐量的最小批量大小
        best_config = None
        for batch_size in sorted(valid_configs.keys()):
            result = valid_configs[batch_size]
            if result['throughput_samples_per_s'] >= target_throughput:
                best_config = (batch_size, result)
                break
        
        if best_config is None:
            # 选择最高吞吐量的配置
            best_batch_size = max(valid_configs.keys(), 
                                key=lambda x: valid_configs[x]['throughput_samples_per_s'])
            best_config = (best_batch_size, valid_configs[best_batch_size])
    else:
        # 选择最高吞吐量的配置
        best_batch_size = max(valid_configs.keys(), 
                            key=lambda x: valid_configs[x]['throughput_samples_per_s'])
        best_config = (best_batch_size, valid_configs[best_batch_size])
    
    batch_size, result = best_config
    
    return {
        'optimal_batch_size': batch_size,
        'expected_throughput': result['throughput_samples_per_s'],
        'expected_memory_gb': result['memory_gb'],
        'expected_latency_ms': result['per_sample_time_ms'],
        'all_results': benchmark_results
    }


class InferenceProfiler:
    """推理性能分析器
    
    用于分析和监控推理性能。
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置统计信息"""
        self.inference_times = []
        self.memory_usage = []
        self.batch_sizes = []
        self.start_time = None
        self.total_samples = 0
    
    def start_profiling(self):
        """开始性能分析"""
        self.start_time = time.time()
    
    def record_inference(self, inference_time: float, batch_size: int, memory_gb: float = 0.0):
        """记录推理信息"""
        self.inference_times.append(inference_time)
        self.batch_sizes.append(batch_size)
        self.memory_usage.append(memory_gb)
        self.total_samples += batch_size
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.inference_times:
            return {}
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'total_samples': self.total_samples,
            'total_time_s': total_time,
            'avg_inference_time_s': np.mean(self.inference_times),
            'std_inference_time_s': np.std(self.inference_times),
            'min_inference_time_s': np.min(self.inference_times),
            'max_inference_time_s': np.max(self.inference_times),
            'avg_throughput_samples_per_s': self.total_samples / total_time if total_time > 0 else 0,
            'avg_batch_size': np.mean(self.batch_sizes),
            'avg_memory_gb': np.mean(self.memory_usage) if self.memory_usage else 0,
            'max_memory_gb': np.max(self.memory_usage) if self.memory_usage else 0
        }
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        
        if not stats:
            print("No profiling data available")
            return
        
        print("\n=== Inference Profiling Results ===")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Total time: {stats['total_time_s']:.2f}s")
        print(f"Average throughput: {stats['avg_throughput_samples_per_s']:.2f} samples/s")
        print(f"Average inference time: {stats['avg_inference_time_s']*1000:.2f}ms")
        print(f"Inference time std: {stats['std_inference_time_s']*1000:.2f}ms")
        print(f"Min/Max inference time: {stats['min_inference_time_s']*1000:.2f}/{stats['max_inference_time_s']*1000:.2f}ms")
        print(f"Average batch size: {stats['avg_batch_size']:.1f}")
        
        if stats['avg_memory_gb'] > 0:
            print(f"Average memory usage: {stats['avg_memory_gb']:.2f}GB")
            print(f"Max memory usage: {stats['max_memory_gb']:.2f}GB")
        
        print("=" * 35)