"""批量推理器实现

包含标准推理器和批量推理器，支持高效的模型推理。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
import time
import logging
from tqdm import tqdm
import gc

from ..models import SigmaMUNet, DoubleMUnet, SimpleSMLMNet
from .post_processing import PostProcessor
from .result_parser import ResultParser
from .utils import auto_batch_size, memory_efficient_inference


class Infer:
    """DECODE推理器
    
    支持单张图像和批量图像的推理，具有以下特性：
    - 自动批量大小调整
    - 内存高效推理
    - 多种输出格式
    - 后处理集成
    - 结果解析
    
    Args:
        model: 训练好的神经网络模型
        device: 推理设备
        post_processor: 后处理器
        result_parser: 结果解析器
        auto_batch: 是否自动调整批量大小
        max_memory_gb: 最大内存使用量（GB）
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: Optional[torch.device] = None,
                 post_processor: Optional[PostProcessor] = None,
                 result_parser: Optional[ResultParser] = None,
                 auto_batch: bool = True,
                 max_memory_gb: float = 4.0,
                 logger: Optional[logging.Logger] = None):
        
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.post_processor = post_processor
        self.result_parser = result_parser
        self.auto_batch = auto_batch
        self.max_memory_gb = max_memory_gb
        self.logger = logger or self._setup_logger()
        
        # 移动模型到设备并设置为评估模式
        self.model.to(self.device)
        self.model.eval()
        
        # 推理统计
        self.inference_stats = {
            'total_samples': 0,
            'total_time': 0.0,
            'avg_time_per_sample': 0.0,
            'memory_usage': []
        }
        
        # 自动确定最佳批量大小
        self.optimal_batch_size = None
        if self.auto_batch:
            self._find_optimal_batch_size()


    def infer_single(self,
                    input_data: Union[torch.Tensor, np.ndarray],
                    return_raw: bool = False,
                    apply_post_processing: bool = True) -> Dict[str, Any]:
        """单张图像推理
        
        Args:
            input_data: 输入数据，形状为 (C, H, W) 或 (H, W)
            return_raw: 是否返回原始输出
            apply_post_processing: 是否应用后处理
            
        Returns:
            推理结果字典
        """
        # 预处理输入
        input_tensor = self._preprocess_input(input_data)
        input_tensor = input_tensor.unsqueeze(0)  # 添加批量维度
        
        # 推理
        start_time = time.time()
        with torch.no_grad():
            raw_output = self.model(input_tensor)
        inference_time = time.time() - start_time
        
        # 移除批量维度
        if isinstance(raw_output, torch.Tensor):
            raw_output = raw_output.squeeze(0)
        elif isinstance(raw_output, (list, tuple)):
            raw_output = [out.squeeze(0) if isinstance(out, torch.Tensor) else out for out in raw_output]
        
        # 构建结果
        result = {
            'inference_time': inference_time,
            'input_shape': input_tensor.shape[1:],  # 不包括批量维度
            'device': str(self.device)
        }
        
        if return_raw:
            result['raw_output'] = raw_output
        
        # 后处理
        if apply_post_processing and self.post_processor is not None:
            processed_output = self.post_processor(raw_output)
            result['processed_output'] = processed_output
        else:
            result['processed_output'] = raw_output
        
        # 结果解析
        if self.result_parser is not None:
            parsed_result = self.result_parser(result['processed_output'])
            result.update(parsed_result)
        
        # 更新统计
        self._update_stats(1, inference_time)
        
        return result
    
    def infer_batch(self,
                   input_data: Union[torch.Tensor, np.ndarray, List],
                   batch_size: Optional[int] = None,
                   return_raw: bool = False,
                   apply_post_processing: bool = True,
                   show_progress: bool = True) -> List[Dict[str, Any]]:
        """批量推理
        
        Args:
            input_data: 输入数据，形状为 (N, C, H, W) 或列表
            batch_size: 批量大小，如果为None则使用自动确定的大小
            return_raw: 是否返回原始输出
            apply_post_processing: 是否应用后处理
            show_progress: 是否显示进度条
            
        Returns:
            推理结果列表
        """
        # 预处理输入
        if isinstance(input_data, list):
            input_tensors = [self._preprocess_input(data) for data in input_data]
            input_tensor = torch.stack(input_tensors, dim=0)
        else:
            input_tensor = self._preprocess_input(input_data)
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)
        
        num_samples = input_tensor.size(0)
        
        # 确定批量大小
        if batch_size is None:
            batch_size = self.optimal_batch_size or 1
        
        self.logger.info(f"Starting batch inference: {num_samples} samples, batch_size={batch_size}")
        
        results = []
        total_time = 0.0
        
        # 批量处理
        iterator = range(0, num_samples, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Inference", unit="batch")
        
        for i in iterator:
            end_idx = min(i + batch_size, num_samples)
            batch_input = input_tensor[i:end_idx]
            
            # 推理
            start_time = time.time()
            with torch.no_grad():
                batch_output = self.model(batch_input)
            batch_time = time.time() - start_time
            total_time += batch_time
            
            # 处理批量输出
            batch_results = self._process_batch_output(
                batch_output, batch_input, batch_time,
                return_raw, apply_post_processing
            )
            results.extend(batch_results)
            
            # 内存清理
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        
        # 更新统计
        self._update_stats(num_samples, total_time)
        
        self.logger.info(f"Batch inference completed: {total_time:.2f}s, {total_time/num_samples:.4f}s/sample")
        
        return results
    
    def infer_dataloader(self,
                        dataloader: DataLoader,
                        return_raw: bool = False,
                        apply_post_processing: bool = True,
                        show_progress: bool = True) -> List[Dict[str, Any]]:
        """数据加载器推理
        
        Args:
            dataloader: 数据加载器
            return_raw: 是否返回原始输出
            apply_post_processing: 是否应用后处理
            show_progress: 是否显示进度条
            
        Returns:
            推理结果列表
        """
        self.logger.info(f"Starting dataloader inference: {len(dataloader)} batches")
        
        results = []
        total_time = 0.0
        total_samples = 0
        
        iterator = dataloader
        if show_progress:
            iterator = tqdm(dataloader, desc="Inference", unit="batch")
        
        for batch_idx, batch in enumerate(iterator):
            # 解析批量数据
            if isinstance(batch, dict):
                batch_input = batch['input'].to(self.device)
                batch_metadata = {k: v for k, v in batch.items() if k != 'input'}
            elif isinstance(batch, (list, tuple)):
                batch_input = batch[0].to(self.device)
                batch_metadata = batch[1] if len(batch) > 1 else {}
            else:
                batch_input = batch.to(self.device)
                batch_metadata = {}
            
            batch_size = batch_input.size(0)
            
            # 推理
            start_time = time.time()
            with torch.no_grad():
                batch_output = self.model(batch_input)
            batch_time = time.time() - start_time
            total_time += batch_time
            total_samples += batch_size
            
            # 处理批量输出
            batch_results = self._process_batch_output(
                batch_output, batch_input, batch_time,
                return_raw, apply_post_processing
            )
            
            # 添加元数据
            for i, result in enumerate(batch_results):
                for key, value in batch_metadata.items():
                    if isinstance(value, (list, tuple)):
                        result[key] = value[i] if i < len(value) else None
                    elif isinstance(value, torch.Tensor) and value.dim() > 0:
                        result[key] = value[i] if i < value.size(0) else None
                    else:
                        result[key] = value
            
            results.extend(batch_results)
            
            # 内存清理
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        
        # 更新统计
        self._update_stats(total_samples, total_time)
        
        self.logger.info(f"Dataloader inference completed: {total_time:.2f}s, {total_time/total_samples:.4f}s/sample")
        
        return results
    
    def _preprocess_input(self, input_data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """预处理输入数据"""
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data.astype(np.float32))
        elif isinstance(input_data, torch.Tensor):
            input_tensor = input_data.float()
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        # 确保输入在正确设备上
        input_tensor = input_tensor.to(self.device)
        
        # 确保输入是3D: (C, H, W)
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(0)  # (1, H, W)
        elif input_tensor.dim() == 4:
            # 如果已经是4D，假设是 (1, C, H, W)
            input_tensor = input_tensor.squeeze(0)
        
        return input_tensor
    
    def _process_batch_output(self,
                             batch_output: Union[torch.Tensor, List, Tuple],
                             batch_input: torch.Tensor,
                             batch_time: float,
                             return_raw: bool,
                             apply_post_processing: bool) -> List[Dict[str, Any]]:
        """处理批量输出"""
        batch_size = batch_input.size(0)
        results = []
        
        for i in range(batch_size):
            # 提取单个样本的输出
            if isinstance(batch_output, torch.Tensor):
                sample_output = batch_output[i]
            elif isinstance(batch_output, (list, tuple)):
                sample_output = [out[i] if isinstance(out, torch.Tensor) else out for out in batch_output]
            else:
                sample_output = batch_output
            
            # 构建结果
            result = {
                'inference_time': batch_time / batch_size,
                'input_shape': batch_input[i].shape,
                'device': str(self.device)
            }
            
            if return_raw:
                result['raw_output'] = sample_output
            
            # 后处理
            if apply_post_processing and self.post_processor is not None:
                processed_output = self.post_processor(sample_output)
                result['processed_output'] = processed_output
            else:
                result['processed_output'] = sample_output
            
            # 结果解析
            if self.result_parser is not None:
                parsed_result = self.result_parser(result['processed_output'])
                result.update(parsed_result)
            
            results.append(result)
        
        return results
    
    def _find_optimal_batch_size(self):
        """自动确定最佳批量大小"""
        try:
            # 创建测试输入
            test_input = torch.randn(1, 3, 64, 64, device=self.device)
            
            # 使用自动批量大小工具
            self.optimal_batch_size = auto_batch_size(
                self.model, test_input, self.max_memory_gb
            )
            
            self.logger.info(f"Optimal batch size determined: {self.optimal_batch_size}")
            
        except Exception as e:
            self.logger.warning(f"Failed to determine optimal batch size: {e}")
            self.optimal_batch_size = 1
    
    def _update_stats(self, num_samples: int, inference_time: float):
        """更新推理统计"""
        self.inference_stats['total_samples'] += num_samples
        self.inference_stats['total_time'] += inference_time
        self.inference_stats['avg_time_per_sample'] = (
            self.inference_stats['total_time'] / self.inference_stats['total_samples']
        )
        
        # 记录内存使用
        if self.device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            self.inference_stats['memory_usage'].append(memory_used)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取推理统计信息"""
        stats = self.inference_stats.copy()
        
        if stats['memory_usage']:
            stats['avg_memory_usage'] = np.mean(stats['memory_usage'])
            stats['max_memory_usage'] = np.max(stats['memory_usage'])
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.inference_stats = {
            'total_samples': 0,
            'total_time': 0.0,
            'avg_time_per_sample': 0.0,
            'memory_usage': []
        }
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('DECODE_Infer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger


class BatchInfer(Infer):
    """批量推理器
    
    专门用于大规模批量推理的优化版本
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: Optional[torch.device] = None,
                 post_processor: Optional[PostProcessor] = None,
                 result_parser: Optional[ResultParser] = None,
                 max_memory_gb: float = 8.0,
                 enable_amp: bool = True,
                 logger: Optional[logging.Logger] = None):
        
        super().__init__(model, device, post_processor, result_parser, 
                        auto_batch=True, max_memory_gb=max_memory_gb, logger=logger)
        
        self.enable_amp = enable_amp and self.device.type == 'cuda'
        
        # 混合精度推理
        if self.enable_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def infer_large_dataset(self,
                           data_path: Union[str, Path],
                           output_path: Optional[Union[str, Path]] = None,
                           chunk_size: int = 1000,
                           save_interval: int = 100,
                           **kwargs) -> List[Dict[str, Any]]:
        """大数据集推理
        
        Args:
            data_path: 数据文件路径
            output_path: 输出路径
            chunk_size: 块大小
            save_interval: 保存间隔
            
        Returns:
            推理结果列表
        """
        self.logger.info(f"Starting large dataset inference: {data_path}")
        
        # 使用内存高效推理
        results = memory_efficient_inference(
            self.model, data_path, self.device,
            chunk_size=chunk_size,
            post_processor=self.post_processor,
            result_parser=self.result_parser,
            enable_amp=self.enable_amp,
            **kwargs
        )
        
        # 保存结果
        if output_path is not None:
            self._save_results(results, output_path)
        
        return results
    
    def infer_with_amp(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """混合精度推理"""
        if self.enable_amp:
            with torch.cuda.amp.autocast():
                return self.model(input_tensor)
        else:
            return self.model(input_tensor)
    
    def _save_results(self, results: List[Dict[str, Any]], output_path: Union[str, Path]):
        """保存推理结果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 根据文件扩展名选择保存格式
        if output_path.suffix == '.npz':
            # 保存为NumPy格式
            save_dict = {}
            for i, result in enumerate(results):
                for key, value in result.items():
                    if isinstance(value, torch.Tensor):
                        save_dict[f'{key}_{i}'] = value.cpu().numpy()
                    elif isinstance(value, np.ndarray):
                        save_dict[f'{key}_{i}'] = value
            
            np.savez_compressed(output_path, **save_dict)
            
        elif output_path.suffix == '.pt':
            # 保存为PyTorch格式
            torch.save(results, output_path)
            
        else:
            # 保存为pickle格式
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
        
        self.logger.info(f"Results saved to {output_path}")


# 为了向后兼容，添加ModelInfer别名
ModelInfer = Infer