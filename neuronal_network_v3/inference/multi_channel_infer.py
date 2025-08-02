"""多通道推理系统

实现双通道联合推理，包括物理约束应用和不确定性量化。
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union, List
import numpy as np

from .infer import Infer
from ..models.ratio_net import RatioNet, FeatureExtractor


class MultiChannelInfer:
    """多通道推理器
    
    支持双通道联合推理，包括：
    - 独立通道预测
    - 比例预测（含不确定性量化）
    - 物理约束应用
    - 结果融合
    
    Args:
        model_ch1: 通道1的模型
        model_ch2: 通道2的模型
        ratio_net: 比例预测网络
        feature_extractor: 特征提取器
        device: 计算设备
        apply_constraints: 是否应用物理约束
    """
    
    def __init__(self, 
                 model_ch1: nn.Module, 
                 model_ch2: nn.Module, 
                 ratio_net: RatioNet,
                 feature_extractor: Optional[FeatureExtractor] = None,
                 device: str = 'cuda',
                 apply_constraints: bool = True):
        
        self.model_ch1 = model_ch1.to(device)
        self.model_ch2 = model_ch2.to(device)
        self.ratio_net = ratio_net.to(device)
        
        if feature_extractor is None:
            feature_extractor = FeatureExtractor()
        self.feature_extractor = feature_extractor.to(device)
        
        self.device = device
        self.apply_constraints = apply_constraints
        
        # 设置为评估模式
        self.model_ch1.eval()
        self.model_ch2.eval()
        self.ratio_net.eval()
        self.feature_extractor.eval()
        
        # 创建单通道推理器
        self.infer_ch1 = Infer(self.model_ch1, device=device)
        self.infer_ch2 = Infer(self.model_ch2, device=device)
    
    def forward(self, input_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向推理
        
        Args:
            input_data: 输入数据，形状为 (B, C, H, W)
            
        Returns:
            包含所有预测结果的字典
        """
        with torch.no_grad():
            # 分别进行通道预测
            pred_ch1 = self.model_ch1(input_data)
            pred_ch2 = self.model_ch2(input_data)
            
            # 提取特征用于比例预测
            ch1_features = self.feature_extractor(pred_ch1)
            ch2_features = self.feature_extractor(pred_ch2)
            
            # 预测比例（均值和不确定性）
            ratio_result = self.ratio_net.predict_with_uncertainty(ch1_features, ch2_features)
            
            # 应用物理约束（如果启用）
            if self.apply_constraints:
                final_pred = self._apply_constraints(pred_ch1, pred_ch2, ratio_result)
            else:
                final_pred = {
                    'channel1': pred_ch1,
                    'channel2': pred_ch2,
                    **ratio_result
                }
            
            return final_pred
    
    def _apply_constraints(self, 
                          pred_ch1: torch.Tensor, 
                          pred_ch2: torch.Tensor, 
                          ratio_result: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """应用物理约束
        
        Args:
            pred_ch1: 通道1预测结果
            pred_ch2: 通道2预测结果
            ratio_result: 比例预测结果
            
        Returns:
            应用约束后的预测结果
        """
        ratio_mean = ratio_result['mean']
        
        # 提取光子数预测（假设在第1个通道）
        if pred_ch1.shape[1] > 1 and pred_ch2.shape[1] > 1:
            photons_ch1 = pred_ch1[:, 1]  
            photons_ch2 = pred_ch2[:, 1]
            
            # 计算总光子数
            total_photons = photons_ch1 + photons_ch2
            
            # 根据比例均值重新分配
            corrected_ch1 = total_photons * ratio_mean.squeeze()
            corrected_ch2 = total_photons * (1 - ratio_mean.squeeze())
            
            # 更新预测结果
            final_pred_ch1 = pred_ch1.clone()
            final_pred_ch2 = pred_ch2.clone()
            final_pred_ch1[:, 1] = corrected_ch1
            final_pred_ch2[:, 1] = corrected_ch2
        else:
            final_pred_ch1 = pred_ch1
            final_pred_ch2 = pred_ch2
            total_photons = None
        
        result = {
            'channel1': final_pred_ch1,
            'channel2': final_pred_ch2,
            'total_photons': total_photons,
            **ratio_result
        }
        
        return result
    
    def predict_batch(self, 
                     data_loader, 
                     return_numpy: bool = True) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """批量预测
        
        Args:
            data_loader: 数据加载器
            return_numpy: 是否返回numpy数组
            
        Returns:
            批量预测结果
        """
        all_results = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    input_data = batch[0].to(self.device)
                else:
                    input_data = batch.to(self.device)
                
                result = self.forward(input_data)
                all_results.append(result)
        
        # 合并所有批次的结果
        merged_results = {}
        for key in all_results[0].keys():
            if isinstance(all_results[0][key], torch.Tensor):
                merged_results[key] = torch.cat([r[key] for r in all_results], dim=0)
                if return_numpy:
                    merged_results[key] = merged_results[key].cpu().numpy()
            elif isinstance(all_results[0][key], tuple):
                # 处理置信区间等元组数据
                lower_bounds = torch.cat([r[key][0] for r in all_results], dim=0)
                upper_bounds = torch.cat([r[key][1] for r in all_results], dim=0)
                if return_numpy:
                    merged_results[key] = (lower_bounds.cpu().numpy(), upper_bounds.cpu().numpy())
                else:
                    merged_results[key] = (lower_bounds, upper_bounds)
        
        return merged_results
    
    def save_models(self, save_dir: str):
        """保存所有模型
        
        Args:
            save_dir: 保存目录
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.model_ch1.state_dict(), os.path.join(save_dir, 'model_ch1.pth'))
        torch.save(self.model_ch2.state_dict(), os.path.join(save_dir, 'model_ch2.pth'))
        torch.save(self.ratio_net.state_dict(), os.path.join(save_dir, 'ratio_net.pth'))
        torch.save(self.feature_extractor.state_dict(), os.path.join(save_dir, 'feature_extractor.pth'))
    
    def load_models(self, save_dir: str):
        """加载所有模型
        
        Args:
            save_dir: 模型保存目录
        """
        import os
        
        self.model_ch1.load_state_dict(torch.load(os.path.join(save_dir, 'model_ch1.pth'), map_location=self.device))
        self.model_ch2.load_state_dict(torch.load(os.path.join(save_dir, 'model_ch2.pth'), map_location=self.device))
        self.ratio_net.load_state_dict(torch.load(os.path.join(save_dir, 'ratio_net.pth'), map_location=self.device))
        self.feature_extractor.load_state_dict(torch.load(os.path.join(save_dir, 'feature_extractor.pth'), map_location=self.device))


class MultiChannelBatchInfer:
    """多通道批量推理器
    
    优化的批量推理实现，支持大规模数据处理。
    """
    
    def __init__(self, multi_channel_infer: MultiChannelInfer, batch_size: Union[int, str] = 'auto'):
        self.multi_channel_infer = multi_channel_infer
        self.batch_size = batch_size
    
    def predict_dataset(self, 
                       dataset, 
                       num_workers: int = 4,
                       return_numpy: bool = True) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """预测整个数据集
        
        Args:
            dataset: 数据集
            num_workers: 数据加载器工作进程数
            return_numpy: 是否返回numpy数组
            
        Returns:
            数据集预测结果
        """
        from torch.utils.data import DataLoader
        
        # 确定批大小
        if self.batch_size == 'auto':
            batch_size = self._determine_batch_size(dataset[0] if hasattr(dataset, '__getitem__') else next(iter(dataset)))
        else:
            batch_size = self.batch_size
        
        # 创建数据加载器
        data_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True if self.multi_channel_infer.device != 'cpu' else False
        )
        
        return self.multi_channel_infer.predict_batch(data_loader, return_numpy)
    
    def _determine_batch_size(self, sample_data) -> int:
        """自动确定批大小
        
        Args:
            sample_data: 样本数据
            
        Returns:
            推荐的批大小
        """
        # 简单的启发式方法
        if isinstance(sample_data, torch.Tensor):
            # 基于数据大小估算
            data_size = sample_data.numel() * 4  # 假设float32
            if data_size < 1e6:  # < 1MB
                return 32
            elif data_size < 1e7:  # < 10MB
                return 16
            else:
                return 8
        else:
            return 16  # 默认值