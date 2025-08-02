"""多通道评估系统

实现多通道模型的性能评估，包括单通道评估、比例预测评估和物理约束评估。
支持不确定性量化的评估指标。
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from scipy import stats

from .evaluator import ModelEvaluator
from .metrics import SegmentationMetrics, RegressionMetrics


class MultiChannelEvaluation:
    """多通道模型评估器
    
    提供全面的多通道模型性能评估，包括：
    - 单通道性能评估
    - 比例预测评估（含不确定性量化）
    - 物理约束评估
    - 可视化分析
    
    Args:
        device: 计算设备
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        # 创建一个虚拟模型用于ModelEvaluator
        import torch.nn as nn
        dummy_model = nn.Identity()
        self.single_channel_eval = ModelEvaluator(model=dummy_model, device=device)
        self.segmentation_metrics = SegmentationMetrics()
        self.regression_metrics = RegressionMetrics()
    
    def evaluate(self, 
                pred_results: Dict[str, torch.Tensor], 
                ground_truth: Dict[str, torch.Tensor]) -> Dict[str, Union[float, Dict]]:
        """全面评估多通道模型性能
        
        Args:
            pred_results: 预测结果字典
            ground_truth: 真实值字典
            
        Returns:
            评估指标字典
        """
        metrics = {}
        
        # 单通道评估
        if 'channel1' in pred_results and 'channel1' in ground_truth:
            metrics['channel1'] = self._evaluate_single_channel(
                pred_results['channel1'], ground_truth['channel1']
            )
        
        if 'channel2' in pred_results and 'channel2' in ground_truth:
            metrics['channel2'] = self._evaluate_single_channel(
                pred_results['channel2'], ground_truth['channel2']
            )
        
        # 比例预测评估（包含不确定性量化）
        if all(key in pred_results for key in ['mean', 'std']) and 'ratio' in ground_truth:
            metrics['ratio'] = self._evaluate_ratio_prediction(
                pred_results['mean'], pred_results['std'], ground_truth['ratio']
            )
        
        # 物理约束评估
        if 'total_photons' in pred_results and 'total_photons' in ground_truth:
            metrics['conservation'] = self._evaluate_conservation(
                pred_results, ground_truth
            )
        
        # 整体性能指标
        metrics['overall'] = self._compute_overall_metrics(metrics)
        
        return metrics
    
    def _evaluate_single_channel(self, 
                               pred: torch.Tensor, 
                               target: torch.Tensor) -> Dict[str, float]:
        """评估单通道性能
        
        Args:
            pred: 预测结果
            target: 真实值
            
        Returns:
            单通道评估指标
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        metrics = {}
        
        # 检测性能（假设第0通道是检测概率）
        if pred.shape[1] > 0 and target.shape[1] > 0:
            # 计算分割指标
            pred_binary = (pred[:, 0] > 0.5).astype(float)
            target_binary = (target[:, 0] > 0.5).astype(float)
            
            intersection = np.sum(pred_binary * target_binary)
            union = np.sum(pred_binary) + np.sum(target_binary) - intersection
            iou = intersection / union if union > 0 else 0.0
            dice = 2 * intersection / (np.sum(pred_binary) + np.sum(target_binary)) if (np.sum(pred_binary) + np.sum(target_binary)) > 0 else 0.0
            pixel_accuracy = np.mean(pred_binary == target_binary)
            
            metrics['detection'] = {
                'iou': iou,
                'dice': dice,
                'pixel_accuracy': pixel_accuracy,
                'mean_iou': iou
            }
        
        # 回归性能（其他通道）
        if pred.shape[1] > 1 and target.shape[1] > 1:
            # 计算回归指标
            pred_reg = pred[:, 1:].flatten()
            target_reg = target[:, 1:].flatten()
            
            mse = np.mean((pred_reg - target_reg) ** 2)
            mae = np.mean(np.abs(pred_reg - target_reg))
            rmse = np.sqrt(mse)
            
            # 计算R²分数
            ss_res = np.sum((target_reg - pred_reg) ** 2)
            ss_tot = np.sum((target_reg - np.mean(target_reg)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            metrics['regression'] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2_score
            }
        
        return metrics
    
    def _evaluate_ratio_prediction(self, 
                                 pred_ratio_mean: torch.Tensor, 
                                 pred_ratio_std: torch.Tensor, 
                                 true_ratio: torch.Tensor) -> Dict[str, float]:
        """评估比例预测性能（包含不确定性量化）
        
        Args:
            pred_ratio_mean: 预测比例均值
            pred_ratio_std: 预测比例标准差
            true_ratio: 真实比例
            
        Returns:
            比例预测评估指标
        """
        if isinstance(pred_ratio_mean, torch.Tensor):
            pred_ratio_mean = pred_ratio_mean.cpu().numpy().flatten()
        if isinstance(pred_ratio_std, torch.Tensor):
            pred_ratio_std = pred_ratio_std.cpu().numpy().flatten()
        if isinstance(true_ratio, torch.Tensor):
            true_ratio = true_ratio.cpu().numpy().flatten()
        
        # 基础误差指标
        abs_error = np.abs(pred_ratio_mean - true_ratio)
        rel_error = abs_error / (true_ratio + 1e-8)
        
        # 不确定性量化评估
        # 1. 校准性评估（预测不确定性与实际误差的相关性）
        calibration_error = np.abs(abs_error - pred_ratio_std)
        
        # 2. 置信区间覆盖率（95%置信区间）
        z_score = 1.96  # 95%置信区间
        lower_bound = pred_ratio_mean - z_score * pred_ratio_std
        upper_bound = pred_ratio_mean + z_score * pred_ratio_std
        coverage = np.mean((true_ratio >= lower_bound) & (true_ratio <= upper_bound))
        
        # 3. 不确定性质量指标
        # 负对数似然（越小越好）
        nll = 0.5 * (np.log(2 * np.pi * pred_ratio_std**2) + 
                     (pred_ratio_mean - true_ratio)**2 / pred_ratio_std**2)
        
        # 4. 分布分析
        ratio_bins = np.linspace(0, 1, 11)
        binned_errors = []
        binned_uncertainties = []
        
        for i in range(len(ratio_bins) - 1):
            mask = (true_ratio >= ratio_bins[i]) & (true_ratio < ratio_bins[i+1])
            if np.sum(mask) > 0:
                binned_errors.append(np.mean(abs_error[mask]))
                binned_uncertainties.append(np.mean(pred_ratio_std[mask]))
        
        # 5. 不确定性与误差的相关性
        if len(pred_ratio_std) > 1:
            uncertainty_error_correlation = np.corrcoef(pred_ratio_std, abs_error)[0, 1]
        else:
            uncertainty_error_correlation = 0.0
        
        return {
            # 基础预测指标
            'mae': np.mean(abs_error),
            'mse': np.mean(abs_error ** 2),
            'mape': np.mean(rel_error * 100),
            'rmse': np.sqrt(np.mean(abs_error ** 2)),
            
            # 不确定性量化指标
            'mean_uncertainty': np.mean(pred_ratio_std),
            'calibration_error': np.mean(calibration_error),
            'coverage_95': coverage,
            'nll': np.mean(nll),
            
            # 分布分析
            'binned_errors': binned_errors,
            'binned_uncertainties': binned_uncertainties,
            
            # 不确定性与误差的相关性
            'uncertainty_error_correlation': uncertainty_error_correlation
        }
    
    def _evaluate_conservation(self, 
                             pred_results: Dict[str, torch.Tensor], 
                             ground_truth: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """评估物理约束保持情况
        
        Args:
            pred_results: 预测结果
            ground_truth: 真实值
            
        Returns:
            物理约束评估指标
        """
        metrics = {}
        
        # 光子数守恒检查
        if all(key in pred_results for key in ['channel1', 'channel2']) and 'total_photons' in ground_truth:
            pred_ch1_photons = pred_results['channel1'][:, 1] if pred_results['channel1'].shape[1] > 1 else None
            pred_ch2_photons = pred_results['channel2'][:, 1] if pred_results['channel2'].shape[1] > 1 else None
            
            if pred_ch1_photons is not None and pred_ch2_photons is not None:
                pred_total = pred_ch1_photons + pred_ch2_photons
                true_total = ground_truth['total_photons']
                
                if isinstance(pred_total, torch.Tensor):
                    pred_total = pred_total.cpu().numpy()
                if isinstance(true_total, torch.Tensor):
                    true_total = true_total.cpu().numpy()
                
                conservation_error = np.abs(pred_total - true_total) / (true_total + 1e-8)
                metrics['conservation_error'] = np.mean(conservation_error)
        
        # 比例一致性检查
        if all(key in pred_results for key in ['channel1', 'channel2', 'mean']):
            pred_ch1_photons = pred_results['channel1'][:, 1] if pred_results['channel1'].shape[1] > 1 else None
            pred_ch2_photons = pred_results['channel2'][:, 1] if pred_results['channel2'].shape[1] > 1 else None
            pred_ratio_direct = pred_results['mean']
            
            if pred_ch1_photons is not None and pred_ch2_photons is not None:
                pred_total = pred_ch1_photons + pred_ch2_photons
                pred_ratio_from_photons = pred_ch1_photons / (pred_total + 1e-8)
                
                if isinstance(pred_ratio_from_photons, torch.Tensor):
                    pred_ratio_from_photons = pred_ratio_from_photons.cpu().numpy()
                if isinstance(pred_ratio_direct, torch.Tensor):
                    pred_ratio_direct = pred_ratio_direct.cpu().numpy().flatten()
                
                ratio_consistency = np.abs(pred_ratio_from_photons - pred_ratio_direct)
                metrics['ratio_consistency'] = np.mean(ratio_consistency)
        
        return metrics
    
    def _compute_overall_metrics(self, metrics: Dict) -> Dict[str, float]:
        """计算整体性能指标
        
        Args:
            metrics: 各组件的评估指标
            
        Returns:
            整体性能指标
        """
        overall = {}
        
        # 平均单通道性能
        channel_maes = []
        if 'channel1' in metrics and 'regression' in metrics['channel1']:
            channel_maes.append(metrics['channel1']['regression'].get('mae', 0))
        if 'channel2' in metrics and 'regression' in metrics['channel2']:
            channel_maes.append(metrics['channel2']['regression'].get('mae', 0))
        
        if channel_maes:
            overall['avg_channel_mae'] = np.mean(channel_maes)
        
        # 比例预测性能
        if 'ratio' in metrics:
            overall['ratio_mae'] = metrics['ratio']['mae']
            overall['ratio_coverage'] = metrics['ratio']['coverage_95']
            overall['ratio_nll'] = metrics['ratio']['nll']
        
        # 物理约束性能
        if 'conservation' in metrics:
            overall['conservation_error'] = metrics['conservation'].get('conservation_error', 0)
            overall['ratio_consistency'] = metrics['conservation'].get('ratio_consistency', 0)
        
        return overall
    
    def visualize_results(self, 
                         pred_results: Dict[str, torch.Tensor], 
                         ground_truth: Dict[str, torch.Tensor],
                         save_path: Optional[str] = None) -> plt.Figure:
        """可视化评估结果
        
        Args:
            pred_results: 预测结果
            ground_truth: 真实值
            save_path: 保存路径（可选）
            
        Returns:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 比例预测散点图
        if 'mean' in pred_results and 'ratio' in ground_truth:
            pred_mean = pred_results['mean'].cpu().numpy().flatten() if isinstance(pred_results['mean'], torch.Tensor) else pred_results['mean'].flatten()
            true_ratio = ground_truth['ratio'].cpu().numpy().flatten() if isinstance(ground_truth['ratio'], torch.Tensor) else ground_truth['ratio'].flatten()
            
            axes[0, 0].scatter(true_ratio, pred_mean, alpha=0.6)
            axes[0, 0].plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
            axes[0, 0].set_xlabel('True Ratio')
            axes[0, 0].set_ylabel('Predicted Ratio')
            axes[0, 0].set_title('Ratio Prediction')
            axes[0, 0].legend()
        
        # 不确定性vs误差
        if all(key in pred_results for key in ['mean', 'std']) and 'ratio' in ground_truth:
            pred_std = pred_results['std'].cpu().numpy().flatten() if isinstance(pred_results['std'], torch.Tensor) else pred_results['std'].flatten()
            abs_error = np.abs(pred_mean - true_ratio)
            
            axes[0, 1].scatter(pred_std, abs_error, alpha=0.6)
            axes[0, 1].set_xlabel('Predicted Uncertainty (std)')
            axes[0, 1].set_ylabel('Absolute Error')
            axes[0, 1].set_title('Uncertainty vs Error')
        
        # 误差分布
        if 'mean' in pred_results and 'ratio' in ground_truth:
            error = pred_mean - true_ratio
            axes[0, 2].hist(error, bins=50, alpha=0.7)
            axes[0, 2].set_xlabel('Prediction Error')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Error Distribution')
        
        # 光子数守恒
        if all(key in pred_results for key in ['channel1', 'channel2']) and 'total_photons' in ground_truth:
            pred_ch1 = pred_results['channel1'][:, 1] if pred_results['channel1'].shape[1] > 1 else None
            pred_ch2 = pred_results['channel2'][:, 1] if pred_results['channel2'].shape[1] > 1 else None
            
            if pred_ch1 is not None and pred_ch2 is not None:
                pred_total = (pred_ch1 + pred_ch2).cpu().numpy() if isinstance(pred_ch1, torch.Tensor) else pred_ch1 + pred_ch2
                true_total = ground_truth['total_photons'].cpu().numpy() if isinstance(ground_truth['total_photons'], torch.Tensor) else ground_truth['total_photons']
                
                axes[1, 0].scatter(true_total, pred_total, alpha=0.6)
                axes[1, 0].plot([true_total.min(), true_total.max()], [true_total.min(), true_total.max()], 'r--')
                axes[1, 0].set_xlabel('True Total Photons')
                axes[1, 0].set_ylabel('Predicted Total Photons')
                axes[1, 0].set_title('Photon Conservation')
        
        # 置信区间覆盖率分析
        if all(key in pred_results for key in ['mean', 'std']) and 'ratio' in ground_truth:
            # 按不确定性分组分析覆盖率
            uncertainty_bins = np.percentile(pred_std, [0, 25, 50, 75, 100])
            coverage_rates = []
            
            for i in range(len(uncertainty_bins) - 1):
                mask = (pred_std >= uncertainty_bins[i]) & (pred_std < uncertainty_bins[i+1])
                if np.sum(mask) > 0:
                    z_score = 1.96
                    lower = pred_mean[mask] - z_score * pred_std[mask]
                    upper = pred_mean[mask] + z_score * pred_std[mask]
                    coverage = np.mean((true_ratio[mask] >= lower) & (true_ratio[mask] <= upper))
                    coverage_rates.append(coverage)
            
            if coverage_rates:
                axes[1, 1].bar(range(len(coverage_rates)), coverage_rates)
                axes[1, 1].axhline(y=0.95, color='r', linestyle='--', label='Target 95%')
                axes[1, 1].set_xlabel('Uncertainty Quartile')
                axes[1, 1].set_ylabel('Coverage Rate')
                axes[1, 1].set_title('Coverage Rate by Uncertainty')
                axes[1, 1].legend()
        
        # 比例预测的分布分析
        if 'mean' in pred_results and 'ratio' in ground_truth:
            axes[1, 2].hist2d(true_ratio, pred_mean, bins=20, alpha=0.7)
            axes[1, 2].set_xlabel('True Ratio')
            axes[1, 2].set_ylabel('Predicted Ratio')
            axes[1, 2].set_title('Ratio Prediction Density')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class RatioEvaluationMetrics:
    """专门用于比例预测的评估指标"""
    
    @staticmethod
    def compute_calibration_metrics(pred_mean: np.ndarray, 
                                  pred_std: np.ndarray, 
                                  true_values: np.ndarray) -> Dict[str, float]:
        """计算校准性指标"""
        abs_error = np.abs(pred_mean - true_values)
        
        # Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # 找到在这个置信度区间内的样本
            in_bin = (pred_std > bin_lower) & (pred_std <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (abs_error[in_bin] <= pred_std[in_bin]).mean()
                avg_confidence_in_bin = pred_std[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return {'ece': ece}
    
    @staticmethod
    def compute_sharpness(pred_std: np.ndarray) -> Dict[str, float]:
        """计算预测的锐度（不确定性的集中程度）"""
        return {
            'mean_uncertainty': np.mean(pred_std),
            'std_uncertainty': np.std(pred_std),
            'uncertainty_range': np.max(pred_std) - np.min(pred_std)
        }
    
    @staticmethod
    def compute_ratio_metrics(pred_mean, pred_std, true_ratio) -> Dict[str, float]:
        """计算比例预测的综合指标"""
        # 转换为numpy数组
        import torch
        import numpy as np
        
        if torch.is_tensor(pred_mean):
            pred_mean = pred_mean.detach().cpu().numpy()
        if torch.is_tensor(pred_std):
            pred_std = pred_std.detach().cpu().numpy()
        if torch.is_tensor(true_ratio):
            true_ratio = true_ratio.detach().cpu().numpy()
        
        # 确保是1D数组
        pred_mean = np.asarray(pred_mean).flatten()
        pred_std = np.asarray(pred_std).flatten()
        true_ratio = np.asarray(true_ratio).flatten()
        
        # 基本回归指标
        mae = np.mean(np.abs(pred_mean - true_ratio))
        mse = np.mean((pred_mean - true_ratio) ** 2)
        rmse = np.sqrt(mse)
        
        # 相关系数
        correlation = np.corrcoef(pred_mean, true_ratio)[0, 1] if len(pred_mean) > 1 else 0.0
        
        # R²分数
        ss_res = np.sum((true_ratio - pred_mean) ** 2)
        ss_tot = np.sum((true_ratio - np.mean(true_ratio)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'r2_score': float(r2_score)
        }