"""性能分析器模块

该模块提供了深度性能分析功能，包括：
- 性能瓶颈分析
- 统计分析
- 趋势分析
- 异常检测
- 性能优化建议
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from .metrics import DetectionMetrics, LocalizationMetrics, PhotonMetrics, ComprehensiveMetrics
from ..utils.logging_utils import get_logger
from ..utils.math_utils import calculate_statistics
from ..utils.io_utils import write_json, create_directory
from ..utils.visualization import save_plot

logger = get_logger(__name__)


@dataclass
class PerformanceProfile:
    """性能概况"""
    model_name: str
    dataset_info: Dict[str, Any]
    overall_score: float
    detection_score: float
    localization_score: float
    photon_score: float
    efficiency_score: float
    stability_score: float
    bottlenecks: List[str]
    recommendations: List[str]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class StatisticalAnalysis:
    """统计分析结果"""
    metric_name: str
    mean: float
    std: float
    median: float
    q25: float
    q75: float
    min_val: float
    max_val: float
    skewness: float
    kurtosis: float
    normality_test: Dict[str, float]
    outliers: List[int]
    confidence_interval: Tuple[float, float]


@dataclass
class TrendAnalysis:
    """趋势分析结果"""
    metric_name: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float
    correlation_coefficient: float
    p_value: float
    seasonal_pattern: bool
    change_points: List[int]
    forecast: Optional[List[float]] = None


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        """
        初始化性能分析器
        """
        self.analysis_history = []
        logger.info("性能分析器初始化完成")
    
    def analyze_performance(self,
                          results: Dict[str, Any],
                          reference_results: Optional[Dict[str, Any]] = None) -> PerformanceProfile:
        """
        分析模型性能
        
        Args:
            results: 评估结果
            reference_results: 参考结果（用于比较）
            
        Returns:
            PerformanceProfile: 性能概况
        """
        logger.info("开始性能分析")
        
        # 提取指标
        detection_metrics = results.get('detection_metrics', {})
        localization_metrics = results.get('localization_metrics', {})
        photon_metrics = results.get('photon_metrics', {})
        comprehensive_metrics = results.get('comprehensive_metrics', {})
        
        # 计算各项得分
        detection_score = self._calculate_detection_score(detection_metrics)
        localization_score = self._calculate_localization_score(localization_metrics)
        photon_score = self._calculate_photon_score(photon_metrics)
        efficiency_score = self._calculate_efficiency_score(results)
        stability_score = self._calculate_stability_score(results)
        
        # 计算总体得分
        overall_score = np.mean([detection_score, localization_score, photon_score, efficiency_score])
        
        # 识别性能瓶颈
        bottlenecks = self._identify_bottlenecks({
            'detection': detection_score,
            'localization': localization_score,
            'photon': photon_score,
            'efficiency': efficiency_score,
            'stability': stability_score
        })
        
        # 生成优化建议
        recommendations = self._generate_recommendations(
            detection_metrics, localization_metrics, photon_metrics, bottlenecks
        )
        
        # 创建性能概况
        profile = PerformanceProfile(
            model_name=results.get('model_name', 'Unknown'),
            dataset_info={
                'name': results.get('dataset_name', 'Unknown'),
                'num_samples': results.get('num_samples', 0),
                'processing_time': results.get('processing_time', 0),
                'memory_usage': results.get('memory_usage', 0)
            },
            overall_score=overall_score,
            detection_score=detection_score,
            localization_score=localization_score,
            photon_score=photon_score,
            efficiency_score=efficiency_score,
            stability_score=stability_score,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
        
        self.analysis_history.append(profile)
        logger.info(f"性能分析完成，总体得分: {overall_score:.3f}")
        
        return profile
    
    def _calculate_detection_score(self, metrics: Dict[str, Any]) -> float:
        """
        计算检测得分
        
        Args:
            metrics: 检测指标
            
        Returns:
            float: 检测得分 (0-1)
        """
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1_score = metrics.get('f1_score', 0)
        jaccard = metrics.get('jaccard_index', 0)
        
        # 加权平均
        weights = [0.3, 0.3, 0.3, 0.1]
        scores = [precision, recall, f1_score, jaccard]
        
        return np.average(scores, weights=weights)
    
    def _calculate_localization_score(self, metrics: Dict[str, Any]) -> float:
        """
        计算定位得分
        
        Args:
            metrics: 定位指标
            
        Returns:
            float: 定位得分 (0-1)
        """
        rmse_x = metrics.get('rmse_x', float('inf'))
        rmse_y = metrics.get('rmse_y', float('inf'))
        rmse_z = metrics.get('rmse_z', float('inf'))
        
        # 转换为得分（RMSE越小得分越高）
        # 假设100nm以下为优秀，500nm以下为良好
        def rmse_to_score(rmse):
            if rmse <= 100:
                return 1.0
            elif rmse <= 500:
                return 1.0 - (rmse - 100) / 400 * 0.5
            else:
                return max(0, 0.5 - (rmse - 500) / 1000 * 0.5)
        
        score_x = rmse_to_score(rmse_x)
        score_y = rmse_to_score(rmse_y)
        score_z = rmse_to_score(rmse_z)
        
        # XY方向权重更高
        return np.average([score_x, score_y, score_z], weights=[0.4, 0.4, 0.2])
    
    def _calculate_photon_score(self, metrics: Dict[str, Any]) -> float:
        """
        计算光子数得分
        
        Args:
            metrics: 光子数指标
            
        Returns:
            float: 光子数得分 (0-1)
        """
        correlation = metrics.get('correlation', 0)
        relative_error = metrics.get('relative_error', 1)
        
        # 相关性得分
        corr_score = max(0, correlation)
        
        # 相对误差得分
        error_score = max(0, 1 - abs(relative_error))
        
        return np.mean([corr_score, error_score])
    
    def _calculate_efficiency_score(self, results: Dict[str, Any]) -> float:
        """
        计算效率得分
        
        Args:
            results: 评估结果
            
        Returns:
            float: 效率得分 (0-1)
        """
        processing_time = results.get('processing_time', float('inf'))
        memory_usage = results.get('memory_usage', float('inf'))
        num_samples = results.get('num_samples', 1)
        
        # 每样本处理时间（秒）
        time_per_sample = processing_time / max(num_samples, 1)
        
        # 时间效率得分（假设1秒/样本为基准）
        time_score = max(0, 1 - time_per_sample / 10)
        
        # 内存效率得分（假设1GB为基准）
        memory_score = max(0, 1 - memory_usage / 1024)
        
        return np.mean([time_score, memory_score])
    
    def _calculate_stability_score(self, results: Dict[str, Any]) -> float:
        """
        计算稳定性得分
        
        Args:
            results: 评估结果
            
        Returns:
            float: 稳定性得分 (0-1)
        """
        # 基于定位指标的标准差计算稳定性
        localization_metrics = results.get('localization_metrics', {})
        
        std_x = localization_metrics.get('std_x', float('inf'))
        std_y = localization_metrics.get('std_y', float('inf'))
        std_z = localization_metrics.get('std_z', float('inf'))
        
        # 标准差越小，稳定性越好
        def std_to_score(std):
            if std <= 50:
                return 1.0
            elif std <= 200:
                return 1.0 - (std - 50) / 150 * 0.5
            else:
                return max(0, 0.5 - (std - 200) / 300 * 0.5)
        
        score_x = std_to_score(std_x)
        score_y = std_to_score(std_y)
        score_z = std_to_score(std_z)
        
        return np.mean([score_x, score_y, score_z])
    
    def _identify_bottlenecks(self, scores: Dict[str, float]) -> List[str]:
        """
        识别性能瓶颈
        
        Args:
            scores: 各项得分
            
        Returns:
            List[str]: 瓶颈列表
        """
        bottlenecks = []
        threshold = 0.6  # 得分阈值
        
        for metric, score in scores.items():
            if score < threshold:
                bottlenecks.append(f"{metric}性能不足 (得分: {score:.3f})")
        
        # 找出最低得分
        min_score = min(scores.values())
        min_metrics = [k for k, v in scores.items() if v == min_score]
        
        if min_score < 0.8:
            bottlenecks.append(f"主要瓶颈: {', '.join(min_metrics)}")
        
        return bottlenecks
    
    def _generate_recommendations(self,
                                detection_metrics: Dict[str, Any],
                                localization_metrics: Dict[str, Any],
                                photon_metrics: Dict[str, Any],
                                bottlenecks: List[str]) -> List[str]:
        """
        生成优化建议
        
        Args:
            detection_metrics: 检测指标
            localization_metrics: 定位指标
            photon_metrics: 光子数指标
            bottlenecks: 瓶颈列表
            
        Returns:
            List[str]: 建议列表
        """
        recommendations = []
        
        # 检测性能建议
        precision = detection_metrics.get('precision', 0)
        recall = detection_metrics.get('recall', 0)
        
        if precision < 0.8:
            recommendations.append("建议调整检测阈值以提高精确率，减少假正例")
        if recall < 0.8:
            recommendations.append("建议降低检测阈值或改进模型以提高召回率")
        if precision < 0.8 and recall < 0.8:
            recommendations.append("建议重新训练模型或调整网络架构")
        
        # 定位性能建议
        rmse_x = localization_metrics.get('rmse_x', 0)
        rmse_y = localization_metrics.get('rmse_y', 0)
        rmse_z = localization_metrics.get('rmse_z', 0)
        
        if max(rmse_x, rmse_y) > 200:
            recommendations.append("横向定位精度不足，建议增加训练数据或调整损失函数权重")
        if rmse_z > 500:
            recommendations.append("轴向定位精度不足，建议改进Z方向的特征提取")
        
        # 光子数建议
        correlation = photon_metrics.get('correlation', 0)
        if correlation < 0.7:
            recommendations.append("光子数预测相关性较低，建议改进光子数回归分支")
        
        # 通用建议
        if len(bottlenecks) > 2:
            recommendations.append("多项指标存在问题，建议进行全面的模型优化")
        
        if not recommendations:
            recommendations.append("模型性能良好，可考虑进一步优化以达到更高精度")
        
        return recommendations
    
    def statistical_analysis(self, data: np.ndarray, metric_name: str) -> StatisticalAnalysis:
        """
        统计分析
        
        Args:
            data: 数据数组
            metric_name: 指标名称
            
        Returns:
            StatisticalAnalysis: 统计分析结果
        """
        logger.info(f"开始统计分析: {metric_name}")
        
        # 基本统计量
        stats_dict = calculate_statistics(data)
        
        # 正态性检验
        if len(data) >= 3:
            shapiro_stat, shapiro_p = stats.shapiro(data)
            normality_test = {'shapiro_stat': shapiro_stat, 'shapiro_p': shapiro_p}
        else:
            normality_test = {'shapiro_stat': np.nan, 'shapiro_p': np.nan}
        
        # 异常值检测（IQR方法）
        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = np.where((data < lower_bound) | (data > upper_bound))[0].tolist()
        
        # 置信区间（95%）
        if len(data) > 1:
            confidence_interval = stats.t.interval(
                0.95, len(data) - 1, 
                loc=stats_dict['mean'], 
                scale=stats.sem(data)
            )
        else:
            confidence_interval = (stats_dict['mean'], stats_dict['mean'])
        
        return StatisticalAnalysis(
            metric_name=metric_name,
            mean=stats_dict['mean'],
            std=stats_dict['std'],
            median=stats_dict['median'],
            q25=q25,
            q75=q75,
            min_val=stats_dict['min'],
            max_val=stats_dict['max'],
            skewness=stats_dict['skewness'],
            kurtosis=stats_dict['kurtosis'],
            normality_test=normality_test,
            outliers=outliers,
            confidence_interval=confidence_interval
        )
    
    def trend_analysis(self, data: np.ndarray, metric_name: str) -> TrendAnalysis:
        """
        趋势分析
        
        Args:
            data: 时间序列数据
            metric_name: 指标名称
            
        Returns:
            TrendAnalysis: 趋势分析结果
        """
        logger.info(f"开始趋势分析: {metric_name}")
        
        if len(data) < 3:
            logger.warning("数据点太少，无法进行趋势分析")
            return TrendAnalysis(
                metric_name=metric_name,
                trend_direction='unknown',
                trend_strength=0.0,
                correlation_coefficient=0.0,
                p_value=1.0,
                seasonal_pattern=False,
                change_points=[]
            )
        
        # 时间索引
        time_index = np.arange(len(data))
        
        # 线性趋势分析
        correlation_coef, p_value = stats.pearsonr(time_index, data)
        
        # 趋势方向
        if p_value < 0.05:
            if correlation_coef > 0:
                trend_direction = 'increasing'
            else:
                trend_direction = 'decreasing'
        else:
            trend_direction = 'stable'
        
        # 趋势强度
        trend_strength = abs(correlation_coef)
        
        # 变点检测（简单方法：滑动窗口方差）
        change_points = self._detect_change_points(data)
        
        # 季节性模式检测（简单方法：自相关）
        seasonal_pattern = self._detect_seasonality(data)
        
        return TrendAnalysis(
            metric_name=metric_name,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            correlation_coefficient=correlation_coef,
            p_value=p_value,
            seasonal_pattern=seasonal_pattern,
            change_points=change_points
        )
    
    def _detect_change_points(self, data: np.ndarray, window_size: int = 5) -> List[int]:
        """
        检测变点
        
        Args:
            data: 数据数组
            window_size: 窗口大小
            
        Returns:
            List[int]: 变点索引列表
        """
        if len(data) < window_size * 2:
            return []
        
        change_points = []
        
        for i in range(window_size, len(data) - window_size):
            left_window = data[i-window_size:i]
            right_window = data[i:i+window_size]
            
            # 使用t检验检测均值变化
            try:
                t_stat, p_value = stats.ttest_ind(left_window, right_window)
                if p_value < 0.01:  # 严格的阈值
                    change_points.append(i)
            except:
                continue
        
        return change_points
    
    def _detect_seasonality(self, data: np.ndarray) -> bool:
        """
        检测季节性模式
        
        Args:
            data: 数据数组
            
        Returns:
            bool: 是否存在季节性
        """
        if len(data) < 10:
            return False
        
        # 计算自相关
        try:
            autocorr = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]
            
            # 寻找显著的周期性峰值
            for lag in range(2, min(len(autocorr) // 2, 20)):
                if autocorr[lag] > 0.3:  # 阈值
                    return True
            
            return False
        except:
            return False
    
    def anomaly_detection(self, data: np.ndarray, method: str = 'isolation') -> Dict[str, Any]:
        """
        异常检测
        
        Args:
            data: 数据数组
            method: 检测方法 ('isolation', 'statistical', 'clustering')
            
        Returns:
            Dict[str, Any]: 异常检测结果
        """
        logger.info(f"开始异常检测，方法: {method}")
        
        if len(data) < 5:
            return {'anomalies': [], 'method': method, 'message': '数据点太少'}
        
        anomalies = []
        
        if method == 'statistical':
            # 基于统计的异常检测（3-sigma规则）
            mean = np.mean(data)
            std = np.std(data)
            threshold = 3 * std
            
            anomalies = np.where(np.abs(data - mean) > threshold)[0].tolist()
            
        elif method == 'clustering':
            # 基于聚类的异常检测
            try:
                # 标准化数据
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data.reshape(-1, 1))
                
                # DBSCAN聚类
                dbscan = DBSCAN(eps=0.5, min_samples=3)
                labels = dbscan.fit_predict(data_scaled)
                
                # 噪声点（标签为-1）视为异常
                anomalies = np.where(labels == -1)[0].tolist()
                
            except Exception as e:
                logger.error(f"聚类异常检测失败: {e}")
                anomalies = []
        
        else:  # isolation forest (简化版)
            # 简化的孤立森林方法
            q25, q75 = np.percentile(data, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 2.5 * iqr
            upper_bound = q75 + 2.5 * iqr
            
            anomalies = np.where((data < lower_bound) | (data > upper_bound))[0].tolist()
        
        return {
            'anomalies': anomalies,
            'anomaly_count': len(anomalies),
            'anomaly_rate': len(anomalies) / len(data),
            'method': method,
            'anomaly_values': data[anomalies].tolist() if anomalies else []
        }
    
    def compare_models(self, profiles: List[PerformanceProfile]) -> Dict[str, Any]:
        """
        比较多个模型的性能
        
        Args:
            profiles: 性能概况列表
            
        Returns:
            Dict[str, Any]: 比较结果
        """
        logger.info(f"开始比较 {len(profiles)} 个模型")
        
        if len(profiles) < 2:
            return {'message': '需要至少2个模型进行比较'}
        
        # 提取各项得分
        model_names = [p.model_name for p in profiles]
        overall_scores = [p.overall_score for p in profiles]
        detection_scores = [p.detection_score for p in profiles]
        localization_scores = [p.localization_score for p in profiles]
        photon_scores = [p.photon_score for p in profiles]
        efficiency_scores = [p.efficiency_score for p in profiles]
        
        # 排名
        overall_ranking = sorted(enumerate(overall_scores), key=lambda x: x[1], reverse=True)
        
        # 最佳模型
        best_model_idx = overall_ranking[0][0]
        best_model = profiles[best_model_idx]
        
        # 各项指标的最佳模型
        best_detection = model_names[np.argmax(detection_scores)]
        best_localization = model_names[np.argmax(localization_scores)]
        best_photon = model_names[np.argmax(photon_scores)]
        best_efficiency = model_names[np.argmax(efficiency_scores)]
        
        # 统计分析
        score_stats = {
            'overall': calculate_statistics(np.array(overall_scores)),
            'detection': calculate_statistics(np.array(detection_scores)),
            'localization': calculate_statistics(np.array(localization_scores)),
            'photon': calculate_statistics(np.array(photon_scores)),
            'efficiency': calculate_statistics(np.array(efficiency_scores))
        }
        
        return {
            'model_count': len(profiles),
            'best_overall_model': {
                'name': best_model.model_name,
                'score': best_model.overall_score,
                'rank': 1
            },
            'rankings': [
                {
                    'rank': i + 1,
                    'model': model_names[idx],
                    'score': score
                }
                for i, (idx, score) in enumerate(overall_ranking)
            ],
            'category_leaders': {
                'detection': best_detection,
                'localization': best_localization,
                'photon': best_photon,
                'efficiency': best_efficiency
            },
            'statistics': score_stats,
            'recommendations': self._generate_comparison_recommendations(profiles)
        }
    
    def _generate_comparison_recommendations(self, profiles: List[PerformanceProfile]) -> List[str]:
        """
        生成模型比较建议
        
        Args:
            profiles: 性能概况列表
            
        Returns:
            List[str]: 建议列表
        """
        recommendations = []
        
        # 分析得分分布
        overall_scores = [p.overall_score for p in profiles]
        score_range = max(overall_scores) - min(overall_scores)
        
        if score_range < 0.1:
            recommendations.append("各模型性能相近，建议考虑计算效率和资源消耗")
        elif score_range > 0.3:
            recommendations.append("模型性能差异较大，建议选择高分模型并分析低分模型的问题")
        
        # 分析瓶颈模式
        all_bottlenecks = []
        for profile in profiles:
            all_bottlenecks.extend(profile.bottlenecks)
        
        if len(all_bottlenecks) > len(profiles):
            recommendations.append("多数模型存在性能瓶颈，建议进行系统性优化")
        
        return recommendations
    
    def generate_analysis_report(self,
                               profile: PerformanceProfile,
                               output_dir: str,
                               include_plots: bool = True) -> str:
        """
        生成分析报告
        
        Args:
            profile: 性能概况
            output_dir: 输出目录
            include_plots: 是否包含图表
            
        Returns:
            str: 报告文件路径
        """
        logger.info("生成性能分析报告")
        
        output_path = Path(output_dir)
        create_directory(str(output_path))
        
        # 保存JSON数据
        profile_path = output_path / "performance_profile.json"
        save_json(profile.to_dict(), str(profile_path))
        
        # 生成图表
        if include_plots:
            self._generate_analysis_plots(profile, output_path)
        
        # 生成HTML报告
        report_path = self._generate_html_analysis_report(profile, output_path)
        
        logger.info(f"分析报告已生成: {report_path}")
        return str(report_path)
    
    def _generate_analysis_plots(self, profile: PerformanceProfile, output_path: Path):
        """
        生成分析图表
        
        Args:
            profile: 性能概况
            output_path: 输出路径
        """
        # 得分雷达图
        categories = ['检测', '定位', '光子数', '效率', '稳定性']
        scores = [
            profile.detection_score,
            profile.localization_score,
            profile.photon_score,
            profile.efficiency_score,
            profile.stability_score
        ]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        scores += scores[:1]  # 闭合图形
        angles += angles[:1]
        
        ax.plot(angles, scores, 'o-', linewidth=2, color='blue')
        ax.fill(angles, scores, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title(f'{profile.model_name} 性能雷达图', y=1.08, fontsize=14)
        
        plt.tight_layout()
        save_plot(fig, str(output_path / "performance_radar.png"))
        plt.close(fig)
        
        # 得分条形图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(categories, scores[:-1], color=['red', 'green', 'blue', 'orange', 'purple'], alpha=0.7)
        ax.set_ylabel('得分')
        ax.set_title(f'{profile.model_name} 各项性能得分')
        ax.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, score in zip(bars, scores[:-1]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        save_plot(fig, str(output_path / "performance_bars.png"))
        plt.close(fig)
    
    def _generate_html_analysis_report(self, profile: PerformanceProfile, output_path: Path) -> Path:
        """
        生成HTML分析报告
        
        Args:
            profile: 性能概况
            output_path: 输出路径
            
        Returns:
            Path: 报告文件路径
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>性能分析报告 - {profile.model_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .score-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
                .score-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; text-align: center; }}
                .score-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
                .bottleneck {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .recommendation {{ background-color: #d1ecf1; border: 1px solid #bee5eb; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .chart {{ text-align: center; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>性能分析报告</h1>
                <h2>{profile.model_name}</h2>
                <p><strong>分析时间:</strong> {profile.timestamp}</p>
                <p><strong>数据集:</strong> {profile.dataset_info['name']}</p>
                <p><strong>样本数量:</strong> {profile.dataset_info['num_samples']}</p>
            </div>
            
            <div class="section">
                <h2>性能得分概览</h2>
                <div class="score-grid">
                    <div class="score-card">
                        <div>总体得分</div>
                        <div class="score-value">{profile.overall_score:.3f}</div>
                    </div>
                    <div class="score-card">
                        <div>检测得分</div>
                        <div class="score-value">{profile.detection_score:.3f}</div>
                    </div>
                    <div class="score-card">
                        <div>定位得分</div>
                        <div class="score-value">{profile.localization_score:.3f}</div>
                    </div>
                    <div class="score-card">
                        <div>光子数得分</div>
                        <div class="score-value">{profile.photon_score:.3f}</div>
                    </div>
                    <div class="score-card">
                        <div>效率得分</div>
                        <div class="score-value">{profile.efficiency_score:.3f}</div>
                    </div>
                    <div class="score-card">
                        <div>稳定性得分</div>
                        <div class="score-value">{profile.stability_score:.3f}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>性能可视化</h2>
                <div class="chart">
                    <img src="performance_radar.png" alt="性能雷达图" style="max-width: 100%; height: auto;">
                </div>
                <div class="chart">
                    <img src="performance_bars.png" alt="性能条形图" style="max-width: 100%; height: auto;">
                </div>
            </div>
            
            <div class="section">
                <h2>性能瓶颈</h2>
                {''.join([f'<div class="bottleneck">⚠️ {bottleneck}</div>' for bottleneck in profile.bottlenecks]) if profile.bottlenecks else '<p>未发现明显性能瓶颈</p>'}
            </div>
            
            <div class="section">
                <h2>优化建议</h2>
                {''.join([f'<div class="recommendation">💡 {rec}</div>' for rec in profile.recommendations])}
            </div>
            
            <div class="section">
                <h2>数据集信息</h2>
                <table>
                    <tr><th>项目</th><th>值</th></tr>
                    <tr><td>数据集名称</td><td>{profile.dataset_info['name']}</td></tr>
                    <tr><td>样本数量</td><td>{profile.dataset_info['num_samples']}</td></tr>
                    <tr><td>处理时间</td><td>{profile.dataset_info['processing_time']:.2f} 秒</td></tr>
                    <tr><td>内存使用</td><td>{profile.dataset_info['memory_usage']:.2f} MB</td></tr>
                    <tr><td>平均处理速度</td><td>{profile.dataset_info['num_samples'] / max(profile.dataset_info['processing_time'], 0.001):.2f} 样本/秒</td></tr>
                </table>
            </div>
        </body>
        </html>
        """
        
        report_path = output_path / "performance_analysis_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path