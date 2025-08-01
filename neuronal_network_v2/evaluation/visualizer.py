"""评估可视化模块

该模块提供了评估结果的可视化功能，包括：
- 指标可视化
- 预测结果可视化
- 比较分析图表
- 交互式可视化
- 报告生成
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import logging
from dataclasses import dataclass

from .metrics import DetectionMetrics, LocalizationMetrics, PhotonMetrics, ComprehensiveMetrics
from ..utils.logging_utils import get_logger
from ..utils.visualization import setup_matplotlib, save_plot
from ..utils.io_utils import create_directory

logger = get_logger(__name__)

# 设置matplotlib
setup_matplotlib()


class MetricsVisualizer:
    """评估指标可视化器"""
    
    def __init__(self, style: str = "seaborn", figsize: Tuple[int, int] = (12, 8)):
        """
        初始化可视化器
        
        Args:
            style: 绘图风格
            figsize: 图形大小
        """
        self.style = style
        self.figsize = figsize
        
        # 设置绘图风格
        plt.style.use(style)
        sns.set_palette("husl")
        
        logger.info(f"指标可视化器初始化完成，风格: {style}")
    
    def plot_detection_metrics(self,
                             metrics: DetectionMetrics,
                             title: str = "检测指标",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制检测指标
        
        Args:
            metrics: 检测指标
            title: 图表标题
            save_path: 保存路径
            
        Returns:
            plt.Figure: 图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # 精确率、召回率、F1分数
        metric_names = ['精确率', '召回率', 'F1分数', 'Jaccard指数']
        metric_values = [metrics.precision, metrics.recall, metrics.f1_score, metrics.jaccard_index]
        
        axes[0, 0].bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'orange', 'pink'])
        axes[0, 0].set_title('检测性能指标')
        axes[0, 0].set_ylabel('分数')
        axes[0, 0].set_ylim(0, 1)
        
        # 添加数值标签
        for i, v in enumerate(metric_values):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 混淆矩阵可视化
        confusion_data = np.array([[metrics.true_positives, metrics.false_negatives],
                                  [metrics.false_positives, 0]])  # TN通常不适用于检测任务
        
        im = axes[0, 1].imshow(confusion_data, cmap='Blues', aspect='auto')
        axes[0, 1].set_title('检测结果统计')
        axes[0, 1].set_xticks([0, 1])
        axes[0, 1].set_yticks([0, 1])
        axes[0, 1].set_xticklabels(['预测正例', '预测负例'])
        axes[0, 1].set_yticklabels(['真实正例', '真实负例'])
        
        # 添加数值标签
        for i in range(2):
            for j in range(2):
                if confusion_data[i, j] > 0:
                    axes[0, 1].text(j, i, str(confusion_data[i, j]), 
                                   ha='center', va='center', color='white', fontweight='bold')
        
        # 精确率-召回率权衡
        axes[1, 0].scatter([metrics.recall], [metrics.precision], s=100, c='red', marker='o')
        axes[1, 0].set_xlabel('召回率')
        axes[1, 0].set_ylabel('精确率')
        axes[1, 0].set_title('精确率-召回率权衡')
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # F1分数等高线
        recall_range = np.linspace(0.01, 1, 100)
        precision_range = np.linspace(0.01, 1, 100)
        R, P = np.meshgrid(recall_range, precision_range)
        F1 = 2 * P * R / (P + R)
        
        contours = axes[1, 0].contour(R, P, F1, levels=[0.2, 0.4, 0.6, 0.8], colors='gray', alpha=0.5)
        axes[1, 0].clabel(contours, inline=True, fontsize=8)
        
        # 检测统计饼图
        labels = ['真正例', '假正例', '假负例']
        sizes = [metrics.true_positives, metrics.false_positives, metrics.false_negatives]
        colors = ['lightgreen', 'lightcoral', 'lightyellow']
        
        # 过滤掉为0的值
        non_zero_indices = [i for i, size in enumerate(sizes) if size > 0]
        if non_zero_indices:
            filtered_labels = [labels[i] for i in non_zero_indices]
            filtered_sizes = [sizes[i] for i in non_zero_indices]
            filtered_colors = [colors[i] for i in non_zero_indices]
            
            axes[1, 1].pie(filtered_sizes, labels=filtered_labels, colors=filtered_colors, 
                          autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('检测结果分布')
        else:
            axes[1, 1].text(0.5, 0.5, '无检测数据', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('检测结果分布')
        
        plt.tight_layout()
        
        if save_path:
            save_plot(fig, save_path)
        
        return fig
    
    def plot_localization_metrics(self,
                                metrics: LocalizationMetrics,
                                title: str = "定位指标",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制定位指标
        
        Args:
            metrics: 定位指标
            title: 图表标题
            save_path: 保存路径
            
        Returns:
            plt.Figure: 图形对象
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # RMSE指标
        rmse_values = [metrics.rmse_x, metrics.rmse_y, metrics.rmse_z]
        rmse_labels = ['RMSE X', 'RMSE Y', 'RMSE Z']
        colors = ['red', 'green', 'blue']
        
        bars = axes[0, 0].bar(rmse_labels, rmse_values, color=colors, alpha=0.7)
        axes[0, 0].set_title('均方根误差 (RMSE)')
        axes[0, 0].set_ylabel('误差 (nm)')
        
        # 添加数值标签
        for bar, value in zip(bars, rmse_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values)*0.01,
                           f'{value:.1f}', ha='center', va='bottom')
        
        # MAE指标
        mae_values = [metrics.mae_x, metrics.mae_y, metrics.mae_z]
        mae_labels = ['MAE X', 'MAE Y', 'MAE Z']
        
        bars = axes[0, 1].bar(mae_labels, mae_values, color=colors, alpha=0.7)
        axes[0, 1].set_title('平均绝对误差 (MAE)')
        axes[0, 1].set_ylabel('误差 (nm)')
        
        for bar, value in zip(bars, mae_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values)*0.01,
                           f'{value:.1f}', ha='center', va='bottom')
        
        # 偏差指标
        bias_values = [metrics.bias_x, metrics.bias_y, metrics.bias_z]
        bias_labels = ['偏差 X', '偏差 Y', '偏差 Z']
        
        bars = axes[0, 2].bar(bias_labels, bias_values, color=colors, alpha=0.7)
        axes[0, 2].set_title('系统偏差')
        axes[0, 2].set_ylabel('偏差 (nm)')
        axes[0, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        for bar, value in zip(bars, bias_values):
            y_pos = bar.get_height() + (max(bias_values) - min(bias_values))*0.01 if value >= 0 else bar.get_height() - (max(bias_values) - min(bias_values))*0.01
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, y_pos,
                           f'{value:.1f}', ha='center', va='bottom' if value >= 0 else 'top')
        
        # 标准差指标
        std_values = [metrics.std_x, metrics.std_y, metrics.std_z]
        std_labels = ['标准差 X', '标准差 Y', '标准差 Z']
        
        bars = axes[1, 0].bar(std_labels, std_values, color=colors, alpha=0.7)
        axes[1, 0].set_title('标准差')
        axes[1, 0].set_ylabel('标准差 (nm)')
        
        for bar, value in zip(bars, std_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(std_values)*0.01,
                           f'{value:.1f}', ha='center', va='bottom')
        
        # 精度比较
        precision_values = [metrics.lateral_precision, metrics.axial_precision]
        precision_labels = ['横向精度', '轴向精度']
        precision_colors = ['orange', 'purple']
        
        bars = axes[1, 1].bar(precision_labels, precision_values, color=precision_colors, alpha=0.7)
        axes[1, 1].set_title('定位精度')
        axes[1, 1].set_ylabel('精度 (nm)')
        
        for bar, value in zip(bars, precision_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(precision_values)*0.01,
                           f'{value:.1f}', ha='center', va='bottom')
        
        # 误差分布雷达图
        categories = ['RMSE X', 'RMSE Y', 'RMSE Z', 'MAE X', 'MAE Y', 'MAE Z']
        values = rmse_values + mae_values
        
        # 归一化值用于雷达图
        max_val = max(values) if max(values) > 0 else 1
        normalized_values = [v / max_val for v in values]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        normalized_values += normalized_values[:1]  # 闭合图形
        angles += angles[:1]
        
        ax_radar = plt.subplot(2, 3, 6, projection='polar')
        ax_radar.plot(angles, normalized_values, 'o-', linewidth=2, color='blue')
        ax_radar.fill(angles, normalized_values, alpha=0.25, color='blue')
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('误差分布雷达图', y=1.08)
        
        plt.tight_layout()
        
        if save_path:
            save_plot(fig, save_path)
        
        return fig
    
    def plot_photon_metrics(self,
                          metrics: PhotonMetrics,
                          title: str = "光子数指标",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制光子数指标
        
        Args:
            metrics: 光子数指标
            title: 图表标题
            save_path: 保存路径
            
        Returns:
            plt.Figure: 图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # 误差指标
        error_metrics = ['RMSE', 'MAE', '标准差']
        error_values = [metrics.rmse, metrics.mae, metrics.std]
        
        bars = axes[0, 0].bar(error_metrics, error_values, color=['red', 'orange', 'yellow'], alpha=0.7)
        axes[0, 0].set_title('光子数误差指标')
        axes[0, 0].set_ylabel('误差值')
        
        for bar, value in zip(bars, error_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(error_values)*0.01,
                           f'{value:.1f}', ha='center', va='bottom')
        
        # 偏差和相关性
        bias_corr_metrics = ['偏差', '相关性']
        bias_corr_values = [metrics.bias, metrics.correlation]
        colors = ['green' if metrics.bias >= 0 else 'red', 'blue']
        
        bars = axes[0, 1].bar(bias_corr_metrics, bias_corr_values, color=colors, alpha=0.7)
        axes[0, 1].set_title('偏差和相关性')
        axes[0, 1].set_ylabel('数值')
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        for bar, value in zip(bars, bias_corr_values):
            y_pos = bar.get_height() + abs(max(bias_corr_values) - min(bias_corr_values))*0.01 if value >= 0 else bar.get_height() - abs(max(bias_corr_values) - min(bias_corr_values))*0.01
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, y_pos,
                           f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top')
        
        # 相对误差
        axes[1, 0].bar(['相对误差'], [metrics.relative_error], color='purple', alpha=0.7)
        axes[1, 0].set_title('相对误差')
        axes[1, 0].set_ylabel('相对误差')
        axes[1, 0].text(0, metrics.relative_error + abs(metrics.relative_error)*0.01,
                       f'{metrics.relative_error:.3f}', ha='center', va='bottom')
        
        # 综合性能雷达图
        categories = ['精度\n(1/RMSE)', '准确性\n(1/MAE)', '一致性\n(相关性)', '稳定性\n(1/标准差)']
        
        # 归一化指标（转换为正向指标）
        rmse_score = 1 / (1 + metrics.rmse / 1000) if metrics.rmse > 0 else 1
        mae_score = 1 / (1 + metrics.mae / 1000) if metrics.mae > 0 else 1
        corr_score = max(0, metrics.correlation)  # 相关性已经是正向指标
        std_score = 1 / (1 + metrics.std / 1000) if metrics.std > 0 else 1
        
        values = [rmse_score, mae_score, corr_score, std_score]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 闭合图形
        angles += angles[:1]
        
        ax_radar = plt.subplot(2, 2, 4, projection='polar')
        ax_radar.plot(angles, values, 'o-', linewidth=2, color='green')
        ax_radar.fill(angles, values, alpha=0.25, color='green')
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('光子数性能雷达图', y=1.08)
        
        plt.tight_layout()
        
        if save_path:
            save_plot(fig, save_path)
        
        return fig
    
    def plot_comprehensive_comparison(self,
                                    results: Dict[str, Dict[str, Any]],
                                    title: str = "模型综合比较",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制多模型综合比较图
        
        Args:
            results: 模型结果字典
            title: 图表标题
            save_path: 保存路径
            
        Returns:
            plt.Figure: 图形对象
        """
        model_names = list(results.keys())
        n_models = len(model_names)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(title, fontsize=16)
        
        # 提取指标数据
        precision_scores = [results[name]['detection_metrics']['precision'] for name in model_names]
        recall_scores = [results[name]['detection_metrics']['recall'] for name in model_names]
        f1_scores = [results[name]['detection_metrics']['f1_score'] for name in model_names]
        rmse_x = [results[name]['localization_metrics']['rmse_x'] for name in model_names]
        rmse_y = [results[name]['localization_metrics']['rmse_y'] for name in model_names]
        rmse_z = [results[name]['localization_metrics']['rmse_z'] for name in model_names]
        
        # 检测指标比较
        x = np.arange(n_models)
        width = 0.25
        
        axes[0, 0].bar(x - width, precision_scores, width, label='精确率', alpha=0.8)
        axes[0, 0].bar(x, recall_scores, width, label='召回率', alpha=0.8)
        axes[0, 0].bar(x + width, f1_scores, width, label='F1分数', alpha=0.8)
        
        axes[0, 0].set_xlabel('模型')
        axes[0, 0].set_ylabel('分数')
        axes[0, 0].set_title('检测性能比较')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1)
        
        # 定位误差比较
        axes[0, 1].bar(x - width, rmse_x, width, label='RMSE X', alpha=0.8)
        axes[0, 1].bar(x, rmse_y, width, label='RMSE Y', alpha=0.8)
        axes[0, 1].bar(x + width, rmse_z, width, label='RMSE Z', alpha=0.8)
        
        axes[0, 1].set_xlabel('模型')
        axes[0, 1].set_ylabel('RMSE (nm)')
        axes[0, 1].set_title('定位误差比较')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(model_names, rotation=45)
        axes[0, 1].legend()
        
        # 精确率-召回率散点图
        colors = plt.cm.Set1(np.linspace(0, 1, n_models))
        for i, name in enumerate(model_names):
            axes[0, 2].scatter(recall_scores[i], precision_scores[i], 
                             s=100, c=[colors[i]], label=name, alpha=0.8)
        
        axes[0, 2].set_xlabel('召回率')
        axes[0, 2].set_ylabel('精确率')
        axes[0, 2].set_title('精确率-召回率分布')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_xlim(0, 1)
        axes[0, 2].set_ylim(0, 1)
        
        # 处理时间比较
        processing_times = [results[name].get('processing_time', 0) for name in model_names]
        bars = axes[1, 0].bar(model_names, processing_times, color=colors, alpha=0.8)
        axes[1, 0].set_xlabel('模型')
        axes[1, 0].set_ylabel('处理时间 (秒)')
        axes[1, 0].set_title('处理时间比较')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        for bar, time in zip(bars, processing_times):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(processing_times)*0.01,
                           f'{time:.2f}s', ha='center', va='bottom')
        
        # 内存使用比较
        memory_usage = [results[name].get('memory_usage', 0) for name in model_names]
        bars = axes[1, 1].bar(model_names, memory_usage, color=colors, alpha=0.8)
        axes[1, 1].set_xlabel('模型')
        axes[1, 1].set_ylabel('内存使用 (MB)')
        axes[1, 1].set_title('内存使用比较')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        for bar, mem in zip(bars, memory_usage):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(memory_usage)*0.01,
                           f'{mem:.1f}MB', ha='center', va='bottom')
        
        # 综合性能雷达图
        categories = ['精确率', '召回率', 'F1分数', '定位精度X', '定位精度Y']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax_radar = plt.subplot(2, 3, 6, projection='polar')
        
        for i, name in enumerate(model_names):
            # 归一化定位精度（转换为正向指标）
            loc_precision_x = 1 / (1 + rmse_x[i] / 100)
            loc_precision_y = 1 / (1 + rmse_y[i] / 100)
            
            values = [precision_scores[i], recall_scores[i], f1_scores[i], 
                     loc_precision_x, loc_precision_y]
            values += values[:1]
            
            ax_radar.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i])
            ax_radar.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('综合性能雷达图', y=1.08)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        if save_path:
            save_plot(fig, save_path)
        
        return fig
    
    def create_interactive_dashboard(self,
                                   results: Dict[str, Dict[str, Any]],
                                   save_path: Optional[str] = None) -> go.Figure:
        """
        创建交互式仪表板
        
        Args:
            results: 模型结果字典
            save_path: 保存路径
            
        Returns:
            go.Figure: Plotly图形对象
        """
        model_names = list(results.keys())
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('检测指标', '定位误差', '光子数指标', '处理性能', '精确率-召回率', '综合雷达图'),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}, {"type": "scatterpolar"}]]
        )
        
        # 提取数据
        precision_scores = [results[name]['detection_metrics']['precision'] for name in model_names]
        recall_scores = [results[name]['detection_metrics']['recall'] for name in model_names]
        f1_scores = [results[name]['detection_metrics']['f1_score'] for name in model_names]
        rmse_x = [results[name]['localization_metrics']['rmse_x'] for name in model_names]
        rmse_y = [results[name]['localization_metrics']['rmse_y'] for name in model_names]
        rmse_z = [results[name]['localization_metrics']['rmse_z'] for name in model_names]
        
        # 检测指标
        fig.add_trace(go.Bar(name='精确率', x=model_names, y=precision_scores, 
                           marker_color='lightblue'), row=1, col=1)
        fig.add_trace(go.Bar(name='召回率', x=model_names, y=recall_scores, 
                           marker_color='lightgreen'), row=1, col=1)
        fig.add_trace(go.Bar(name='F1分数', x=model_names, y=f1_scores, 
                           marker_color='orange'), row=1, col=1)
        
        # 定位误差
        fig.add_trace(go.Bar(name='RMSE X', x=model_names, y=rmse_x, 
                           marker_color='red'), row=1, col=2)
        fig.add_trace(go.Bar(name='RMSE Y', x=model_names, y=rmse_y, 
                           marker_color='green'), row=1, col=2)
        fig.add_trace(go.Bar(name='RMSE Z', x=model_names, y=rmse_z, 
                           marker_color='blue'), row=1, col=2)
        
        # 光子数指标
        photon_rmse = [results[name]['photon_metrics']['rmse'] for name in model_names]
        photon_mae = [results[name]['photon_metrics']['mae'] for name in model_names]
        photon_corr = [results[name]['photon_metrics']['correlation'] for name in model_names]
        
        fig.add_trace(go.Bar(name='光子RMSE', x=model_names, y=photon_rmse, 
                           marker_color='purple'), row=1, col=3)
        fig.add_trace(go.Bar(name='光子MAE', x=model_names, y=photon_mae, 
                           marker_color='pink'), row=1, col=3)
        
        # 处理性能
        processing_times = [results[name].get('processing_time', 0) for name in model_names]
        memory_usage = [results[name].get('memory_usage', 0) for name in model_names]
        
        fig.add_trace(go.Bar(name='处理时间(s)', x=model_names, y=processing_times, 
                           marker_color='yellow'), row=2, col=1)
        fig.add_trace(go.Bar(name='内存使用(MB)', x=model_names, y=memory_usage, 
                           marker_color='cyan'), row=2, col=1)
        
        # 精确率-召回率散点图
        fig.add_trace(go.Scatter(x=recall_scores, y=precision_scores, 
                               mode='markers+text', text=model_names,
                               textposition='top center',
                               marker=dict(size=10, color=list(range(len(model_names))), 
                                         colorscale='viridis'),
                               name='模型性能'), row=2, col=2)
        
        # 综合雷达图
        categories = ['精确率', '召回率', 'F1分数', '定位精度X', '定位精度Y']
        
        for i, name in enumerate(model_names):
            loc_precision_x = 1 / (1 + rmse_x[i] / 100)
            loc_precision_y = 1 / (1 + rmse_y[i] / 100)
            
            values = [precision_scores[i], recall_scores[i], f1_scores[i], 
                     loc_precision_x, loc_precision_y]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=name
            ), row=2, col=3)
        
        # 更新布局
        fig.update_layout(
            title_text="模型评估交互式仪表板",
            showlegend=True,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"交互式仪表板已保存到: {save_path}")
        
        return fig


class PredictionVisualizer:
    """预测结果可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        初始化预测可视化器
        
        Args:
            figsize: 图形大小
        """
        self.figsize = figsize
        logger.info("预测结果可视化器初始化完成")
    
    def plot_prediction_comparison(self,
                                 image: np.ndarray,
                                 true_positions: np.ndarray,
                                 pred_positions: np.ndarray,
                                 true_photons: Optional[np.ndarray] = None,
                                 pred_photons: Optional[np.ndarray] = None,
                                 title: str = "预测结果比较",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制预测结果比较图
        
        Args:
            image: 输入图像
            true_positions: 真实位置
            pred_positions: 预测位置
            true_photons: 真实光子数
            pred_photons: 预测光子数
            title: 图表标题
            save_path: 保存路径
            
        Returns:
            plt.Figure: 图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # 原始图像
        im1 = axes[0, 0].imshow(image, cmap='hot', aspect='auto')
        axes[0, 0].set_title('原始图像')
        axes[0, 0].set_xlabel('X (像素)')
        axes[0, 0].set_ylabel('Y (像素)')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 真实位置标注
        im2 = axes[0, 1].imshow(image, cmap='gray', aspect='auto', alpha=0.7)
        if len(true_positions) > 0:
            axes[0, 1].scatter(true_positions[:, 0], true_positions[:, 1], 
                             c='green', s=50, marker='o', alpha=0.8, label='真实位置')
        axes[0, 1].set_title('真实位置')
        axes[0, 1].set_xlabel('X (像素)')
        axes[0, 1].set_ylabel('Y (像素)')
        axes[0, 1].legend()
        
        # 预测位置标注
        im3 = axes[1, 0].imshow(image, cmap='gray', aspect='auto', alpha=0.7)
        if len(pred_positions) > 0:
            axes[1, 0].scatter(pred_positions[:, 0], pred_positions[:, 1], 
                             c='red', s=50, marker='x', alpha=0.8, label='预测位置')
        axes[1, 0].set_title('预测位置')
        axes[1, 0].set_xlabel('X (像素)')
        axes[1, 0].set_ylabel('Y (像素)')
        axes[1, 0].legend()
        
        # 对比图
        im4 = axes[1, 1].imshow(image, cmap='gray', aspect='auto', alpha=0.7)
        if len(true_positions) > 0:
            axes[1, 1].scatter(true_positions[:, 0], true_positions[:, 1], 
                             c='green', s=50, marker='o', alpha=0.8, label='真实位置')
        if len(pred_positions) > 0:
            axes[1, 1].scatter(pred_positions[:, 0], pred_positions[:, 1], 
                             c='red', s=30, marker='x', alpha=0.8, label='预测位置')
        
        axes[1, 1].set_title('位置对比')
        axes[1, 1].set_xlabel('X (像素)')
        axes[1, 1].set_ylabel('Y (像素)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            save_plot(fig, save_path)
        
        return fig
    
    def plot_error_distribution(self,
                              errors: np.ndarray,
                              error_type: str = "定位误差",
                              bins: int = 50,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制误差分布图
        
        Args:
            errors: 误差数组
            error_type: 误差类型
            bins: 直方图箱数
            save_path: 保存路径
            
        Returns:
            plt.Figure: 图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(f'{error_type}分布分析', fontsize=16)
        
        # 直方图
        axes[0, 0].hist(errors, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('误差直方图')
        axes[0, 0].set_xlabel('误差值')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 累积分布
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[0, 1].plot(sorted_errors, cumulative, linewidth=2, color='red')
        axes[0, 1].set_title('累积分布函数')
        axes[0, 1].set_xlabel('误差值')
        axes[0, 1].set_ylabel('累积概率')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 箱线图
        axes[1, 0].boxplot(errors, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightgreen', alpha=0.7))
        axes[1, 0].set_title('误差箱线图')
        axes[1, 0].set_ylabel('误差值')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q图（与正态分布比较）
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q图 (正态分布)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_plot(fig, save_path)
        
        return fig