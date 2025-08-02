"""评估器模块

该模块提供了模型评估功能，包括：
- 模型性能评估
- 指标计算和分析
- 结果可视化
- 基准测试
- 性能分析
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import logging
import json
from datetime import datetime
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from tqdm import tqdm

from .metrics import (
    DetectionMetrics, LocalizationMetrics, PhotonMetrics, ComprehensiveMetrics,
    calculate_detection_metrics, calculate_localization_metrics,
    calculate_photon_metrics, calculate_comprehensive_metrics,
    calculate_precision_recall_curve, calculate_roc_curve
)
from ..inference.infer import Infer
from ..inference.post_processing import PostProcessor
from ..utils.logging_utils import get_logger
from ..utils.visualization import plot_metrics, plot_comparison, save_plot
from ..utils.io_utils import write_json, create_directory

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """评估结果"""
    timestamp: str
    model_name: str
    dataset_name: str
    num_samples: int
    detection_metrics: DetectionMetrics
    localization_metrics: LocalizationMetrics
    photon_metrics: PhotonMetrics
    comprehensive_metrics: ComprehensiveMetrics
    processing_time: float
    memory_usage: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def save(self, filepath: str):
        """保存结果"""
        save_json(self.to_dict(), filepath)
        logger.info(f"评估结果已保存到: {filepath}")


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self,
                 model: nn.Module,
                 post_processor: Optional[PostProcessor] = None,
                 device: str = "auto"):
        """
        初始化评估器
        
        Args:
            model: 要评估的模型
            post_processor: 后处理器
            device: 计算设备
        """
        self.model = model
        self.post_processor = post_processor
        
        # 设置设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        # 创建推理器
        self.infer = Infer(model, post_processor, device=str(self.device))
        
        logger.info(f"模型评估器初始化完成，使用设备: {self.device}")
    
    def evaluate_dataset(self,
                        dataloader: DataLoader,
                        tolerance_xy: float = 250.0,
                        tolerance_z: float = 500.0,
                        min_photons: int = 50,
                        max_photons: int = 10000,
                        confidence_level: float = 0.95,
                        save_predictions: bool = False,
                        output_dir: Optional[str] = None) -> EvaluationResult:
        """
        评估数据集
        
        Args:
            dataloader: 数据加载器
            tolerance_xy: XY方向容差（nm）
            tolerance_z: Z方向容差（nm）
            min_photons: 最小光子数
            max_photons: 最大光子数
            confidence_level: 置信水平
            save_predictions: 是否保存预测结果
            output_dir: 输出目录
            
        Returns:
            EvaluationResult: 评估结果
        """
        logger.info("开始评估数据集")
        start_time = datetime.now()
        
        # 收集所有预测和真实值
        all_predictions = []
        all_targets = []
        all_images = []
        
        # 内存使用监控
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        else:
            initial_memory = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="评估进度")):
                try:
                    if isinstance(batch, (list, tuple)):
                        images = batch[0]
                        targets = batch[1] if len(batch) > 1 else None
                    else:
                        images = batch
                        targets = None
                    
                    # 移动到设备
                    images = images.to(self.device)
                    
                    # 推理
                    predictions = self.infer.infer_batch(images)
                    
                    # 收集结果
                    all_predictions.extend(predictions)
                    if targets is not None:
                        all_targets.extend(targets)
                    
                    if save_predictions:
                        all_images.extend(images.cpu().numpy())
                    
                except Exception as e:
                    logger.error(f"处理批次 {batch_idx} 时出错: {e}")
                    continue
        
        # 计算处理时间
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 计算内存使用
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            memory_usage = (peak_memory - initial_memory) / 1024 / 1024  # MB
        else:
            memory_usage = 0.0
        
        logger.info(f"数据集评估完成，处理了 {len(all_predictions)} 个样本")
        
        # 计算评估指标
        if all_targets:
            metrics = self._calculate_metrics(
                all_predictions, all_targets,
                tolerance_xy, tolerance_z,
                min_photons, max_photons,
                confidence_level
            )
        else:
            # 如果没有真实标签，创建空指标
            metrics = self._create_empty_metrics()
        
        # 创建评估结果
        result = EvaluationResult(
            timestamp=start_time.isoformat(),
            model_name=self.model.__class__.__name__,
            dataset_name=getattr(dataloader.dataset, 'name', 'Unknown'),
            num_samples=len(all_predictions),
            detection_metrics=metrics[0],
            localization_metrics=metrics[1],
            photon_metrics=metrics[2],
            comprehensive_metrics=metrics[3],
            processing_time=processing_time,
            memory_usage=memory_usage
        )
        
        # 保存结果
        if output_dir:
            self._save_evaluation_results(
                result, all_predictions, all_targets, all_images,
                output_dir, save_predictions
            )
        
        return result
    
    def _calculate_metrics(self,
                          predictions: List[Any],
                          targets: List[Any],
                          tolerance_xy: float,
                          tolerance_z: float,
                          min_photons: int,
                          max_photons: int,
                          confidence_level: float) -> Tuple[DetectionMetrics, LocalizationMetrics, PhotonMetrics, ComprehensiveMetrics]:
        """
        计算评估指标
        
        Args:
            predictions: 预测结果列表
            targets: 真实标签列表
            tolerance_xy: XY方向容差
            tolerance_z: Z方向容差
            min_photons: 最小光子数
            max_photons: 最大光子数
            confidence_level: 置信水平
            
        Returns:
            Tuple: 各种评估指标
        """
        logger.info("计算评估指标")
        
        # 提取位置和光子数信息
        pred_positions = []
        pred_photons = []
        true_positions = []
        true_photons = []
        
        for pred, target in zip(predictions, targets):
            # 处理预测结果
            if hasattr(pred, 'positions'):
                pred_positions.append(pred.positions)
                if hasattr(pred, 'photons'):
                    pred_photons.append(pred.photons)
                else:
                    pred_photons.append(np.ones(len(pred.positions)) * 1000)  # 默认光子数
            else:
                pred_positions.append(np.array([]).reshape(0, 3))
                pred_photons.append(np.array([]))
            
            # 处理真实标签
            if isinstance(target, dict):
                if 'positions' in target:
                    true_positions.append(target['positions'])
                else:
                    true_positions.append(np.array([]).reshape(0, 3))
                
                if 'photons' in target:
                    true_photons.append(target['photons'])
                else:
                    true_photons.append(np.ones(len(true_positions[-1])) * 1000)
            else:
                # 假设target是位置数组
                true_positions.append(target)
                true_photons.append(np.ones(len(target)) * 1000)
        
        # 计算检测指标
        detection_metrics = calculate_detection_metrics(
            pred_positions, true_positions,
            tolerance_xy, tolerance_z
        )
        
        # 计算定位指标
        localization_metrics = calculate_localization_metrics(
            pred_positions, true_positions,
            tolerance_xy, tolerance_z
        )
        
        # 计算光子数指标
        photon_metrics = calculate_photon_metrics(
            pred_photons, true_photons,
            pred_positions, true_positions,
            tolerance_xy, tolerance_z
        )
        
        # 计算综合指标
        comprehensive_metrics = calculate_comprehensive_metrics(
            detection_metrics, localization_metrics, photon_metrics
        )
        
        return detection_metrics, localization_metrics, photon_metrics, comprehensive_metrics
    
    def _create_empty_metrics(self) -> Tuple[DetectionMetrics, LocalizationMetrics, PhotonMetrics, ComprehensiveMetrics]:
        """创建空指标（当没有真实标签时）"""
        detection_metrics = DetectionMetrics(
            precision=0.0, recall=0.0, f1_score=0.0,
            true_positives=0, false_positives=0, false_negatives=0,
            jaccard_index=0.0
        )
        
        localization_metrics = LocalizationMetrics(
            rmse_x=0.0, rmse_y=0.0, rmse_z=0.0,
            mae_x=0.0, mae_y=0.0, mae_z=0.0,
            bias_x=0.0, bias_y=0.0, bias_z=0.0,
            std_x=0.0, std_y=0.0, std_z=0.0,
            lateral_precision=0.0, axial_precision=0.0
        )
        
        photon_metrics = PhotonMetrics(
            rmse=0.0, mae=0.0, bias=0.0, std=0.0,
            correlation=0.0, relative_error=0.0
        )
        
        comprehensive_metrics = ComprehensiveMetrics(
            efficiency=0.0, accuracy=0.0, precision=0.0,
            jaccard_index=0.0, crlb_ratio_x=0.0, crlb_ratio_y=0.0, crlb_ratio_z=0.0
        )
        
        return detection_metrics, localization_metrics, photon_metrics, comprehensive_metrics
    
    def _save_evaluation_results(self,
                               result: EvaluationResult,
                               predictions: List[Any],
                               targets: List[Any],
                               images: List[np.ndarray],
                               output_dir: str,
                               save_predictions: bool):
        """
        保存评估结果
        
        Args:
            result: 评估结果
            predictions: 预测结果
            targets: 真实标签
            images: 图像数据
            output_dir: 输出目录
            save_predictions: 是否保存预测结果
        """
        output_path = Path(output_dir)
        create_directory(str(output_path))
        
        # 保存评估指标
        result.save(str(output_path / "evaluation_metrics.json"))
        
        # 保存预测结果
        if save_predictions:
            pred_path = output_path / "predictions"
            create_directory(str(pred_path))
            
            for i, pred in enumerate(predictions):
                if hasattr(pred, 'to_dict'):
                    save_json(pred.to_dict(), str(pred_path / f"prediction_{i:06d}.json"))
                else:
                    np.save(str(pred_path / f"prediction_{i:06d}.npy"), pred)
        
        # 生成可视化报告
        self._generate_visualization_report(result, output_path)
        
        logger.info(f"评估结果已保存到: {output_dir}")
    
    def _generate_visualization_report(self, result: EvaluationResult, output_path: Path):
        """
        生成可视化报告
        
        Args:
            result: 评估结果
            output_path: 输出路径
        """
        try:
            # 创建指标可视化
            metrics_data = {
                '检测指标': {
                    '精确率': result.detection_metrics.precision,
                    '召回率': result.detection_metrics.recall,
                    'F1分数': result.detection_metrics.f1_score,
                    'Jaccard指数': result.detection_metrics.jaccard_index
                },
                '定位指标': {
                    'RMSE_X': result.localization_metrics.rmse_x,
                    'RMSE_Y': result.localization_metrics.rmse_y,
                    'RMSE_Z': result.localization_metrics.rmse_z,
                    '横向精度': result.localization_metrics.lateral_precision,
                    '轴向精度': result.localization_metrics.axial_precision
                },
                '光子数指标': {
                    'RMSE': result.photon_metrics.rmse,
                    'MAE': result.photon_metrics.mae,
                    '偏差': result.photon_metrics.bias,
                    '相关性': result.photon_metrics.correlation
                }
            }
            
            # 绘制指标图表
            fig = plot_metrics(metrics_data, plot_type='bar')
            save_plot(fig, str(output_path / "metrics_overview.png"))
            plt.close(fig)
            
            # 生成HTML报告
            self._generate_html_report(result, output_path)
            
        except Exception as e:
            logger.error(f"生成可视化报告失败: {e}")
    
    def _generate_html_report(self, result: EvaluationResult, output_path: Path):
        """
        生成HTML报告
        
        Args:
            result: 评估结果
            output_path: 输出路径
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>模型评估报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .metric-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .metric-title {{ font-weight: bold; color: #333; margin-bottom: 10px; }}
                .metric-value {{ font-size: 1.2em; color: #007bff; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>模型评估报告</h1>
                <p><strong>模型:</strong> {result.model_name}</p>
                <p><strong>数据集:</strong> {result.dataset_name}</p>
                <p><strong>评估时间:</strong> {result.timestamp}</p>
                <p><strong>样本数量:</strong> {result.num_samples}</p>
                <p><strong>处理时间:</strong> {result.processing_time:.2f} 秒</p>
                <p><strong>内存使用:</strong> {result.memory_usage:.2f} MB</p>
            </div>
            
            <div class="section">
                <h2>评估指标</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-title">检测指标</div>
                        <p>精确率: <span class="metric-value">{result.detection_metrics.precision:.4f}</span></p>
                        <p>召回率: <span class="metric-value">{result.detection_metrics.recall:.4f}</span></p>
                        <p>F1分数: <span class="metric-value">{result.detection_metrics.f1_score:.4f}</span></p>
                        <p>Jaccard指数: <span class="metric-value">{result.detection_metrics.jaccard_index:.4f}</span></p>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">定位指标</div>
                        <p>RMSE X: <span class="metric-value">{result.localization_metrics.rmse_x:.2f} nm</span></p>
                        <p>RMSE Y: <span class="metric-value">{result.localization_metrics.rmse_y:.2f} nm</span></p>
                        <p>RMSE Z: <span class="metric-value">{result.localization_metrics.rmse_z:.2f} nm</span></p>
                        <p>横向精度: <span class="metric-value">{result.localization_metrics.lateral_precision:.2f} nm</span></p>
                        <p>轴向精度: <span class="metric-value">{result.localization_metrics.axial_precision:.2f} nm</span></p>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">光子数指标</div>
                        <p>RMSE: <span class="metric-value">{result.photon_metrics.rmse:.2f}</span></p>
                        <p>MAE: <span class="metric-value">{result.photon_metrics.mae:.2f}</span></p>
                        <p>偏差: <span class="metric-value">{result.photon_metrics.bias:.2f}</span></p>
                        <p>相关性: <span class="metric-value">{result.photon_metrics.correlation:.4f}</span></p>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">综合指标</div>
                        <p>效率: <span class="metric-value">{result.comprehensive_metrics.efficiency:.4f}</span></p>
                        <p>准确率: <span class="metric-value">{result.comprehensive_metrics.accuracy:.4f}</span></p>
                        <p>精确度: <span class="metric-value">{result.comprehensive_metrics.precision:.4f}</span></p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>详细统计</h2>
                <table>
                    <tr><th>指标类别</th><th>指标名称</th><th>数值</th></tr>
                    <tr><td rowspan="4">检测</td><td>真正例</td><td>{result.detection_metrics.true_positives}</td></tr>
                    <tr><td>假正例</td><td>{result.detection_metrics.false_positives}</td></tr>
                    <tr><td>假负例</td><td>{result.detection_metrics.false_negatives}</td></tr>
                    <tr><td>Jaccard指数</td><td>{result.detection_metrics.jaccard_index:.4f}</td></tr>
                    
                    <tr><td rowspan="6">定位偏差</td><td>偏差 X</td><td>{result.localization_metrics.bias_x:.2f} nm</td></tr>
                    <tr><td>偏差 Y</td><td>{result.localization_metrics.bias_y:.2f} nm</td></tr>
                    <tr><td>偏差 Z</td><td>{result.localization_metrics.bias_z:.2f} nm</td></tr>
                    <tr><td>标准差 X</td><td>{result.localization_metrics.std_x:.2f} nm</td></tr>
                    <tr><td>标准差 Y</td><td>{result.localization_metrics.std_y:.2f} nm</td></tr>
                    <tr><td>标准差 Z</td><td>{result.localization_metrics.std_z:.2f} nm</td></tr>
                    
                    <tr><td rowspan="2">光子数</td><td>标准差</td><td>{result.photon_metrics.std:.2f}</td></tr>
                    <tr><td>相对误差</td><td>{result.photon_metrics.relative_error:.4f}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>可视化图表</h2>
                <img src="metrics_overview.png" alt="指标概览" style="max-width: 100%; height: auto;">
            </div>
        </body>
        </html>
        """
        
        with open(output_path / "evaluation_report.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def compare_models(self,
                      other_evaluators: List['ModelEvaluator'],
                      dataloader: DataLoader,
                      model_names: Optional[List[str]] = None,
                      output_dir: Optional[str] = None) -> Dict[str, EvaluationResult]:
        """
        比较多个模型
        
        Args:
            other_evaluators: 其他评估器列表
            dataloader: 数据加载器
            model_names: 模型名称列表
            output_dir: 输出目录
            
        Returns:
            Dict[str, EvaluationResult]: 各模型的评估结果
        """
        logger.info(f"开始比较 {len(other_evaluators) + 1} 个模型")
        
        all_evaluators = [self] + other_evaluators
        
        if model_names is None:
            model_names = [f"Model_{i}" for i in range(len(all_evaluators))]
        
        results = {}
        
        # 评估每个模型
        for evaluator, name in zip(all_evaluators, model_names):
            logger.info(f"评估模型: {name}")
            result = evaluator.evaluate_dataset(dataloader)
            results[name] = result
        
        # 生成比较报告
        if output_dir:
            self._generate_comparison_report(results, output_dir)
        
        return results
    
    def _generate_comparison_report(self, results: Dict[str, EvaluationResult], output_dir: str):
        """
        生成模型比较报告
        
        Args:
            results: 评估结果字典
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        create_directory(str(output_path))
        
        # 保存比较结果
        comparison_data = {name: result.to_dict() for name, result in results.items()}
        save_json(comparison_data, str(output_path / "model_comparison.json"))
        
        # 生成比较图表
        try:
            self._plot_model_comparison(results, output_path)
        except Exception as e:
            logger.error(f"生成比较图表失败: {e}")
        
        logger.info(f"模型比较报告已保存到: {output_dir}")
    
    def _plot_model_comparison(self, results: Dict[str, EvaluationResult], output_path: Path):
        """
        绘制模型比较图表
        
        Args:
            results: 评估结果字典
            output_path: 输出路径
        """
        model_names = list(results.keys())
        
        # 提取指标数据
        precision_scores = [results[name].detection_metrics.precision for name in model_names]
        recall_scores = [results[name].detection_metrics.recall for name in model_names]
        f1_scores = [results[name].detection_metrics.f1_score for name in model_names]
        rmse_x = [results[name].localization_metrics.rmse_x for name in model_names]
        rmse_y = [results[name].localization_metrics.rmse_y for name in model_names]
        
        # 创建比较图表
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('模型性能比较', fontsize=16)
        
        # 检测指标
        axes[0, 0].bar(model_names, precision_scores)
        axes[0, 0].set_title('精确率')
        axes[0, 0].set_ylabel('精确率')
        
        axes[0, 1].bar(model_names, recall_scores)
        axes[0, 1].set_title('召回率')
        axes[0, 1].set_ylabel('召回率')
        
        axes[0, 2].bar(model_names, f1_scores)
        axes[0, 2].set_title('F1分数')
        axes[0, 2].set_ylabel('F1分数')
        
        # 定位指标
        axes[1, 0].bar(model_names, rmse_x)
        axes[1, 0].set_title('RMSE X (nm)')
        axes[1, 0].set_ylabel('RMSE (nm)')
        
        axes[1, 1].bar(model_names, rmse_y)
        axes[1, 1].set_title('RMSE Y (nm)')
        axes[1, 1].set_ylabel('RMSE (nm)')
        
        # 综合性能雷达图
        axes[1, 2].remove()
        ax_radar = fig.add_subplot(2, 3, 6, projection='polar')
        
        # 归一化指标用于雷达图
        metrics = ['精确率', '召回率', 'F1分数', '定位精度X', '定位精度Y']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        for name in model_names:
            values = [
                results[name].detection_metrics.precision,
                results[name].detection_metrics.recall,
                results[name].detection_metrics.f1_score,
                1.0 / (1.0 + results[name].localization_metrics.rmse_x / 100),  # 归一化定位精度
                1.0 / (1.0 + results[name].localization_metrics.rmse_y / 100)
            ]
            values += values[:1]  # 闭合图形
            
            ax_radar.plot(angles, values, 'o-', linewidth=2, label=name)
            ax_radar.fill(angles, values, alpha=0.25)
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metrics)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('综合性能比较')
        ax_radar.legend()
        
        plt.tight_layout()
        save_plot(fig, str(output_path / "model_comparison.png"))
        plt.close(fig)


class BenchmarkEvaluator:
    """基准测试评估器"""
    
    def __init__(self):
        self.benchmark_results = {}
    
    def run_benchmark(self,
                     evaluator: ModelEvaluator,
                     test_cases: List[Dict[str, Any]],
                     output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        运行基准测试
        
        Args:
            evaluator: 模型评估器
            test_cases: 测试用例列表
            output_dir: 输出目录
            
        Returns:
            Dict[str, Any]: 基准测试结果
        """
        logger.info(f"开始基准测试，共 {len(test_cases)} 个测试用例")
        
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'model_name': evaluator.model.__class__.__name__,
            'test_cases': [],
            'summary': {}
        }
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"运行测试用例 {i+1}/{len(test_cases)}: {test_case.get('name', f'TestCase_{i}')}")
            
            try:
                # 运行测试用例
                result = evaluator.evaluate_dataset(
                    test_case['dataloader'],
                    **test_case.get('params', {})
                )
                
                test_result = {
                    'name': test_case.get('name', f'TestCase_{i}'),
                    'description': test_case.get('description', ''),
                    'result': result.to_dict(),
                    'status': 'success'
                }
                
            except Exception as e:
                logger.error(f"测试用例 {i} 执行失败: {e}")
                test_result = {
                    'name': test_case.get('name', f'TestCase_{i}'),
                    'description': test_case.get('description', ''),
                    'error': str(e),
                    'status': 'failed'
                }
            
            benchmark_results['test_cases'].append(test_result)
        
        # 计算汇总统计
        successful_cases = [case for case in benchmark_results['test_cases'] if case['status'] == 'success']
        
        if successful_cases:
            avg_precision = np.mean([case['result']['detection_metrics']['precision'] for case in successful_cases])
            avg_recall = np.mean([case['result']['detection_metrics']['recall'] for case in successful_cases])
            avg_f1 = np.mean([case['result']['detection_metrics']['f1_score'] for case in successful_cases])
            avg_rmse_x = np.mean([case['result']['localization_metrics']['rmse_x'] for case in successful_cases])
            avg_rmse_y = np.mean([case['result']['localization_metrics']['rmse_y'] for case in successful_cases])
            
            benchmark_results['summary'] = {
                'total_cases': len(test_cases),
                'successful_cases': len(successful_cases),
                'failed_cases': len(test_cases) - len(successful_cases),
                'average_precision': avg_precision,
                'average_recall': avg_recall,
                'average_f1_score': avg_f1,
                'average_rmse_x': avg_rmse_x,
                'average_rmse_y': avg_rmse_y
            }
        
        # 保存结果
        if output_dir:
            output_path = Path(output_dir)
            create_directory(str(output_path))
            save_json(benchmark_results, str(output_path / "benchmark_results.json"))
            logger.info(f"基准测试结果已保存到: {output_dir}")
        
        self.benchmark_results = benchmark_results
        return benchmark_results