"""基准测试模块

该模块提供了标准化的基准测试功能，包括：
- 标准基准测试套件
- 性能基准测试
- 跨平台基准测试
- 基准测试结果管理
- 基准测试报告生成
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
from datetime import datetime
import platform
import psutil
import matplotlib.pyplot as plt
from tqdm import tqdm

from .evaluator import ModelEvaluator, EvaluationResult
from .analyzer import PerformanceAnalyzer, PerformanceProfile
from ..utils.logging_utils import get_logger
from ..utils.io_utils import write_json, create_directory
from ..utils.visualization import save_plot
from ..utils.data_utils import normalize_data

logger = get_logger(__name__)


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    name: str
    description: str
    test_cases: List[Dict[str, Any]]
    metrics: List[str]
    tolerance_xy: float = 250.0
    tolerance_z: float = 500.0
    min_photons: int = 50
    max_photons: int = 10000
    confidence_level: float = 0.95
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    benchmark_name: str
    model_name: str
    timestamp: str
    system_info: Dict[str, Any]
    test_results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    performance_profile: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def save(self, filepath: str):
        """保存结果"""
        save_json(self.to_dict(), filepath)
        logger.info(f"基准测试结果已保存到: {filepath}")


class StandardBenchmark:
    """标准基准测试"""
    
    def __init__(self):
        """
        初始化标准基准测试
        """
        self.benchmark_configs = {}
        self.results_history = []
        self._load_standard_benchmarks()
        logger.info("标准基准测试初始化完成")
    
    def _load_standard_benchmarks(self):
        """
        加载标准基准测试配置
        """
        # SMLM基础基准测试
        self.benchmark_configs['smlm_basic'] = BenchmarkConfig(
            name="SMLM基础基准测试",
            description="单分子定位显微镜的基础性能测试",
            test_cases=[
                {
                    'name': '低密度检测',
                    'description': '测试低发射器密度下的检测性能',
                    'emitter_density': 0.1,  # 每平方微米
                    'image_size': (64, 64),
                    'num_frames': 100,
                    'snr_range': (5, 20),
                    'background_level': 100
                },
                {
                    'name': '中密度检测',
                    'description': '测试中等发射器密度下的检测性能',
                    'emitter_density': 0.5,
                    'image_size': (64, 64),
                    'num_frames': 100,
                    'snr_range': (3, 15),
                    'background_level': 150
                },
                {
                    'name': '高密度检测',
                    'description': '测试高发射器密度下的检测性能',
                    'emitter_density': 1.0,
                    'image_size': (64, 64),
                    'num_frames': 100,
                    'snr_range': (2, 10),
                    'background_level': 200
                }
            ],
            metrics=['precision', 'recall', 'f1_score', 'rmse_x', 'rmse_y', 'rmse_z']
        )
        
        # 3D SMLM基准测试
        self.benchmark_configs['smlm_3d'] = BenchmarkConfig(
            name="3D SMLM基准测试",
            description="三维单分子定位显微镜性能测试",
            test_cases=[
                {
                    'name': '3D定位精度',
                    'description': '测试三维定位精度',
                    'emitter_density': 0.3,
                    'image_size': (64, 64),
                    'num_frames': 200,
                    'z_range': (-1000, 1000),  # nm
                    'snr_range': (3, 15),
                    'background_level': 120
                },
                {
                    'name': '深度范围测试',
                    'description': '测试不同深度范围的性能',
                    'emitter_density': 0.2,
                    'image_size': (64, 64),
                    'num_frames': 150,
                    'z_range': (-2000, 2000),
                    'snr_range': (4, 12),
                    'background_level': 100
                }
            ],
            metrics=['precision', 'recall', 'rmse_x', 'rmse_y', 'rmse_z', 'axial_precision']
        )
        
        # 光子数估计基准测试
        self.benchmark_configs['photon_estimation'] = BenchmarkConfig(
            name="光子数估计基准测试",
            description="光子数估计精度测试",
            test_cases=[
                {
                    'name': '低光子数',
                    'description': '测试低光子数条件下的估计精度',
                    'photon_range': (50, 500),
                    'emitter_density': 0.2,
                    'image_size': (64, 64),
                    'num_frames': 100,
                    'background_level': 100
                },
                {
                    'name': '高光子数',
                    'description': '测试高光子数条件下的估计精度',
                    'photon_range': (1000, 10000),
                    'emitter_density': 0.3,
                    'image_size': (64, 64),
                    'num_frames': 100,
                    'background_level': 150
                }
            ],
            metrics=['photon_rmse', 'photon_mae', 'photon_correlation']
        )
        
        # 性能基准测试
        self.benchmark_configs['performance'] = BenchmarkConfig(
            name="性能基准测试",
            description="计算性能和资源使用测试",
            test_cases=[
                {
                    'name': '小图像处理',
                    'description': '测试小尺寸图像的处理速度',
                    'image_size': (32, 32),
                    'batch_size': 32,
                    'num_batches': 50
                },
                {
                    'name': '中等图像处理',
                    'description': '测试中等尺寸图像的处理速度',
                    'image_size': (64, 64),
                    'batch_size': 16,
                    'num_batches': 50
                },
                {
                    'name': '大图像处理',
                    'description': '测试大尺寸图像的处理速度',
                    'image_size': (128, 128),
                    'batch_size': 8,
                    'num_batches': 50
                }
            ],
            metrics=['processing_time', 'memory_usage', 'throughput']
        )
    
    def run_benchmark(self,
                     model: nn.Module,
                     benchmark_name: str,
                     data_generator: Optional[Callable] = None,
                     output_dir: Optional[str] = None) -> BenchmarkResult:
        """
        运行基准测试
        
        Args:
            model: 要测试的模型
            benchmark_name: 基准测试名称
            data_generator: 数据生成器函数
            output_dir: 输出目录
            
        Returns:
            BenchmarkResult: 基准测试结果
        """
        if benchmark_name not in self.benchmark_configs:
            raise ValueError(f"未知的基准测试: {benchmark_name}")
        
        config = self.benchmark_configs[benchmark_name]
        logger.info(f"开始运行基准测试: {config.name}")
        
        # 获取系统信息
        system_info = self._get_system_info()
        
        # 创建评估器
        evaluator = ModelEvaluator(model)
        
        # 运行测试用例
        test_results = []
        for test_case in config.test_cases:
            logger.info(f"运行测试用例: {test_case['name']}")
            
            try:
                # 生成测试数据
                if data_generator:
                    dataloader = data_generator(test_case)
                else:
                    dataloader = self._generate_test_data(test_case)
                
                # 运行评估
                if benchmark_name == 'performance':
                    result = self._run_performance_test(model, test_case)
                else:
                    result = evaluator.evaluate_dataset(
                        dataloader,
                        tolerance_xy=config.tolerance_xy,
                        tolerance_z=config.tolerance_z,
                        min_photons=config.min_photons,
                        max_photons=config.max_photons,
                        confidence_level=config.confidence_level
                    )
                
                test_results.append({
                    'test_case': test_case,
                    'result': result.to_dict() if hasattr(result, 'to_dict') else result,
                    'status': 'success'
                })
                
            except Exception as e:
                logger.error(f"测试用例 {test_case['name']} 失败: {e}")
                test_results.append({
                    'test_case': test_case,
                    'error': str(e),
                    'status': 'failed'
                })
        
        # 计算汇总统计
        summary = self._calculate_benchmark_summary(test_results, config.metrics)
        
        # 性能分析
        analyzer = PerformanceAnalyzer()
        performance_profile = None
        if test_results and test_results[0]['status'] == 'success':
            try:
                profile = analyzer.analyze_performance(test_results[0]['result'])
                performance_profile = profile.to_dict()
            except Exception as e:
                logger.warning(f"性能分析失败: {e}")
        
        # 创建基准测试结果
        benchmark_result = BenchmarkResult(
            benchmark_name=config.name,
            model_name=model.__class__.__name__,
            timestamp=datetime.now().isoformat(),
            system_info=system_info,
            test_results=test_results,
            summary=summary,
            performance_profile=performance_profile
        )
        
        # 保存结果
        if output_dir:
            self._save_benchmark_results(benchmark_result, output_dir)
        
        self.results_history.append(benchmark_result)
        logger.info(f"基准测试完成: {config.name}")
        
        return benchmark_result
    
    def _get_system_info(self) -> Dict[str, Any]:
        """
        获取系统信息
        
        Returns:
            Dict[str, Any]: 系统信息
        """
        system_info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'pytorch_version': torch.__version__,
        }
        
        # GPU信息
        if torch.cuda.is_available():
            system_info.update({
                'cuda_available': True,
                'cuda_version': torch.version.cuda,
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'Unknown'
            })
        else:
            system_info['cuda_available'] = False
        
        return system_info
    
    def _generate_test_data(self, test_case: Dict[str, Any]) -> DataLoader:
        """
        生成测试数据
        
        Args:
            test_case: 测试用例配置
            
        Returns:
            DataLoader: 数据加载器
        """
        # 简化的数据生成（实际应用中应该使用更复杂的模拟）
        image_size = test_case.get('image_size', (64, 64))
        num_frames = test_case.get('num_frames', 100)
        batch_size = test_case.get('batch_size', 16)
        
        # 生成随机图像数据
        images = torch.randn(num_frames, 1, image_size[0], image_size[1])
        
        # 生成随机目标（位置和光子数）
        targets = []
        for _ in range(num_frames):
            num_emitters = np.random.poisson(test_case.get('emitter_density', 0.3) * np.prod(image_size) / 100)
            positions = np.random.rand(num_emitters, 3) * [image_size[0], image_size[1], 2000] - [0, 0, 1000]
            photons = np.random.uniform(
                test_case.get('photon_range', [500, 5000])[0],
                test_case.get('photon_range', [500, 5000])[1],
                num_emitters
            )
            targets.append({
                'positions': positions,
                'photons': photons
            })
        
        dataset = TensorDataset(images)
        # 手动添加targets到dataset
        dataset.targets = targets
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    def _run_performance_test(self, model: nn.Module, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行性能测试
        
        Args:
            model: 模型
            test_case: 测试用例
            
        Returns:
            Dict[str, Any]: 性能测试结果
        """
        image_size = test_case['image_size']
        batch_size = test_case['batch_size']
        num_batches = test_case['num_batches']
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # 预热
        dummy_input = torch.randn(batch_size, 1, image_size[0], image_size[1]).to(device)
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input)
        
        # 性能测试
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            for _ in tqdm(range(num_batches), desc="性能测试"):
                batch_input = torch.randn(batch_size, 1, image_size[0], image_size[1]).to(device)
                _ = model(batch_input)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 计算指标
        total_samples = num_batches * batch_size
        throughput = total_samples / processing_time
        
        # 内存使用
        if torch.cuda.is_available():
            memory_usage = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        else:
            memory_usage = 0
        
        return {
            'processing_time': processing_time,
            'memory_usage': memory_usage,
            'throughput': throughput,
            'samples_per_second': throughput,
            'total_samples': total_samples,
            'batch_size': batch_size,
            'image_size': image_size
        }
    
    def _calculate_benchmark_summary(self,
                                   test_results: List[Dict[str, Any]],
                                   metrics: List[str]) -> Dict[str, Any]:
        """
        计算基准测试汇总
        
        Args:
            test_results: 测试结果列表
            metrics: 指标列表
            
        Returns:
            Dict[str, Any]: 汇总统计
        """
        successful_results = [r for r in test_results if r['status'] == 'success']
        
        if not successful_results:
            return {
                'total_tests': len(test_results),
                'successful_tests': 0,
                'failed_tests': len(test_results),
                'success_rate': 0.0
            }
        
        summary = {
            'total_tests': len(test_results),
            'successful_tests': len(successful_results),
            'failed_tests': len(test_results) - len(successful_results),
            'success_rate': len(successful_results) / len(test_results)
        }
        
        # 计算各指标的统计
        for metric in metrics:
            values = []
            for result in successful_results:
                value = self._extract_metric_value(result['result'], metric)
                if value is not None:
                    values.append(value)
            
            if values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_min'] = np.min(values)
                summary[f'{metric}_max'] = np.max(values)
                summary[f'{metric}_median'] = np.median(values)
        
        return summary
    
    def _extract_metric_value(self, result: Dict[str, Any], metric: str) -> Optional[float]:
        """
        从结果中提取指标值
        
        Args:
            result: 结果字典
            metric: 指标名称
            
        Returns:
            Optional[float]: 指标值
        """
        # 检测指标
        if metric in ['precision', 'recall', 'f1_score']:
            return result.get('detection_metrics', {}).get(metric)
        
        # 定位指标
        if metric in ['rmse_x', 'rmse_y', 'rmse_z', 'lateral_precision', 'axial_precision']:
            return result.get('localization_metrics', {}).get(metric)
        
        # 光子数指标
        if metric.startswith('photon_'):
            photon_metric = metric.replace('photon_', '')
            return result.get('photon_metrics', {}).get(photon_metric)
        
        # 性能指标
        if metric in ['processing_time', 'memory_usage', 'throughput']:
            return result.get(metric)
        
        return None
    
    def _save_benchmark_results(self, result: BenchmarkResult, output_dir: str):
        """
        保存基准测试结果
        
        Args:
            result: 基准测试结果
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        create_directory(str(output_path))
        
        # 保存JSON结果
        result.save(str(output_path / "benchmark_result.json"))
        
        # 生成可视化报告
        self._generate_benchmark_report(result, output_path)
        
        logger.info(f"基准测试结果已保存到: {output_dir}")
    
    def _generate_benchmark_report(self, result: BenchmarkResult, output_path: Path):
        """
        生成基准测试报告
        
        Args:
            result: 基准测试结果
            output_path: 输出路径
        """
        # 生成图表
        self._generate_benchmark_plots(result, output_path)
        
        # 生成HTML报告
        self._generate_html_benchmark_report(result, output_path)
    
    def _generate_benchmark_plots(self, result: BenchmarkResult, output_path: Path):
        """
        生成基准测试图表
        
        Args:
            result: 基准测试结果
            output_path: 输出路径
        """
        successful_results = [r for r in result.test_results if r['status'] == 'success']
        
        if not successful_results:
            return
        
        # 测试用例性能对比
        test_names = [r['test_case']['name'] for r in successful_results]
        
        # 提取关键指标
        metrics_to_plot = ['precision', 'recall', 'f1_score', 'rmse_x', 'rmse_y']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{result.benchmark_name} - 测试结果', fontsize=16)
        
        for i, metric in enumerate(metrics_to_plot):
            if i >= 5:  # 只绘制前5个指标
                break
            
            row, col = i // 3, i % 3
            
            values = []
            for r in successful_results:
                value = self._extract_metric_value(r['result'], metric)
                values.append(value if value is not None else 0)
            
            if any(v != 0 for v in values):
                bars = axes[row, col].bar(test_names, values, alpha=0.7)
                axes[row, col].set_title(metric.replace('_', ' ').title())
                axes[row, col].tick_params(axis='x', rotation=45)
                
                # 添加数值标签
                for bar, value in zip(bars, values):
                    if value > 0:
                        axes[row, col].text(bar.get_x() + bar.get_width()/2, 
                                           bar.get_height() + max(values)*0.01,
                                           f'{value:.3f}', ha='center', va='bottom')
        
        # 隐藏空的子图
        if len(metrics_to_plot) < 6:
            axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        save_plot(fig, str(output_path / "benchmark_metrics.png"))
        plt.close(fig)
        
        # 成功率饼图
        fig, ax = plt.subplots(figsize=(8, 8))
        
        success_count = len(successful_results)
        fail_count = len(result.test_results) - success_count
        
        if fail_count > 0:
            labels = ['成功', '失败']
            sizes = [success_count, fail_count]
            colors = ['lightgreen', 'lightcoral']
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        else:
            ax.text(0.5, 0.5, '所有测试通过\n✓', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=20, color='green')
        
        ax.set_title('测试成功率')
        save_plot(fig, str(output_path / "test_success_rate.png"))
        plt.close(fig)
    
    def _generate_html_benchmark_report(self, result: BenchmarkResult, output_path: Path):
        """
        生成HTML基准测试报告
        
        Args:
            result: 基准测试结果
            output_path: 输出路径
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>基准测试报告 - {result.benchmark_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
                .summary-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; text-align: center; }}
                .summary-value {{ font-size: 1.5em; font-weight: bold; color: #007bff; }}
                .test-case {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
                .success {{ border-left: 5px solid #28a745; }}
                .failed {{ border-left: 5px solid #dc3545; }}
                .chart {{ text-align: center; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .system-info {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>基准测试报告</h1>
                <h2>{result.benchmark_name}</h2>
                <p><strong>模型:</strong> {result.model_name}</p>
                <p><strong>测试时间:</strong> {result.timestamp}</p>
            </div>
            
            <div class="section">
                <h2>测试概览</h2>
                <div class="summary-grid">
                    <div class="summary-card">
                        <div>总测试数</div>
                        <div class="summary-value">{result.summary['total_tests']}</div>
                    </div>
                    <div class="summary-card">
                        <div>成功测试</div>
                        <div class="summary-value">{result.summary['successful_tests']}</div>
                    </div>
                    <div class="summary-card">
                        <div>失败测试</div>
                        <div class="summary-value">{result.summary['failed_tests']}</div>
                    </div>
                    <div class="summary-card">
                        <div>成功率</div>
                        <div class="summary-value">{result.summary['success_rate']:.1%}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>测试结果可视化</h2>
                <div class="chart">
                    <img src="benchmark_metrics.png" alt="基准测试指标" style="max-width: 100%; height: auto;">
                </div>
                <div class="chart">
                    <img src="test_success_rate.png" alt="测试成功率" style="max-width: 100%; height: auto;">
                </div>
            </div>
            
            <div class="section">
                <h2>详细测试结果</h2>
        """
        
        # 添加每个测试用例的详细结果
        for i, test_result in enumerate(result.test_results):
            test_case = test_result['test_case']
            status_class = 'success' if test_result['status'] == 'success' else 'failed'
            status_icon = '✓' if test_result['status'] == 'success' else '✗'
            
            html_content += f"""
                <div class="test-case {status_class}">
                    <h3>{status_icon} {test_case['name']}</h3>
                    <p><strong>描述:</strong> {test_case['description']}</p>
                    <p><strong>状态:</strong> {test_result['status']}</p>
            """
            
            if test_result['status'] == 'success' and 'result' in test_result:
                # 显示关键指标
                result_data = test_result['result']
                if 'detection_metrics' in result_data:
                    dm = result_data['detection_metrics']
                    html_content += f"""
                        <p><strong>检测指标:</strong> 
                        精确率={dm.get('precision', 0):.3f}, 
                        召回率={dm.get('recall', 0):.3f}, 
                        F1分数={dm.get('f1_score', 0):.3f}</p>
                    """
                
                if 'localization_metrics' in result_data:
                    lm = result_data['localization_metrics']
                    html_content += f"""
                        <p><strong>定位指标:</strong> 
                        RMSE_X={lm.get('rmse_x', 0):.1f}nm, 
                        RMSE_Y={lm.get('rmse_y', 0):.1f}nm, 
                        RMSE_Z={lm.get('rmse_z', 0):.1f}nm</p>
                    """
            
            elif test_result['status'] == 'failed':
                html_content += f"<p><strong>错误:</strong> {test_result.get('error', '未知错误')}</p>"
            
            html_content += "</div>"
        
        # 系统信息
        html_content += f"""
            </div>
            
            <div class="section">
                <h2>系统信息</h2>
                <div class="system-info">
                    <table>
                        <tr><th>项目</th><th>值</th></tr>
                        <tr><td>操作系统</td><td>{result.system_info['platform']}</td></tr>
                        <tr><td>处理器</td><td>{result.system_info['processor']}</td></tr>
                        <tr><td>CPU核心数</td><td>{result.system_info['cpu_count']}</td></tr>
                        <tr><td>内存总量</td><td>{result.system_info['memory_total']:.1f} GB</td></tr>
                        <tr><td>Python版本</td><td>{result.system_info['python_version']}</td></tr>
                        <tr><td>PyTorch版本</td><td>{result.system_info['pytorch_version']}</td></tr>
                        <tr><td>CUDA可用</td><td>{'是' if result.system_info['cuda_available'] else '否'}</td></tr>
        """
        
        if result.system_info['cuda_available']:
            html_content += f"""
                        <tr><td>CUDA版本</td><td>{result.system_info.get('cuda_version', 'N/A')}</td></tr>
                        <tr><td>GPU数量</td><td>{result.system_info.get('gpu_count', 0)}</td></tr>
                        <tr><td>GPU型号</td><td>{result.system_info.get('gpu_name', 'Unknown')}</td></tr>
            """
        
        html_content += """
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        
        report_path = output_path / "benchmark_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"基准测试HTML报告已生成: {report_path}")
    
    def compare_benchmarks(self,
                         results: List[BenchmarkResult],
                         output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        比较多个基准测试结果
        
        Args:
            results: 基准测试结果列表
            output_dir: 输出目录
            
        Returns:
            Dict[str, Any]: 比较结果
        """
        logger.info(f"开始比较 {len(results)} 个基准测试结果")
        
        if len(results) < 2:
            return {'message': '需要至少2个基准测试结果进行比较'}
        
        # 提取比较数据
        comparison_data = {
            'models': [r.model_name for r in results],
            'benchmarks': [r.benchmark_name for r in results],
            'timestamps': [r.timestamp for r in results],
            'success_rates': [r.summary['success_rate'] for r in results]
        }
        
        # 找出最佳性能
        best_success_rate_idx = np.argmax(comparison_data['success_rates'])
        best_model = results[best_success_rate_idx]
        
        comparison_result = {
            'comparison_timestamp': datetime.now().isoformat(),
            'total_results': len(results),
            'best_model': {
                'name': best_model.model_name,
                'benchmark': best_model.benchmark_name,
                'success_rate': best_model.summary['success_rate']
            },
            'comparison_data': comparison_data,
            'summary': {
                'avg_success_rate': np.mean(comparison_data['success_rates']),
                'std_success_rate': np.std(comparison_data['success_rates']),
                'min_success_rate': np.min(comparison_data['success_rates']),
                'max_success_rate': np.max(comparison_data['success_rates'])
            }
        }
        
        # 保存比较结果
        if output_dir:
            output_path = Path(output_dir)
            create_directory(str(output_path))
            save_json(comparison_result, str(output_path / "benchmark_comparison.json"))
            
            # 生成比较图表
            self._generate_comparison_plots(results, output_path)
            
            logger.info(f"基准测试比较结果已保存到: {output_dir}")
        
        return comparison_result
    
    def _generate_comparison_plots(self, results: List[BenchmarkResult], output_path: Path):
        """
        生成比较图表
        
        Args:
            results: 基准测试结果列表
            output_path: 输出路径
        """
        model_names = [r.model_name for r in results]
        success_rates = [r.summary['success_rate'] for r in results]
        
        # 成功率比较
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(model_names, success_rates, alpha=0.7, color='skyblue')
        ax.set_ylabel('成功率')
        ax.set_title('模型基准测试成功率比较')
        ax.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, rate in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{rate:.1%}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_plot(fig, str(output_path / "success_rate_comparison.png"))
        plt.close(fig)
        
        logger.info("基准测试比较图表已生成")
    
    def get_available_benchmarks(self) -> List[str]:
        """
        获取可用的基准测试列表
        
        Returns:
            List[str]: 基准测试名称列表
        """
        return list(self.benchmark_configs.keys())
    
    def get_benchmark_config(self, benchmark_name: str) -> Optional[BenchmarkConfig]:
        """
        获取基准测试配置
        
        Args:
            benchmark_name: 基准测试名称
            
        Returns:
            Optional[BenchmarkConfig]: 基准测试配置
        """
        return self.benchmark_configs.get(benchmark_name)