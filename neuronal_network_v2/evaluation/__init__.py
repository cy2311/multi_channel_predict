"""评估模块

该模块提供了DECODE神经网络的评估功能，包括：
- 模型性能评估
- 基准测试
- 结果可视化
- 性能分析
- 评估报告生成
"""

from .metrics import (
    DetectionMetrics,
    LocalizationMetrics,
    PhotonMetrics,
    ComprehensiveMetrics,
    calculate_detection_metrics,
    calculate_localization_metrics,
    calculate_photon_metrics,
    calculate_comprehensive_metrics
)

from .evaluator import (
    ModelEvaluator,
    BenchmarkEvaluator
)

from .visualizer import (
    MetricsVisualizer,
    PredictionVisualizer
)

from .benchmark import (
    StandardBenchmark
)

from .analyzer import (
    PerformanceAnalyzer
)

# Utils functions would be imported from individual modules as needed

__all__ = [
    # 指标计算
    'DetectionMetrics',
    'LocalizationMetrics', 
    'PhotonMetrics',
    'ComprehensiveMetrics',
    'calculate_detection_metrics',
    'calculate_localization_metrics',
    'calculate_photon_metrics',
    'calculate_comprehensive_metrics',
    
    # 评估器
    'ModelEvaluator',
    'BatchEvaluator',
    'CrossValidationEvaluator',
    'BenchmarkEvaluator',
    
    # 可视化
    'MetricsVisualizer',
    'LocalizationVisualizer',
    'ComparisonVisualizer',
    'InteractiveVisualizer',
    'plot_detection_metrics',
    'plot_localization_accuracy',
    'plot_photon_accuracy',
    'plot_comparison_results',
    
    # 基准测试
    'StandardBenchmark',
    'CustomBenchmark',
    'BenchmarkSuite',
    'run_standard_benchmark',
    'compare_models',
    'generate_benchmark_report',
    
    # 分析工具
    'PerformanceAnalyzer',
    'ErrorAnalyzer',
    'StatisticalAnalyzer',
    'TrendAnalyzer',
    'analyze_performance',
    'analyze_errors',
    'statistical_comparison',
    'trend_analysis',
    
    # 工具函数
    'match_emitters',
    'calculate_distances',
    'filter_by_distance',
    'bootstrap_confidence_interval',
    'statistical_significance_test',
    'load_ground_truth',
    'save_evaluation_results',
    'export_metrics_report'
]