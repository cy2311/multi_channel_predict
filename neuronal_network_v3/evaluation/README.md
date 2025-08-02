# Evaluation 评估模块

本模块包含DECODE神经网络v3的全面评估系统，支持单通道和多通道SMLM数据的性能评估、不确定性分析和可视化。

## 📋 模块概览

### 核心组件

#### 🔹 MultiChannelEvaluation (`multi_channel_evaluation.py`)
- **功能**: 多通道系统综合评估
- **特点**:
  - 双通道性能评估
  - 比例预测评估
  - 物理约束验证
  - 不确定性校准分析
- **用途**: 多通道模型的全面性能评估

#### 🔹 ModelEvaluator (`evaluator.py`)
- **功能**: 基础模型评估器
- **特点**:
  - 标准评估指标
  - 批量评估处理
  - 可配置评估流程
  - 结果统计分析
- **用途**: 单通道模型和基础评估任务

#### 🔹 ResultAnalyzer (`analyzer.py`)
- **功能**: 结果深度分析
- **特点**:
  - 误差分析
  - 性能瓶颈识别
  - 数据分布分析
  - 相关性分析
- **用途**: 模型性能的深入理解和优化指导

#### 🔹 BenchmarkEvaluator (`benchmark.py`)
- **功能**: 基准测试和对比
- **特点**:
  - 多模型对比
  - 标准数据集评估
  - 性能基准建立
  - 回归测试
- **用途**: 模型性能基准测试和版本对比

#### 🔹 EvaluationMetrics (`metrics.py`)
- **功能**: 评估指标计算
- **特点**:
  - 丰富的评估指标
  - 自定义指标支持
  - 统计显著性测试
  - 置信区间计算
- **用途**: 标准化的性能指标计算

#### 🔹 ResultVisualizer (`visualizer.py`)
- **功能**: 结果可视化
- **特点**:
  - 多维度可视化
  - 交互式图表
  - 自动报告生成
  - 自定义可视化
- **用途**: 评估结果的直观展示和分析

## 🚀 使用示例

### 多通道评估

```python
from neuronal_network_v3.evaluation import MultiChannelEvaluation
from neuronal_network_v3.inference import MultiChannelInfer

# 初始化评估器
evaluator = MultiChannelEvaluation(
    device='cuda',
    save_plots=True,
    plot_dir='evaluation_plots/'
)

# 加载测试数据和模型预测
test_data = load_test_data('data/test_multi_channel.h5')
predictions = load_predictions('results/multi_channel_predictions.h5')

# 执行评估
metrics = evaluator.evaluate(
    predictions=predictions,
    ground_truth=test_data,
    compute_uncertainty_metrics=True,
    validate_physics_constraints=True
)

# 打印主要指标
print("=== 多通道评估结果 ===")
print(f"通道1 RMSE: {metrics['channel1']['rmse']:.4f}")
print(f"通道2 RMSE: {metrics['channel2']['rmse']:.4f}")
print(f"比例预测 MAE: {metrics['ratio']['mae']:.4f}")
print(f"光子数守恒误差: {metrics['conservation']['error']:.4f}")
print(f"不确定性校准误差: {metrics['uncertainty']['calibration_error']:.4f}")

# 生成详细报告
report = evaluator.generate_report(metrics, save_path='evaluation_report.html')
```

### 单通道评估

```python
from neuronal_network_v3.evaluation import ModelEvaluator

# 初始化评估器
evaluator = ModelEvaluator(
    metrics=['rmse', 'mae', 'jaccard', 'precision', 'recall'],
    confidence_level=0.95
)

# 评估模型
results = evaluator.evaluate_model(
    model=trained_model,
    test_loader=test_dataloader,
    return_predictions=True
)

# 查看结果
for metric, value in results['metrics'].items():
    ci_lower, ci_upper = results['confidence_intervals'][metric]
    print(f"{metric}: {value:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
```

### 基准测试

```python
from neuronal_network_v3.evaluation import BenchmarkEvaluator

# 初始化基准测试
benchmark = BenchmarkEvaluator(
    benchmark_datasets=['standard_test', 'high_density', 'low_snr'],
    reference_methods=['decode_v2', 'thunderstorm', 'rapidstorm']
)

# 添加待测试模型
benchmark.add_model('decode_v3_single', single_channel_model)
benchmark.add_model('decode_v3_multi', multi_channel_model)

# 运行基准测试
benchmark_results = benchmark.run_benchmark(
    save_results=True,
    output_dir='benchmark_results/'
)

# 生成对比报告
benchmark.generate_comparison_report(
    results=benchmark_results,
    save_path='benchmark_report.pdf'
)
```

### 不确定性评估

```python
from neuronal_network_v3.evaluation.metrics import UncertaintyMetrics

# 初始化不确定性评估
unc_metrics = UncertaintyMetrics()

# 计算校准指标
calibration_results = unc_metrics.compute_calibration(
    predictions=pred_mean,
    uncertainties=pred_std,
    targets=ground_truth,
    num_bins=10
)

print(f"期望校准误差 (ECE): {calibration_results['ece']:.4f}")
print(f"最大校准误差 (MCE): {calibration_results['mce']:.4f}")
print(f"Brier分数: {calibration_results['brier_score']:.4f}")

# 计算覆盖率
coverage_results = unc_metrics.compute_coverage(
    predictions=pred_mean,
    uncertainties=pred_std,
    targets=ground_truth,
    confidence_levels=[0.68, 0.95, 0.99]
)

for level, coverage in coverage_results.items():
    print(f"{level*100}% 置信区间覆盖率: {coverage:.3f}")
```

### 结果可视化

```python
from neuronal_network_v3.evaluation import ResultVisualizer

# 初始化可视化器
visualizer = ResultVisualizer(
    style='publication',
    dpi=300,
    save_format='both'  # 保存为PNG和PDF
)

# 创建评估图表
fig = visualizer.create_evaluation_dashboard(
    metrics=evaluation_metrics,
    predictions=model_predictions,
    ground_truth=test_targets
)

# 保存图表
visualizer.save_figure(fig, 'evaluation_dashboard.png')

# 创建不确定性可视化
unc_fig = visualizer.plot_uncertainty_analysis(
    predictions=pred_mean,
    uncertainties=pred_std,
    targets=ground_truth
)

# 创建比例预测可视化（多通道）
ratio_fig = visualizer.plot_ratio_predictions(
    predicted_ratios=pred_ratios,
    true_ratios=true_ratios,
    uncertainties=ratio_uncertainties
)
```

## 📊 评估指标详解

### 基础回归指标

#### 均方根误差 (RMSE)
```python
def rmse(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))
```

#### 平均绝对误差 (MAE)
```python
def mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))
```

#### 皮尔逊相关系数
```python
def pearson_correlation(predictions, targets):
    return torch.corrcoef(torch.stack([predictions.flatten(), targets.flatten()]))[0, 1]
```

### 检测性能指标

#### 精确率和召回率
```python
def precision_recall(detections, ground_truth, distance_threshold=2.0):
    # 基于距离阈值的匹配
    matches = match_detections(detections, ground_truth, distance_threshold)
    
    precision = len(matches) / len(detections) if len(detections) > 0 else 0
    recall = len(matches) / len(ground_truth) if len(ground_truth) > 0 else 0
    
    return precision, recall
```

#### Jaccard指数
```python
def jaccard_index(detections, ground_truth, distance_threshold=2.0):
    matches = match_detections(detections, ground_truth, distance_threshold)
    union = len(detections) + len(ground_truth) - len(matches)
    
    return len(matches) / union if union > 0 else 0
```

### 多通道特定指标

#### 比例预测误差
```python
def ratio_prediction_error(pred_ratios, true_ratios):
    return {
        'mae': torch.mean(torch.abs(pred_ratios - true_ratios)),
        'rmse': torch.sqrt(torch.mean((pred_ratios - true_ratios) ** 2)),
        'mape': torch.mean(torch.abs((pred_ratios - true_ratios) / true_ratios)) * 100
    }
```

#### 光子数守恒误差
```python
def conservation_error(ch1_photons, ch2_photons, total_photons):
    predicted_total = ch1_photons + ch2_photons
    relative_error = torch.abs(predicted_total - total_photons) / total_photons
    return torch.mean(relative_error)
```

### 不确定性量化指标

#### 期望校准误差 (ECE)
```python
def expected_calibration_error(predictions, uncertainties, targets, num_bins=10):
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 计算每个bin的校准误差
        in_bin = (uncertainties > bin_lower) & (uncertainties <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (torch.abs(predictions[in_bin] - targets[in_bin]) < uncertainties[in_bin]).float().mean()
            avg_confidence_in_bin = uncertainties[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece
```

#### 覆盖率
```python
def coverage_probability(predictions, uncertainties, targets, confidence_level=0.95):
    z_score = torch.distributions.Normal(0, 1).icdf(torch.tensor((1 + confidence_level) / 2))
    lower_bound = predictions - z_score * uncertainties
    upper_bound = predictions + z_score * uncertainties
    
    coverage = ((targets >= lower_bound) & (targets <= upper_bound)).float().mean()
    return coverage
```

## 🔧 高级评估功能

### 交叉验证评估

```python
from sklearn.model_selection import KFold

class CrossValidationEvaluator:
    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    def evaluate_with_cv(self, model_class, dataset, model_params):
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(dataset)):
            print(f"评估第 {fold + 1}/{self.n_folds} 折...")
            
            # 分割数据
            train_data = torch.utils.data.Subset(dataset, train_idx)
            val_data = torch.utils.data.Subset(dataset, val_idx)
            
            # 训练模型
            model = model_class(**model_params)
            trained_model = self.train_model(model, train_data)
            
            # 评估模型
            fold_metrics = self.evaluate_model(trained_model, val_data)
            fold_results.append(fold_metrics)
        
        # 计算平均性能和标准差
        avg_metrics = self.aggregate_fold_results(fold_results)
        return avg_metrics
    
    def aggregate_fold_results(self, fold_results):
        aggregated = {}
        for metric in fold_results[0].keys():
            values = [result[metric] for result in fold_results]
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        return aggregated
```

### 统计显著性测试

```python
from scipy import stats

class StatisticalTester:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
    
    def paired_t_test(self, model1_results, model2_results):
        """配对t检验比较两个模型"""
        statistic, p_value = stats.ttest_rel(model1_results, model2_results)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'effect_size': self.cohens_d(model1_results, model2_results)
        }
    
    def wilcoxon_test(self, model1_results, model2_results):
        """Wilcoxon符号秩检验（非参数）"""
        statistic, p_value = stats.wilcoxon(model1_results, model2_results)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha
        }
    
    def cohens_d(self, group1, group2):
        """计算Cohen's d效应量"""
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1) + 
                             (len(group2) - 1) * np.var(group2)) / 
                            (len(group1) + len(group2) - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std
```

### 性能剖析

```python
import time
import psutil
import torch.profiler

class PerformanceProfiler:
    def __init__(self):
        self.metrics = {}
    
    def profile_inference(self, model, test_loader, num_batches=10):
        """推理性能剖析"""
        model.eval()
        
        # GPU内存使用
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        
        # CPU使用率
        cpu_percent_start = psutil.cpu_percent()
        
        # 推理时间测量
        inference_times = []
        
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True
        ) as prof:
            
            for i, batch in enumerate(test_loader):
                if i >= num_batches:
                    break
                
                start_time = time.time()
                
                with torch.no_grad():
                    _ = model(batch[0])
                
                end_time = time.time()
                inference_times.append(end_time - start_time)
        
        # 收集性能指标
        self.metrics['inference_time'] = {
            'mean': np.mean(inference_times),
            'std': np.std(inference_times),
            'min': np.min(inference_times),
            'max': np.max(inference_times)
        }
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            self.metrics['gpu_memory'] = {
                'peak_mb': (peak_memory - initial_memory) / 1024**2,
                'peak_gb': (peak_memory - initial_memory) / 1024**3
            }
        
        self.metrics['cpu_usage'] = psutil.cpu_percent() - cpu_percent_start
        
        # 保存详细的profiler报告
        prof.export_chrome_trace("inference_trace.json")
        
        return self.metrics
```

### 错误分析

```python
class ErrorAnalyzer:
    def __init__(self):
        self.error_categories = {
            'high_error': [],
            'medium_error': [],
            'low_error': []
        }
    
    def analyze_prediction_errors(self, predictions, targets, metadata):
        """分析预测误差的分布和模式"""
        errors = torch.abs(predictions - targets)
        
        # 按误差大小分类
        high_error_threshold = torch.quantile(errors, 0.9)
        medium_error_threshold = torch.quantile(errors, 0.7)
        
        high_error_mask = errors > high_error_threshold
        medium_error_mask = (errors > medium_error_threshold) & (errors <= high_error_threshold)
        low_error_mask = errors <= medium_error_threshold
        
        # 分析高误差样本的特征
        high_error_analysis = self.analyze_error_patterns(
            predictions[high_error_mask],
            targets[high_error_mask],
            metadata[high_error_mask] if metadata is not None else None
        )
        
        return {
            'error_distribution': {
                'high_error_count': high_error_mask.sum().item(),
                'medium_error_count': medium_error_mask.sum().item(),
                'low_error_count': low_error_mask.sum().item()
            },
            'high_error_analysis': high_error_analysis,
            'error_statistics': {
                'mean': errors.mean().item(),
                'std': errors.std().item(),
                'median': errors.median().item(),
                'q95': torch.quantile(errors, 0.95).item()
            }
        }
    
    def analyze_error_patterns(self, predictions, targets, metadata):
        """分析误差模式"""
        analysis = {}
        
        # 误差与目标值的关系
        errors = torch.abs(predictions - targets)
        correlation = torch.corrcoef(torch.stack([errors.flatten(), targets.flatten()]))[0, 1]
        analysis['error_target_correlation'] = correlation.item()
        
        # 误差的空间分布（如果有位置信息）
        if metadata is not None and 'positions' in metadata:
            positions = metadata['positions']
            analysis['spatial_error_pattern'] = self.analyze_spatial_errors(errors, positions)
        
        return analysis
```

## 📊 可视化功能

### 评估仪表板

```python
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

class EvaluationDashboard:
    def __init__(self, style='seaborn', figsize=(15, 10)):
        plt.style.use(style)
        self.figsize = figsize
    
    def create_dashboard(self, metrics, predictions, targets):
        """创建综合评估仪表板"""
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(3, 4, figure=fig)
        
        # 1. 误差分布直方图
        ax1 = fig.add_subplot(gs[0, 0])
        errors = torch.abs(predictions - targets)
        ax1.hist(errors.numpy(), bins=50, alpha=0.7, edgecolor='black')
        ax1.set_title('误差分布')
        ax1.set_xlabel('绝对误差')
        ax1.set_ylabel('频次')
        
        # 2. 预测vs真实值散点图
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(targets.numpy(), predictions.numpy(), alpha=0.5)
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax2.set_title('预测 vs 真实值')
        ax2.set_xlabel('真实值')
        ax2.set_ylabel('预测值')
        
        # 3. 残差图
        ax3 = fig.add_subplot(gs[0, 2])
        residuals = predictions - targets
        ax3.scatter(targets.numpy(), residuals.numpy(), alpha=0.5)
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_title('残差图')
        ax3.set_xlabel('真实值')
        ax3.set_ylabel('残差')
        
        # 4. 指标雷达图
        ax4 = fig.add_subplot(gs[0, 3], projection='polar')
        self.plot_metrics_radar(ax4, metrics)
        
        # 5. 误差随时间变化（如果有时间信息）
        ax5 = fig.add_subplot(gs[1, :])
        ax5.plot(errors.numpy())
        ax5.set_title('误差随样本变化')
        ax5.set_xlabel('样本索引')
        ax5.set_ylabel('绝对误差')
        
        # 6. 指标汇总表
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('tight')
        ax6.axis('off')
        self.create_metrics_table(ax6, metrics)
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_radar(self, ax, metrics):
        """绘制指标雷达图"""
        # 标准化指标到0-1范围
        metric_names = list(metrics.keys())
        metric_values = [metrics[name] for name in metric_names]
        
        # 归一化（这里需要根据具体指标调整）
        normalized_values = [(1 - val) if 'error' in name.lower() else val 
                           for name, val in zip(metric_names, metric_values)]
        
        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        normalized_values = normalized_values + [normalized_values[0]]
        
        ax.plot(angles, normalized_values, 'o-', linewidth=2)
        ax.fill(angles, normalized_values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_title('性能指标雷达图')
    
    def create_metrics_table(self, ax, metrics):
        """创建指标汇总表"""
        table_data = []
        for metric_name, value in metrics.items():
            if isinstance(value, dict):
                for sub_metric, sub_value in value.items():
                    table_data.append([f"{metric_name}_{sub_metric}", f"{sub_value:.4f}"])
            else:
                table_data.append([metric_name, f"{value:.4f}"])
        
        table = ax.table(cellText=table_data,
                        colLabels=['指标', '数值'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.set_title('评估指标汇总')
```

## 🐛 常见问题

### Q: 评估指标不稳定怎么办？
A: 建议措施：
- 增加测试样本数量
- 使用交叉验证
- 计算置信区间
- 检查数据分布
- 使用多个随机种子

### Q: 如何选择合适的评估指标？
A: 选择原则：
- 根据任务类型选择
- 考虑业务需求
- 平衡多个指标
- 包含不确定性评估
- 考虑计算成本

### Q: 评估速度慢怎么办？
A: 优化方法：
- 并行计算
- 采样评估
- 简化指标
- 缓存中间结果
- 使用GPU加速

### Q: 如何解释评估结果？
A: 分析策略：
- 对比基准方法
- 分析误差分布
- 考虑应用场景
- 查看可视化结果
- 进行统计检验

## 📚 相关文档

- [模型文档](../models/README.md)
- [推理模块文档](../inference/README.md)
- [训练模块文档](../training/README.md)
- [数据处理文档](../data/README.md)
- [多通道训练指南](../README_MultiChannel.md)
- [评估指标参考](./METRICS_REFERENCE.md)