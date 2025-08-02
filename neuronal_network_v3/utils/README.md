# Utils 工具模块

本模块包含DECODE神经网络v3的通用工具函数和实用程序，为整个项目提供基础支持功能。

## 📋 模块概览

### 核心组件

#### 🔹 ConfigManager (`config_utils.py`)
- **功能**: 配置文件管理
- **特点**:
  - YAML/JSON配置解析
  - 配置验证和合并
  - 环境变量支持
  - 配置模板生成
- **用途**: 统一的配置管理系统

#### 🔹 FileUtils (`file_utils.py`)
- **功能**: 文件操作工具
- **特点**:
  - 安全文件操作
  - 批量文件处理
  - 路径管理
  - 文件格式转换
- **用途**: 文件系统相关操作

#### 🔹 DataUtils (`data_utils.py`)
- **功能**: 数据处理工具
- **特点**:
  - 数据格式转换
  - 数组操作
  - 统计计算
  - 数据验证
- **用途**: 通用数据处理功能

#### 🔹 PlotUtils (`plot_utils.py`)
- **功能**: 可视化工具
- **特点**:
  - 标准化绘图
  - 多种图表类型
  - 自定义样式
  - 批量绘图
- **用途**: 数据可视化和结果展示

#### 🔹 MathUtils (`math_utils.py`)
- **功能**: 数学计算工具
- **特点**:
  - 数值计算
  - 统计函数
  - 几何变换
  - 信号处理
- **用途**: 数学运算和算法支持

#### 🔹 LogUtils (`log_utils.py`)
- **功能**: 日志管理
- **特点**:
  - 结构化日志
  - 多级别日志
  - 文件和控制台输出
  - 日志轮转
- **用途**: 系统日志记录和调试

#### 🔹 DeviceUtils (`device_utils.py`)
- **功能**: 设备管理工具
- **特点**:
  - GPU/CPU检测
  - 内存监控
  - 设备优化
  - 资源管理
- **用途**: 硬件资源管理和优化

## 🚀 使用示例

### 配置管理

```python
from neuronal_network_v3.utils import ConfigManager

# 初始化配置管理器
config_manager = ConfigManager()

# 加载配置文件
config = config_manager.load_config('configs/training_config.yaml')

# 合并多个配置
base_config = config_manager.load_config('configs/base_config.yaml')
training_config = config_manager.load_config('configs/training_config.yaml')
merged_config = config_manager.merge_configs(base_config, training_config)

# 验证配置
schema = {
    'model': {'type': str, 'required': True},
    'learning_rate': {'type': float, 'min': 1e-6, 'max': 1.0},
    'batch_size': {'type': int, 'min': 1, 'max': 1024}
}

is_valid, errors = config_manager.validate_config(config, schema)
if not is_valid:
    print(f"配置验证失败: {errors}")

# 保存配置
config_manager.save_config(config, 'configs/current_config.yaml')

# 从环境变量更新配置
config_with_env = config_manager.update_from_env(
    config, 
    env_mapping={
        'LEARNING_RATE': 'training.learning_rate',
        'BATCH_SIZE': 'training.batch_size',
        'GPU_ID': 'device.gpu_id'
    }
)

print(f"最终配置: {config_with_env}")
```

### 文件操作工具

```python
from neuronal_network_v3.utils import FileUtils

# 初始化文件工具
file_utils = FileUtils()

# 安全创建目录
file_utils.ensure_dir('results/experiment_001/')

# 批量文件操作
data_files = file_utils.find_files(
    directory='data/',
    pattern='*.h5',
    recursive=True
)

print(f"找到 {len(data_files)} 个数据文件")

# 文件格式转换
file_utils.convert_format(
    input_file='data/raw_data.csv',
    output_file='data/processed_data.h5',
    input_format='csv',
    output_format='hdf5'
)

# 安全文件复制
file_utils.safe_copy(
    src='models/best_model.pth',
    dst='backup/best_model_backup.pth',
    overwrite=False
)

# 文件完整性检查
checksum = file_utils.calculate_checksum('data/important_data.h5')
file_utils.save_checksum(checksum, 'data/important_data.h5.md5')

# 验证文件完整性
is_valid = file_utils.verify_checksum(
    'data/important_data.h5',
    'data/important_data.h5.md5'
)

if is_valid:
    print("文件完整性验证通过")
else:
    print("警告: 文件可能已损坏")

# 清理临时文件
file_utils.cleanup_temp_files(
    directory='temp/',
    max_age_hours=24,
    pattern='*.tmp'
)
```

### 数据处理工具

```python
from neuronal_network_v3.utils import DataUtils
import numpy as np
import torch

# 初始化数据工具
data_utils = DataUtils()

# 数据格式转换
numpy_array = np.random.randn(100, 100)
torch_tensor = data_utils.numpy_to_torch(numpy_array, device='cuda')
back_to_numpy = data_utils.torch_to_numpy(torch_tensor)

# 数据标准化
data = np.random.randn(1000, 5)
normalized_data, stats = data_utils.normalize_data(
    data, 
    method='zscore',
    return_stats=True
)

# 使用保存的统计信息标准化新数据
new_data = np.random.randn(100, 5)
normalized_new_data = data_utils.apply_normalization(new_data, stats)

# 数据分割
train_data, val_data, test_data = data_utils.split_data(
    data,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42
)

# 数据验证
validation_results = data_utils.validate_data(
    data,
    checks={
        'no_nan': True,
        'no_inf': True,
        'range_check': (-10, 10),
        'shape_check': (None, 5)  # 任意行数，5列
    }
)

if not validation_results['is_valid']:
    print(f"数据验证失败: {validation_results['errors']}")

# 数据统计
stats = data_utils.compute_statistics(data)
print(f"数据统计: {stats}")

# 异常值检测
outliers = data_utils.detect_outliers(
    data,
    method='iqr',
    threshold=1.5
)

print(f"检测到 {len(outliers)} 个异常值")

# 数据插值
data_with_missing = data.copy()
data_with_missing[np.random.choice(1000, 50, replace=False)] = np.nan

interpolated_data = data_utils.interpolate_missing(
    data_with_missing,
    method='linear'
)
```

### 可视化工具

```python
from neuronal_network_v3.utils import PlotUtils
import matplotlib.pyplot as plt

# 初始化绘图工具
plot_utils = PlotUtils(
    style='publication',
    dpi=300,
    figsize=(10, 8)
)

# 设置绘图样式
plot_utils.set_style({
    'font.size': 12,
    'axes.linewidth': 1.5,
    'grid.alpha': 0.3
})

# 绘制训练曲线
training_history = {
    'train_loss': [1.0, 0.8, 0.6, 0.4, 0.3],
    'val_loss': [1.1, 0.9, 0.7, 0.5, 0.4],
    'train_acc': [0.6, 0.7, 0.8, 0.85, 0.9],
    'val_acc': [0.55, 0.65, 0.75, 0.8, 0.85]
}

fig = plot_utils.plot_training_curves(
    training_history,
    save_path='plots/training_curves.png'
)

# 绘制预测结果对比
predictions = np.random.randn(100)
targets = predictions + np.random.randn(100) * 0.1

fig = plot_utils.plot_prediction_comparison(
    predictions=predictions,
    targets=targets,
    title='预测结果对比',
    save_path='plots/prediction_comparison.png'
)

# 绘制分布图
data = np.random.randn(1000)
fig = plot_utils.plot_distribution(
    data,
    bins=50,
    title='数据分布',
    xlabel='数值',
    ylabel='频次'
)

# 绘制热力图
correlation_matrix = np.corrcoef(np.random.randn(10, 100))
fig = plot_utils.plot_heatmap(
    correlation_matrix,
    title='相关性矩阵',
    cmap='coolwarm',
    center=0
)

# 批量绘图
data_dict = {
    'experiment_1': np.random.randn(100),
    'experiment_2': np.random.randn(100) + 1,
    'experiment_3': np.random.randn(100) - 1
}

plot_utils.plot_multiple_distributions(
    data_dict,
    save_dir='plots/distributions/',
    format='both'  # 保存PNG和PDF
)

# 创建子图网格
fig, axes = plot_utils.create_subplot_grid(
    nrows=2, ncols=2,
    figsize=(12, 10),
    titles=['图1', '图2', '图3', '图4']
)

# 在子图中绘制
for i, ax in enumerate(axes.flat):
    ax.plot(np.random.randn(100).cumsum())
    ax.set_title(f'子图 {i+1}')

plot_utils.save_figure(fig, 'plots/subplot_example.png')
```

### 数学工具

```python
from neuronal_network_v3.utils import MathUtils
import numpy as np

# 初始化数学工具
math_utils = MathUtils()

# 几何变换
points = np.random.randn(100, 2)

# 旋转变换
rotated_points = math_utils.rotate_points(
    points, 
    angle=np.pi/4,  # 45度
    center=(0, 0)
)

# 缩放变换
scaled_points = math_utils.scale_points(
    points,
    scale_x=2.0,
    scale_y=1.5
)

# 平移变换
translated_points = math_utils.translate_points(
    points,
    dx=5.0,
    dy=3.0
)

# 统计计算
data = np.random.randn(1000)

# 计算置信区间
ci_lower, ci_upper = math_utils.confidence_interval(
    data,
    confidence=0.95
)

print(f"95% 置信区间: [{ci_lower:.3f}, {ci_upper:.3f}]")

# 计算效应量
group1 = np.random.randn(100)
group2 = np.random.randn(100) + 0.5

cohens_d = math_utils.cohens_d(group1, group2)
print(f"Cohen's d: {cohens_d:.3f}")

# 信号处理
signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1000))  # 5Hz信号
noise = np.random.randn(1000) * 0.1
noisy_signal = signal + noise

# 滤波
filtered_signal = math_utils.lowpass_filter(
    noisy_signal,
    cutoff_freq=10,
    sampling_rate=1000
)

# 峰值检测
peaks = math_utils.find_peaks(
    signal,
    height=0.5,
    distance=50
)

print(f"检测到 {len(peaks)} 个峰值")

# 插值
x = np.linspace(0, 10, 11)
y = np.sin(x)
x_new = np.linspace(0, 10, 101)

y_interp = math_utils.interpolate(
    x, y, x_new,
    method='cubic'
)

# 数值积分
integral = math_utils.integrate(
    lambda x: x**2,
    a=0, b=1,
    method='simpson'
)

print(f"∫₀¹ x² dx = {integral:.6f}")

# 优化
result = math_utils.minimize(
    lambda x: (x[0] - 1)**2 + (x[1] - 2)**2,
    x0=[0, 0],
    method='BFGS'
)

print(f"优化结果: x = {result.x}, f(x) = {result.fun}")
```

### 日志管理

```python
from neuronal_network_v3.utils import LogUtils
import logging

# 初始化日志工具
log_utils = LogUtils()

# 设置日志配置
logger = log_utils.setup_logger(
    name='decode_v3',
    log_file='logs/training.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    max_bytes=10*1024*1024,  # 10MB
    backup_count=5
)

# 使用结构化日志
log_utils.log_structured(
    logger,
    level='info',
    message='训练开始',
    extra={
        'epoch': 1,
        'batch_size': 32,
        'learning_rate': 0.001,
        'model': 'DoubleMUNet'
    }
)

# 记录性能指标
log_utils.log_metrics(
    logger,
    metrics={
        'train_loss': 0.456,
        'val_loss': 0.523,
        'train_acc': 0.89,
        'val_acc': 0.85
    },
    step=100
)

# 记录异常
try:
    # 一些可能出错的代码
    result = 1 / 0
except Exception as e:
    log_utils.log_exception(
        logger,
        e,
        context={
            'function': 'train_step',
            'batch_idx': 42,
            'epoch': 5
        }
    )

# 创建上下文日志记录器
with log_utils.log_context(logger, 'model_training'):
    logger.info('开始模型训练')
    # 训练代码
    logger.info('模型训练完成')

# 日志分析
log_stats = log_utils.analyze_logs(
    log_file='logs/training.log',
    start_time='2024-01-01 00:00:00',
    end_time='2024-01-02 00:00:00'
)

print(f"日志统计: {log_stats}")
```

### 设备管理

```python
from neuronal_network_v3.utils import DeviceUtils
import torch

# 初始化设备工具
device_utils = DeviceUtils()

# 获取最佳设备
device = device_utils.get_best_device()
print(f"使用设备: {device}")

# 检查GPU信息
if torch.cuda.is_available():
    gpu_info = device_utils.get_gpu_info()
    print(f"GPU信息: {gpu_info}")
    
    # 选择最佳GPU
    best_gpu = device_utils.select_best_gpu()
    print(f"最佳GPU: {best_gpu}")

# 内存监控
memory_info = device_utils.get_memory_info(device)
print(f"内存信息: {memory_info}")

# 设置内存管理
device_utils.setup_memory_management(
    device=device,
    memory_fraction=0.8,  # 使用80%的GPU内存
    allow_growth=True
)

# 监控资源使用
with device_utils.resource_monitor(device) as monitor:
    # 执行一些计算密集的操作
    data = torch.randn(1000, 1000, device=device)
    result = torch.matmul(data, data.T)
    
    # 获取资源使用情况
    usage = monitor.get_usage()
    print(f"资源使用: {usage}")

# 优化设备设置
device_utils.optimize_device_settings(
    device=device,
    benchmark=True,
    deterministic=False
)

# 清理GPU内存
device_utils.cleanup_memory(device)

# 设备兼容性检查
compatibility = device_utils.check_compatibility(
    required_cuda_version='11.0',
    required_memory_gb=8
)

if not compatibility['is_compatible']:
    print(f"设备兼容性问题: {compatibility['issues']}")
```

## 🔧 高级工具功能

### 性能分析器

```python
class PerformanceProfiler:
    def __init__(self):
        self.timers = {}
        self.counters = {}
        self.memory_tracker = {}
    
    def timer(self, name):
        """计时器装饰器"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                if name not in self.timers:
                    self.timers[name] = []
                self.timers[name].append(end_time - start_time)
                
                return result
            return wrapper
        return decorator
    
    def count(self, name, increment=1):
        """计数器"""
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += increment
    
    def track_memory(self, name, device='cuda'):
        """内存追踪"""
        if device == 'cuda' and torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(device) / 1024**2  # MB
        else:
            import psutil
            memory_used = psutil.virtual_memory().used / 1024**2  # MB
        
        if name not in self.memory_tracker:
            self.memory_tracker[name] = []
        self.memory_tracker[name].append(memory_used)
    
    def get_report(self):
        """生成性能报告"""
        report = {
            'timers': {},
            'counters': self.counters,
            'memory': {}
        }
        
        # 计时器统计
        for name, times in self.timers.items():
            report['timers'][name] = {
                'count': len(times),
                'total': sum(times),
                'mean': np.mean(times),
                'std': np.std(times),
                'min': min(times),
                'max': max(times)
            }
        
        # 内存统计
        for name, memories in self.memory_tracker.items():
            report['memory'][name] = {
                'count': len(memories),
                'mean': np.mean(memories),
                'max': max(memories),
                'min': min(memories)
            }
        
        return report

# 使用示例
profiler = PerformanceProfiler()

@profiler.timer('data_loading')
def load_data():
    # 模拟数据加载
    time.sleep(0.1)
    return torch.randn(100, 100)

@profiler.timer('model_forward')
def model_forward(data):
    # 模拟模型前向传播
    time.sleep(0.05)
    return data * 2

# 执行操作
for i in range(10):
    profiler.track_memory('before_loading')
    data = load_data()
    profiler.count('data_loaded')
    
    profiler.track_memory('after_loading')
    result = model_forward(data)
    profiler.count('forward_pass')
    
    profiler.track_memory('after_forward')

# 生成报告
report = profiler.get_report()
print(json.dumps(report, indent=2))
```

### 缓存管理器

```python
class CacheManager:
    def __init__(self, cache_dir='cache/', max_size_gb=10):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_gb * 1024**3
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self):
        index_file = self.cache_dir / 'cache_index.json'
        if index_file.exists():
            with open(index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache_index(self):
        index_file = self.cache_dir / 'cache_index.json'
        with open(index_file, 'w') as f:
            json.dump(self.cache_index, f, indent=2)
    
    def _get_cache_key(self, *args, **kwargs):
        """生成缓存键"""
        key_data = {'args': args, 'kwargs': kwargs}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key):
        """获取缓存"""
        if key in self.cache_index:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                # 更新访问时间
                self.cache_index[key]['last_accessed'] = time.time()
                self._save_cache_index()
                
                return data
        return None
    
    def set(self, key, data):
        """设置缓存"""
        cache_file = self.cache_dir / f"{key}.pkl"
        
        # 保存数据
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        
        # 更新索引
        file_size = cache_file.stat().st_size
        self.cache_index[key] = {
            'size': file_size,
            'created': time.time(),
            'last_accessed': time.time()
        }
        
        # 检查缓存大小
        self._cleanup_if_needed()
        self._save_cache_index()
    
    def _cleanup_if_needed(self):
        """清理缓存"""
        total_size = sum(item['size'] for item in self.cache_index.values())
        
        if total_size > self.max_size_bytes:
            # 按最后访问时间排序，删除最旧的
            sorted_items = sorted(
                self.cache_index.items(),
                key=lambda x: x[1]['last_accessed']
            )
            
            for key, info in sorted_items:
                cache_file = self.cache_dir / f"{key}.pkl"
                if cache_file.exists():
                    cache_file.unlink()
                
                del self.cache_index[key]
                total_size -= info['size']
                
                if total_size <= self.max_size_bytes * 0.8:  # 清理到80%
                    break
    
    def cached(self, func):
        """缓存装饰器"""
        def wrapper(*args, **kwargs):
            key = self._get_cache_key(func.__name__, *args, **kwargs)
            
            # 尝试从缓存获取
            cached_result = self.get(key)
            if cached_result is not None:
                return cached_result
            
            # 计算结果并缓存
            result = func(*args, **kwargs)
            self.set(key, result)
            
            return result
        return wrapper

# 使用示例
cache_manager = CacheManager()

@cache_manager.cached
def expensive_computation(n):
    """模拟耗时计算"""
    time.sleep(1)  # 模拟计算时间
    return sum(i**2 for i in range(n))

# 第一次调用会计算并缓存
start_time = time.time()
result1 = expensive_computation(1000)
print(f"第一次调用耗时: {time.time() - start_time:.2f}秒")

# 第二次调用会从缓存获取
start_time = time.time()
result2 = expensive_computation(1000)
print(f"第二次调用耗时: {time.time() - start_time:.2f}秒")

assert result1 == result2
```

### 并行处理工具

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count

class ParallelProcessor:
    def __init__(self, max_workers=None, use_processes=False):
        self.max_workers = max_workers or cpu_count()
        self.use_processes = use_processes
    
    def map(self, func, iterable, chunksize=1):
        """并行映射"""
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            if self.use_processes:
                results = list(executor.map(func, iterable, chunksize=chunksize))
            else:
                results = list(executor.map(func, iterable))
        
        return results
    
    def map_async(self, func, iterable, callback=None):
        """异步并行映射"""
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            futures = [executor.submit(func, item) for item in iterable]
            
            results = []
            for future in futures:
                result = future.result()
                results.append(result)
                
                if callback:
                    callback(result)
        
        return results
    
    def batch_process(self, func, data, batch_size=100):
        """批量处理"""
        results = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_results = self.map(func, batch)
            results.extend(batch_results)
            
            # 进度报告
            progress = min((i + batch_size) / len(data), 1.0)
            print(f"处理进度: {progress:.1%}")
        
        return results

# 使用示例
processor = ParallelProcessor(max_workers=4)

def process_item(item):
    # 模拟处理
    time.sleep(0.1)
    return item ** 2

# 并行处理
data = list(range(100))
results = processor.map(process_item, data)
print(f"处理了 {len(results)} 个项目")

# 批量处理大数据
large_data = list(range(10000))
batch_results = processor.batch_process(process_item, large_data, batch_size=500)
```

## 🐛 常见问题

### Q: 配置文件格式不正确怎么办？
A: 解决方法：
1. 使用配置验证功能
2. 检查YAML/JSON语法
3. 参考配置模板
4. 使用配置生成工具

### Q: 文件操作权限问题？
A: 解决方案：
1. 检查文件权限
2. 使用安全文件操作
3. 创建必要的目录
4. 处理权限异常

### Q: 内存使用过多怎么办？
A: 优化策略：
1. 使用内存监控
2. 及时清理变量
3. 使用生成器
4. 分批处理数据

### Q: 日志文件过大？
A: 管理方法：
1. 设置日志轮转
2. 调整日志级别
3. 定期清理旧日志
4. 使用压缩存储

### Q: 并行处理效率低？
A: 优化建议：
1. 选择合适的并行方式
2. 调整工作进程数
3. 优化任务分割
4. 减少进程间通信

## 📚 相关文档

- [配置管理指南](./CONFIG_GUIDE.md)
- [性能优化指南](./PERFORMANCE_GUIDE.md)
- [调试工具使用](./DEBUG_TOOLS.md)
- [最佳实践](./BEST_PRACTICES.md)
- [API参考](./API_REFERENCE.md)