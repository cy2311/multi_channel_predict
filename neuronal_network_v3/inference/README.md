# Inference 推理模块

本模块包含DECODE神经网络v3的推理引擎，支持单通道和多通道SMLM数据的高效推理、后处理和结果解析。

## 📋 模块概览

### 核心组件

#### 🔹 MultiChannelInfer (`multi_channel_infer.py`)
- **功能**: 多通道联合推理引擎
- **特点**:
  - 双通道协同推理
  - 比例预测和不确定性量化
  - 物理约束集成
  - 批量处理优化
- **用途**: 多通道SMLM数据的主要推理接口

#### 🔹 BaseInfer (`infer.py`)
- **功能**: 基础推理器
- **特点**:
  - 标准化推理流程
  - 内存高效处理
  - 可配置的后处理
  - 多种输出格式
- **用途**: 单通道推理和基础推理操作

#### 🔹 PostProcessing (`post_processing.py`)
- **功能**: 推理结果后处理
- **特点**:
  - 峰值检测和聚类
  - 坐标精化
  - 质量过滤
  - 结果优化
- **用途**: 原始推理结果的精化和优化

#### 🔹 ResultParser (`result_parser.py`)
- **功能**: 结果解析和格式转换
- **特点**:
  - 多种输出格式支持
  - 元数据管理
  - 结果验证
  - 统计信息生成
- **用途**: 推理结果的标准化输出

#### 🔹 InferenceUtils (`utils.py`)
- **功能**: 推理工具函数
- **特点**:
  - 图像预处理
  - 批处理优化
  - 内存管理
  - 性能监控
- **用途**: 推理过程的辅助功能

## 🚀 使用示例

### 多通道推理

```python
from neuronal_network_v3.inference import MultiChannelInfer
from neuronal_network_v3.models import SigmaMUNet, RatioNet
import torch

# 加载模型
channel1_net = SigmaMUNet(n_inp=1, n_out=10)
channel2_net = SigmaMUNet(n_inp=1, n_out=10)
ratio_net = RatioNet(input_channels=20, hidden_dim=128)

# 加载权重
checkpoint = torch.load('models/multi_channel_model.pth')
channel1_net.load_state_dict(checkpoint['channel1_net'])
channel2_net.load_state_dict(checkpoint['channel2_net'])
ratio_net.load_state_dict(checkpoint['ratio_net'])

# 初始化推理器
inferrer = MultiChannelInfer(
    channel1_net=channel1_net,
    channel2_net=channel2_net,
    ratio_net=ratio_net,
    device='cuda',
    apply_conservation=True,      # 应用光子数守恒
    apply_consistency=True,       # 应用比例一致性
    batch_size=16                 # 批处理大小
)

# 推理
results = inferrer.predict(
    channel1_images=ch1_images,   # [N, 1, H, W]
    channel2_images=ch2_images,   # [N, 1, H, W]
    return_uncertainty=True,      # 返回不确定性
    apply_postprocessing=True     # 应用后处理
)

# 结果包含
print(f"通道1预测: {results['channel1']['predictions'].shape}")
print(f"通道2预测: {results['channel2']['predictions'].shape}")
print(f"比例预测: {results['ratio']['mean'].shape}")
print(f"比例不确定性: {results['ratio']['std'].shape}")
print(f"检测到的发射体: {len(results['detections'])}")
```

### 单通道推理

```python
from neuronal_network_v3.inference import BaseInfer
from neuronal_network_v3.models import SigmaMUNet

# 加载模型
model = SigmaMUNet(n_inp=1, n_out=10)
model.load_state_dict(torch.load('models/single_channel_model.pth'))

# 初始化推理器
inferrer = BaseInfer(
    model=model,
    device='cuda',
    batch_size=32
)

# 推理
results = inferrer.predict(
    images=images,                # [N, 1, H, W]
    return_raw=False,             # 不返回原始输出
    apply_postprocessing=True     # 应用后处理
)

# 结果处理
detections = results['detections']
for detection in detections:
    x, y, z = detection['position']
    intensity = detection['intensity']
    uncertainty = detection['uncertainty']
    print(f"位置: ({x:.2f}, {y:.2f}, {z:.2f}), 强度: {intensity:.2f}")
```

### 批量推理

```python
from neuronal_network_v3.inference import MultiChannelBatchInfer

# 大规模数据推理
batch_inferrer = MultiChannelBatchInfer(
    channel1_net=channel1_net,
    channel2_net=channel2_net,
    ratio_net=ratio_net,
    auto_batch_size=True,         # 自动批大小
    max_memory_gb=8.0,            # 最大内存使用
    progress_bar=True             # 显示进度条
)

# 处理大型数据集
results = batch_inferrer.predict_dataset(
    dataset_path='data/large_test_set.h5',
    output_path='results/predictions.h5',
    chunk_size=1000               # 分块处理
)
```

### 后处理配置

```python
from neuronal_network_v3.inference.post_processing import PostProcessor

# 配置后处理参数
post_processor = PostProcessor(
    detection_threshold=0.5,      # 检测阈值
    nms_threshold=0.3,            # 非极大值抑制阈值
    min_distance=2.0,             # 最小距离（像素）
    quality_threshold=0.7,        # 质量阈值
    coordinate_refinement=True,   # 坐标精化
    uncertainty_filtering=True    # 不确定性过滤
)

# 应用后处理
processed_results = post_processor.process(
    raw_predictions=raw_results,
    metadata=image_metadata
)
```

## ⚙️ 推理配置

### MultiChannelInfer参数

```python
infer_config = {
    'device': 'cuda',                    # 计算设备
    'batch_size': 16,                    # 批处理大小
    'apply_conservation': True,          # 光子数守恒
    'apply_consistency': True,           # 比例一致性
    'conservation_weight': 1.0,          # 守恒约束权重
    'consistency_weight': 0.5,           # 一致性约束权重
    'uncertainty_threshold': 0.1,        # 不确定性阈值
    'output_format': 'dict',             # 输出格式
    'return_intermediate': False,        # 返回中间结果
    'memory_efficient': True             # 内存高效模式
}
```

### 后处理参数

```python
postprocess_config = {
    'detection_threshold': 0.5,          # 检测概率阈值
    'nms_threshold': 0.3,                # NMS IoU阈值
    'min_distance': 2.0,                 # 最小检测距离
    'max_detections': 1000,              # 最大检测数量
    'quality_threshold': 0.7,            # 质量分数阈值
    'coordinate_refinement': True,       # 亚像素精化
    'uncertainty_filtering': True,       # 基于不确定性过滤
    'clustering_method': 'dbscan',       # 聚类方法
    'cluster_eps': 1.5,                  # 聚类参数
    'filter_edge_detections': True       # 过滤边缘检测
}
```

## 🔧 高级功能

### 物理约束推理

```python
class PhysicsConstrainedInfer(MultiChannelInfer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint_optimizer = ConstraintOptimizer()
    
    def apply_physics_constraints(self, predictions):
        """应用物理约束优化预测结果"""
        # 光子数守恒
        total_photons = predictions['total_photons']
        ch1_photons = predictions['channel1']['photons']
        ch2_photons = predictions['channel2']['photons']
        
        # 约束优化
        optimized_predictions = self.constraint_optimizer.optimize(
            ch1_photons, ch2_photons, total_photons
        )
        
        return optimized_predictions
```

### 不确定性量化

```python
class UncertaintyQuantifier:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
    
    def monte_carlo_inference(self, model, inputs):
        """蒙特卡洛推理获取不确定性"""
        model.train()  # 启用dropout
        predictions = []
        
        for _ in range(self.num_samples):
            with torch.no_grad():
                pred = model(inputs)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std
    
    def epistemic_uncertainty(self, predictions):
        """计算认知不确定性"""
        return predictions.var(dim=0)
    
    def aleatoric_uncertainty(self, model_variance):
        """计算偶然不确定性"""
        return model_variance
```

### 自适应批处理

```python
class AdaptiveBatchInfer:
    def __init__(self, model, max_memory_gb=8.0):
        self.model = model
        self.max_memory = max_memory_gb * 1024**3  # 转换为字节
        self.optimal_batch_size = self.find_optimal_batch_size()
    
    def find_optimal_batch_size(self):
        """自动寻找最优批大小"""
        batch_size = 1
        while True:
            try:
                # 测试当前批大小
                dummy_input = torch.randn(batch_size, 1, 64, 64)
                memory_before = torch.cuda.memory_allocated()
                
                with torch.no_grad():
                    _ = self.model(dummy_input)
                
                memory_after = torch.cuda.memory_allocated()
                memory_used = memory_after - memory_before
                
                if memory_used > self.max_memory:
                    return batch_size - 1
                
                batch_size *= 2
                
            except RuntimeError:  # OOM
                return batch_size // 2
    
    def predict(self, inputs):
        """使用最优批大小进行推理"""
        results = []
        for i in range(0, len(inputs), self.optimal_batch_size):
            batch = inputs[i:i + self.optimal_batch_size]
            with torch.no_grad():
                batch_results = self.model(batch)
            results.append(batch_results)
        
        return torch.cat(results, dim=0)
```

### 结果验证

```python
class ResultValidator:
    def __init__(self, validation_config):
        self.config = validation_config
    
    def validate_detections(self, detections, metadata):
        """验证检测结果的合理性"""
        validation_report = {
            'total_detections': len(detections),
            'valid_detections': 0,
            'warnings': [],
            'errors': []
        }
        
        for detection in detections:
            # 位置合理性检查
            if self.is_position_valid(detection['position'], metadata):
                validation_report['valid_detections'] += 1
            else:
                validation_report['warnings'].append(
                    f"Invalid position: {detection['position']}"
                )
            
            # 强度合理性检查
            if not self.is_intensity_valid(detection['intensity']):
                validation_report['warnings'].append(
                    f"Unusual intensity: {detection['intensity']}"
                )
        
        return validation_report
    
    def is_position_valid(self, position, metadata):
        """检查位置是否在合理范围内"""
        x, y, z = position
        image_shape = metadata['image_shape']
        
        return (0 <= x < image_shape[1] and 
                0 <= y < image_shape[0] and 
                -1000 <= z <= 1000)  # Z范围检查
    
    def is_intensity_valid(self, intensity):
        """检查强度是否合理"""
        return 0 < intensity < 1e6  # 合理的光子数范围
```

## 📊 性能优化

### GPU内存优化

```python
# 内存高效推理
with torch.cuda.amp.autocast():  # 混合精度
    with torch.no_grad():         # 禁用梯度计算
        predictions = model(inputs)

# 清理GPU缓存
torch.cuda.empty_cache()
```

### 并行推理

```python
from torch.nn import DataParallel
from concurrent.futures import ThreadPoolExecutor

# 多GPU推理
if torch.cuda.device_count() > 1:
    model = DataParallel(model)

# CPU并行后处理
def parallel_postprocess(predictions, num_workers=4):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for pred in predictions:
            future = executor.submit(postprocess_single, pred)
            futures.append(future)
        
        results = [future.result() for future in futures]
    return results
```

### 流式推理

```python
class StreamingInfer:
    def __init__(self, model, buffer_size=10):
        self.model = model
        self.buffer_size = buffer_size
        self.input_queue = queue.Queue(maxsize=buffer_size)
        self.output_queue = queue.Queue()
    
    def inference_worker(self):
        """推理工作线程"""
        while True:
            batch = self.input_queue.get()
            if batch is None:  # 停止信号
                break
            
            with torch.no_grad():
                results = self.model(batch)
            
            self.output_queue.put(results)
    
    def stream_predict(self, data_stream):
        """流式推理"""
        # 启动推理线程
        worker_thread = threading.Thread(target=self.inference_worker)
        worker_thread.start()
        
        # 处理数据流
        for batch in data_stream:
            self.input_queue.put(batch)
            
            # 获取结果（如果可用）
            try:
                result = self.output_queue.get_nowait()
                yield result
            except queue.Empty:
                continue
        
        # 停止推理线程
        self.input_queue.put(None)
        worker_thread.join()
```

## 🐛 常见问题

### Q: 推理速度慢怎么办？
A: 优化建议：
- 增加批大小
- 使用混合精度
- 启用模型编译
- 使用多GPU并行
- 优化数据加载

### Q: GPU内存不足？
A: 解决方案：
- 减少批大小
- 使用梯度检查点
- 启用内存高效模式
- 分块处理大图像
- 使用CPU推理

### Q: 推理结果不准确？
A: 检查项目：
- 模型是否正确加载
- 输入数据预处理
- 后处理参数设置
- 物理约束是否合理
- 模型训练质量

### Q: 如何处理大规模数据？
A: 策略：
- 使用批量推理
- 启用流式处理
- 分块处理
- 并行计算
- 结果缓存

## 📚 相关文档

- [模型文档](../models/README.md)
- [训练模块文档](../training/README.md)
- [评估模块文档](../evaluation/README.md)
- [数据处理文档](../data/README.md)
- [多通道训练指南](../README_MultiChannel.md)