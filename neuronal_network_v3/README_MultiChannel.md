# DECODE神经网络v3 - 多通道扩展

基于`DECODE_Network_Analysis.md`文档实现的多通道DECODE网络扩展，支持双通道联合推理、不确定性量化和物理约束。

## 🚀 新功能特性

### 1. 多通道架构
- **双通道独立网络**: 两个独立的SigmaMUNet处理不同通道数据
- **比例预测网络**: RatioNet预测光子数在两通道间的分配比例
- **不确定性量化**: 预测比例的均值和方差，提供置信度估计
- **特征提取器**: 从SigmaMUNet输出中提取用于比例预测的特征

### 2. 三阶段训练策略
- **阶段1**: 双通道独立训练 - 分别训练两个通道的SigmaMUNet
- **阶段2**: 比例网络训练 - 冻结通道网络，训练RatioNet
- **阶段3**: 联合微调 - 端到端优化所有组件

### 3. 物理约束
- **光子数守恒**: 确保两通道光子数之和等于总光子数
- **比例一致性**: 保证直接预测的比例与从光子数计算的比例一致
- **约束损失**: 在训练和推理中强制执行物理约束

### 4. 高级损失函数
- **RatioGaussianNLLLoss**: 基于高斯负对数似然的比例预测损失
- **MultiChannelLossWithGaussianRatio**: 集成多通道损失和物理约束
- **不确定性正则化**: 防止过度自信的预测

### 5. 全面评估系统
- **多维度评估**: 单通道性能、比例预测、物理约束评估
- **不确定性评估**: 校准性、覆盖率、锐度分析
- **可视化分析**: 自动生成评估图表和分析报告

## 📁 项目结构

```
neuronal_network_v3/
├── 📁 models/                     # 神经网络模型定义
│   ├── ratio_net.py              # 比例预测网络
│   ├── sigma_munet.py            # Sigma MUNet模型
│   ├── double_munet.py           # 双通道MUNet
│   ├── simple_smlm_net.py        # 简单SMLM网络
│   ├── unet2d.py                 # 2D UNet基础模型
│   └── __init__.py               # 模型模块初始化
├── 📁 loss/                       # 损失函数定义
│   ├── ratio_loss.py             # 比例预测损失函数
│   ├── gaussian_mm_loss.py       # 高斯混合模型损失
│   ├── ppxyzb_loss.py            # PPXYZB损失函数
│   ├── unified_loss.py           # 统一损失函数
│   └── __init__.py               # 损失函数模块初始化
├── 📁 trainer/                    # 训练器模块
│   ├── multi_channel_trainer.py  # 多通道训练器
│   └── __init__.py               # 训练器模块初始化
├── 📁 training/                   # 训练相关工具
│   ├── trainer.py                # 基础训练器
│   ├── dataset.py                # 训练数据集
│   ├── target_generator.py       # 目标生成器
│   ├── callbacks.py              # 训练回调函数
│   ├── utils.py                  # 训练工具函数
│   └── __init__.py               # 训练模块初始化
├── 📁 inference/                  # 推理引擎
│   ├── multi_channel_infer.py    # 多通道推理引擎
│   ├── infer.py                  # 基础推理器
│   ├── post_processing.py        # 后处理模块
│   ├── result_parser.py          # 结果解析器
│   ├── utils.py                  # 推理工具函数
│   └── __init__.py               # 推理模块初始化
├── 📁 evaluation/                 # 评估系统
│   ├── multi_channel_evaluation.py # 多通道评估系统
│   ├── evaluator.py              # 基础评估器
│   ├── analyzer.py               # 结果分析器
│   ├── benchmark.py              # 基准测试
│   ├── metrics.py                # 评估指标
│   ├── visualizer.py             # 可视化工具
│   └── __init__.py               # 评估模块初始化
├── 📁 data/                       # 数据处理模块
│   ├── multi_channel_dataset.py  # 多通道数据集
│   ├── dataset.py                # 基础数据集
│   ├── transforms.py             # 数据变换
│   └── __init__.py               # 数据模块初始化
├── 📁 utils/                      # 工具函数库 [📖 详细文档](utils/README.md)
│   ├── config_utils.py           # 配置管理工具
│   ├── file_utils.py             # 文件操作工具
│   ├── data_utils.py             # 数据处理工具
│   ├── plot_utils.py             # 可视化工具
│   ├── math_utils.py             # 数学计算工具
│   ├── log_utils.py              # 日志管理工具
│   ├── device_utils.py           # 设备管理工具
│   └── __init__.py               # 工具模块初始化
├── 📁 checkpoints/                # 模型检查点存储
├── 📁 logs/                       # 训练日志
├── 📁 results/                    # 结果输出
├── 📄 multi_channel_config.yaml   # 多通道训练配置
├── 📄 training_config.yaml        # 基础训练配置
├── 📄 train_multi_channel.py      # 多通道训练脚本
├── 📄 infer_multi_channel.py      # 多通道推理脚本
├── 📄 example_multi_channel.py    # 多通道使用示例
├── 📄 test_multi_channel.py       # 多通道测试脚本
├── 📄 train.py                    # 基础训练脚本
├── 📄 submit_training.sh          # SLURM训练提交脚本
├── 📄 submit_training_resume.sh   # SLURM恢复训练脚本
├── 📄 DECODE_Network_Analysis.md  # 网络架构分析文档
├── 📄 neuronal_network_architecture.md # 神经网络架构文档
├── 📄 README_MultiChannel.md      # 多通道扩展文档（本文档）
└── 📄 __init__.py                 # 主模块初始化
```

## 📚 模块文档

本项目包含多个专业模块，每个模块都有详细的文档说明：

- **[🧠 Models 模型模块](models/README.md)** - DECODE神经网络模型定义和架构
- **[📊 Loss 损失函数模块](loss/README.md)** - 多通道损失函数和优化目标
- **[📁 Data 数据模块](data/README.md)** - 数据加载、处理和变换
- **[🔍 Inference 推理模块](inference/README.md)** - 模型推理和后处理
- **[📈 Evaluation 评估模块](evaluation/README.md)** - 性能评估和指标计算
- **[🎯 Training 训练模块](training/README.md)** - 模型训练和优化策略
- **[🛠️ Utils 工具模块](utils/README.md)** - 通用工具和实用程序

## 🛠️ 安装和环境

### 依赖要求
```bash
# 基础依赖
pip install torch torchvision
pip install numpy scipy matplotlib
pip install h5py pyyaml tqdm

# 可选依赖（用于高级功能）
pip install tensorboard wandb
pip install scikit-learn seaborn
```

### 环境配置
```python
# 设置PYTHONPATH
export PYTHONPATH="/path/to/DECODE_rewrite:$PYTHONPATH"

# 或在Python中
import sys
sys.path.append('/path/to/DECODE_rewrite')
```

## 🚀 快速开始

### 1. 数据准备

多通道数据应包含以下结构：

```python
# HDF5格式
data_file.h5:
├── train/
│   ├── channel1/
│   │   ├── images      # 通道1输入图像 [N, H, W]
│   │   ├── targets     # 通道1目标 [N, C, H, W]
│   │   └── photons     # 通道1光子数 [N]
│   ├── channel2/
│   │   ├── images      # 通道2输入图像 [N, H, W]
│   │   ├── targets     # 通道2目标 [N, C, H, W]
│   │   └── photons     # 通道2光子数 [N]
│   └── metadata        # 元数据（JSON格式）
├── val/
│   └── ...
└── test/
    └── ...
```

### 2. 配置文件

复制并修改`multi_channel_config.yaml`：

```yaml
# 基本配置示例
model:
  n_inp: 1
  n_out: 10
  sigma_munet_params:
    depth: 3
    initial_features: 64
  ratio_net:
    hidden_dim: 128
    num_layers: 3

data:
  train_path: "data/train_multi_channel.h5"
  val_path: "data/val_multi_channel.h5"
  patch_size: 64
  
training:
  stage1_epochs: 100
  stage2_epochs: 50
  stage3_epochs: 30
  batch_size: 16
```

### 3. 训练模型

```bash
# 完整三阶段训练
python train_multi_channel.py --config multi_channel_config.yaml

# 单独训练某个阶段
python train_multi_channel.py --config multi_channel_config.yaml --stage 1

# 从检查点恢复训练
python train_multi_channel.py --config multi_channel_config.yaml --resume outputs/stage2_models.pth
```

### 4. 模型推理

```bash
# 基本推理
python infer_multi_channel.py \
    --model outputs/stage3_models.pth \
    --input data/test_images.h5 \
    --output results/

# 带评估和可视化的推理
python infer_multi_channel.py \
    --model outputs/stage3_models.pth \
    --input data/test_images.h5 \
    --output results/ \
    --apply-constraints \
    --evaluate \
    --visualize
```

## 📊 API使用示例

### 1. 多通道训练

```python
from neuronal_network_v3.trainer.multi_channel_trainer import MultiChannelTrainer
from neuronal_network_v3.data.multi_channel_dataset import MultiChannelDataModule

# 加载配置
with open('multi_channel_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 初始化数据模块
data_module = MultiChannelDataModule(config)
data_module.setup()

# 初始化训练器
trainer = MultiChannelTrainer(config, device='cuda')

# 执行三阶段训练
results = trainer.train_full_pipeline(
    train_loader=data_module.train_loader,
    val_loader=data_module.val_loader,
    save_dir='outputs/'
)
```

### 2. 多通道推理

```python
from neuronal_network_v3.inference.multi_channel_infer import MultiChannelInfer
from neuronal_network_v3.models.sigma_munet import SigmaMUNet
from neuronal_network_v3.models.ratio_net import RatioNet

# 加载模型
channel1_net = SigmaMUNet(n_inp=1, n_out=10)
channel2_net = SigmaMUNet(n_inp=1, n_out=10)
ratio_net = RatioNet(input_channels=20, hidden_dim=128)

# 加载权重
checkpoint = torch.load('outputs/stage3_models.pth')
channel1_net.load_state_dict(checkpoint['channel1_net'])
channel2_net.load_state_dict(checkpoint['channel2_net'])
ratio_net.load_state_dict(checkpoint['ratio_net'])

# 初始化推理器
inferrer = MultiChannelInfer(
    channel1_net=channel1_net,
    channel2_net=channel2_net,
    ratio_net=ratio_net,
    apply_conservation=True,
    apply_consistency=True
)

# 推理
results = inferrer.predict(channel1_images, channel2_images)
print(f"Predicted ratios: {results['ratio_mean']}")
print(f"Uncertainties: {results['ratio_std']}")
```

### 3. 评估和可视化

```python
from neuronal_network_v3.evaluation.multi_channel_evaluation import MultiChannelEvaluation

# 初始化评估器
evaluator = MultiChannelEvaluation(device='cuda')

# 评估
metrics = evaluator.evaluate(pred_results, ground_truth)
print(f"Ratio MAE: {metrics['ratio']['mae']:.4f}")
print(f"Coverage 95%: {metrics['ratio']['coverage_95']:.4f}")
print(f"Conservation error: {metrics['conservation']['conservation_error']:.4f}")

# 可视化
fig = evaluator.visualize_results(
    pred_results, ground_truth,
    save_path='evaluation_plots.png'
)
```

## 🔧 高级功能

### 1. 自定义损失函数

```python
from neuronal_network_v3.loss.ratio_loss import RatioGaussianNLLLoss

# 创建自定义比例损失
ratio_loss = RatioGaussianNLLLoss(
    photon_conservation_weight=1.0,
    ratio_consistency_weight=0.5,
    uncertainty_regularization=0.1
)

# 计算损失
loss = ratio_loss(ratio_mean, ratio_std, true_ratio)
```

### 2. 批量推理优化

```python
from neuronal_network_v3.inference.multi_channel_infer import MultiChannelBatchInfer

# 自动批大小优化
batch_inferrer = MultiChannelBatchInfer(
    channel1_net=channel1_net,
    channel2_net=channel2_net,
    ratio_net=ratio_net,
    auto_batch_size=True,
    max_memory_gb=8.0
)

# 大规模数据推理
results = batch_inferrer.predict_large_dataset(large_dataset)
```

### 3. 不确定性分析

```python
from neuronal_network_v3.evaluation.multi_channel_evaluation import RatioEvaluationMetrics

# 校准性分析
calibration_metrics = RatioEvaluationMetrics.compute_calibration_metrics(
    pred_mean, pred_std, true_values
)
print(f"Expected Calibration Error: {calibration_metrics['ece']:.4f}")

# 锐度分析
sharpness_metrics = RatioEvaluationMetrics.compute_sharpness(pred_std)
print(f"Mean uncertainty: {sharpness_metrics['mean_uncertainty']:.4f}")
```

## 📈 性能优化

### 1. 内存优化
- 使用梯度检查点减少内存使用
- 自动批大小调整
- 混合精度训练

### 2. 计算优化
- GPU并行处理
- 异步数据加载
- 模型编译优化

### 3. 训练技巧
- 学习率调度
- 早停机制
- 模型集成

## 🐛 故障排除

### 常见问题

1. **内存不足**
   ```python
   # 减少批大小
   config['training']['batch_size'] = 8
   
   # 启用梯度检查点
   config['hardware']['gradient_checkpointing'] = True
   ```

2. **训练不稳定**
   ```python
   # 降低学习率
   config['training']['optimizer']['lr'] = 1e-4
   
   # 增加梯度裁剪
   config['hardware']['gradient_clipping'] = 0.5
   ```

3. **物理约束违反**
   ```python
   # 增加约束权重
   config['loss']['joint_params']['conservation_weight'] = 1.0
   config['loss']['joint_params']['consistency_weight'] = 0.5
   ```

### 调试技巧

```python
# 启用详细日志
logging.getLogger().setLevel(logging.DEBUG)

# 检查数据统计
data_stats = dataset.get_statistics()
print(f"Data statistics: {data_stats}")

# 验证物理约束
conservation_error = torch.abs(pred_ch1_photons + pred_ch2_photons - total_photons)
print(f"Conservation error: {conservation_error.mean():.4f}")
```

## 🎯 最佳实践

### 数据管理
- 使用HDF5格式存储大型数据集
- 实施数据版本控制和校验
- 定期备份重要数据和模型
- 使用数据缓存提高训练效率

### 模型开发
- 遵循模块化设计原则
- 使用配置文件管理超参数
- 实施全面的单元测试
- 记录模型架构和训练过程

### 训练策略
- 使用混合精度训练节省内存
- 实施早停和学习率调度
- 定期验证和保存检查点
- 监控训练指标和资源使用

### 代码质量
- 遵循PEP 8代码规范
- 编写清晰的文档字符串
- 使用类型提示提高代码可读性
- 实施代码审查流程

## 🔧 项目管理

### 目录结构建议
```
project_workspace/
├── data/
│   ├── raw/                    # 原始数据
│   ├── processed/              # 处理后数据
│   └── cache/                  # 数据缓存
├── experiments/
│   ├── exp_001_baseline/       # 实验1：基线模型
│   ├── exp_002_multi_channel/  # 实验2：多通道模型
│   └── exp_003_optimization/   # 实验3：优化版本
├── models/
│   ├── checkpoints/            # 模型检查点
│   ├── final/                  # 最终模型
│   └── pretrained/             # 预训练模型
├── results/
│   ├── figures/                # 结果图表
│   ├── reports/                # 分析报告
│   └── metrics/                # 评估指标
└── logs/
    ├── training/               # 训练日志
    ├── inference/              # 推理日志
    └── system/                 # 系统日志
```

### 实验管理
- 为每个实验创建独立目录
- 记录实验配置和结果
- 使用版本控制跟踪代码变更
- 建立实验结果对比机制

### 性能监控
- 监控GPU/CPU使用率
- 跟踪内存消耗情况
- 记录训练时间和收敛性
- 建立性能基准测试

## 📚 参考文献

1. DECODE原始论文: [Deep learning enables fast and dense single-molecule localization with high accuracy](https://www.nature.com/articles/s41592-021-01236-x)
2. 不确定性量化: [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977)
3. 物理约束学习: [Physics-informed neural networks](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
4. 多任务学习: [Multi-Task Learning Using Uncertainty to Weigh Losses](https://arxiv.org/abs/1705.07115)
5. 深度学习最佳实践: [Deep Learning Best Practices](https://www.deeplearningbook.org/)

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出改进建议！

### 贡献流程
1. Fork项目到个人仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

### 代码贡献规范
- 遵循现有代码风格
- 添加适当的测试用例
- 更新相关文档
- 确保所有测试通过

### 问题报告
- 使用清晰的标题描述问题
- 提供详细的重现步骤
- 包含系统环境信息
- 附上相关的错误日志

## 📄 许可证

本项目遵循原DECODE项目的许可证条款。详细信息请参考LICENSE文件。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 创建GitHub Issue（推荐）
- 发送邮件至项目维护者
- 参与项目讨论区

## 🙏 致谢

感谢以下贡献者和项目：
- 原DECODE项目团队
- PyTorch社区
- 科学计算Python生态系统
- 所有测试用户和反馈提供者

---

**注意**: 这是基于`DECODE_Network_Analysis.md`文档实现的多通道扩展。使用前请确保理解相关的理论基础和实现细节。建议先阅读各模块的详细文档，然后根据具体需求选择合适的功能组件。