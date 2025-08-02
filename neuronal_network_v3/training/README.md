# Training 训练模块

本模块包含DECODE神经网络v3的完整训练系统，支持单通道和多通道SMLM数据的模型训练、优化和监控。

## 📋 模块概览

### 核心组件

#### 🔹 MultiChannelTrainer (`multi_channel_trainer.py`)
- **功能**: 多通道系统专用训练器
- **特点**:
  - 双通道联合训练
  - 比例预测训练
  - 物理约束集成
  - 不确定性训练
- **用途**: 多通道模型的端到端训练

#### 🔹 BaseTrainer (`trainer.py`)
- **功能**: 基础训练器框架
- **特点**:
  - 灵活的训练流程
  - 多种优化器支持
  - 学习率调度
  - 早停机制
- **用途**: 单通道模型和基础训练任务

#### 🔹 TrainingConfig (`config.py`)
- **功能**: 训练配置管理
- **特点**:
  - 参数验证
  - 配置继承
  - 动态配置
  - 配置保存/加载
- **用途**: 统一的训练参数管理

#### 🔹 TrainingMonitor (`monitor.py`)
- **功能**: 训练过程监控
- **特点**:
  - 实时指标追踪
  - 可视化监控
  - 异常检测
  - 性能分析
- **用途**: 训练过程的实时监控和分析

#### 🔹 CheckpointManager (`checkpoint.py`)
- **功能**: 模型检查点管理
- **特点**:
  - 自动保存
  - 版本管理
  - 最佳模型追踪
  - 断点续训
- **用途**: 训练状态的保存和恢复

#### 🔹 LossScheduler (`scheduler.py`)
- **功能**: 损失函数调度
- **特点**:
  - 动态权重调整
  - 多阶段训练
  - 自适应调度
  - 损失平衡
- **用途**: 复杂损失函数的动态调整

## 🚀 使用示例

### 多通道训练

```python
from neuronal_network_v3.training import MultiChannelTrainer
from neuronal_network_v3.models import DoubleMUNet
from neuronal_network_v3.loss import UnifiedLoss
from neuronal_network_v3.data import create_multi_channel_dataloader

# 创建数据加载器
train_loader = create_multi_channel_dataloader(
    data_path='data/train_multi_channel.h5',
    config={
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True
    },
    mode='train'
)

val_loader = create_multi_channel_dataloader(
    data_path='data/val_multi_channel.h5',
    config={'batch_size': 32, 'shuffle': False},
    mode='val'
)

# 初始化模型
model = DoubleMUNet(
    in_channels=1,
    out_channels_ch1=5,  # x, y, z, photons, bg
    out_channels_ch2=5,
    out_channels_ratio=1,
    uncertainty=True
)

# 初始化损失函数
loss_fn = UnifiedLoss(
    ch1_weight=1.0,
    ch2_weight=1.0,
    ratio_weight=0.5,
    conservation_weight=0.1,
    uncertainty_weight=0.1
)

# 初始化训练器
trainer = MultiChannelTrainer(
    model=model,
    loss_fn=loss_fn,
    optimizer_config={
        'type': 'AdamW',
        'lr': 1e-3,
        'weight_decay': 1e-4
    },
    scheduler_config={
        'type': 'CosineAnnealingLR',
        'T_max': 100,
        'eta_min': 1e-6
    },
    device='cuda',
    mixed_precision=True
)

# 开始训练
training_history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    save_dir='checkpoints/multi_channel/',
    save_every=10,
    validate_every=5,
    early_stopping_patience=15
)

# 查看训练历史
print("=== 训练完成 ===")
print(f"最佳验证损失: {training_history['best_val_loss']:.6f}")
print(f"最佳模型epoch: {training_history['best_epoch']}")
print(f"训练总时长: {training_history['total_time']:.2f}秒")
```

### 单通道训练

```python
from neuronal_network_v3.training import BaseTrainer
from neuronal_network_v3.models import SigmaMUNet
from neuronal_network_v3.loss import RatioGaussianNLLLoss

# 初始化模型
model = SigmaMUNet(
    in_channels=1,
    out_channels=5,  # x, y, z, photons, bg
    uncertainty=True
)

# 初始化损失函数
loss_fn = RatioGaussianNLLLoss()

# 初始化训练器
trainer = BaseTrainer(
    model=model,
    loss_fn=loss_fn,
    optimizer_config={
        'type': 'Adam',
        'lr': 2e-3,
        'betas': (0.9, 0.999)
    },
    scheduler_config={
        'type': 'StepLR',
        'step_size': 30,
        'gamma': 0.5
    }
)

# 训练配置
training_config = {
    'epochs': 80,
    'gradient_clip_val': 1.0,
    'accumulate_grad_batches': 2,
    'log_every': 100,
    'validate_every': 5
}

# 开始训练
results = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    **training_config
)
```

### 高级训练配置

```python
from neuronal_network_v3.training import TrainingConfig, LossScheduler

# 创建训练配置
config = TrainingConfig(
    # 基础配置
    epochs=100,
    batch_size=16,
    learning_rate=1e-3,
    
    # 优化器配置
    optimizer={
        'type': 'AdamW',
        'weight_decay': 1e-4,
        'amsgrad': True
    },
    
    # 学习率调度
    scheduler={
        'type': 'OneCycleLR',
        'max_lr': 1e-2,
        'pct_start': 0.3,
        'anneal_strategy': 'cos'
    },
    
    # 正则化
    regularization={
        'dropout': 0.1,
        'batch_norm': True,
        'weight_decay': 1e-4
    },
    
    # 训练策略
    training_strategy={
        'mixed_precision': True,
        'gradient_clipping': 1.0,
        'accumulate_grad_batches': 1,
        'early_stopping_patience': 20
    },
    
    # 损失函数权重调度
    loss_scheduling={
        'enable': True,
        'schedule_type': 'linear',
        'warmup_epochs': 10,
        'final_weights': {
            'ch1_weight': 1.0,
            'ch2_weight': 1.0,
            'ratio_weight': 0.8,
            'conservation_weight': 0.2
        }
    },
    
    # 监控配置
    monitoring={
        'log_every': 50,
        'validate_every': 5,
        'save_every': 10,
        'plot_every': 20
    }
)

# 使用配置初始化训练器
trainer = MultiChannelTrainer.from_config(config)
```

### 损失函数调度

```python
from neuronal_network_v3.training import LossScheduler

# 创建损失调度器
loss_scheduler = LossScheduler(
    total_epochs=100,
    schedule_config={
        'ch1_weight': {
            'type': 'constant',
            'value': 1.0
        },
        'ch2_weight': {
            'type': 'constant', 
            'value': 1.0
        },
        'ratio_weight': {
            'type': 'linear',
            'start_value': 0.1,
            'end_value': 0.8,
            'start_epoch': 10,
            'end_epoch': 50
        },
        'conservation_weight': {
            'type': 'exponential',
            'start_value': 0.01,
            'end_value': 0.2,
            'decay_rate': 0.1
        },
        'uncertainty_weight': {
            'type': 'cosine',
            'min_value': 0.05,
            'max_value': 0.15,
            'period': 20
        }
    }
)

# 在训练循环中使用
for epoch in range(100):
    # 更新损失权重
    current_weights = loss_scheduler.get_weights(epoch)
    loss_fn.update_weights(**current_weights)
    
    # 训练一个epoch
    train_loss = trainer.train_epoch(train_loader)
    
    print(f"Epoch {epoch}: Loss={train_loss:.6f}, Weights={current_weights}")
```

### 训练监控

```python
from neuronal_network_v3.training import TrainingMonitor
import wandb

# 初始化监控器
monitor = TrainingMonitor(
    log_dir='logs/training/',
    use_tensorboard=True,
    use_wandb=True,
    wandb_project='decode_v3',
    wandb_config={
        'model': 'DoubleMUNet',
        'dataset': 'multi_channel_smlm',
        'experiment': 'baseline_training'
    }
)

# 在训练循环中记录指标
for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_metrics = {'loss': 0, 'ch1_loss': 0, 'ch2_loss': 0, 'ratio_loss': 0}
    
    for batch_idx, batch in enumerate(train_loader):
        # ... 训练代码 ...
        
        # 记录批次指标
        monitor.log_batch_metrics(
            epoch=epoch,
            batch_idx=batch_idx,
            metrics={
                'train_loss': loss.item(),
                'train_ch1_loss': ch1_loss.item(),
                'train_ch2_loss': ch2_loss.item(),
                'train_ratio_loss': ratio_loss.item(),
                'learning_rate': optimizer.param_groups[0]['lr']
            }
        )
    
    # 验证阶段
    val_metrics = trainer.validate(val_loader)
    
    # 记录epoch指标
    monitor.log_epoch_metrics(
        epoch=epoch,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        model=model
    )
    
    # 生成可视化
    if epoch % 10 == 0:
        monitor.plot_training_curves()
        monitor.plot_model_predictions(model, val_loader)
```

### 检查点管理

```python
from neuronal_network_v3.training import CheckpointManager

# 初始化检查点管理器
checkpoint_manager = CheckpointManager(
    save_dir='checkpoints/multi_channel/',
    save_top_k=3,  # 保存最好的3个模型
    monitor_metric='val_loss',
    mode='min',
    save_every_n_epochs=5,
    save_last=True
)

# 在训练循环中使用
for epoch in range(epochs):
    # ... 训练代码 ...
    
    # 验证
    val_metrics = trainer.validate(val_loader)
    
    # 保存检查点
    checkpoint_manager.save_checkpoint(
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=val_metrics,
        config=training_config
    )
    
    # 检查是否需要早停
    if checkpoint_manager.should_stop_early(patience=15):
        print(f"Early stopping at epoch {epoch}")
        break

# 加载最佳模型
best_checkpoint = checkpoint_manager.load_best_checkpoint()
model.load_state_dict(best_checkpoint['model_state_dict'])

# 断点续训
if checkpoint_manager.has_checkpoint():
    checkpoint = checkpoint_manager.load_last_checkpoint()
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"从epoch {start_epoch}继续训练")
```

## 🔧 高级训练技术

### 混合精度训练

```python
from torch.cuda.amp import GradScaler, autocast

class MixedPrecisionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = GradScaler()
        self.use_amp = True
    
    def train_step(self, batch):
        inputs, targets = batch
        
        with autocast(enabled=self.use_amp):
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
        
        # 反向传播
        self.scaler.scale(loss).backward()
        
        # 梯度裁剪
        if self.gradient_clip_val > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
        
        # 优化器步骤
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        return loss.item()
```

### 梯度累积

```python
class GradientAccumulationTrainer(BaseTrainer):
    def __init__(self, accumulate_grad_batches=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulate_grad_batches = accumulate_grad_batches
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            loss = self.train_step(batch)
            
            # 标准化损失
            loss = loss / self.accumulate_grad_batches
            loss.backward()
            
            # 每accumulate_grad_batches步更新一次
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulate_grad_batches
        
        return total_loss / len(train_loader)
```

### 自适应学习率

```python
class AdaptiveLRTrainer(BaseTrainer):
    def __init__(self, patience=10, factor=0.5, min_lr=1e-7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def adjust_learning_rate(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                current_lr = self.optimizer.param_groups[0]['lr']
                new_lr = max(current_lr * self.factor, self.min_lr)
                
                if new_lr < current_lr:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"学习率降低到: {new_lr:.2e}")
                    self.patience_counter = 0
```

### 多GPU训练

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

class DistributedTrainer(BaseTrainer):
    def __init__(self, local_rank, world_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_rank = local_rank
        self.world_size = world_size
        
        # 初始化分布式训练
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        # 包装模型
        self.model = DDP(self.model, device_ids=[local_rank])
    
    def create_distributed_dataloader(self, dataset, batch_size, shuffle=True):
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.local_rank,
            shuffle=shuffle
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
    
    def train_epoch(self, train_loader):
        # 设置epoch以确保数据shuffle
        train_loader.sampler.set_epoch(self.current_epoch)
        
        return super().train_epoch(train_loader)
```

### 课程学习

```python
class CurriculumTrainer(BaseTrainer):
    def __init__(self, curriculum_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_config = curriculum_config
        self.current_stage = 0
    
    def get_current_curriculum(self, epoch):
        """根据epoch获取当前课程设置"""
        for stage, config in enumerate(self.curriculum_config):
            if epoch < config['end_epoch']:
                return stage, config
        return len(self.curriculum_config) - 1, self.curriculum_config[-1]
    
    def train_epoch(self, train_loader):
        stage, config = self.get_current_curriculum(self.current_epoch)
        
        if stage != self.current_stage:
            print(f"切换到课程阶段 {stage}: {config['description']}")
            self.current_stage = stage
            
            # 调整学习率
            if 'learning_rate' in config:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = config['learning_rate']
            
            # 调整损失权重
            if 'loss_weights' in config:
                self.loss_fn.update_weights(**config['loss_weights'])
        
        return super().train_epoch(train_loader)
```

### 知识蒸馏

```python
class DistillationTrainer(BaseTrainer):
    def __init__(self, teacher_model, temperature=4.0, alpha=0.7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def distillation_loss(self, student_outputs, teacher_outputs, targets):
        # 硬目标损失
        hard_loss = self.loss_fn(student_outputs, targets)
        
        # 软目标损失
        student_soft = F.log_softmax(student_outputs / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_outputs / self.temperature, dim=1)
        soft_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # 组合损失
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        return total_loss
    
    def train_step(self, batch):
        inputs, targets = batch
        
        # 学生模型前向传播
        student_outputs = self.model(inputs)
        
        # 教师模型前向传播
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
        
        # 计算蒸馏损失
        loss = self.distillation_loss(student_outputs, teacher_outputs, targets)
        
        # 反向传播
        loss.backward()
        
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
```

## 📊 训练策略

### 学习率调度策略

```python
# 1. 余弦退火
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)

# 2. 一周期学习率
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-2, steps_per_epoch=len(train_loader), epochs=100
)

# 3. 指数衰减
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=0.95
)

# 4. 自适应调整
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7
)

# 5. 分段常数
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[30, 60, 90], gamma=0.1
)
```

### 数据增强策略

```python
from neuronal_network_v3.data.transforms import *

# 训练时数据增强
train_transforms = Compose([
    RandomRotation(angle_range=(-180, 180)),
    RandomFlip(p=0.5),
    GaussianNoise(noise_std=0.1),
    RandomBrightness(factor_range=(0.8, 1.2)),
    RandomContrast(factor_range=(0.8, 1.2)),
    Normalize(mean=0.5, std=0.5)
])

# 验证时只做标准化
val_transforms = Compose([
    Normalize(mean=0.5, std=0.5)
])
```

### 正则化技术

```python
# 1. Dropout
model = SigmaMUNet(
    in_channels=1,
    out_channels=5,
    dropout=0.1
)

# 2. 权重衰减
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

# 3. 标签平滑
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        # 实现标签平滑
        pass

# 4. 梯度惩罚
def gradient_penalty(model, real_data, fake_data):
    # 实现梯度惩罚
    pass
```

## 🐛 常见问题

### Q: 训练过程中损失不下降怎么办？
A: 排查步骤：
1. 检查学习率是否合适
2. 验证数据预处理
3. 检查模型架构
4. 确认损失函数实现
5. 检查梯度流动
6. 尝试不同的优化器

### Q: 训练速度慢怎么办？
A: 优化方法：
1. 使用混合精度训练
2. 增加batch size
3. 使用多GPU训练
4. 优化数据加载
5. 使用更快的存储
6. 减少不必要的计算

### Q: 模型过拟合怎么办？
A: 解决方案：
1. 增加数据增强
2. 使用正则化技术
3. 减少模型复杂度
4. 早停训练
5. 使用更多训练数据
6. 交叉验证

### Q: 内存不足怎么办？
A: 解决方法：
1. 减少batch size
2. 使用梯度累积
3. 使用梯度检查点
4. 优化模型架构
5. 使用数据并行
6. 清理不必要的变量

### Q: 如何调试训练过程？
A: 调试技巧：
1. 可视化损失曲线
2. 监控梯度范数
3. 检查学习率变化
4. 分析验证指标
5. 使用tensorboard
6. 保存中间结果

## 📚 相关文档

- [模型文档](../models/README.md)
- [损失函数文档](../loss/README.md)
- [数据处理文档](../data/README.md)
- [评估模块文档](../evaluation/README.md)
- [推理模块文档](../inference/README.md)
- [多通道训练指南](../README_MultiChannel.md)
- [训练最佳实践](./TRAINING_BEST_PRACTICES.md)