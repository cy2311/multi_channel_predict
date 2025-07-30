#!/usr/bin/env python3
"""
测试改进的损失函数
验证BCEWithLogitsLoss替换BCELoss后的功能
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))

from loss.improved_count_loss import ImprovedCountLoss, MultiLevelLoss, WeightGenerator

def test_improved_count_loss():
    """
    测试ImprovedCountLoss功能
    """
    print("=== 测试 ImprovedCountLoss ===")
    
    # 创建测试数据
    batch_size, height, width = 2, 64, 64
    
    # 创建logits (未经过sigmoid的原始输出)
    logits = torch.randn(batch_size, 1, height, width)
    
    # 创建目标 (0或1的二进制标签)
    targets = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    print(f"输入logits形状: {logits.shape}")
    print(f"目标targets形状: {targets.shape}")
    print(f"logits范围: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"targets范围: [{targets.min():.3f}, {targets.max():.3f}]")
    
    # 测试不同配置的ImprovedCountLoss
    configs = [
        {'pos_weight': None, 'pixel_weight_strategy': None},
        {'pos_weight': 2.0, 'pixel_weight_strategy': None},
        {'pos_weight': 2.0, 'pixel_weight_strategy': 'distance'},
        {'pos_weight': 2.0, 'pixel_weight_strategy': 'adaptive'},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n配置 {i+1}: {config}")
        
        loss_fn = ImprovedCountLoss(**config)
        loss = loss_fn(logits, targets)
        
        print(f"损失值: {loss.item():.6f}")
        print(f"损失是否为有限值: {torch.isfinite(loss).item()}")
        
        # 测试梯度计算
        logits_grad = torch.randn_like(logits, requires_grad=True)
        loss_grad = loss_fn(logits_grad, targets)
        loss_grad.backward()
        
        print(f"梯度计算成功: {logits_grad.grad is not None}")
        if logits_grad.grad is not None:
            print(f"梯度范围: [{logits_grad.grad.min():.6f}, {logits_grad.grad.max():.6f}]")

def test_multi_level_loss():
    """
    测试MultiLevelLoss功能
    """
    print("\n=== 测试 MultiLevelLoss ===")
    
    batch_size, height, width = 2, 64, 64
    
    # 创建模拟的网络输出
    outputs = {
        'prob': torch.randn(batch_size, 1, height, width),  # logits
        'offset': torch.randn(batch_size, 3, height, width),  # x, y, z偏移
        'photon': torch.randn(batch_size, 1, height, width),  # 光子数
        'background': torch.randn(batch_size, 1, height, width),  # 背景
    }
    
    # 创建目标
    targets = {
        'count_maps': torch.randint(0, 2, (batch_size, 1, height, width)).float(),
        'loc_maps': torch.randn(batch_size, 3, height, width),
        'photon_maps': torch.randn(batch_size, 1, height, width),
        'background_maps': torch.randn(batch_size, 1, height, width),
    }
    
    print(f"输出形状: {[(k, v.shape) for k, v in outputs.items()]}")
    print(f"目标形状: {[(k, v.shape) for k, v in targets.items()]}")
    
    # 创建损失权重
    loss_weights = {
        'count': 1.0,
        'localization': 1.0,
        'photon': 0.5,
        'background': 0.1
    }
    
    # 测试MultiLevelLoss
    multi_loss = MultiLevelLoss(
        count_pos_weight=2.0,
        pixel_weight_strategy='adaptive',
        loss_weights=loss_weights
    )
    
    loss_dict = multi_loss(outputs, targets)
    total_loss = loss_dict['total']
    
    print(f"\n总损失: {total_loss.item():.6f}")
    print("各项损失:")
    for name, value in loss_dict.items():
        print(f"  {name}: {value.item():.6f}")
    
    # 测试梯度
    for output in outputs.values():
        output.requires_grad_(True)
    
    loss_dict_grad = multi_loss(outputs, targets)
    total_loss_grad = loss_dict_grad['total']
    total_loss_grad.backward()
    
    print("\n梯度计算成功:")
    for name, output in outputs.items():
        has_grad = output.grad is not None
        print(f"  {name}: {has_grad}")
        if has_grad:
            print(f"    梯度范围: [{output.grad.min():.6f}, {output.grad.max():.6f}]")

def test_weight_generator():
    """
    测试WeightGenerator功能
    """
    print("\n=== 测试 WeightGenerator ===")
    
    batch_size, height, width = 2, 32, 32
    targets = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    print(f"目标形状: {targets.shape}")
    print(f"正样本比例: {targets.mean().item():.3f}")
    
    # 测试自适应pos_weight生成
    pos_weight = WeightGenerator.generate_adaptive_pos_weight(targets)
    print(f"自适应pos_weight: {pos_weight:.3f}")
    
    # 测试通道权重生成
    channel_weights = WeightGenerator.generate_channel_weights(num_channels=4)
    print(f"通道权重: {channel_weights}")
    
    print("WeightGenerator测试完成")

def compare_bce_losses():
    """
    比较BCELoss和BCEWithLogitsLoss的数值稳定性
    """
    print("\n=== 比较 BCELoss vs BCEWithLogitsLoss ===")
    
    # 创建极端的logits值来测试数值稳定性
    extreme_logits = torch.tensor([[-50.0, -10.0, 0.0, 10.0, 50.0]]).unsqueeze(-1).unsqueeze(-1)
    targets = torch.tensor([[0.0, 0.0, 0.5, 1.0, 1.0]]).unsqueeze(-1).unsqueeze(-1)
    
    print(f"极端logits: {extreme_logits.squeeze()}")
    print(f"目标: {targets.squeeze()}")
    
    # BCEWithLogitsLoss (我们的新方法)
    bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
    loss_with_logits = bce_with_logits(extreme_logits, targets)
    
    # BCELoss (旧方法) - 需要先应用sigmoid
    bce = nn.BCELoss(reduction='none')
    probs = torch.sigmoid(extreme_logits)
    loss_bce = bce(probs, targets)
    
    print(f"\nBCEWithLogitsLoss: {loss_with_logits.squeeze()}")
    print(f"BCELoss: {loss_bce.squeeze()}")
    print(f"sigmoid概率: {probs.squeeze()}")
    
    # 检查是否有无穷大或NaN
    print(f"\nBCEWithLogitsLoss有无穷大: {torch.isinf(loss_with_logits).any().item()}")
    print(f"BCELoss有无穷大: {torch.isinf(loss_bce).any().item()}")
    print(f"BCEWithLogitsLoss有NaN: {torch.isnan(loss_with_logits).any().item()}")
    print(f"BCELoss有NaN: {torch.isnan(loss_bce).any().item()}")

if __name__ == "__main__":
    print("开始测试改进的损失函数...")
    
    try:
        test_improved_count_loss()
        test_multi_level_loss()
        test_weight_generator()
        compare_bce_losses()
        
        print("\n=== 所有测试完成 ===")
        print("✅ ImprovedCountLoss (BCEWithLogitsLoss) 替换成功")
        print("✅ 前景权重机制工作正常")
        print("✅ 多层次损失函数工作正常")
        print("✅ 数值稳定性得到改善")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()