#!/usr/bin/env python3
"""
多通道DECODE网络使用示例

本脚本演示如何使用多通道扩展功能，包括：
1. 数据准备和加载
2. 模型初始化
3. 三阶段训练
4. 推理和评估
5. 结果可视化

作者: DECODE团队
日期: 2024
"""

import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from neuronal_network_v3.models.sigma_munet import SigmaMUNet
from neuronal_network_v3.models.ratio_net import RatioNet, FeatureExtractor
from neuronal_network_v3.trainer.multi_channel_trainer import MultiChannelTrainer
from neuronal_network_v3.data.multi_channel_dataset import (
    MultiChannelSMLMDataset, 
    MultiChannelDataModule,
    create_multi_channel_dataloader
)
from neuronal_network_v3.inference.multi_channel_infer import MultiChannelInfer
from neuronal_network_v3.evaluation.multi_channel_evaluation import MultiChannelEvaluation
from neuronal_network_v3.loss.ratio_loss import (
    RatioGaussianNLLLoss,
    MultiChannelLossWithGaussianRatio
)


def create_synthetic_data(num_samples: int = 100, image_size: int = 64) -> Dict[str, np.ndarray]:
    """
    创建合成的多通道数据用于演示
    
    Args:
        num_samples: 样本数量
        image_size: 图像尺寸
        
    Returns:
        包含多通道数据的字典
    """
    print("创建合成多通道数据...")
    
    # 生成随机图像
    channel1_images = np.random.randn(num_samples, image_size, image_size).astype(np.float32)
    channel2_images = np.random.randn(num_samples, image_size, image_size).astype(np.float32)
    
    # 生成目标（10通道输出）
    channel1_targets = np.random.randn(num_samples, 10, image_size, image_size).astype(np.float32)
    channel2_targets = np.random.randn(num_samples, 10, image_size, image_size).astype(np.float32)
    
    # 生成光子数
    total_photons = np.random.uniform(100, 1000, num_samples).astype(np.float32)
    ratios = np.random.uniform(0.2, 0.8, num_samples).astype(np.float32)
    
    channel1_photons = total_photons * ratios
    channel2_photons = total_photons * (1 - ratios)
    
    return {
        'channel1_images': channel1_images,
        'channel2_images': channel2_images,
        'channel1_targets': channel1_targets,
        'channel2_targets': channel2_targets,
        'channel1_photons': channel1_photons,
        'channel2_photons': channel2_photons,
        'total_photons': total_photons,
        'ratios': ratios
    }


def create_example_config() -> Dict[str, Any]:
    """
    创建示例配置
    
    Returns:
        配置字典
    """
    return {
        'model': {
            'n_inp': 1,
            'n_out': 10,
            'sigma_munet_params': {
                'depth': 2,
                'initial_features': 32,
                'norm': 'BatchNorm2d',
                'activation': 'ReLU',
                'final_activation': 'Sigmoid'
            },
            'ratio_net': {
                'hidden_dim': 64,
                'num_layers': 2,
                'dropout': 0.1,
                'activation': 'ReLU',
                'output_activation': 'Sigmoid'
            }
        },
        'loss': {
            'channel_loss_type': 'gaussian_mm',
            'ratio_params': {
                'photon_conservation_weight': 1.0,
                'ratio_consistency_weight': 0.5,
                'uncertainty_regularization': 0.1
            },
            'joint_params': {
                'channel_weight': 1.0,
                'ratio_weight': 0.5,
                'conservation_weight': 1.0,
                'consistency_weight': 0.5
            }
        },
        'training': {
            'stage1_epochs': 5,
            'stage2_epochs': 3,
            'stage3_epochs': 2,
            'batch_size': 8,
            'optimizer': {
                'type': 'Adam',
                'lr': 1e-3,
                'weight_decay': 1e-5
            },
            'scheduler': {
                'type': 'StepLR',
                'step_size': 10,
                'gamma': 0.5
            }
        },
        'data': {
            'patch_size': 64,
            'ratio_calculation': 'photon_based',
            'apply_transforms': True,
            'normalize': True
        },
        'hardware': {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_workers': 2,
            'pin_memory': True
        }
    }


def demonstrate_model_initialization(config: Dict[str, Any]) -> Tuple[SigmaMUNet, SigmaMUNet, RatioNet]:
    """
    演示模型初始化
    
    Args:
        config: 配置字典
        
    Returns:
        初始化的模型
    """
    print("\n=== 模型初始化演示 ===")
    
    # 初始化通道网络
    channel1_net = SigmaMUNet(
        n_inp=config['model']['n_inp'],
        n_out=config['model']['n_out'],
        **config['model']['sigma_munet_params']
    )
    
    channel2_net = SigmaMUNet(
        n_inp=config['model']['n_inp'],
        n_out=config['model']['n_out'],
        **config['model']['sigma_munet_params']
    )
    
    # 初始化比例网络
    ratio_net = RatioNet(
        input_channels=config['model']['n_out'] * 2,  # 两个通道的输出
        **config['model']['ratio_net']
    )
    
    print(f"通道1网络参数数量: {sum(p.numel() for p in channel1_net.parameters()):,}")
    print(f"通道2网络参数数量: {sum(p.numel() for p in channel2_net.parameters()):,}")
    print(f"比例网络参数数量: {sum(p.numel() for p in ratio_net.parameters()):,}")
    
    return channel1_net, channel2_net, ratio_net


def demonstrate_data_loading(data: Dict[str, np.ndarray], config: Dict[str, Any]):
    """
    演示数据加载
    
    Args:
        data: 合成数据
        config: 配置字典
    """
    print("\n=== 数据加载演示 ===")
    
    # 创建数据集
    dataset = MultiChannelSMLMDataset(
        channel1_images=data['channel1_images'],
        channel2_images=data['channel2_images'],
        channel1_targets=data['channel1_targets'],
        channel2_targets=data['channel2_targets'],
        channel1_photons=data['channel1_photons'],
        channel2_photons=data['channel2_photons'],
        ratio_calculation=config['data']['ratio_calculation']
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 获取一个样本
    sample = dataset[0]
    print(f"样本键: {list(sample.keys())}")
    print(f"通道1图像形状: {sample['channel1_image'].shape}")
    print(f"通道2图像形状: {sample['channel2_image'].shape}")
    print(f"真实比例: {sample['true_ratio']:.4f}")
    
    # 创建数据加载器
    dataloader = create_multi_channel_dataloader(
        dataset=dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers']
    )
    
    # 获取一个批次
    batch = next(iter(dataloader))
    print(f"批次大小: {batch['channel1_image'].shape[0]}")
    print(f"批次通道1图像形状: {batch['channel1_image'].shape}")
    
    return dataset, dataloader


def demonstrate_training(models: Tuple[SigmaMUNet, SigmaMUNet, RatioNet], 
                        dataloader, config: Dict[str, Any]):
    """
    演示训练过程
    
    Args:
        models: 模型元组
        dataloader: 数据加载器
        config: 配置字典
    """
    print("\n=== 训练演示 ===")
    
    channel1_net, channel2_net, ratio_net = models
    device = config['hardware']['device']
    
    # 初始化训练器
    trainer = MultiChannelTrainer(config, device=device)
    trainer.setup_models(channel1_net, channel2_net, ratio_net)
    
    print("开始三阶段训练...")
    
    # 执行训练（使用较少的epoch用于演示）
    results = trainer.train_full_pipeline(
        train_loader=dataloader,
        val_loader=dataloader,  # 演示中使用相同的数据
        save_dir='demo_outputs'
    )
    
    print("训练完成！")
    print(f"阶段1最佳损失: {results['stage1']['best_val_loss']:.4f}")
    print(f"阶段2最佳损失: {results['stage2']['best_val_loss']:.4f}")
    print(f"阶段3最佳损失: {results['stage3']['best_val_loss']:.4f}")
    
    return trainer, results


def demonstrate_inference(models: Tuple[SigmaMUNet, SigmaMUNet, RatioNet],
                         data: Dict[str, np.ndarray], config: Dict[str, Any]):
    """
    演示推理过程
    
    Args:
        models: 训练好的模型
        data: 测试数据
        config: 配置字典
    """
    print("\n=== 推理演示 ===")
    
    channel1_net, channel2_net, ratio_net = models
    device = config['hardware']['device']
    
    # 初始化推理器
    inferrer = MultiChannelInfer(
        channel1_net=channel1_net,
        channel2_net=channel2_net,
        ratio_net=ratio_net,
        device=device,
        apply_conservation=True,
        apply_consistency=True
    )
    
    # 准备测试数据
    test_ch1 = torch.from_numpy(data['channel1_images'][:10]).unsqueeze(1)  # 添加通道维度
    test_ch2 = torch.from_numpy(data['channel2_images'][:10]).unsqueeze(1)
    
    print(f"测试数据形状: {test_ch1.shape}")
    
    # 执行推理
    with torch.no_grad():
        results = inferrer.predict(test_ch1, test_ch2)
    
    print("推理完成！")
    print(f"预测比例形状: {results['ratio_mean'].shape}")
    print(f"不确定性形状: {results['ratio_std'].shape}")
    print(f"平均预测比例: {results['ratio_mean'].mean():.4f}")
    print(f"平均不确定性: {results['ratio_std'].mean():.4f}")
    
    return results


def demonstrate_evaluation(pred_results: Dict[str, torch.Tensor],
                         true_data: Dict[str, np.ndarray], config: Dict[str, Any]):
    """
    演示评估过程
    
    Args:
        pred_results: 预测结果
        true_data: 真实数据
        config: 配置字典
    """
    print("\n=== 评估演示 ===")
    
    device = config['hardware']['device']
    
    # 初始化评估器
    evaluator = MultiChannelEvaluation(device=device)
    
    # 准备真实数据
    ground_truth = {
        'ratios': torch.from_numpy(true_data['ratios'][:10]),
        'channel1_photons': torch.from_numpy(true_data['channel1_photons'][:10]),
        'channel2_photons': torch.from_numpy(true_data['channel2_photons'][:10]),
        'total_photons': torch.from_numpy(true_data['total_photons'][:10])
    }
    
    # 执行评估
    metrics = evaluator.evaluate(pred_results, ground_truth)
    
    print("评估完成！")
    print(f"比例MAE: {metrics['ratio']['mae']:.4f}")
    print(f"比例RMSE: {metrics['ratio']['rmse']:.4f}")
    print(f"95%覆盖率: {metrics['ratio']['coverage_95']:.4f}")
    print(f"光子数守恒误差: {metrics['conservation']['conservation_error']:.4f}")
    print(f"比例一致性误差: {metrics['conservation']['consistency_error']:.4f}")
    
    return metrics


def demonstrate_visualization(pred_results: Dict[str, torch.Tensor],
                            true_data: Dict[str, np.ndarray]):
    """
    演示结果可视化
    
    Args:
        pred_results: 预测结果
        true_data: 真实数据
    """
    print("\n=== 可视化演示 ===")
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 比例预测 vs 真实值
    pred_ratios = pred_results['ratio_mean'].cpu().numpy()
    true_ratios = true_data['ratios'][:len(pred_ratios)]
    
    axes[0, 0].scatter(true_ratios, pred_ratios, alpha=0.7)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
    axes[0, 0].set_xlabel('True Ratio')
    axes[0, 0].set_ylabel('Predicted Ratio')
    axes[0, 0].set_title('Ratio Prediction')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 不确定性分布
    uncertainties = pred_results['ratio_std'].cpu().numpy()
    axes[0, 1].hist(uncertainties, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Predicted Uncertainty')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Uncertainty Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 预测误差 vs 不确定性
    errors = np.abs(pred_ratios - true_ratios)
    axes[1, 0].scatter(uncertainties, errors, alpha=0.7)
    axes[1, 0].set_xlabel('Predicted Uncertainty')
    axes[1, 0].set_ylabel('Prediction Error')
    axes[1, 0].set_title('Error vs Uncertainty')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 光子数守恒检查
    pred_ch1_photons = pred_results.get('channel1_photons', torch.zeros_like(pred_results['ratio_mean']))
    pred_ch2_photons = pred_results.get('channel2_photons', torch.zeros_like(pred_results['ratio_mean']))
    total_pred = (pred_ch1_photons + pred_ch2_photons).cpu().numpy()
    true_total = true_data['total_photons'][:len(total_pred)]
    
    axes[1, 1].scatter(true_total, total_pred, alpha=0.7)
    axes[1, 1].plot([true_total.min(), true_total.max()], 
                    [true_total.min(), true_total.max()], 'r--', label='Perfect conservation')
    axes[1, 1].set_xlabel('True Total Photons')
    axes[1, 1].set_ylabel('Predicted Total Photons')
    axes[1, 1].set_title('Photon Conservation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("可视化结果已保存为 'demo_results.png'")


def main():
    """
    主函数：运行完整的多通道DECODE演示
    """
    print("=" * 60)
    print("多通道DECODE网络演示")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建输出目录
    os.makedirs('demo_outputs', exist_ok=True)
    
    try:
        # 1. 创建配置
        config = create_example_config()
        print(f"使用设备: {config['hardware']['device']}")
        
        # 2. 创建合成数据
        data = create_synthetic_data(num_samples=50, image_size=64)
        
        # 3. 演示模型初始化
        models = demonstrate_model_initialization(config)
        
        # 4. 演示数据加载
        dataset, dataloader = demonstrate_data_loading(data, config)
        
        # 5. 演示训练（可选，需要较长时间）
        print("\n是否执行训练演示？(y/n): ", end="")
        if input().lower().startswith('y'):
            trainer, train_results = demonstrate_training(models, dataloader, config)
        else:
            print("跳过训练演示")
        
        # 6. 演示推理
        pred_results = demonstrate_inference(models, data, config)
        
        # 7. 演示评估
        metrics = demonstrate_evaluation(pred_results, data, config)
        
        # 8. 演示可视化
        demonstrate_visualization(pred_results, data)
        
        print("\n=" * 60)
        print("演示完成！")
        print("=" * 60)
        
        # 保存配置和结果
        with open('demo_outputs/config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print("\n演示文件已保存到 'demo_outputs/' 目录")
        print("配置文件: demo_outputs/config.yaml")
        print("可视化结果: demo_results.png")
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n感谢使用多通道DECODE网络演示！")


if __name__ == "__main__":
    main()