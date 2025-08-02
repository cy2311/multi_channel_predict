#!/usr/bin/env python3
"""
DECODE神经网络训练脚本
使用neuronal_network_v2框架训练40x40像素数据集
"""

import os
import sys
import yaml
import torch
import argparse
from pathlib import Path

# 添加neuronal_network_v2到路径
sys.path.append('/home/guest/Others/DECODE_rewrite')

from neuronal_network_v2.utils.config import ModelConfig, DataConfig
from neuronal_network_v2.utils.factories import create_model, create_loss_function, create_optimizer, create_scheduler
from neuronal_network_v2.training.dataset import SMLMStaticDataset
from neuronal_network_v2.training.target_generator import UnifiedEmbeddingTarget
from neuronal_network_v2.training.trainer import Trainer
from neuronal_network_v2.utils.logging_utils import setup_logging

def load_config(config_path: str):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='DECODE神经网络训练')
    parser.add_argument('--config', type=str, default='training_config.yaml',
                       help='训练配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    setup_logging(log_level='INFO', log_file=f"{config['logging']['log_dir']}/training.log")
    
    # 设置设备
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型配置
    model_config = ModelConfig(
        architecture=config['model']['architecture'],
        input_channels=config['model']['input_channels'],
        output_channels=config['model']['output_channels'],
        **config['model'].get('unet', {})
    )
    
    # 创建数据配置
    data_config = DataConfig(
        train_data_path=config['data']['data_root'],
        val_data_path=config['data']['data_root'],
        test_data_path=config['data']['data_root'],
        data_format=config['data'].get('file_format', 'hdf5'),
        batch_size=config['data'].get('batch_size', 16)
    )
    
    # 创建模型
    model = create_model(
        model_config.architecture,
        channels_in=model_config.input_channels,
        channels_out=model_config.output_channels,
        depth=model_config.depth,
        initial_features=model_config.initial_features
    )
    model = model.to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建损失函数
    loss_function = create_loss_function(
        config['loss']['type'],
        loss_configs=config['loss']['loss_configs']
    )
    
    # 创建优化器
    optimizer = create_optimizer(
        config['optimization']['optimizer'],
        model.parameters(),
        **{k: v for k, v in config['optimization'].items() if k not in ['optimizer', 'scheduler']}
    )
    
    # 创建学习率调度器
    scheduler = create_scheduler(
        config['optimization']['scheduler']['type'],
        optimizer,
        **{k: v for k, v in config['optimization']['scheduler'].items() if k != 'type'}
    )
    
    # 创建数据集
    print("创建数据集...")
    
    # 获取所有样本路径
    data_root = Path(config['data']['data_root'])
    sample_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.startswith('sample_')])
    
    print(f"找到 {len(sample_dirs)} 个样本目录")
    
    # 分割数据集
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    
    n_train = int(len(sample_dirs) * train_ratio)
    n_val = int(len(sample_dirs) * val_ratio)
    
    train_dirs = sample_dirs[:n_train]
    val_dirs = sample_dirs[n_train:n_train + n_val]
    test_dirs = sample_dirs[n_train + n_val:]
    
    print(f"训练样本: {len(train_dirs)}, 验证样本: {len(val_dirs)}, 测试样本: {len(test_dirs)}")
    
    # 创建数据集
    train_h5_files = [os.path.join(d, 'emitters.h5') for d in train_dirs if os.path.exists(os.path.join(d, 'emitters.h5'))]
    val_h5_files = [os.path.join(d, 'emitters.h5') for d in val_dirs if os.path.exists(os.path.join(d, 'emitters.h5'))]
    
    print(f"找到训练数据文件: {len(train_h5_files)} 个")
    print(f"找到验证数据文件: {len(val_h5_files)} 个")
    
    # 创建目标生成器
    target_generator = UnifiedEmbeddingTarget(
        target_size=(64, 64),
        sigma=1.0,
        output_format='ppxyzb',
        coordinate_system='subpixel'
    )
    
    train_dataset = SMLMStaticDataset(
        data_path=train_h5_files,
        target_generator=target_generator,
        frame_window=3  # 使用3帧，匹配模型的3个输入通道
    )
    
    val_dataset = SMLMStaticDataset(
        data_path=val_h5_files,
        target_generator=target_generator,
        frame_window=3  # 使用3帧，匹配模型的3个输入通道
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    print(f"训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        loss_fn=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    
    # 恢复训练（如果指定）
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"从检查点恢复训练: {args.resume}")
    
    # 开始训练
    print("开始训练...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['epochs']
    )
    
    print("训练完成！")

if __name__ == '__main__':
    main()