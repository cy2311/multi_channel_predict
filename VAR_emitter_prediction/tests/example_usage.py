#!/usr/bin/env python3
"""
VAR-based Emitter Prediction - Example Usage

这个脚本展示了如何使用VAR-based emitter预测系统进行训练和推理。
包含了完整的工作流程示例。
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from var_emitter_model import VAREmitterPredictor
from var_emitter_loss import VAREmitterLoss
from var_dataset import create_dataloaders, InferenceDataset
from train_var_emitter import VAREmitterTrainer, create_model_and_optimizer
from inference import VAREmitterInference, load_model


def create_sample_config():
    """
    创建示例配置文件
    """
    config = {
        "model": {
            "input_size": [40, 40],
            "target_sizes": [[80, 80], [160, 160]],  # 简化版本，只用两个尺度
            "base_channels": 32,  # 减小模型大小用于演示
            "embed_dim": 256,
            "num_heads": 4,
            "num_layers": 3,
            "codebook_size": 512,
            "commitment_cost": 0.25
        },
        "training": {
            "num_epochs": 10,  # 短期训练用于演示
            "batch_size": 4,
            "num_workers": 2,
            "train_val_split": 0.8,
            "use_amp": True,
            "log_interval": 5,
            "save_interval": 5,
            "progressive": True,
            "warmup_epochs": 3,
            "scale_schedule": "linear"
        },
        "loss": {
            "count_weight": 1.0,
            "loc_weight": 1.0,
            "recon_weight": 0.1,
            "uncertainty_weight": 0.5,
            "scale_weights": [0.7, 1.0]  # 对应两个尺度
        },
        "optimizer": {
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "betas": [0.9, 0.999]
        },
        "scheduler": {
            "type": "cosine",
            "min_lr": 1e-6,
            "factor": 0.5,
            "patience": 5
        }
    }
    return config


def create_dummy_data(data_dir: str, num_samples: int = 20):
    """
    创建虚拟数据用于演示
    
    Args:
        data_dir: 数据目录
        num_samples: 样本数量
    """
    import tifffile
    import h5py
    
    data_dir = Path(data_dir)
    
    # 创建目录
    tiff_dir = data_dir / 'tiff'
    emitter_dir = data_dir / 'emitters'
    inference_dir = data_dir / 'inference'
    
    for dir_path in [tiff_dir, emitter_dir, inference_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_samples} dummy samples...")
    
    for i in range(num_samples):
        # 创建高分辨率训练TIFF (160x160)
        high_res_img = np.random.randn(160, 160).astype(np.float32)
        # 添加一些"emitter"信号
        for _ in range(np.random.randint(5, 15)):
            y, x = np.random.randint(20, 140, 2)
            high_res_img[y-2:y+3, x-2:x+3] += np.random.exponential(2.0)
        
        # 保存高分辨率TIFF
        tiff_path = tiff_dir / f'frame_{i:03d}.tif'
        tifffile.imwrite(tiff_path, high_res_img)
        
        # 创建对应的emitter数据
        num_emitters = np.random.randint(5, 15)
        emitter_data = {
            'x': np.random.uniform(0, 160, num_emitters),
            'y': np.random.uniform(0, 160, num_emitters),
            'intensity': np.random.exponential(1000, num_emitters),
            'frame': np.full(num_emitters, i)
        }
        
        # 保存emitter H5文件
        h5_path = emitter_dir / f'frame_{i:03d}.h5'
        with h5py.File(h5_path, 'w') as f:
            for key, value in emitter_data.items():
                f.create_dataset(key, data=value)
        
        # 创建低分辨率推理TIFF (40x40)
        # 通过下采样高分辨率图像
        from scipy.ndimage import zoom
        low_res_img = zoom(high_res_img, 40/160, order=1)
        
        inference_path = inference_dir / f'test_{i:03d}.tif'
        tifffile.imwrite(inference_path, low_res_img.astype(np.float32))
    
    print(f"Dummy data created in {data_dir}")
    print(f"- Training TIFF: {tiff_dir} ({num_samples} files, 160x160)")
    print(f"- Emitter data: {emitter_dir} ({num_samples} files)")
    print(f"- Inference TIFF: {inference_dir} ({num_samples} files, 40x40)")


def example_training(config: dict, data_dir: str, output_dir: str):
    """
    示例训练流程
    """
    print("\n=== 开始训练示例 ===")
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    try:
        # 创建数据加载器
        print("创建数据加载器...")
        train_loader, val_loader = create_dataloaders(
            tiff_dir=str(data_dir / 'tiff'),
            emitter_dir=str(data_dir / 'emitters'),
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            train_val_split=config['training']['train_val_split'],
            low_res_size=config['model']['input_size'],
            high_res_sizes=config['model']['target_sizes']
        )
        
        print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")
        
        # 创建模型和优化器
        print("创建模型...")
        model, loss_fn, optimizer, scheduler = create_model_and_optimizer(config)
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数总数: {total_params:,}")
        
        # 创建训练器
        trainer = VAREmitterTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=str(output_dir),
            use_amp=config['training']['use_amp'],
            log_interval=config['training']['log_interval'],
            save_interval=config['training']['save_interval']
        )
        
        # 开始训练
        print(f"开始训练 {config['training']['num_epochs']} 个epoch...")
        trainer.train(config['training']['num_epochs'])
        
        print(f"训练完成！模型保存在: {output_dir}")
        return True
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def example_inference(config_path: str, checkpoint_path: str, data_dir: str, output_dir: str):
    """
    示例推理流程
    """
    print("\n=== 开始推理示例 ===")
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查文件是否存在
    if not Path(checkpoint_path).exists():
        print(f"错误: 检查点文件不存在: {checkpoint_path}")
        return False
    
    if not Path(config_path).exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return False
    
    inference_dir = data_dir / 'inference'
    if not inference_dir.exists():
        print(f"错误: 推理数据目录不存在: {inference_dir}")
        return False
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    try:
        # 加载模型
        print("加载训练好的模型...")
        model = load_model(checkpoint_path, config_path, device)
        
        # 创建推理数据集
        print("创建推理数据集...")
        dataset = InferenceDataset(
            tiff_dir=str(inference_dir),
            input_size=(40, 40)
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"找到 {len(dataset)} 张图像用于推理")
        
        # 创建推理对象
        inference = VAREmitterInference(
            model=model,
            device=device,
            use_amp=True
        )
        
        # 运行推理
        print("运行推理...")
        results = inference.predict_batch(
            dataloader=dataloader,
            output_dir=str(output_dir),
            save_visualizations=True,
            save_raw_outputs=True
        )
        
        print(f"推理完成！结果保存在: {output_dir}")
        print(f"处理了 {len(results)} 张图像")
        
        # 显示一些统计信息
        if results:
            count_estimates = [r['count_estimate'] for r in results]
            print(f"计数估计统计:")
            print(f"  平均值: {np.mean(count_estimates):.2f}")
            print(f"  标准差: {np.std(count_estimates):.2f}")
            print(f"  范围: {np.min(count_estimates):.2f} - {np.max(count_estimates):.2f}")
        
        return True
        
    except Exception as e:
        print(f"推理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    主函数 - 完整的示例工作流程
    """
    print("VAR-based Emitter Prediction - 示例使用")
    print("=" * 50)
    
    # 设置路径
    base_dir = Path('./example_run')
    data_dir = base_dir / 'data'
    output_dir = base_dir / 'outputs'
    inference_output_dir = base_dir / 'inference_results'
    
    # 创建示例配置
    config = create_sample_config()
    
    print("步骤 1: 创建虚拟数据")
    create_dummy_data(str(data_dir), num_samples=20)
    
    print("\n步骤 2: 训练模型")
    training_success = example_training(config, str(data_dir), str(output_dir))
    
    if training_success:
        print("\n步骤 3: 运行推理")
        config_path = output_dir / 'config.json'
        checkpoint_path = output_dir / 'best_model.pth'
        
        # 检查是否有保存的模型
        if checkpoint_path.exists():
            inference_success = example_inference(
                str(config_path),
                str(checkpoint_path),
                str(data_dir),
                str(inference_output_dir)
            )
            
            if inference_success:
                print("\n=== 示例完成 ===")
                print(f"所有结果保存在: {base_dir}")
                print("\n文件结构:")
                print(f"├── data/                    # 虚拟数据")
                print(f"├── outputs/                 # 训练输出")
                print(f"│   ├── config.json         # 配置文件")
                print(f"│   ├── best_model.pth      # 最佳模型")
                print(f"│   └── tensorboard/        # 训练日志")
                print(f"└── inference_results/      # 推理结果")
                print(f"    ├── visualizations/     # 可视化图像")
                print(f"    ├── raw_outputs/        # 原始预测数据")
                print(f"    └── summary.json        # 结果摘要")
            else:
                print("推理失败")
        else:
            print(f"未找到训练好的模型: {checkpoint_path}")
    else:
        print("训练失败")
    
    print("\n示例脚本执行完毕")


if __name__ == '__main__':
    # 检查依赖
    try:
        import tifffile
        import h5py
        import scipy
    except ImportError as e:
        print(f"缺少依赖包: {e}")
        print("请安装: pip install tifffile h5py scipy")
        sys.exit(1)
    
    main()