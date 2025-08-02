#!/usr/bin/env python3
"""多通道DECODE网络推理脚本

使用训练好的多通道DECODE模型进行推理，支持：
- 双通道联合推理
- 不确定性量化
- 物理约束应用
- 批量处理

使用方法:
    python infer_multi_channel.py --model models/stage3_models.pth --input data/test_images.h5 --output results/
"""

import argparse
import logging
import yaml
from pathlib import Path
import torch
import numpy as np
import h5py
from datetime import datetime
from tqdm import tqdm

from neuronal_network_v3.models.sigma_munet import SigmaMUNet
from neuronal_network_v3.models.ratio_net import RatioNet
from neuronal_network_v3.inference.multi_channel_infer import MultiChannelInfer
from neuronal_network_v3.evaluation.multi_channel_evaluation import MultiChannelEvaluation


def setup_logging() -> logging.Logger:
    """设置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_models(model_path: str, device: str) -> tuple:
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # 初始化模型
    model_config = config.get('model', {})
    
    channel1_net = SigmaMUNet(
        n_inp=model_config.get('n_inp', 1),
        n_out=model_config.get('n_out', 10),
        **model_config.get('sigma_munet_params', {})
    ).to(device)
    
    channel2_net = SigmaMUNet(
        n_inp=model_config.get('n_inp', 1),
        n_out=model_config.get('n_out', 10),
        **model_config.get('sigma_munet_params', {})
    ).to(device)
    
    ratio_config = model_config.get('ratio_net', {})
    ratio_net = RatioNet(
        input_channels=ratio_config.get('input_channels', 20),
        hidden_dim=ratio_config.get('hidden_dim', 128),
        num_layers=ratio_config.get('num_layers', 3),
        dropout=ratio_config.get('dropout', 0.1)
    ).to(device)
    
    # 加载权重
    channel1_net.load_state_dict(checkpoint['channel1_net'])
    channel2_net.load_state_dict(checkpoint['channel2_net'])
    ratio_net.load_state_dict(checkpoint['ratio_net'])
    
    # 设置为评估模式
    channel1_net.eval()
    channel2_net.eval()
    ratio_net.eval()
    
    return channel1_net, channel2_net, ratio_net, config


def load_input_data(input_path: str) -> dict:
    """加载输入数据"""
    input_path = Path(input_path)
    
    if input_path.suffix == '.h5':
        with h5py.File(input_path, 'r') as f:
            data = {
                'channel1_images': f['channel1_images'][:],
                'channel2_images': f['channel2_images'][:]
            }
            
            # 如果有真实值，也加载
            if 'channel1_targets' in f:
                data['channel1_targets'] = f['channel1_targets'][:]
            if 'channel2_targets' in f:
                data['channel2_targets'] = f['channel2_targets'][:]
            if 'ratios' in f:
                data['ratios'] = f['ratios'][:]
                
    elif input_path.suffix == '.npz':
        npz_data = np.load(input_path)
        data = {
            'channel1_images': npz_data['channel1_images'],
            'channel2_images': npz_data['channel2_images']
        }
        
        # 如果有真实值，也加载
        if 'channel1_targets' in npz_data:
            data['channel1_targets'] = npz_data['channel1_targets']
        if 'channel2_targets' in npz_data:
            data['channel2_targets'] = npz_data['channel2_targets']
        if 'ratios' in npz_data:
            data['ratios'] = npz_data['ratios']
    else:
        raise ValueError(f"Unsupported input format: {input_path.suffix}")
    
    return data


def save_results(results: dict, output_path: str, format: str = 'h5'):
    """保存推理结果"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'h5':
        with h5py.File(output_path, 'w') as f:
            for key, value in results.items():
                if isinstance(value, torch.Tensor):
                    f[key] = value.cpu().numpy()
                else:
                    f[key] = value
                    
    elif format == 'npz':
        save_dict = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                save_dict[key] = value.cpu().numpy()
            else:
                save_dict[key] = value
        np.savez_compressed(output_path, **save_dict)
        
    else:
        raise ValueError(f"Unsupported output format: {format}")


def main():
    """主推理函数"""
    parser = argparse.ArgumentParser(description='Multi-channel DECODE inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input data file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory or file path')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for inference')
    parser.add_argument('--apply-constraints', action='store_true',
                       help='Apply physical constraints during inference')
    parser.add_argument('--uncertainty-threshold', type=float, default=0.1,
                       help='Uncertainty threshold for filtering')
    parser.add_argument('--output-format', type=str, default='h5',
                       choices=['h5', 'npz'], help='Output file format')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate results if ground truth is available')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging()
    logger.info("Starting multi-channel DECODE inference")
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        logger.warning("CUDA not available, using CPU")
    
    logger.info(f"Using device: {args.device}")
    
    try:
        # 加载模型
        logger.info(f"Loading models from: {args.model}")
        channel1_net, channel2_net, ratio_net, config = load_models(args.model, args.device)
        
        # 加载输入数据
        logger.info(f"Loading input data from: {args.input}")
        input_data = load_input_data(args.input)
        
        n_samples = len(input_data['channel1_images'])
        logger.info(f"Loaded {n_samples} samples")
        
        # 初始化推理器
        logger.info("Initializing multi-channel inference engine")
        inferrer = MultiChannelInfer(
            channel1_net=channel1_net,
            channel2_net=channel2_net,
            ratio_net=ratio_net,
            device=args.device,
            apply_conservation=args.apply_constraints,
            apply_consistency=args.apply_constraints,
            uncertainty_threshold=args.uncertainty_threshold
        )
        
        # 准备输入张量
        channel1_images = torch.from_numpy(input_data['channel1_images']).float()
        channel2_images = torch.from_numpy(input_data['channel2_images']).float()
        
        # 添加通道维度如果需要
        if len(channel1_images.shape) == 3:  # [N, H, W] -> [N, 1, H, W]
            channel1_images = channel1_images.unsqueeze(1)
        if len(channel2_images.shape) == 3:
            channel2_images = channel2_images.unsqueeze(1)
        
        # 批量推理
        logger.info("Running inference...")
        all_results = {
            'channel1_pred': [],
            'channel2_pred': [],
            'ratio_mean': [],
            'ratio_std': [],
            'fused_results': []
        }
        
        # 分批处理
        n_batches = (n_samples + args.batch_size - 1) // args.batch_size
        
        with torch.no_grad():
            for i in tqdm(range(n_batches), desc="Processing batches"):
                start_idx = i * args.batch_size
                end_idx = min((i + 1) * args.batch_size, n_samples)
                
                batch_ch1 = channel1_images[start_idx:end_idx].to(args.device)
                batch_ch2 = channel2_images[start_idx:end_idx].to(args.device)
                
                # 推理
                results = inferrer.predict(batch_ch1, batch_ch2)
                
                # 收集结果
                for key in all_results:
                    if key in results:
                        all_results[key].append(results[key].cpu())
        
        # 合并结果
        final_results = {}
        for key, value_list in all_results.items():
            if value_list:
                final_results[key] = torch.cat(value_list, dim=0)
        
        logger.info("Inference completed")
        
        # 保存结果
        output_path = Path(args.output)
        if output_path.is_dir():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_path / f"inference_results_{timestamp}.{args.output_format}"
        else:
            output_file = output_path
        
        logger.info(f"Saving results to: {output_file}")
        save_results(final_results, str(output_file), args.output_format)
        
        # 评估（如果有真实值）
        if args.evaluate and 'channel1_targets' in input_data:
            logger.info("Running evaluation...")
            
            # 准备真实值
            ground_truth = {
                'channel1': torch.from_numpy(input_data['channel1_targets']).float(),
                'channel2': torch.from_numpy(input_data['channel2_targets']).float()
            }
            
            if 'ratios' in input_data:
                ground_truth['ratio'] = torch.from_numpy(input_data['ratios']).float()
            
            # 评估
            evaluator = MultiChannelEvaluation(device=args.device)
            metrics = evaluator.evaluate(final_results, ground_truth)
            
            logger.info(f"Evaluation metrics: {metrics}")
            
            # 保存评估结果
            import json
            metrics_file = output_file.parent / f"evaluation_metrics_{output_file.stem}.json"
            
            # 转换为可序列化格式
            def serialize_metrics(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: serialize_metrics(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [serialize_metrics(item) for item in obj]
                else:
                    return obj
            
            serializable_metrics = serialize_metrics(metrics)
            
            with open(metrics_file, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
            
            logger.info(f"Evaluation metrics saved to: {metrics_file}")
        
        # 可视化（如果启用且有真实值）
        if args.visualize and args.evaluate and 'channel1_targets' in input_data:
            logger.info("Generating visualization...")
            
            fig = evaluator.visualize_results(
                final_results, ground_truth,
                save_path=str(output_file.parent / f"visualization_{output_file.stem}.png")
            )
            fig.close()
            
            logger.info("Visualization saved")
        
        # 打印统计信息
        logger.info("\n=== Inference Statistics ===")
        logger.info(f"Total samples processed: {n_samples}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Device: {args.device}")
        
        if 'ratio_mean' in final_results:
            ratio_mean = final_results['ratio_mean'].numpy()
            ratio_std = final_results['ratio_std'].numpy()
            
            logger.info(f"Ratio predictions:")
            logger.info(f"  Mean: {np.mean(ratio_mean):.4f} ± {np.std(ratio_mean):.4f}")
            logger.info(f"  Range: [{np.min(ratio_mean):.4f}, {np.max(ratio_mean):.4f}]")
            logger.info(f"Average uncertainty: {np.mean(ratio_std):.4f}")
        
        logger.info("Inference completed successfully!")
        
    except Exception as e:
        logger.error(f"Inference failed with error: {str(e)}")
        raise


if __name__ == '__main__':
    main()