#!/usr/bin/env python3
"""多通道DECODE网络训练脚本

基于DECODE_Network_Analysis.md文档实现的多通道扩展训练流程。
支持三阶段训练策略和不确定性量化。

使用方法:
    python train_multi_channel.py --config multi_channel_config.yaml
"""

import argparse
import logging
import yaml
from pathlib import Path
import torch
import numpy as np
from datetime import datetime

from neuronal_network_v3.trainer.multi_channel_trainer import MultiChannelTrainer
from neuronal_network_v3.data.multi_channel_dataset import MultiChannelDataModule
from neuronal_network_v3.evaluation.multi_channel_evaluation import MultiChannelEvaluation


def setup_logging(config: dict) -> logging.Logger:
    """设置日志系统"""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    
    # 创建日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 文件处理器（如果启用）
    handlers = [console_handler]
    if log_config.get('save_logs', False):
        output_dir = Path(config['experiment']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # 配置根日志器
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True
    )
    
    return logging.getLogger(__name__)


def set_random_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_experiment_dir(config: dict) -> Path:
    """设置实验目录"""
    exp_config = config['experiment']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    exp_dir = Path(exp_config['output_dir']) / f"{exp_config['name']}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置文件
    config_save_path = exp_dir / 'config.yaml'
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return exp_dir


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='Multi-channel DECODE training')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--stage', type=int, default=None,
                       help='Specific training stage to run (1, 2, or 3)')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置实验目录
    exp_dir = setup_experiment_dir(config)
    config['experiment']['exp_dir'] = str(exp_dir)
    
    # 设置日志
    logger = setup_logging(config)
    logger.info(f"Starting multi-channel DECODE training")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Configuration: {args.config}")
    
    # 设置随机种子
    seed = config['experiment'].get('seed', 42)
    set_random_seed(seed)
    logger.info(f"Random seed set to: {seed}")
    
    # 设置设备
    device = config['hardware'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        logger.warning("CUDA not available, using CPU")
    
    logger.info(f"Using device: {device}")
    
    try:
        # 初始化数据模块
        logger.info("Initializing data module...")
        data_module = MultiChannelDataModule(config)
        data_module.setup()
        
        # 打印数据统计信息
        data_stats = data_module.get_statistics()
        logger.info(f"Data statistics: {data_stats}")
        
        # 初始化训练器
        logger.info("Initializing multi-channel trainer...")
        trainer = MultiChannelTrainer(
            config=config,
            device=device,
            logger=logger
        )
        
        # 恢复训练（如果指定）
        if args.resume:
            logger.info(f"Resuming training from: {args.resume}")
            trainer.load_models(args.resume)
        
        # 执行训练
        if args.stage is None:
            # 完整的三阶段训练
            logger.info("Starting full three-stage training pipeline")
            training_results = trainer.train_full_pipeline(
                train_loader=data_module.train_loader,
                val_loader=data_module.val_loader,
                save_dir=str(exp_dir)
            )
            
        else:
            # 指定阶段训练
            logger.info(f"Starting stage {args.stage} training")
            
            if args.stage == 1:
                training_results = trainer.train_stage1(
                    train_loader=data_module.train_loader,
                    val_loader=data_module.val_loader
                )
            elif args.stage == 2:
                training_results = trainer.train_stage2(
                    train_loader=data_module.train_loader,
                    val_loader=data_module.val_loader
                )
            elif args.stage == 3:
                training_results = trainer.train_stage3(
                    train_loader=data_module.train_loader,
                    val_loader=data_module.val_loader
                )
            else:
                raise ValueError(f"Invalid stage: {args.stage}. Must be 1, 2, or 3.")
            
            # 保存阶段模型
            trainer.save_stage_models(str(exp_dir), stage=args.stage)
        
        # 最终评估
        if data_module.test_loader is not None:
            logger.info("Running final evaluation on test set...")
            
            # 设置为评估模式
            trainer.channel1_net.eval()
            trainer.channel2_net.eval()
            trainer.ratio_net.eval()
            
            # 收集测试结果
            all_predictions = {
                'channel1': [], 'channel2': [],
                'mean': [], 'std': []
            }
            all_targets = {
                'channel1': [], 'channel2': [], 'ratio': []
            }
            
            with torch.no_grad():
                for batch in data_module.test_loader:
                    # 移动到设备
                    for key in batch:
                        batch[key] = batch[key].to(device)
                    
                    # 前向传播
                    pred_ch1 = trainer.channel1_net(batch['channel1_input'])
                    pred_ch2 = trainer.channel2_net(batch['channel2_input'])
                    ratio_mean, ratio_std = trainer.ratio_net(pred_ch1, pred_ch2)
                    
                    # 收集结果
                    all_predictions['channel1'].append(pred_ch1.cpu())
                    all_predictions['channel2'].append(pred_ch2.cpu())
                    all_predictions['mean'].append(ratio_mean.cpu())
                    all_predictions['std'].append(ratio_std.cpu())
                    all_targets['channel1'].append(batch['channel1_target'].cpu())
                    all_targets['channel2'].append(batch['channel2_target'].cpu())
                    all_targets['ratio'].append(batch['ratio'].cpu())
            
            # 合并结果
            pred_results = {
                'channel1': torch.cat(all_predictions['channel1'], dim=0),
                'channel2': torch.cat(all_predictions['channel2'], dim=0),
                'mean': torch.cat(all_predictions['mean'], dim=0),
                'std': torch.cat(all_predictions['std'], dim=0)
            }
            ground_truth = {
                'channel1': torch.cat(all_targets['channel1'], dim=0),
                'channel2': torch.cat(all_targets['channel2'], dim=0),
                'ratio': torch.cat(all_targets['ratio'], dim=0)
            }
            
            # 评估
            evaluator = MultiChannelEvaluation(device=device)
            test_metrics = evaluator.evaluate(pred_results, ground_truth)
            
            logger.info(f"Test metrics: {test_metrics}")
            
            # 保存评估结果
            import json
            with open(exp_dir / 'test_metrics.json', 'w') as f:
                # 转换tensor为可序列化格式
                serializable_metrics = trainer._serialize_metrics(test_metrics)
                json.dump(serializable_metrics, f, indent=2)
            
            # 生成可视化
            if config.get('evaluation', {}).get('visualization', {}).get('save_plots', False):
                logger.info("Generating evaluation plots...")
                fig = evaluator.visualize_results(
                    pred_results, ground_truth,
                    save_path=str(exp_dir / 'evaluation_plots.png')
                )
                fig.close()
        
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {exp_dir}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == '__main__':
    main()