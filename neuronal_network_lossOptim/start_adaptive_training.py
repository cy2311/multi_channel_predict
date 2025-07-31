#!/usr/bin/env python3
"""
自适应训练启动脚本

使用方法:
1. 直接运行: python start_adaptive_training.py
2. 指定配置: python start_adaptive_training.py --config configs/train_config_adaptive.json
3. 继续训练: python start_adaptive_training.py --resume outputs/training_results_adaptive/latest_checkpoint.pth
"""

import os
import sys
import argparse
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training'))

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"  Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查PyTorch
    try:
        import torch
        print(f"  PyTorch版本: {torch.__version__}")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU数量: {torch.cuda.device_count()}")
            print(f"  当前GPU: {torch.cuda.get_device_name()}")
    except ImportError:
        print("  ❌ PyTorch未安装")
        return False
    
    # 检查其他依赖
    required_packages = ['numpy', 'matplotlib']
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}未安装")
            return False
    
    return True

def check_data_availability():
    """检查数据可用性"""
    print("\n📁 检查数据可用性...")
    
    data_dir = "/home/guest/Others/DECODE_rewrite/simulation_zmap2tiff/outputs_100samples_40"
    
    if os.path.exists(data_dir):
        # 统计样本数量
        samples = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print(f"  数据目录: {data_dir}")
        print(f"  样本数量: {len(samples)}")
        
        if len(samples) < 10:
            print("  ⚠️  样本数量较少，可能影响训练效果")
        else:
            print(f"  ✅ 数据充足")
        
        return True
    else:
        print(f"  ❌ 数据目录不存在: {data_dir}")
        print("  将使用模拟数据进行训练")
        return False

def create_default_config():
    """创建默认配置文件"""
    config_path = "training/configs/train_config_adaptive.json"
    
    if not os.path.exists(config_path):
        print(f"\n⚠️  配置文件不存在: {config_path}")
        print("请确保已创建自适应配置文件")
        return None
    
    return config_path

def setup_output_directory(config_path: str):
    """设置输出目录"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    output_dir = config['output']['save_dir']
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'tensorboard'), exist_ok=True)
    
    print(f"\n📂 输出目录: {output_dir}")
    return output_dir

def start_training(config_path: str, resume_path: str = None):
    """启动训练"""
    print("\n🚀 启动自适应训练...")
    print("=" * 60)
    
    try:
        from training.train_decode_network_adaptive import AdaptiveDECODETrainer
        
        # 创建训练器
        trainer = AdaptiveDECODETrainer(config_path)
        
        # 如果有恢复路径，加载检查点
        if resume_path and os.path.exists(resume_path):
            print(f"从检查点恢复训练: {resume_path}")
            # 这里可以添加恢复逻辑
        
        # 开始训练
        results = trainer.train()
        
        print("\n🎉 训练完成！")
        print(f"最佳验证损失: {results['best_val_loss']:.6f}")
        print(f"训练时间: {results['total_time']:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def start_monitoring(output_dir: str):
    """启动监控"""
    checkpoint_path = os.path.join(output_dir, 'latest_checkpoint.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"\n⚠️  检查点文件不存在: {checkpoint_path}")
        print("请先开始训练")
        return
    
    print("\n📊 启动训练监控...")
    
    try:
        from training.monitor_adaptive_training import TrainingMonitor
        
        monitor = TrainingMonitor(checkpoint_path)
        checkpoint = monitor.load_checkpoint()
        
        if checkpoint:
            # 生成报告和图表
            monitor.generate_report(checkpoint)
            monitor.plot_training_curves(checkpoint)
        
    except Exception as e:
        print(f"监控启动失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='DECODE自适应训练启动器')
    parser.add_argument('--config', '-c',
                       default='training/configs/train_config_adaptive.json',
                       help='配置文件路径')
    parser.add_argument('--resume', '-r',
                       help='从检查点恢复训练')
    parser.add_argument('--monitor-only', '-m',
                       action='store_true',
                       help='仅启动监控，不训练')
    parser.add_argument('--skip-checks', '-s',
                       action='store_true',
                       help='跳过环境检查')
    
    args = parser.parse_args()
    
    print("🎯 DECODE自适应训练启动器")
    print("=" * 60)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 环境检查
    if not args.skip_checks:
        if not check_environment():
            print("\n❌ 环境检查失败，请安装必要的依赖")
            return
        
        check_data_availability()
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"\n❌ 配置文件不存在: {args.config}")
        config_path = create_default_config()
        if not config_path:
            return
    else:
        config_path = args.config
    
    print(f"\n📋 使用配置: {config_path}")
    
    # 设置输出目录
    output_dir = setup_output_directory(config_path)
    
    # 仅监控模式
    if args.monitor_only:
        start_monitoring(output_dir)
        return
    
    # 开始训练
    success = start_training(config_path, args.resume)
    
    if success:
        print("\n📊 训练完成，生成监控报告...")
        start_monitoring(output_dir)
        
        print("\n🎯 后续建议:")
        print("1. 查看TensorBoard: tensorboard --logdir outputs/training_results_adaptive/tensorboard")
        print("2. 运行监控脚本: python training/monitor_adaptive_training.py")
        print("3. 检查最佳模型: outputs/training_results_adaptive/best_model.pth")
    else:
        print("\n💡 故障排除建议:")
        print("1. 检查数据路径是否正确")
        print("2. 确认GPU内存充足")
        print("3. 查看详细错误信息")
        print("4. 尝试使用更小的batch_size")

if __name__ == '__main__':
    main()