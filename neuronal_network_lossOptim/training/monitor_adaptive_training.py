#!/usr/bin/env python3
"""
自适应训练监控脚本

功能:
1. 实时监控训练进度
2. 可视化学习率变化
3. 分析loss收敛趋势
4. 生成训练报告
"""

import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import argparse
from datetime import datetime

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, checkpoint_path: str, output_dir: str = "monitoring_outputs"):
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def load_checkpoint(self) -> Optional[Dict]:
        """加载检查点数据"""
        if not os.path.exists(self.checkpoint_path):
            print(f"检查点文件不存在: {self.checkpoint_path}")
            return None
            
        try:
            import torch
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            return checkpoint
        except Exception as e:
            print(f"加载检查点失败: {e}")
            return None
    
    def plot_training_curves(self, checkpoint: Dict):
        """绘制训练曲线"""
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        lr_values = checkpoint.get('lr_values', [])
        
        if not train_losses:
            print("没有找到训练损失数据")
            return
            
        epochs = range(len(train_losses))
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('自适应训练监控报告', fontsize=16, fontweight='bold')
        
        # 1. 损失曲线
        ax1.plot(epochs, train_losses, label='训练损失', color='blue', alpha=0.7)
        if val_losses:
            ax1.plot(epochs[:len(val_losses)], val_losses, label='验证损失', color='red', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('损失变化曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 学习率变化
        if lr_values:
            ax2.plot(epochs[:len(lr_values)], lr_values, color='green', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('学习率变化')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, '无学习率数据', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('学习率变化')
        
        # 3. 损失改善分析
        if len(train_losses) > 10:
            # 计算移动平均
            window_size = min(20, len(train_losses) // 5)
            train_ma = self._moving_average(train_losses, window_size)
            val_ma = self._moving_average(val_losses, window_size) if val_losses else []
            
            ma_epochs = range(window_size-1, len(train_losses))
            ax3.plot(ma_epochs, train_ma, label=f'训练损失MA({window_size})', color='blue')
            if val_ma:
                ax3.plot(ma_epochs[:len(val_ma)], val_ma, label=f'验证损失MA({window_size})', color='red')
            
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss (Moving Average)')
            ax3.set_title('损失移动平均')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('损失移动平均')
        
        # 4. 收敛分析
        if len(train_losses) > 5:
            # 计算损失变化率
            train_diff = np.diff(train_losses)
            val_diff = np.diff(val_losses) if val_losses else []
            
            diff_epochs = range(1, len(train_losses))
            ax4.plot(diff_epochs, train_diff, label='训练损失变化率', alpha=0.7, color='blue')
            if val_diff:
                ax4.plot(diff_epochs[:len(val_diff)], val_diff, label='验证损失变化率', alpha=0.7, color='red')
            
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss Change')
            ax4.set_title('损失变化率')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('损失变化率')
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.output_dir, f"training_curves_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存: {plot_path}")
        
        plt.show()
        
    def _moving_average(self, data: List[float], window_size: int) -> List[float]:
        """计算移动平均"""
        if len(data) < window_size:
            return data
            
        ma = []
        for i in range(window_size-1, len(data)):
            ma.append(np.mean(data[i-window_size+1:i+1]))
        return ma
    
    def analyze_convergence(self, checkpoint: Dict) -> Dict:
        """分析收敛情况"""
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        lr_values = checkpoint.get('lr_values', [])
        
        if not train_losses:
            return {"error": "没有训练数据"}
        
        analysis = {
            "total_epochs": len(train_losses),
            "initial_train_loss": train_losses[0] if train_losses else None,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "best_train_loss": min(train_losses) if train_losses else None,
            "initial_val_loss": val_losses[0] if val_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "best_val_loss": min(val_losses) if val_losses else None,
            "initial_lr": lr_values[0] if lr_values else None,
            "final_lr": lr_values[-1] if lr_values else None,
            "min_lr": min(lr_values) if lr_values else None,
            "max_lr": max(lr_values) if lr_values else None
        }
        
        # 计算改善率
        if train_losses and len(train_losses) > 1:
            train_improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
            analysis["train_improvement_percent"] = train_improvement
            
        if val_losses and len(val_losses) > 1:
            val_improvement = (val_losses[0] - val_losses[-1]) / val_losses[0] * 100
            analysis["val_improvement_percent"] = val_improvement
        
        # 检测停滞
        if len(train_losses) > 20:
            recent_losses = train_losses[-20:]
            loss_variance = np.var(recent_losses)
            analysis["recent_loss_variance"] = loss_variance
            analysis["is_stagnating"] = loss_variance < 0.001
        
        # 学习率调整次数
        if lr_values and len(lr_values) > 1:
            lr_changes = sum(1 for i in range(1, len(lr_values)) if abs(lr_values[i] - lr_values[i-1]) > 1e-8)
            analysis["lr_adjustment_count"] = lr_changes
        
        return analysis
    
    def generate_report(self, checkpoint: Dict):
        """生成训练报告"""
        analysis = self.analyze_convergence(checkpoint)
        
        report = []
        report.append("=" * 60)
        report.append("自适应训练分析报告")
        report.append("=" * 60)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 基本信息
        report.append("📊 训练基本信息:")
        report.append(f"  总训练轮数: {analysis.get('total_epochs', 'N/A')}")
        report.append(f"  当前epoch: {checkpoint.get('epoch', 'N/A')}")
        report.append("")
        
        # 损失分析
        report.append("📈 损失分析:")
        if analysis.get('initial_train_loss') and analysis.get('final_train_loss'):
            report.append(f"  训练损失: {analysis['initial_train_loss']:.6f} → {analysis['final_train_loss']:.6f}")
            if 'train_improvement_percent' in analysis:
                report.append(f"  训练改善: {analysis['train_improvement_percent']:.2f}%")
        
        if analysis.get('initial_val_loss') and analysis.get('final_val_loss'):
            report.append(f"  验证损失: {analysis['initial_val_loss']:.6f} → {analysis['final_val_loss']:.6f}")
            if 'val_improvement_percent' in analysis:
                report.append(f"  验证改善: {analysis['val_improvement_percent']:.2f}%")
        
        if analysis.get('best_val_loss'):
            report.append(f"  最佳验证损失: {analysis['best_val_loss']:.6f}")
        report.append("")
        
        # 学习率分析
        report.append("🎯 学习率分析:")
        if analysis.get('initial_lr') and analysis.get('final_lr'):
            report.append(f"  学习率: {analysis['initial_lr']:.2e} → {analysis['final_lr']:.2e}")
        if analysis.get('lr_adjustment_count'):
            report.append(f"  学习率调整次数: {analysis['lr_adjustment_count']}")
        report.append("")
        
        # 收敛状态
        report.append("🔍 收敛状态:")
        if 'is_stagnating' in analysis:
            status = "停滞" if analysis['is_stagnating'] else "正常"
            report.append(f"  当前状态: {status}")
            if analysis.get('recent_loss_variance'):
                report.append(f"  近期损失方差: {analysis['recent_loss_variance']:.6f}")
        
        # 建议
        report.append("")
        report.append("💡 优化建议:")
        
        if analysis.get('is_stagnating', False):
            report.append("  ⚠️  检测到训练停滞，建议:")
            report.append("     - 增加学习率")
            report.append("     - 调整损失函数权重")
            report.append("     - 检查数据质量")
        
        if analysis.get('final_lr', 1) < 1e-6:
            report.append("  ⚠️  学习率过小，可能需要:")
            report.append("     - 重新启动训练")
            report.append("     - 使用更大的初始学习率")
        
        if analysis.get('train_improvement_percent', 0) < 5:
            report.append("  ⚠️  训练改善有限，建议:")
            report.append("     - 增加数据集大小")
            report.append("     - 调整模型架构")
            report.append("     - 检查损失函数设计")
        
        report_text = "\n".join(report)
        print(report_text)
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"training_report_{timestamp}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n报告已保存: {report_path}")
        
        return analysis
    
    def monitor_realtime(self, interval: int = 30):
        """实时监控训练进度"""
        print(f"开始实时监控，检查间隔: {interval}秒")
        print("按 Ctrl+C 停止监控")
        
        last_epoch = -1
        
        try:
            while True:
                checkpoint = self.load_checkpoint()
                if checkpoint:
                    current_epoch = checkpoint.get('epoch', 0)
                    
                    if current_epoch > last_epoch:
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Epoch {current_epoch} 完成")
                        
                        train_losses = checkpoint.get('train_losses', [])
                        val_losses = checkpoint.get('val_losses', [])
                        lr_values = checkpoint.get('lr_values', [])
                        
                        if train_losses:
                            print(f"  训练损失: {train_losses[-1]:.6f}")
                        if val_losses:
                            print(f"  验证损失: {val_losses[-1]:.6f}")
                        if lr_values:
                            print(f"  当前学习率: {lr_values[-1]:.2e}")
                        
                        last_epoch = current_epoch
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n监控已停止")

def main():
    parser = argparse.ArgumentParser(description='自适应训练监控工具')
    parser.add_argument('--checkpoint', '-c', 
                       default='outputs/training_results_adaptive/latest_checkpoint.pth',
                       help='检查点文件路径')
    parser.add_argument('--output', '-o', 
                       default='monitoring_outputs',
                       help='输出目录')
    parser.add_argument('--mode', '-m', 
                       choices=['plot', 'report', 'monitor', 'all'],
                       default='all',
                       help='运行模式')
    parser.add_argument('--interval', '-i', 
                       type=int, default=30,
                       help='实时监控间隔（秒）')
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.checkpoint, args.output)
    
    if args.mode in ['plot', 'all']:
        print("生成训练曲线...")
        checkpoint = monitor.load_checkpoint()
        if checkpoint:
            monitor.plot_training_curves(checkpoint)
    
    if args.mode in ['report', 'all']:
        print("生成训练报告...")
        checkpoint = monitor.load_checkpoint()
        if checkpoint:
            monitor.generate_report(checkpoint)
    
    if args.mode == 'monitor':
        monitor.monitor_realtime(args.interval)

if __name__ == '__main__':
    main()