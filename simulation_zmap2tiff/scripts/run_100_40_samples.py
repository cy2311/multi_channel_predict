#!/usr/bin/env python3
"""
运行100样本40x40尺寸TIFF生成

这个脚本用于生成完整的100个样本数据集，每个样本包含200帧。
总共生成20,000帧数据用于神经网络训练。
"""

import sys
import time
import argparse
from pathlib import Path
from batch_tiff_generator import BatchTiffGenerator

def main():
    """运行100样本生成"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成100样本40x40尺寸TIFF数据集')
    parser.add_argument('--auto-confirm', action='store_true', 
                       help='自动确认开始生成，跳过交互式提示（用于SLURM等自动化环境）')
    args = parser.parse_args()
    
    print("=== 开始生成100样本40x40尺寸TIFF数据集 ===")
    print("这将生成100个样本，每个样本200帧，总共20,000帧数据")
    print("预计处理时间: 2-4小时（取决于硬件性能）")
    print("")
    
    # 确认用户想要继续（除非使用auto-confirm）
    if not args.auto_confirm:
        response = input("是否继续？(y/N): ")
        if response.lower() not in ['y', 'yes', '是']:
            print("已取消生成")
            return 0
    else:
        print("[AUTO-CONFIRM] 自动确认开始生成...")
    
    config_path = Path("configs/batch_config_100samples_40.json")
    
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return 1
    
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 运行批量生成器
        generator = BatchTiffGenerator(config_path)
        summary = generator.run_batch()
        
        # 计算总时间
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("=== 生成完成 ===")
        print(f"成功生成的样本数: {summary['completed_jobs']}")
        print(f"失败的样本数: {summary['failed_jobs']}")
        print(f"成功率: {summary['success_rate']:.1f}%")
        print(f"总处理时间: {summary['total_time_formatted']}")
        print(f"输出目录: {summary['output_directory']}")
        
        if summary['completed_jobs'] >= 90:  # 至少90%成功
            print("\n✅ 数据集生成成功！")
            print("\n数据集统计:")
            print(f"- 样本数量: {summary['completed_jobs']}")
            print(f"- 每样本帧数: 200")
            print(f"- 总帧数: {summary['completed_jobs'] * 200}")
            print(f"- 图像尺寸: 40x40")
            print(f"- 数据格式: OME-TIFF")
            
            print("\n建议的渐进式训练方案:")
            print("1. 20样本测试 (4,000帧) - 快速验证")
            print("2. 50样本测试 (10,000帧) - 中等规模")
            print(f"3. {summary['completed_jobs']}样本测试 ({summary['completed_jobs'] * 200}帧) - 完整数据集")
            
            print("\n数据集可用于:")
            print("- 神经网络训练和验证")
            print("- 算法性能评估")
            print("- 渐进式学习实验")
            
        else:
            print("\n⚠️ 部分样本生成失败")
            print("请检查失败的作业详情并考虑重新运行")
            
            if summary['failed_job_details']:
                print("\n失败作业详情:")
                for failed_job in summary['failed_job_details'][:5]:  # 只显示前5个
                    print(f"- {failed_job['job_id']}: {failed_job['error']}")
        
        return 0 if summary['completed_jobs'] >= 90 else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断了生成过程")
        print("注意: 批量生成器支持断点续传功能")
        print("重新运行此脚本将从上次中断的地方继续")
        return 1
        
    except Exception as e:
        print(f"\n❌ 生成过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())