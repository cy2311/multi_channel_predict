#!/usr/bin/env python3
"""
性能比较脚本

比较原始单个生成方法和新的批量生成方法的性能
"""

import os
import json
import time
import psutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from batch_tiff_generator import BatchTiffGenerator


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.peak_memory = None
        self.process = psutil.Process()
    
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
    
    def update_peak_memory(self):
        """更新峰值内存"""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
    
    def stop(self):
        """停止监控"""
        self.end_time = time.time()
        self.update_peak_memory()
    
    def get_results(self) -> Dict[str, float]:
        """获取监控结果"""
        return {
            'duration': self.end_time - self.start_time if self.end_time else 0,
            'start_memory_mb': self.start_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': self.peak_memory - self.start_memory
        }


def create_comparison_configs() -> Tuple[Dict, List[Dict]]:
    """创建比较配置"""
    # 批量配置
    batch_config = {
        "base_output_dir": "./performance_test_batch",
        "max_workers": 2,
        "base_config": {
            "emitters": {
                "num_emitters": 100,
                "density_mu_sig": [1.5, 0.3],
                "intensity_mu_sig": [8000, 1000],
                "xy_unit": "px",
                "z_range": [-750, 750],
                "lifetime": 1
            },
            "zernike": {
                "num_modes": 100,
                "z_range": [-750, 750],
                "mode_weights": {
                    "piston": 0.0,
                    "tip": 0.05,
                    "tilt": 0.05,
                    "defocus": 0.1,
                    "astigmatism": 0.08,
                    "coma": 0.06,
                    "spherical": 0.04,
                    "higher_order": 0.02
                }
            },
            "tiff": {
                "filename": "simulation.tiff",
                "roi_size": 800,
                "use_direct_rendering": True,
                "add_noise": True,
                "noise_params": {
                    "background": 100,
                    "readout_noise": 10,
                    "shot_noise": True
                }
            },
            "optical": {
                "description": "从Zmap插值获取Zernike系数"
            },
            "output": {
                "save_intermediate": False,
                "generate_plots": False,
                "verbose": False
            }
        },
        "variable_configs": {
            "num_emitters": [50, 100, 150],
            "roi_size": [600, 800]
        }
    }
    
    # 单个配置列表（对应批量配置的所有组合）
    single_configs = []
    for num_emitters in [50, 100, 150]:
        for roi_size in [600, 800]:
            config = batch_config["base_config"].copy()
            config["emitters"]["num_emitters"] = num_emitters
            config["tiff"]["roi_size"] = roi_size
            config["tiff"]["filename"] = f"simulation_e{num_emitters}_r{roi_size}.tiff"
            single_configs.append(config)
    
    return batch_config, single_configs


def run_single_generation(config: Dict, output_dir: str) -> Dict[str, float]:
    """运行单个生成"""
    print(f"运行单个生成: emitters={config['emitters']['num_emitters']}, "
          f"roi_size={config['tiff']['roi_size']}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 监控性能
    monitor = PerformanceMonitor()
    monitor.start()
    
    try:
        # 运行原始脚本
        cmd = [
            "python", "main.py",
            "--config", config_path,
            "--output_dir", output_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        if result.returncode != 0:
            print(f"错误: {result.stderr}")
            return {"error": True}
        
    except Exception as e:
        print(f"运行错误: {e}")
        return {"error": True}
    
    finally:
        monitor.stop()
    
    return monitor.get_results()


def run_batch_generation(batch_config: Dict) -> Dict[str, float]:
    """运行批量生成"""
    print("运行批量生成")
    
    # 保存批量配置
    config_path = "batch_performance_config.json"
    with open(config_path, 'w') as f:
        json.dump(batch_config, f, indent=2)
    
    # 监控性能
    monitor = PerformanceMonitor()
    monitor.start()
    
    try:
        # 运行批量生成
        generator = BatchTiffGenerator(config_path)
        generator.run_batch()
        
    except Exception as e:
        print(f"批量生成错误: {e}")
        return {"error": True}
    
    finally:
        monitor.stop()
        # 清理配置文件
        if os.path.exists(config_path):
            os.remove(config_path)
    
    return monitor.get_results()


def compare_performance():
    """比较性能"""
    print("=== 性能比较测试 ===")
    print("比较原始单个生成方法和批量生成方法")
    
    # 创建配置
    batch_config, single_configs = create_comparison_configs()
    
    print(f"\n将生成 {len(single_configs)} 个TIFF文件")
    for i, config in enumerate(single_configs):
        print(f"  {i+1}. emitters={config['emitters']['num_emitters']}, "
              f"roi_size={config['tiff']['roi_size']}")
    
    # 测试单个生成方法
    print("\n--- 测试单个生成方法 ---")
    single_results = []
    single_total_time = 0
    
    for i, config in enumerate(single_configs):
        output_dir = f"./performance_test_single/job_{i+1}"
        result = run_single_generation(config, output_dir)
        
        if "error" not in result:
            single_results.append(result)
            single_total_time += result['duration']
            print(f"  作业 {i+1}: {result['duration']:.2f}s, "
                  f"内存: {result['memory_increase_mb']:.1f}MB")
        else:
            print(f"  作业 {i+1}: 失败")
    
    # 测试批量生成方法
    print("\n--- 测试批量生成方法 ---")
    batch_result = run_batch_generation(batch_config)
    
    if "error" not in batch_result:
        print(f"  批量生成: {batch_result['duration']:.2f}s, "
              f"内存: {batch_result['memory_increase_mb']:.1f}MB")
    else:
        print("  批量生成: 失败")
        return
    
    # 分析结果
    print("\n=== 性能分析 ===")
    
    if single_results and "error" not in batch_result:
        # 时间比较
        avg_single_time = single_total_time / len(single_results)
        batch_time = batch_result['duration']
        time_improvement = (single_total_time - batch_time) / single_total_time * 100
        
        print(f"时间性能:")
        print(f"  单个生成总时间: {single_total_time:.2f}s")
        print(f"  单个生成平均时间: {avg_single_time:.2f}s")
        print(f"  批量生成时间: {batch_time:.2f}s")
        print(f"  时间改进: {time_improvement:.1f}%")
        
        # 内存比较
        avg_single_memory = sum(r['memory_increase_mb'] for r in single_results) / len(single_results)
        batch_memory = batch_result['memory_increase_mb']
        memory_improvement = (avg_single_memory - batch_memory) / avg_single_memory * 100 if avg_single_memory > 0 else 0
        
        print(f"\n内存性能:")
        print(f"  单个生成平均内存增长: {avg_single_memory:.1f}MB")
        print(f"  批量生成内存增长: {batch_memory:.1f}MB")
        print(f"  内存改进: {memory_improvement:.1f}%")
        
        # 效率分析
        single_throughput = len(single_configs) / single_total_time
        batch_throughput = len(single_configs) / batch_time
        throughput_improvement = (batch_throughput - single_throughput) / single_throughput * 100
        
        print(f"\n吞吐量:")
        print(f"  单个生成: {single_throughput:.2f} 作业/秒")
        print(f"  批量生成: {batch_throughput:.2f} 作业/秒")
        print(f"  吞吐量改进: {throughput_improvement:.1f}%")
        
        # 文件大小比较
        print(f"\n文件输出:")
        single_dir = Path("./performance_test_single")
        batch_dir = Path("./performance_test_batch")
        
        if single_dir.exists():
            single_size = sum(f.stat().st_size for f in single_dir.rglob("*.tiff")) / 1024 / 1024
            print(f"  单个生成输出大小: {single_size:.1f}MB")
        
        if batch_dir.exists():
            batch_size = sum(f.stat().st_size for f in batch_dir.rglob("*.tiff")) / 1024 / 1024
            print(f"  批量生成输出大小: {batch_size:.1f}MB")
    
    print("\n性能比较完成！")


def cleanup_test_files():
    """清理测试文件"""
    import shutil
    
    test_dirs = [
        "./performance_test_single",
        "./performance_test_batch",
        "./batch_output_test"
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            try:
                shutil.rmtree(test_dir)
                print(f"已清理: {test_dir}")
            except Exception as e:
                print(f"清理失败 {test_dir}: {e}")


def main():
    """主函数"""
    print("批量TIFF生成性能比较")
    print("=" * 50)
    
    try:
        # 运行性能比较
        compare_performance()
        
        # 询问是否清理测试文件
        print("\n是否清理测试文件? (y/n): ", end="")
        response = input().strip().lower()
        if response in ['y', 'yes']:
            cleanup_test_files()
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)