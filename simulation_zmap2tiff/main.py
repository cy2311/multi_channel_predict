#!/usr/bin/env python3
"""
主程序 - 统一的Zmap到TIFF模拟流程
直接使用phase_retrieval_tiff2h5的result.h5文件
"""

import sys
import time
from pathlib import Path
import argparse
import numpy as np
from typing import Dict, Any

# 添加模块路径
sys.path.append(str(Path(__file__).parent / 'modules'))

from config import Config
from zmap_processor import ZmapProcessor
from emitter_manager import EmitterManager
from image_generator import ImageGenerator
from post_processor import PostProcessor


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Zmap到TIFF模拟流程')
    parser.add_argument('--config', type=str, default=None, 
                       help='配置文件路径 (可选，使用默认配置)')
    parser.add_argument('--zmap-file', type=str, default=None,
                       help='Zmap HDF5文件路径 (覆盖配置文件中的路径)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录 (覆盖配置文件中的路径)')
    parser.add_argument('--num-frames', type=int, default=None,
                       help='生成帧数 (覆盖配置文件中的设置)')
    parser.add_argument('--num-emitters', type=int, default=None,
                       help='发射器数量 (覆盖配置文件中的设置)')
    parser.add_argument('--skip-visualization', action='store_true',
                       help='跳过可视化步骤')
    parser.add_argument('--skip-camera-model', action='store_true',
                       help='跳过相机模型，只生成理想光子图像')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Zmap到TIFF模拟流程")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # 1. 初始化配置
        print("\n1. 初始化配置...")
        config = Config(config_file=args.config)
        
        # 命令行参数覆盖配置
        config.update_from_args(args)
        
        # 确保输出目录存在
        config.ensure_output_dirs()
        
        print(f"配置加载完成")
        print(f"Zmap文件: {config.get_paths()['zmap_file']}")
        print(f"输出目录: {config.get_paths()['output_dir']}")
        
        # 2. 处理Zmap数据
        print("\n2. 处理Zmap数据...")
        zmap_processor = ZmapProcessor(config)
        
        # 加载和处理Zmap数据
        zmap_data = zmap_processor.load_zmap_data()
        print(f"Zmap数据加载完成: {len(zmap_data['coefficients'])}个系数")
        
        # 可视化Zernike系数 (如果不跳过)
        if not args.skip_visualization:
            print("生成Zernike系数可视化...")
            zmap_processor.visualize_coefficients(
                zmap_data['coefficients'], 
                config.get_paths()['visualization_dir'],
                prefix="zernike_"
            )
        
        # 3. 生成和管理发射器
        print("\n3. 生成发射器数据...")
        emitter_manager = EmitterManager(config)
        
        # 生成发射器
        emitters_data = emitter_manager.generate_emitters()
        print(f"生成了{len(emitters_data['xyz'])}个发射器")
        
        # 分配Zernike系数
        zernike_coeffs = emitter_manager.assign_zernike_coefficients(zmap_processor)
        print("Zernike系数分配完成")
        
        # 转换为每帧记录
        frame_records = emitter_manager.bin_emitters_to_frames()
        print(f"转换为每帧记录完成，总共{len(frame_records['frame_ix'])}条记录")
        
        # 保存发射器数据
        emitter_file = config.get_paths()['output_dir'] / 'emitters_data.h5'
        emitter_manager.save_to_h5(emitter_file)
        print(f"发射器数据保存到: {emitter_file}")
        
        # 发射器可视化 (如果不跳过)
        if not args.skip_visualization:
            print("生成发射器可视化...")
            emitter_manager.visualize_emitters(
                config.get_paths()['visualization_dir']
            )
        
        # 4. 生成图像
        print("\n4. 生成模拟图像...")
        image_generator = ImageGenerator(config)
        
        # 生成多帧图像堆栈
        photon_stack = image_generator.generate_multi_frame_stack(frame_records, zernike_coeffs)
        print(f"生成了{photon_stack.shape[0]}帧图像，尺寸: {photon_stack.shape[1:]}")
        
        # 保存理想光子图像
        photon_file = config.get_paths()['tiff_dir'] / 'photon_stack.tiff'
        image_generator.save_tiff_stack(photon_stack, photon_file, 
                                      metadata={'ImageType': 'IdealPhotons'})
        
        # 图像可视化 (如果不跳过)
        if not args.skip_visualization:
            print("生成图像可视化...")
            # PSF示例
            image_generator.visualize_psf_examples(
                frame_records, config.get_paths()['visualization_dir']
            )
            
            # 帧蒙太奇
            image_generator.visualize_frame_montage(
                photon_stack, config.get_paths()['visualization_dir'],
                prefix="photon_"
            )
            
            # 图像统计
            image_generator.analyze_image_statistics(
                photon_stack, config.get_paths()['visualization_dir'],
                prefix="photon_"
            )
        
        # 5. 后处理 (相机模型)
        if not args.skip_camera_model:
            print("\n5. 应用相机模型...")
            post_processor = PostProcessor(config)
            
            # 添加背景噪声
            photon_with_bg = post_processor.add_background(photon_stack)
            
            # 应用相机模型
            camera_stack = post_processor.apply_camera_model(photon_with_bg)
            print(f"相机模型处理完成")
            
            # 保存相机输出
            camera_file = config.get_paths()['tiff_dir'] / 'camera_stack.tiff'
            post_processor.save_camera_stack(
                camera_stack, camera_file,
                metadata={
                    'ImageType': 'CameraOutput',
                    'ProcessingTime': time.time() - start_time
                }
            )
            
            # 噪声分析和可视化 (如果不跳过)
            if not args.skip_visualization:
                print("生成噪声分析可视化...")
                post_processor.visualize_noise_analysis(
                    photon_with_bg, camera_stack,
                    config.get_paths()['visualization_dir'],
                    prefix="camera_"
                )
                
                # 堆栈比较
                noise_stats = post_processor.compare_stacks(
                    photon_with_bg, camera_stack,
                    config.get_paths()['visualization_dir'],
                    prefix="camera_"
                )
                
                print(f"平均信噪比: {noise_stats['snr_db']:.1f} dB")
        
        # 6. 生成处理报告
        print("\n6. 生成处理报告...")
        generate_processing_report(config, zmap_data, emitters_data, 
                                 frame_records, photon_stack, start_time)
        
        total_time = time.time() - start_time
        print(f"\n处理完成! 总耗时: {total_time:.1f}秒")
        print(f"结果保存在: {config.get_paths()['output_dir']}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def generate_processing_report(config: Config, zmap_data: Dict[str, Any], 
                             emitters_data: Dict[str, Any], 
                             frame_records: list,
                             photon_stack: np.ndarray,
                             start_time: float):
    """生成处理报告
    
    Args:
        config: 配置对象
        zmap_data: Zmap数据
        emitters_data: 发射器数据
        frame_records: 帧记录
        photon_stack: 光子图像堆栈
        start_time: 开始时间
    """
    report_file = config.get_paths()['output_dir'] / 'processing_report.md'
    
    # 计算统计信息
    processing_time = time.time() - start_time
    num_emitters = len(emitters_data['xyz'])
    num_frames = photon_stack.shape[0]
    image_size = photon_stack.shape[1:]
    total_photons = int(photon_stack.sum())
    avg_photons_per_frame = total_photons / num_frames
    
    # 发射器生命周期统计
    lifetimes = emitters_data['on_time'].numpy()
    avg_lifetime = np.mean(lifetimes)
    median_lifetime = np.median(lifetimes)
    
    # 每帧发射器数量统计
    frame_emitter_counts = []
    for frame_idx in range(num_frames):
        # 直接从emitters_data计算每帧发射器数量
        # 尝试不同的键名
        if 'frame_ix' in emitters_data:
            frame_mask = emitters_data['frame_ix'] == frame_idx
        elif 'frame' in emitters_data:
            frame_mask = emitters_data['frame'] == frame_idx
        else:
            # 如果都没有，使用默认值
            count = len(emitters_data['xyz']) // num_frames
            frame_emitter_counts.append(count)
            continue
            
        if hasattr(frame_mask, 'numpy'):
            frame_mask = frame_mask.numpy()
        count = np.sum(frame_mask)
        frame_emitter_counts.append(count)
    
    avg_emitters_per_frame = np.mean(frame_emitter_counts)
    
    report_content = f"""# 处理报告

## 基本信息
- **处理时间**: {processing_time:.1f} 秒
- **Zmap文件**: {config.get_paths()['zmap_file']}
- **输出目录**: {config.get_paths()['output_dir']}
- **生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Zmap数据
- **数据类型**: {type(zmap_data).__name__}
- **数据已加载**: ✅

## 发射器统计
- **总发射器数量**: {num_emitters}
- **平均生命周期**: {avg_lifetime:.1f} 帧
- **中位生命周期**: {median_lifetime:.1f} 帧
- **生命周期范围**: {lifetimes.min():.0f} - {lifetimes.max():.0f} 帧
- **平均每帧发射器数**: {avg_emitters_per_frame:.1f}

## 图像统计
- **总帧数**: {num_frames}
- **图像尺寸**: {image_size[0]} × {image_size[1]} 像素
- **总光子数**: {total_photons:,}
- **平均每帧光子数**: {avg_photons_per_frame:.0f}
- **图像数据类型**: {photon_stack.dtype}

## 配置参数

### 光学参数
```
{config.get_optical_params()}
```

### 相机参数
```
{config.get_camera_params()}
```

### 模拟参数
```
{config.get_simulation_params()}
```

## 输出文件
- **发射器数据**: `emitters_data.h5`
- **理想光子图像**: `photon_stack.tiff`
- **相机输出图像**: `camera_stack.tiff` (如果启用)
- **可视化结果**: `visualization/` 目录

## 处理流程
1. ✅ 加载Zmap数据 ({len(zmap_data['coefficients'])}个系数)
2. ✅ 生成发射器 ({num_emitters}个)
3. ✅ 分配Zernike系数
4. ✅ 转换为帧记录 ({len(frame_records)}条)
5. ✅ 生成图像堆栈 ({num_frames}帧)
6. ✅ 保存结果文件

---
*报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # 保存报告
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"处理报告保存到: {report_file}")


if __name__ == '__main__':
    main()