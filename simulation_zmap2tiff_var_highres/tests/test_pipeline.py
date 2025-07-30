#!/usr/bin/env python3
"""
流水线测试脚本

用于测试整合方案的各个组件是否正常工作
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))


def test_imports():
    """测试所有必要的导入"""
    print("=== 测试导入 ===")
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        import scipy
        print("✓ scipy")
    except ImportError as e:
        print(f"✗ scipy: {e}")
        return False
    
    try:
        import h5py
        print("✓ h5py")
    except ImportError as e:
        print(f"✗ h5py: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib")
    except ImportError as e:
        print(f"✗ matplotlib: {e}")
        return False
    
    try:
        import tifffile
        print("✓ tifffile")
    except ImportError as e:
        print(f"✗ tifffile: {e}")
        return False
    
    try:
        import skimage
        print("✓ scikit-image")
    except ImportError as e:
        print(f"✗ scikit-image: {e}")
        return False
    
    try:
        import torch
        print("✓ torch")
    except ImportError as e:
        print(f"✗ torch: {e}")
        return False
    
    try:
        import tqdm
        print("✓ tqdm")
    except ImportError as e:
        print(f"✗ tqdm: {e}")
        return False
    
    return True


def test_module_imports():
    """测试自定义模块导入"""
    print("\n=== 测试模块导入 ===")
    
    try:
        from trainset_simulation.generate_emitters import sample_emitters
        print("✓ generate_emitters")
    except ImportError as e:
        print(f"✗ generate_emitters: {e}")
        return False
    
    try:
        from trainset_simulation.compute_zernike_coeffs import load_data
        print("✓ compute_zernike_coeffs")
    except ImportError as e:
        print(f"✗ compute_zernike_coeffs: {e}")
        return False
    
    try:
        from tiff_generator import generate_tiff_stack
        print("✓ tiff_generator")
    except ImportError as e:
        print(f"✗ tiff_generator: {e}")
        return False
    
    return True


def test_config_files():
    """测试配置文件"""
    print("\n=== 测试配置文件 ===")
    
    # 测试默认光学配置
    default_config_path = Path(__file__).parent.parent / "configs" / "default_config.json"
    if default_config_path.exists():
        try:
            with open(default_config_path, 'r') as f:
                config = json.load(f)
            if 'optical' in config:
                print("✓ default_config.json")
            else:
                print("✗ default_config.json: 缺少optical配置")
                return False
        except Exception as e:
            print(f"✗ default_config.json: {e}")
            return False
    else:
        print(f"✗ default_config.json: 文件不存在 {default_config_path}")
        return False
    
    # 测试流水线配置
    pipeline_config_path = Path(__file__).parent / "pipeline_config.json"
    if pipeline_config_path.exists():
        try:
            with open(pipeline_config_path, 'r') as f:
                config = json.load(f)
            print("✓ pipeline_config.json")
        except Exception as e:
            print(f"✗ pipeline_config.json: {e}")
            return False
    else:
        print("✗ pipeline_config.json: 文件不存在")
        return False
    
    return True


def test_zernike_basis():
    """测试Zernike基函数文件"""
    print("\n=== 测试Zernike基函数 ===")
    
    zernike_dir = Path(__file__).parent.parent / "simulated_data" / "zernike_polynomials"
    
    if not zernike_dir.exists():
        print(f"✗ Zernike目录不存在: {zernike_dir}")
        return False
    
    import glob
    pattern = str(zernike_dir / "zernike_*_n*_m*.npy")
    files = glob.glob(pattern)
    
    if len(files) >= 21:
        print(f"✓ 找到 {len(files)} 个Zernike基函数文件")
        
        # 测试加载第一个文件
        try:
            import numpy as np
            basis = np.load(files[0])
            print(f"✓ 基函数形状: {basis.shape}")
        except Exception as e:
            print(f"✗ 加载基函数失败: {e}")
            return False
    else:
        print(f"✗ Zernike基函数文件不足: 找到 {len(files)} 个，需要至少 21 个")
        return False
    
    return True


def test_emitter_generation():
    """测试发射器生成功能"""
    print("\n=== 测试发射器生成 ===")
    
    try:
        from trainset_simulation.generate_emitters import sample_emitters, bin_emitters_to_frames
        
        # 生成少量测试数据
        em_attrs = sample_emitters(
            num_emitters=10,
            frame_range=(0, 2),
            area_px=100.0,
            intensity_mu=1000.0,
            intensity_sigma=200.0,
            lifetime_avg=1.5,
            z_range_um=0.5,
            seed=42
        )
        
        records = bin_emitters_to_frames(em_attrs, (0, 2))
        
        print(f"✓ 生成 {len(em_attrs['id'])} 个发射器")
        print(f"✓ 生成 {len(records['id'])} 条记录")
        
        return True
        
    except Exception as e:
        print(f"✗ 发射器生成失败: {e}")
        return False


def test_tiff_generator_components():
    """测试TIFF生成器的组件"""
    print("\n=== 测试TIFF生成器组件 ===")
    
    try:
        from tiff_generator import (
            load_config, load_zernike_basis, build_pupil_mask,
            construct_pupil, generate_psf
        )
        
        # 测试配置加载
        wavelength_nm, pix_x_nm, pix_y_nm, NA = load_config()
        print(f"✓ 光学参数: λ={wavelength_nm}nm, NA={NA}")
        
        # 测试基函数加载
        basis = load_zernike_basis()
        print(f"✓ Zernike基函数: {basis.shape}")
        
        # 测试瞳孔掩码
        wavelength_m = wavelength_nm * 1e-9
        pix_x_m = pix_x_nm * 1e-9
        pix_y_m = pix_y_nm * 1e-9
        
        pupil_mask = build_pupil_mask(128, pix_x_m, pix_y_m, NA, wavelength_m)
        print(f"✓ 瞳孔掩码: {pupil_mask.shape}, 有效像素: {pupil_mask.sum():.0f}")
        
        # 测试PSF生成
        import numpy as np
        coeff_mag = np.random.normal(0, 0.1, 21)
        coeff_phase = np.random.normal(0, 0.5, 21)
        
        pupil = construct_pupil(coeff_mag, coeff_phase, basis, pupil_mask)
        psf = generate_psf(pupil)
        
        print(f"✓ PSF生成: {psf.shape}, 总强度: {psf.sum():.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ TIFF生成器组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_dummy_zmap():
    """创建虚拟的Zmap文件用于测试"""
    print("\n=== 创建测试用Zmap文件 ===")
    
    try:
        import numpy as np
        import h5py
        
        # 创建临时文件
        temp_dir = Path(tempfile.gettempdir())
        zmap_path = temp_dir / "test_patches.h5"
        
        with h5py.File(zmap_path, 'w') as f:
            # 创建虚拟相位图
            phase_maps = np.random.normal(0, 0.5, (21, 64, 64)).astype(np.float32)
            f.create_dataset('z_maps/phase', data=phase_maps)
            
            # 创建虚拟坐标
            coords = np.random.uniform(0, 64, (100, 2)).astype(np.float32)
            f.create_dataset('coords', data=coords)
            
            # 创建虚拟幅度系数
            coeff_mag = np.random.normal(0, 0.1, (100, 21)).astype(np.float32)
            f.create_dataset('zernike/coeff_mag', data=coeff_mag)
        
        print(f"✓ 创建测试Zmap文件: {zmap_path}")
        return str(zmap_path)
        
    except Exception as e:
        print(f"✗ 创建测试Zmap文件失败: {e}")
        return None


def run_integration_test():
    """运行集成测试"""
    print("\n=== 集成测试 ===")
    
    # 创建测试用的Zmap文件
    zmap_path = create_dummy_zmap()
    if not zmap_path:
        return False
    
    try:
        # 创建临时输出目录
        temp_dir = Path(tempfile.gettempdir())
        output_dir = temp_dir / "test_output"
        output_dir.mkdir(exist_ok=True)
        
        # 测试主流水线（只测试发射器生成）
        from main import step1_generate_emitters
        
        config = {
            'emitters': {
                'num_emitters': 50,
                'frames': 3,
                'area_px': 100.0,
                'intensity_mu': 1000.0,
                'intensity_sigma': 200.0,
                'lifetime_avg': 1.5,
                'z_range_um': 0.5,
                'seed': 42,
                'no_plot': True
            }
        }
        
        emitters_path = step1_generate_emitters(config, output_dir)
        
        if emitters_path.exists():
            print(f"✓ 集成测试成功: {emitters_path}")
            
            # 检查文件内容
            import h5py
            with h5py.File(emitters_path, 'r') as f:
                print(f"  - 发射器数量: {len(f['emitters/id'])}")
                print(f"  - 记录数量: {len(f['records/id'])}")
            
            return True
        else:
            print("✗ 集成测试失败: 输出文件不存在")
            return False
            
    except Exception as e:
        print(f"✗ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时文件
        if zmap_path and Path(zmap_path).exists():
            Path(zmap_path).unlink()


def main():
    """运行所有测试"""
    print("流水线测试脚本")
    print("=" * 50)
    
    tests = [
        ("导入测试", test_imports),
        ("模块导入测试", test_module_imports),
        ("配置文件测试", test_config_files),
        ("Zernike基函数测试", test_zernike_basis),
        ("发射器生成测试", test_emitter_generation),
        ("TIFF生成器组件测试", test_tiff_generator_components),
        ("集成测试", run_integration_test)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} 异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "="*50)
    print("测试总结")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！流水线准备就绪。")
        print("\n下一步:")
        print("1. 准备真实的patches.h5文件")
        print("2. 运行: python main.py --zmap patches.h5 --output_dir output/")
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，请检查依赖和配置。")
        print("\n常见解决方案:")
        print("1. 安装缺失的Python包: pip install numpy scipy h5py matplotlib tifffile scikit-image torch tqdm")
        print("2. 检查Zernike基函数文件是否存在")
        print("3. 检查配置文件是否正确")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)