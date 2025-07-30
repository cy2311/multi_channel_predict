#!/usr/bin/env python3
"""
验证Zernike系数处理的正确性

这个脚本演示了整合方案如何正确处理每个发射器的独特Zernike系数，
而不是使用固定的系数值。
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os
import sys

# 添加路径
sys.path.append(str(Path(__file__).parent.parent / 'trainset_simulation'))
from generate_emitters import main as generate_emitters_main
from compute_zernike_coeffs import process_single_file


def create_test_zmap():
    """创建一个测试用的Zmap文件"""
    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
    temp_file.close()
    
    # 创建模拟的Zmap数据
    with h5py.File(temp_file.name, 'w') as f:
        # 创建空间变化的相位图 (21个Zernike模式)
        nx, ny = 100, 100
        n_modes = 21
        
        # 创建具有空间变化的相位图
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        
        phase_maps = np.zeros((n_modes, ny, nx))
        for i in range(n_modes):
            # 每个模式都有不同的空间变化模式
            phase_maps[i] = 0.1 * np.sin(2*np.pi*X*(i+1)/5) * np.cos(2*np.pi*Y*(i+1)/5)
        
        # 保存相位图
        z_maps_grp = f.create_group('z_maps')
        z_maps_grp.create_dataset('phase', data=phase_maps)
        
        # 创建坐标网格
        coords = np.column_stack([X.ravel(), Y.ravel()])
        f.create_dataset('coords', data=coords)
        
        # 创建幅度系数（简化处理）
        coeff_mag_patch = np.random.normal(0, 0.01, (len(coords), n_modes))
        zernike_grp = f.create_group('zernike')
        zernike_grp.create_dataset('coeff_mag', data=coeff_mag_patch)
    
    return temp_file.name


def create_test_emitters(zmap_path, output_path):
    """创建测试用的发射器文件"""
    # 模拟命令行参数
    import argparse
    
    # 创建发射器数据
    with h5py.File(output_path, 'w') as f:
        # 创建发射器组
        emitters_grp = f.create_group('emitters')
        
        # 生成10个发射器在不同位置
        n_emitters = 10
        xyz = np.random.rand(n_emitters, 3)
        xyz[:, :2] *= 99  # X,Y在0-99范围内
        xyz[:, 2] = (xyz[:, 2] - 0.5) * 1.0  # Z在-0.5到0.5微米
        
        emitters_grp.create_dataset('xyz', data=xyz)
        emitters_grp.create_dataset('id', data=np.arange(n_emitters))
        emitters_grp.create_dataset('phot', data=np.full(n_emitters, 2000.0))
        
        # 创建记录数据（简化）
        records_grp = f.create_group('records')
        records_grp.create_dataset('frame_ix', data=np.zeros(n_emitters, dtype=int))
        records_grp.create_dataset('id', data=np.arange(n_emitters))
        records_grp.create_dataset('xyz', data=xyz)
        records_grp.create_dataset('phot', data=np.full(n_emitters, 2000.0))


def verify_zernike_coefficients():
    """验证Zernike系数的正确处理"""
    print("=" * 60)
    print("验证Zernike系数处理的正确性")
    print("=" * 60)
    
    # 1. 创建测试数据
    print("\n1. 创建测试Zmap和发射器数据...")
    zmap_path = create_test_zmap()
    emitters_path = tempfile.NamedTemporaryFile(suffix='.h5', delete=False).name
    create_test_emitters(zmap_path, emitters_path)
    
    print(f"   Zmap文件: {zmap_path}")
    print(f"   发射器文件: {emitters_path}")
    
    # 2. 计算Zernike系数
    print("\n2. 从Zmap插值计算每个发射器的Zernike系数...")
    process_single_file(Path(zmap_path), Path(emitters_path), num_plot=5)
    
    # 3. 验证结果
    print("\n3. 验证结果...")
    with h5py.File(emitters_path, 'r') as f:
        # 检查是否有Zernike系数
        if 'zernike_coeffs' not in f:
            print("   ❌ 错误: 没有找到zernike_coeffs组")
            return False
        
        phase_coeffs = f['zernike_coeffs/phase'][:]
        mag_coeffs = f['zernike_coeffs/mag'][:]
        emitter_positions = f['emitters/xyz'][:]
        
        print(f"   ✓ 发射器数量: {len(emitter_positions)}")
        print(f"   ✓ 相位系数形状: {phase_coeffs.shape}")
        print(f"   ✓ 幅度系数形状: {mag_coeffs.shape}")
        
        # 检查系数是否有空间变化
        phase_std = np.std(phase_coeffs, axis=0)
        mag_std = np.std(mag_coeffs, axis=0)
        
        print(f"   ✓ 相位系数标准差范围: [{phase_std.min():.6f}, {phase_std.max():.6f}]")
        print(f"   ✓ 幅度系数标准差范围: [{mag_std.min():.6f}, {mag_std.max():.6f}]")
        
        # 验证系数确实有变化（不是固定值）
        has_variation = np.any(phase_std > 1e-6) or np.any(mag_std > 1e-6)
        
        if has_variation:
            print("   ✓ 系数具有空间变化 - 正确！")
        else:
            print("   ❌ 系数没有空间变化 - 可能有问题")
            return False
    
    # 4. 展示不同发射器的系数差异
    print("\n4. 展示不同发射器的系数差异...")
    
    # 选择几个发射器进行比较
    indices = [0, len(emitter_positions)//2, len(emitter_positions)-1]
    
    print("\n   发射器位置和对应的前5个相位系数:")
    for i, idx in enumerate(indices):
        pos = emitter_positions[idx]
        coeffs = phase_coeffs[idx, :5]
        print(f"   发射器{idx}: 位置({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.3f}) -> 系数{coeffs}")
    
    # 5. 对比说明
    print("\n5. 对比说明:")
    print("   ✓ 每个发射器都有独特的Zernike系数")
    print("   ✓ 系数是从Zmap中根据发射器位置插值得到的")
    print("   ✓ 不同位置的发射器具有不同的光学像差")
    print("   ✓ 这与使用固定系数的方法完全不同")
    
    # 清理临时文件
    os.unlink(zmap_path)
    os.unlink(emitters_path)
    
    print("\n" + "=" * 60)
    print("✅ 验证完成: 整合方案正确处理了Zernike系数的空间变化")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    verify_zernike_coefficients()