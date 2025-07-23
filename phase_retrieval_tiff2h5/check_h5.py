#!/usr/bin/env python3
"""
检查HDF5文件结构和内容
"""

import h5py
import numpy as np

# 打开HDF5文件
with h5py.File('result/result.h5', 'r') as f:
    # 打印文件结构
    print('HDF5文件结构:')
    for key in f.keys():
        print(f'/{key}')
    
    # 打印zernike组结构
    print('\nZernike组结构:')
    for key in f['zernike'].keys():
        print(f'/zernike/{key}')
    
    # 检查z_maps组是否存在
    if 'z_maps' in f:
        print('\nZernike maps shape:', f['z_maps']['phase'].shape)
    
    # 计算有效发射体数量
    valid_count = (~np.isnan(f['zernike']['mean_ncc'][:])).sum()
    print(f'\n有效发射体数量: {valid_count}')
    
    # 计算平均NCC值
    mean_ncc = np.nanmean(f['zernike']['mean_ncc'][:])
    print(f'\n平均NCC值: {mean_ncc:.4f}')
    
    # 打印补丁形状
    print(f'\n补丁形状: {f["patches"].shape}')
    
    # 打印坐标形状
    print(f'\n坐标形状: {f["coords"].shape}')