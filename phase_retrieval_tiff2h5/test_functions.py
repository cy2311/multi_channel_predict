#!/usr/bin/env python3
"""
测试tiff2h5_phase_retrieval.py中的关键函数

这个脚本测试相位恢复流程中的关键函数，包括：
1. 发射体检测和过滤
2. 补丁提取
3. Zernike多项式生成
4. 相位恢复函数

使用方法：
    python test_functions.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# 导入主脚本中的函数
from tiff2h5_phase_retrieval import (
    load_config, load_tiff_stack, detect_emitters, filter_close_emitters,
    extract_patch, normalize_patch, select_frames, generate_zernike_basis,
    load_zernike_basis, normalized_cross_correlation
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量和配置
CONFIG_PATH = os.path.join('..', 'configs', 'default_config.json')
CAMERA_PARAMS_PATH = os.path.join('..', 'beads', 'spool_100mW_30ms_3D_1_2', 'camera_parameters.json')
TIFF_PATH = os.path.join('..', 'beads', 'spool_100mW_30ms_3D_1_2', 'spool_100mW_30ms_3D_1_2_MMStack_Default.ome.tif')
OUTPUT_DIR = os.path.join('test_output')

# 创建测试输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_load_config():
    """测试配置加载功能"""
    logger.info("测试配置加载...")
    try:
        cfg = load_config(CONFIG_PATH)
        logger.info(f"配置加载成功，包含键: {list(cfg.keys())}")
        return True
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        return False


def test_load_camera_params():
    """测试相机参数加载功能"""
    logger.info("测试相机参数加载...")
    try:
        cam_params = load_config(CAMERA_PARAMS_PATH)
        logger.info(f"相机参数加载成功，包含键: {list(cam_params.keys())}")
        return True
    except Exception as e:
        logger.error(f"相机参数加载失败: {e}")
        return False


def test_load_tiff():
    """测试TIFF加载功能"""
    logger.info("测试TIFF加载...")
    try:
        # 只加载前10帧以加快测试速度
        stack = load_tiff_stack(TIFF_PATH)[:10]
        logger.info(f"TIFF加载成功，形状: {stack.shape}")
        
        # 保存第一帧的图像以进行可视化检查
        plt.figure(figsize=(6, 6))
        plt.imshow(stack[0], cmap='gray')
        plt.title("TIFF第一帧")
        plt.colorbar()
        plt.savefig(os.path.join(OUTPUT_DIR, "tiff_first_frame.png"))
        plt.close()
        
        return True
    except Exception as e:
        logger.error(f"TIFF加载失败: {e}")
        return False


def test_emitter_detection():
    """测试发射体检测功能"""
    logger.info("测试发射体检测...")
    try:
        # 只加载前10帧以加快测试速度
        stack = load_tiff_stack(TIFF_PATH)[:10]
        emitter_coords = detect_emitters(stack, n_emitters=20, min_distance=25)
        logger.info(f"检测到{len(emitter_coords)}个发射体")
        
        # 过滤太近的发射体
        filtered_coords = filter_close_emitters(emitter_coords, min_distance=25)
        logger.info(f"过滤后剩余{len(filtered_coords)}个发射体")
        
        # 可视化检测结果
        plt.figure(figsize=(8, 8))
        plt.imshow(stack[0], cmap='gray')
        plt.title("检测到的发射体")
        
        # 绘制原始检测点
        y_coords, x_coords = zip(*emitter_coords)
        plt.scatter(x_coords, y_coords, c='red', marker='x', label='原始检测')
        
        # 绘制过滤后的点
        if filtered_coords:
            y_filtered, x_filtered = zip(*filtered_coords)
            plt.scatter(x_filtered, y_filtered, c='green', marker='o', 
                       facecolors='none', s=100, label='过滤后')
        
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, "detected_emitters.png"))
        plt.close()
        
        return True
    except Exception as e:
        logger.error(f"发射体检测失败: {e}")
        return False


def test_patch_extraction():
    """测试补丁提取功能"""
    logger.info("测试补丁提取...")
    try:
        # 只加载前201帧以确保有足够的Z深度
        stack = load_tiff_stack(TIFF_PATH)[:201]
        emitter_coords = detect_emitters(stack, n_emitters=5, min_distance=25)
        
        if not emitter_coords:
            logger.warning("没有检测到发射体，跳过补丁提取测试")
            return False
        
        # 提取第一个发射体的补丁
        coord = emitter_coords[0]
        patch = extract_patch(stack, coord, patch_z=201, patch_xy=25)
        logger.info(f"补丁提取成功，形状: {patch.shape}")
        
        # 归一化补丁
        patch_norm = normalize_patch(patch)
        
        # 选择特定帧
        frames_selected = select_frames(patch_norm, center=75, step=2, n_each_side=20)
        logger.info(f"选择的帧形状: {frames_selected.shape}")
        
        # 可视化中心帧
        plt.figure(figsize=(6, 6))
        plt.imshow(frames_selected[20], cmap='gray')  # 中心帧索引为20
        plt.title(f"发射体在{coord}的中心帧")
        plt.colorbar()
        plt.savefig(os.path.join(OUTPUT_DIR, "patch_center_frame.png"))
        plt.close()
        
        # 可视化Z剖面
        plt.figure(figsize=(10, 6))
        center_y, center_x = 12, 12  # 25x25补丁的中心
        z_profile = frames_selected[:, center_y, center_x]
        plt.plot(range(len(z_profile)), z_profile)
        plt.title(f"发射体在{coord}的Z剖面")
        plt.xlabel("帧索引")
        plt.ylabel("归一化强度")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(OUTPUT_DIR, "patch_z_profile.png"))
        plt.close()
        
        return True
    except Exception as e:
        logger.error(f"补丁提取失败: {e}")
        return False


def test_zernike_basis():
    """测试Zernike基生成功能"""
    logger.info("测试Zernike基生成...")
    try:
        # 尝试加载现有的基
        try:
            basis = load_zernike_basis()
            logger.info(f"成功加载现有的Zernike基，形状: {basis.shape}")
        except Exception:
            # 如果加载失败，生成新的基
            logger.info("加载失败，生成新的Zernike基...")
            generate_zernike_basis()
            basis = load_zernike_basis()
            logger.info(f"成功生成新的Zernike基，形状: {basis.shape}")
        
        # 可视化前9个基
        plt.figure(figsize=(9, 9))
        for i in range(min(9, basis.shape[0])):
            plt.subplot(3, 3, i+1)
            plt.imshow(basis[i], cmap='seismic')
            plt.title(f"Zernike {i+1}")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "zernike_basis.png"))
        plt.close()
        
        return True
    except Exception as e:
        logger.error(f"Zernike基生成失败: {e}")
        return False


def test_ncc():
    """测试归一化互相关功能"""
    logger.info("测试归一化互相关...")
    try:
        # 创建两个相似的图像
        img1 = np.random.rand(25, 25)
        img2 = img1 + 0.1 * np.random.rand(25, 25)  # 添加一些噪声
        
        # 计算NCC
        ncc = normalized_cross_correlation(img1, img2)
        logger.info(f"相似图像的NCC: {ncc:.4f}")
        
        # 创建两个不相关的图像
        img3 = np.random.rand(25, 25)
        img4 = np.random.rand(25, 25)
        
        # 计算NCC
        ncc2 = normalized_cross_correlation(img3, img4)
        logger.info(f"不相关图像的NCC: {ncc2:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"NCC计算失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    tests = [
        ("配置加载", test_load_config),
        ("相机参数加载", test_load_camera_params),
        ("TIFF加载", test_load_tiff),
        ("发射体检测", test_emitter_detection),
        ("补丁提取", test_patch_extraction),
        ("Zernike基生成", test_zernike_basis),
        ("归一化互相关", test_ncc),
    ]
    
    results = []
    for name, test_func in tests:
        logger.info(f"\n{'='*50}\n运行测试: {name}\n{'='*50}")
        success = test_func()
        results.append((name, success))
    
    # 打印摘要
    logger.info("\n\n测试摘要:")
    logger.info("-" * 40)
    for name, success in results:
        status = "通过" if success else "失败"
        logger.info(f"{name:20s}: {status}")
    logger.info("-" * 40)
    
    # 计算通过率
    passed = sum(1 for _, success in results if success)
    total = len(results)
    logger.info(f"通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    logger.info(f"\n测试输出保存在: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    run_all_tests()