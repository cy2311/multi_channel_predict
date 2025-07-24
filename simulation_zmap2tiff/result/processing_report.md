# 处理报告

## 基本信息
- **处理时间**: 2.2 秒
- **Zmap文件**: ../phase_retrieval_tiff2h5/result/result.h5
- **输出目录**: result
- **生成时间**: 2025-07-24 21:56:58

## Zmap数据
- **数据类型**: dict
- **数据已加载**: ✅

## 发射器统计
- **总发射器数量**: 200
- **平均生命周期**: 10.7 帧
- **中位生命周期**: 7.5 帧
- **生命周期范围**: 0 - 52 帧
- **平均每帧发射器数**: 20.0

## 图像统计
- **总帧数**: 10
- **图像尺寸**: 256 × 256 像素
- **总光子数**: 778,419
- **平均每帧光子数**: 77842
- **图像数据类型**: float32

## 配置参数

### 光学参数
```
{'wavelength': 6.7e-07, 'NA': 1.4, 'n': 1.518, 'pixel_size': 6.5e-08, 'magnification': 100, 'zernike_modes': 15}
```

### 相机参数
```
{'QE': 0.9, 'EMGain': 30.0, 'read_noise_e': 1.0, 'offset': 100.0, 'A2D': 1.0, 'max_adu': 65535}
```

### 模拟参数
```
{'num_frames': 10, 'num_emitters': 200, 'image_size': [256, 256], 'upsampling_factor': 4, 'seed': 42, 'emitter_intensity_range': [1000, 5000], 'emitter_lifetime_range': [1, 10], 'background_level': 10.0}
```

## 输出文件
- **发射器数据**: `emitters_data.h5`
- **理想光子图像**: `photon_stack.tiff`
- **相机输出图像**: `camera_stack.tiff` (如果启用)
- **可视化结果**: `visualization/` 目录

## 处理流程
1. ✅ 加载Zmap数据 (245个系数)
2. ✅ 生成发射器 (200个)
3. ✅ 分配Zernike系数
4. ✅ 转换为帧记录 (4条)
5. ✅ 生成图像堆栈 (10帧)
6. ✅ 保存结果文件

---
*报告生成时间: 2025-07-24 21:56:58*
