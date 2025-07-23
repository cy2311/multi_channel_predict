# 相位恢复与Zernike系数计算流程

这个项目实现了从OME-TIFF文件中检测发射体，提取PSF补丁，进行相位恢复，计算Zernike系数，并生成全视场Zernike系数图的完整流程。

## 功能概述

- 从OME-TIFF文件中检测发射体位置
- 以发射体像素为中心裁剪25×25×201的图像块
- 剔除距离小于25像素的发射体
- 以第75帧为中心，间隔2帧，两侧各取20帧，共41帧作为原始PSF
- 使用Gerchberg-Saxton算法进行相位恢复
- 计算Zernike系数并生成全视场Zernike系数图
- 可视化重建的PSF与原始PSF的对比

## 使用方法

```bash
python tiff2h5_phase_retrieval.py
```

## 输入文件

- OME-TIFF文件：`../beads/spool_100mW_30ms_3D_1_2/spool_100mW_30ms_3D_1_2_MMStack_Default.ome.tif`
- 相机参数文件：`../beads/spool_100mW_30ms_3D_1_2/camera_parameters.json`
- 配置文件：`../configs/default_config.json`

## 输出文件

所有结果保存在`result`目录下：

- `result.h5`：包含所有处理结果的HDF5文件
  - `/coords`：发射体坐标
  - `/patches`：选定的PSF补丁
  - `/zernike/coeff_mag`：幅度Zernike系数
  - `/zernike/coeff_phase`：相位Zernike系数
  - `/zernike/mean_ncc`：平均NCC值
  - `/z_maps/phase`：全视场Zernike系数图
- `psf_comparison.png`：中心PSF对比图
- `psf_comparison_five.png`：5帧PSF对比图
- `zernike_coefficients.png`：单个发射体的Zernike系数图
- `zernike_coefficients_random.png`：随机发射体的Zernike系数图
- `zernike_phase_maps.png`：全视场Zernike系数图

## 算法流程

1. 检测发射体位置
2. 提取PSF补丁
3. 选择特定帧
4. 运行GS相位恢复
5. 应用OTF高斯低通滤波
6. 计算Zernike系数
7. 生成全视场Zernike系数图
8. 可视化结果

## 参数设置

- 补丁大小：25×25×201
- 发射体最小距离：25像素
- 中心帧：75
- 帧间隔：2
- 每侧帧数：20
- 最大迭代次数：100
- NCC阈值：0.7