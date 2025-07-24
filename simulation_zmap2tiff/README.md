# Simulation Zmap2TIFF

统一的Zmap到TIFF模拟流程，直接使用`phase_retrieval_tiff2h5`的结果进行模拟数据集生成。

## 功能概述

本系统整合了从Zernike系数到模拟TIFF图像的完整流程，消除了原有系统中的冗余步骤，提供了模块化的设计和完整的可视化功能。

### 主要特性

- **直接使用Zmap数据**: 无需中间转换，直接从`result.h5`文件读取Zernike系数
- **统一处理流程**: 一个主程序管理整个流程，减少中间文件
- **模块化设计**: 清晰的模块分离，便于维护和扩展
- **完整可视化**: 包含发射器生命周期、统计分析、图像质量等多维度可视化
- **灵活配置**: 支持配置文件和命令行参数
- **相机模型**: 完整的相机噪声模型，包括量子效率、EM增益、读取噪声等

## 系统架构

```
simulation_zmap2tiff/
├── main.py                 # 主程序
├── README.md              # 说明文档
├── modules/               # 核心模块
│   ├── config.py         # 配置管理
│   ├── zmap_processor.py # Zmap数据处理
│   ├── emitter_manager.py # 发射器管理
│   ├── image_generator.py # 图像生成
│   └── post_processor.py # 后处理(相机模型)
└── result/               # 输出结果
    ├── visualization/    # 可视化结果
    └── tiff/            # TIFF图像输出
```

## 模块说明

### 1. 配置管理模块 (`config.py`)
- 统一管理光学参数、相机参数、文件路径等
- 支持配置文件和默认配置
- 自动创建输出目录

### 2. Zmap处理模块 (`zmap_processor.py`)
- 从`result.h5`文件加载Zernike系数
- 提供系数插值和可视化功能
- 支持相位图重建和分析

### 3. 发射器管理模块 (`emitter_manager.py`)
- 生成发射器的空间位置、强度、时间属性
- 管理发射器生命周期和闪烁模式
- 分配Zernike系数给每个发射器
- 完整的发射器可视化和统计分析

### 4. 图像生成模块 (`image_generator.py`)
- 基于Zernike系数构建复数瞳孔函数
- 生成点扩散函数(PSF)
- 模拟单帧和多帧图像
- 图像质量分析和可视化

### 5. 后处理模块 (`post_processor.py`)
- 应用相机模型(量子效率、EM增益、噪声)
- 添加背景噪声
- 噪声特性分析和可视化
- 信噪比计算

## 使用方法

### 基本使用

```bash
# 使用默认配置
python main.py

# 指定Zmap文件
python main.py --zmap-file /path/to/result.h5

# 指定输出目录
python main.py --output-dir /path/to/output

# 设置帧数和发射器数量
python main.py --num-frames 100 --num-emitters 1000
```

### 高级选项

```bash
# 跳过可视化(加快处理速度)
python main.py --skip-visualization

# 只生成理想光子图像，跳过相机模型
python main.py --skip-camera-model

# 使用自定义配置文件
python main.py --config custom_config.json
```

### 配置文件示例

创建`config.json`文件:

```json
{
    "optical": {
        "wavelength": 670e-9,
        "NA": 1.4,
        "n": 1.518,
        "pixel_size": 65e-9,
        "magnification": 100
    },
    "camera": {
        "QE": 0.9,
        "EMGain": 30.0,
        "read_noise_e": 1.0,
        "offset": 100.0,
        "A2D": 1.0,
        "max_adu": 65535
    },
    "simulation": {
        "num_frames": 50,
        "num_emitters": 500,
        "image_size": [256, 256],
        "upsampling_factor": 4,
        "seed": 42
    },
    "paths": {
        "zmap_file": "../phase_retrieval_tiff2h5/result/result.h5",
        "output_dir": "./result"
    }
}
```

## 输出结果

### TIFF图像
- `photon_stack.tiff`: 理想光子图像堆栈
- `camera_stack.tiff`: 相机输出图像堆栈(包含噪声)

### 发射器数据
- `emitters_data.h5`: 完整的发射器数据和帧记录

### 可视化结果
- `zernike_coefficients.png`: Zernike系数可视化
- `emitter_distribution.png`: 发射器空间分布
- `emitter_timeline.png`: 发射器时间线
- `emitter_lifetime_stats.png`: 生命周期统计
- `emitter_intensity_distribution.png`: 强度分布
- `emitter_frame_statistics.png`: 每帧统计
- `psf_examples.png`: PSF示例
- `photon_frame_montage.png`: 帧蒙太奇
- `photon_image_statistics.png`: 图像统计
- `camera_noise_analysis.png`: 噪声分析
- `camera_stack_comparison.png`: 堆栈比较

### 处理报告
- `processing_report.md`: 详细的处理报告，包含所有统计信息

## 发射器生命周期可视化

系统提供了完整的发射器生命周期分析:

1. **空间分布**: 发射器在图像中的位置分布
2. **时间线**: 每个发射器的开启和关闭时间
3. **生命周期统计**: 持续时间的分布和统计
4. **强度分布**: 发射器强度的统计分析
5. **每帧统计**: 每帧中活跃发射器的数量变化

## 技术特点

### 性能优化
- 向量化计算，提高处理速度
- 内存高效的图像生成
- 可选的可视化步骤，平衡速度和分析需求

### 数据完整性
- 完整的元数据保存
- 可重现的随机数生成
- 详细的处理日志

### 扩展性
- 模块化设计，易于添加新功能
- 灵活的配置系统
- 标准化的数据接口

## 依赖要求

```
numpy
scipy
matplotlib
h5py
tifffile
tqdm
```

## 安装依赖

```bash
pip install numpy scipy matplotlib h5py tifffile tqdm
```

## 故障排除

### 常见问题

1. **找不到Zmap文件**
   - 确保`phase_retrieval_tiff2h5/result/result.h5`文件存在
   - 使用`--zmap-file`参数指定正确路径

2. **内存不足**
   - 减少`num_frames`或`num_emitters`
   - 降低`image_size`或`upsampling_factor`

3. **处理速度慢**
   - 使用`--skip-visualization`跳过可视化
   - 减少发射器数量或帧数

4. **输出目录权限问题**
   - 确保对输出目录有写权限
   - 使用`--output-dir`指定其他目录

## 开发说明

### 添加新的可视化
1. 在相应模块中添加可视化函数
2. 在主程序中调用
3. 更新配置选项(如需要)

### 修改相机模型
1. 编辑`post_processor.py`中的相机参数
2. 修改噪声模型函数
3. 更新配置文件

### 扩展发射器模型
1. 修改`emitter_manager.py`中的生成函数
2. 添加新的属性和可视化
3. 更新HDF5保存格式

## 版本历史

- **v1.0**: 初始版本，整合所有核心功能
- 统一处理流程
- 完整的可视化系统
- 模块化架构

## 许可证

本项目遵循MIT许可证。