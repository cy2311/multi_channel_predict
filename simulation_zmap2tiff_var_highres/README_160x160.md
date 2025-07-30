# 160x160分辨率TIFF样本生成指南

## 概述

本指南说明如何使用修改后的程序生成160x160分辨率的TIFF样本，同时保持与原始40x40样本相同的物理尺寸。

## 关键特性

### 1. 分辨率提升
- **输出分辨率**: 160x160像素（相比原来的40x40提升4倍）
- **物理FOV保持不变**: 4.044 x 3.953 μm
- **像素物理尺寸**: 自动重新计算以保持相同的物理覆盖范围

### 2. 像素尺寸计算

原始40x40配置：
- X方向像素尺寸: 101.11 nm
- Y方向像素尺寸: 98.83 nm
- 物理FOV: 4.044 x 3.953 μm

新的160x160配置：
- X方向像素尺寸: 25.2775 nm (101.11/4)
- Y方向像素尺寸: 24.7075 nm (98.83/4)
- 物理FOV: 4.044 x 3.953 μm（保持不变）

## 配置文件

### 主要配置文件

`configs/batch_config_100samples_160.json` - 用于生成100个160x160样本的批量配置

### 关键配置参数

```json
{
  "base_config": {
    "emitters": {
      "area_px": 160.0,  // 发射器生成区域
      "num_emitters": 5000,
      "frames": 200
    },
    "tiff": {
      "roi_size": 160,  // TIFF图像尺寸
      "filename": "simulation_160.ome.tiff"
    },
    "optical": {
      "use_default_config": false,  // 使用自定义光学参数
      "pixel_size_nm_x": 25.2775,  // 重新计算的像素尺寸
      "pixel_size_nm_y": 24.7075,
      "wavelength_nm": 660,
      "NA": 1.4
    },
    "zernike": {
      "crop_size": 160  // Zernike系数处理尺寸
    }
  },
  "sample_configs": {
    "num_samples": 100,
    "frames_per_sample": 200,
    "sample_naming": "sample_160_{sample_id:03d}"
  }
}
```

## 使用方法

### 1. 生成160x160样本

```bash
# 进入工作目录
cd /home/guest/Others/DECODE_rewrite/simulation_zmap2tiff_var_highres

# 运行批量生成
python batch_tiff_generator.py --batch_config configs/batch_config_100samples_160.json
```

### 2. 输出结果

生成的文件将保存在 `outputs_100samples_160/` 目录下：

```
outputs_100samples_160/
├── sample_160_001/
│   ├── emitters.h5              # 发射器数据
│   └── sample_160_001.ome.tiff  # 160x160 TIFF图像
├── sample_160_002/
│   ├── emitters.h5
│   └── sample_160_002.ome.tiff
└── ...
```

### 3. 验证结果

可以使用以下Python代码验证生成的TIFF文件：

```python
import tifffile as tiff
import numpy as np

# 读取TIFF文件
with tiff.TiffFile('outputs_100samples_160/sample_160_001/sample_160_001.ome.tiff') as tf:
    images = tf.asarray()
    print(f'图像形状: {images.shape}')  # 应该是 (200, 160, 160)
    
    # 验证物理FOV
    pixel_size_x_nm = 25.2775
    pixel_size_y_nm = 24.7075
    fov_x_um = 160 * pixel_size_x_nm / 1000
    fov_y_um = 160 * pixel_size_y_nm / 1000
    print(f'物理FOV: {fov_x_um:.3f} x {fov_y_um:.3f} μm')  # 应该是 4.044 x 3.953 μm
```

## 技术实现

### 1. 修改的文件

- `tiff_generator.py`: 修改了 `load_config()` 函数以支持自定义光学参数
- `configs/batch_config_100samples_160.json`: 新的160x160配置文件

### 2. 关键修改点

#### `load_config()` 函数修改

```python
def load_config(optical_config: Optional[Dict[str, Any]] = None) -> Tuple[float, float, float, float]:
    """加载光学参数配置
    
    Parameters
    ----------
    optical_config : dict, optional
        自定义光学参数配置，如果为None则使用默认配置文件
    """
    if optical_config is not None:
        # 使用传入的光学参数
        opt_cfg = optical_config
    else:
        # 使用默认配置文件
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default_config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        opt_cfg = cfg["optical"]
    
    return (
        float(opt_cfg["wavelength_nm"]),
        float(opt_cfg["pixel_size_nm_x"]),
        float(opt_cfg["pixel_size_nm_y"]),
        float(opt_cfg["NA"]),
    )
```

#### `generate_tiff_stack()` 函数修改

```python
# 加载光学参数
optical_config = config.get('optical', {})
use_custom_optical = not optical_config.get('use_default_config', True)

if use_custom_optical:
    # 使用配置文件中的自定义光学参数
    wavelength_nm, pix_x_nm, pix_y_nm, NA = load_config(optical_config)
else:
    # 使用默认配置文件
    wavelength_nm, pix_x_nm, pix_y_nm, NA = load_config()
```

## 验证测试

### 测试脚本

`test_160_generation.py` - 用于验证160x160配置的测试脚本

运行测试：
```bash
python test_160_generation.py
```

### 测试结果

✅ **图像尺寸**: 正确生成160x160像素的TIFF图像  
✅ **物理FOV**: 与40x40样本完全一致（4.044 x 3.953 μm）  
✅ **发射器位置**: 正确缩放到160x160坐标系统  
✅ **像素尺寸**: 正确计算为25.2775 x 24.7075 nm  
✅ **数据完整性**: 发射器数据和TIFF图像正确生成  

## 与40x40样本的对比

| 参数 | 40x40样本 | 160x160样本 | 说明 |
|------|-----------|-------------|------|
| 图像分辨率 | 40x40像素 | 160x160像素 | 4倍分辨率提升 |
| X方向像素尺寸 | 101.11 nm | 25.2775 nm | 1/4缩小 |
| Y方向像素尺寸 | 98.83 nm | 24.7075 nm | 1/4缩小 |
| 物理FOV | 4.044 x 3.953 μm | 4.044 x 3.953 μm | 完全一致 |
| 发射器坐标范围 | [0, 40) | [0, 160) | 按比例缩放 |

## 注意事项

1. **内存使用**: 160x160图像的内存使用量是40x40的16倍，请确保有足够的内存
2. **处理时间**: 生成时间会相应增加，建议使用并行处理
3. **存储空间**: TIFF文件大小会显著增加
4. **兼容性**: 生成的160x160样本与现有的40x40处理流程需要相应调整

## 故障排除

### 常见问题

1. **内存不足错误**
   - 减少 `max_workers` 参数
   - 减少 `frames_per_sample` 参数
   - 分批处理样本

2. **像素尺寸计算错误**
   - 检查配置文件中的 `pixel_size_nm_x` 和 `pixel_size_nm_y` 参数
   - 确保 `use_default_config` 设置为 `false`

3. **发射器位置超出范围**
   - 检查 `area_px` 参数是否设置为160.0
   - 验证 `crop_size` 参数是否为160

## 扩展使用

### 生成其他分辨率

要生成其他分辨率（如320x320），只需：

1. 复制160x160配置文件
2. 修改以下参数：
   - `area_px`: 320.0
   - `roi_size`: 320
   - `crop_size`: 320
   - `pixel_size_nm_x`: 101.11/8 = 12.63875
   - `pixel_size_nm_y`: 98.83/8 = 12.35375

### 自定义物理FOV

如果需要不同的物理FOV，调整像素尺寸参数：

```
新的像素尺寸 = 目标FOV(μm) * 1000 / 图像尺寸(像素)
```

例如，对于160x160像素覆盖8x8 μm的FOV：
- `pixel_size_nm_x`: 8000 / 160 = 50 nm
- `pixel_size_nm_y`: 8000 / 160 = 50 nm