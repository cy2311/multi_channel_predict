# Zernike系数处理说明

## 问题描述

用户提出了一个重要的技术问题：在配置文件中设置了固定的Zernike系数，但实际上每个发射器的Zernike系数应该从Zmap中插值获取，而不是使用固定值。

### 原始配置文件中的问题

在 `default_config.json` 中有如下设置：
```json
"psf": {
  "type": "zernike",
  "zernike_coefficients": {
    "4": 0.1,
    "5": 0.05,
    "6": 0.03
  }
}
```

这种设置会导致所有发射器使用相同的固定Zernike系数，这与实际的光学像差分布不符。

## 整合方案的正确处理方式

### 1. 数据流程

我们的整合方案正确实现了从Zmap到个体发射器Zernike系数的完整流程：

```
Zmap (patches.h5) → 插值计算 → 每个发射器的独特系数 → PSF生成 → TIFF图像
```

### 2. 关键步骤

#### 步骤1: Zernike系数插值计算

在 `compute_zernike_coeffs.py` 中：

```python
def compute_phase_coeffs(phase_maps: np.ndarray, em_xy: np.ndarray) -> np.ndarray:
    """从相位图中为每个发射器位置插值计算相位系数"""
    # 使用三次样条插值
    for idx in range(n_coeff):
        phase_coeffs[:, idx] = ndi.map_coordinates(
            phase_maps[idx], [y, x], order=3, mode='nearest'
        )

def compute_mag_coeffs(coords: np.ndarray, coeff_mag_patch: np.ndarray, em_xy: np.ndarray) -> np.ndarray:
    """为每个发射器位置插值计算幅度系数"""
    for idx in range(n_coeff):
        vals = interp.griddata(coords, coeff_mag_patch[:, idx], em_xy, method='cubic')
```

#### 步骤2: 个体系数存储

计算得到的系数被保存到HDF5文件中：
```python
def save_coeffs(emitters_path: Path, phase_coeffs: np.ndarray, mag_coeffs: np.ndarray):
    with h5py.File(emitters_path, 'a') as f:
        grp = f.create_group('zernike_coeffs')
        grp.create_dataset('phase', data=phase_coeffs)  # 每个发射器的相位系数
        grp.create_dataset('mag', data=mag_coeffs)      # 每个发射器的幅度系数
```

#### 步骤3: PSF生成时使用个体系数

在 `tiff_generator.py` 中，为每个发射器使用其独特的系数：

```python
def simulate_frame(frame_idx: int, emitters_data: Dict[str, np.ndarray], ...):
    # 获取该帧的活跃发射器
    active_ids = emitters_data['ids_rec'][frame_mask]
    
    # 获取对应的Zernike系数（每个发射器都不同）
    coeff_mag = emitters_data['coeff_mag_all'][active_ids]
    coeff_phase = emitters_data['coeff_phase_all'][active_ids]
    
    # 为每个发射器生成独特的PSF
    for i in range(len(active_ids)):
        pupil = construct_pupil(coeff_mag[i], coeff_phase[i], basis, pupil_mask)
        psf = generate_psf(pupil_defocus)
```

### 3. 配置文件修正

我们已经修正了 `pipeline_config.json`：

```json
"optical": {
  "description": "光学参数（从default_config.json继承，但不使用固定的PSF Zernike系数）",
  "use_default_config": true,
  "ignore_fixed_psf_coeffs": true,
  "note": "每个发射器的Zernike系数将从Zmap中插值获取，而不是使用固定值"
}
```

## 技术优势

### 1. 真实性
- 每个发射器根据其空间位置获得独特的光学像差
- 反映了真实显微镜系统中的空间变化像差

### 2. 精确性
- 使用三次样条插值确保平滑的空间变化
- 支持亚像素精度的位置插值

### 3. 灵活性
- 支持任意复杂的像差分布
- 可以处理不同类型的Zernike系数（相位和幅度）

## 验证方法

可以通过以下方式验证系数的正确性：

1. **检查系数分布**：
   ```python
   with h5py.File('emitters.h5', 'r') as f:
       phase_coeffs = f['zernike_coeffs/phase'][:]
       print(f"相位系数范围: {phase_coeffs.min():.3f} 到 {phase_coeffs.max():.3f}")
       print(f"相位系数标准差: {phase_coeffs.std():.3f}")
   ```

2. **可视化系数变化**：
   - 运行时会自动生成 `*_zernike_coeffs.png` 图像
   - 显示随机选择的发射器的系数分布

3. **PSF差异检查**：
   - 不同位置的发射器应该产生不同的PSF形状
   - 可以通过比较生成的TIFF图像中的PSF来验证

## 总结

整合方案正确地解决了固定Zernike系数的问题：

✅ **不使用** `default_config.json` 中的固定PSF系数  
✅ **从Zmap插值** 为每个发射器计算独特的系数  
✅ **保存个体系数** 到HDF5文件中  
✅ **使用个体系数** 生成每个发射器的PSF  
✅ **支持空间变化** 的光学像差模拟  

这确保了生成的TIFF图像具有真实的空间变化光学像差特性，符合实际显微镜系统的物理特性。