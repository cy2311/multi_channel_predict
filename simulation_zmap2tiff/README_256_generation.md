# 256x256尺寸TIFF生成功能使用指南

本指南介绍如何使用新增的256x256尺寸TIFF生成功能，包括Zernike图裁剪和批量样本生成。

## 新增功能

### 1. 可调整的TIFF输出尺寸
- 支持自定义输出TIFF的尺寸（如256x256）
- 通过配置文件中的`tiff.roi_size`参数控制

### 2. Zernike图裁剪功能
- 支持从指定角落开始裁剪Zernike相位图
- 通过`zernike.crop_size`和`zernike.crop_offset`参数控制
- 同时支持TIFF生成时的裁剪偏移

### 3. 大规模样本生成
- 支持生成100个样本，每个样本200帧
- 总共20,000帧数据用于神经网络训练
- 支持不同数据量的渐进式训练测试

## 配置参数说明

### Zernike配置
```json
"zernike": {
  "crop_size": 256,           // 裁剪后的尺寸
  "crop_offset": [0, 0],      // 裁剪起始位置 [x_offset, y_offset]
  "interpolation_method": "cubic"
}
```

### TIFF配置
```json
"tiff": {
  "roi_size": 256,            // 输出TIFF尺寸
  "crop_offset": [0, 0],      // TIFF生成时的裁剪偏移
  "use_direct_rendering": true,
  "add_noise": true
}
```

### 样本配置
```json
"sample_configs": {
  "num_samples": 100,         // 生成样本数量
  "frames_per_sample": 200,   // 每个样本的帧数
  "sample_naming": "sample_{sample_id:03d}"
}
```

## 使用步骤

### 1. 快速测试（推荐先执行）
```bash
# 运行测试脚本，生成2个小样本验证功能
python scripts/test_256_generation.py
```

### 2. 生成完整的100样本数据集

#### 本地运行（占用IDE）
```bash
# 方法1: 使用便捷脚本（推荐）
# 交互式运行（需要手动确认）
python scripts/run_100_samples.py

# 自动运行（跳过确认，适用于脚本化环境）
python scripts/run_100_samples.py --auto-confirm

# 方法2: 直接使用批量生成器
python batch_tiff_generator.py configs/batch_config_100samples_256.json
```

#### SLURM集群运行（推荐，不占用IDE）
```bash
# 一键提交到集群
./slurm/quick_submit.sh gpu    # 使用GPU节点
./slurm/quick_submit.sh cpu    # 使用CPU节点

# 或手动提交
sbatch slurm/submit_100_samples.slurm      # GPU版本
sbatch slurm/submit_100_samples_cpu.slurm  # CPU版本
```

#### 监控集群任务

```bash
# 检查任务进度
./slurm/check_progress.sh

# 查看SLURM任务状态
squeue -u $USER

# 查看实时日志
tail -f logs/tiff_generation_*.out
```

## 故障排除

### 常见问题

#### 1. 任务提交后立即结束（EOFError）
**问题**: 任务在SLURM环境中遇到交互式输入提示
```
EOFError: EOF when reading a line
```

**解决方案**: 脚本已更新支持`--auto-confirm`参数
- SLURM脚本会自动使用此参数
- 本地测试时也可使用: `python scripts/run_100_samples.py --auto-confirm`

#### 2. 分区不存在错误
**问题**: `sbatch: error: invalid partition specified`

**解决方案**: 检查并更新SLURM脚本中的分区名称
```bash
# 查看可用分区
sinfo

# 修改脚本中的分区设置
#SBATCH --partition=your_partition_name
```

#### 3. 任务状态检查
```bash
# 检查任务是否在运行
squeue -j <job_id>

# 查看任务详细信息
scontrol show job <job_id>

# 查看任务日志
cat logs/tiff_generation_<job_id>.out
cat logs/tiff_generation_<job_id>.err
```

### 3. 自定义配置
可以基于`configs/batch_config_100samples_256.json`修改参数：
- 调整`crop_offset`来从不同角落开始裁剪
- 修改`num_samples`和`frames_per_sample`来生成不同数量的数据
- 调整`roi_size`和`crop_size`来使用不同的尺寸

## 渐进式训练建议

生成完整数据集后，可以按以下方式进行渐进式训练测试：

1. **20样本测试**: 使用前20个样本（4,000帧）
2. **50样本测试**: 使用前50个样本（10,000帧）
3. **100样本测试**: 使用前100个样本（20,000帧）
4. **150样本测试**: 生成额外50个样本
5. **200样本测试**: 生成额外50个样本

## 文件说明

### 核心文件
- `batch_config_100samples_256.json` - 100样本生成配置
- `scripts/test_256_generation.py` - 功能测试脚本
- `scripts/run_100_samples.py` - 本地运行脚本

### SLURM集群文件
- `slurm/submit_100_samples.slurm` - GPU节点作业脚本
- `slurm/submit_100_samples_cpu.slurm` - CPU节点作业脚本
- `slurm/quick_submit.sh` - 一键提交工具
- `slurm/check_progress.sh` - 进度检查工具
- `slurm/SLURM_GUIDE.md` - 详细使用指南

## 输出结构

```
outputs_100samples_256/
├── sample_001/
│   ├── emitters_sample_001.h5
│   └── simulation_256.ome.tiff
├── sample_002/
│   ├── emitters_sample_002.h5
│   └── simulation_256.ome.tiff
├── ...
├── sample_100/
│   ├── emitters_sample_100.h5
│   └── simulation_256.ome.tiff
├── batch_status.json
└── logs/                    # SLURM日志（集群运行时）
    ├── tiff_generation_*.out
    └── tiff_generation_*.err
```

## 性能优化

- 256x256尺寸相比1200x1200大幅减少了处理时间
- 支持多进程并行生成（通过`max_workers`参数控制）
- 支持断点续传功能
- 内存优化的TIFF生成算法

## 测试结果验证

测试脚本成功验证了以下功能：
- ✅ 256x256尺寸TIFF生成正常工作
- ✅ Zernike图裁剪功能正常
- ✅ 批量样本生成流程完整
- ✅ 生成的TIFF格式: (10, 256, 256) float32
- ✅ 像素值范围合理: [34.39, 958.82]

## 注意事项

1. 确保Zernike相位图文件路径正确
2. 裁剪参数不要超出原始图像边界
3. 建议先运行测试脚本验证配置正确性
4. 大规模生成时注意磁盘空间（100样本约需要几GB空间）
5. 支持断点续传，中断后可重新运行继续生成