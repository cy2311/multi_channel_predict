# SLURM 集群任务提交指南

本指南说明如何在SLURM集群上提交256x256 TIFF生成任务，避免占用本地IDE资源。

## 📋 可用的SLURM脚本

### 1. GPU版本 (推荐)
- **文件**: `submit_100_samples.slurm`
- **资源**: 1个GPU节点，8核CPU，32GB内存
- **预计时间**: 4-6小时
- **适用**: 有GPU资源的集群

### 2. CPU版本
- **文件**: `submit_100_samples_cpu.slurm`
- **资源**: 16核CPU，64GB内存
- **预计时间**: 6-8小时
- **适用**: 仅CPU的集群

## 🚀 使用步骤

### 1. 准备工作
```bash
# 进入工作目录
cd /home/guest/Others/DECODE_rewrite/simulation_zmap2tiff\ copy

# 创建日志目录
mkdir -p logs

# 检查配置文件
ls configs/batch_config_100samples_256.json
```

### 2. 修改SLURM脚本（根据集群配置）
编辑相应的`.slurm`文件，调整以下参数：

```bash
# 根据集群情况修改分区名称
#SBATCH --partition=gpu        # 或 compute, normal 等

# 根据需要调整资源
#SBATCH --cpus-per-task=8      # CPU核心数
#SBATCH --mem=32G              # 内存大小
#SBATCH --time=06:00:00        # 最大运行时间

# 如果需要特定GPU类型
#SBATCH --gres=gpu:v100:1      # 指定GPU型号
```

### 3. 激活环境设置
在SLURM脚本中取消注释并修改环境设置：

```bash
# 方法1: 使用module系统
module load anaconda3
conda activate your_env_name

# 方法2: 直接激活conda
source /path/to/conda/etc/profile.d/conda.sh
conda activate your_env_name

# 方法3: 使用虚拟环境
source /path/to/venv/bin/activate
```

### 4. 提交任务
```bash
# 提交GPU版本
sbatch submit_100_samples.slurm

# 或提交CPU版本
sbatch submit_100_samples_cpu.slurm
```

## 📊 监控任务

### 查看任务状态
```bash
# 查看所有任务
squeue -u $USER

# 查看特定任务
squeue -j <job_id>

# 查看任务详情
scontrol show job <job_id>
```

### 查看实时日志
```bash
# 查看输出日志
tail -f logs/tiff_generation_<job_id>.out

# 查看错误日志
tail -f logs/tiff_generation_<job_id>.err
```

### 取消任务
```bash
# 取消特定任务
scancel <job_id>

# 取消所有任务
scancel -u $USER
```

## 📈 任务进度监控

### 检查生成进度
```bash
# 查看已生成的样本数
ls -1d outputs_100samples_256/sample_* | wc -l

# 查看总文件数
find outputs_100samples_256 -name "*.tiff" | wc -l

# 查看数据大小
du -sh outputs_100samples_256/
```

### 检查批量状态
```bash
# 查看批量生成状态
cat outputs_100samples_256/batch_status.json | jq '.summary'
```

## ⚠️ 重要注意事项

### 1. 路径配置
- 确保所有路径都是绝对路径
- 检查Zernike图文件路径是否正确
- 验证输出目录权限

### 2. 资源估算
- **内存**: 每个样本约需要2-4GB内存
- **存储**: 100样本约需要5-10GB磁盘空间
- **时间**: GPU约4-6小时，CPU约6-8小时

### 3. 环境依赖
确保集群节点上有以下依赖：
- Python 3.8+
- NumPy, SciPy
- h5py, tifffile
- 其他项目依赖

### 4. 断点续传
如果任务中断，重新提交会自动从断点继续：
```bash
# 重新提交相同的脚本即可
sbatch submit_100_samples.slurm
```

## 🔧 故障排除

### 常见问题

1. **任务排队时间长**
   - 调整资源请求（减少CPU/内存/时间）
   - 选择负载较轻的分区

2. **内存不足**
   - 增加 `--mem` 参数
   - 减少并行处理数量

3. **超时**
   - 增加 `--time` 参数
   - 考虑分批处理

4. **环境问题**
   - 检查Python环境是否正确激活
   - 验证依赖包是否安装

### 调试命令
```bash
# 测试环境
srun --pty bash  # 获取交互式节点
python -c "import numpy, h5py, tifffile; print('环境OK')"

# 测试小样本
python test_256_generation.py
```

## 📋 任务完成后

任务完成后，数据将保存在 `outputs_100samples_256/` 目录中：
- 100个样本目录 (`sample_001` 到 `sample_100`)
- 每个样本包含200帧的256x256 TIFF文件
- 总计20,000帧训练数据

可以直接用于神经网络训练或其他分析任务。