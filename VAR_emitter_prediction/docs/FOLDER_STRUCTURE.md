# 文件夹结构说明

本项目已重新组织文件结构，以提高代码的可维护性和清晰度。

## 文件夹结构

```
VAR_emitter_prediction/
├── configs/                    # 配置文件
│   ├── config.json
│   ├── config_true_var.json
│   └── config_true_var_slurm.json
├── docs/                       # 文档文件
│   ├── FOLDER_STRUCTURE.md    # 文件夹结构说明
│   ├── README.md              # 项目说明
│   ├── README_SLURM.md        # SLURM使用说明
│   └── README_TrueVAR.md      # TrueVAR模型说明
├── models/                     # 训练好的模型文件 (.pt)
├── outputs/                    # 训练输出和中间文件
├── logs/                       # 日志文件
│   └── tensorboard/           # TensorBoard事件文件
├── scripts/                    # 脚本文件
│   ├── submit_training.sh     # 训练提交脚本
│   └── submit_var_training.slurm # SLURM作业脚本
├── tests/                      # 测试和演示文件
│   ├── demo_true_var.py       # VAR模型演示
│   ├── example_usage.py       # 使用示例
│   ├── inference.py           # 推理脚本
│   └── inference_true_var.py  # TrueVAR推理脚本
├── train_true_var.py          # 主训练脚本
├── train_var_emitter.py       # VAR训练脚本
├── run_training_direct.py     # 直接训练脚本
├── var_emitter_model_true.py  # TrueVAR模型定义
├── var_emitter_model.py       # VAR模型定义
├── var_emitter_loss.py        # 损失函数
├── var_dataset.py             # 数据集定义
└── requirements.txt           # 依赖包列表
```

## 主要变更

1. **配置文件**: 所有 `.json` 配置文件移动到 `configs/` 文件夹
2. **模型文件**: 训练生成的 `.pt` 模型文件将保存到 `models/` 文件夹
3. **文档文件**: 所有 `.md` 文档文件移动到 `docs/` 文件夹
4. **脚本文件**: SLURM脚本和Shell脚本移动到 `scripts/` 文件夹
5. **测试文件**: 演示、推理和示例文件移动到 `tests/` 文件夹
6. **输出文件**: 训练过程中的输出文件保存到 `outputs/` 文件夹
7. **日志文件**: TensorBoard和其他日志文件统一保存在 `logs/` 文件夹

## 使用说明

### 训练
```bash
# 使用默认配置训练
python train_true_var.py

# 使用指定配置文件训练
python train_true_var.py --config configs/config_true_var_slurm.json

# 直接训练（不使用SLURM）
python run_training_direct.py
```

### 测试和演示
```bash
# 运行演示
cd tests
python demo_true_var.py

# 运行推理
cd tests
python inference_true_var.py --model ../models/best_model.pt --config ../configs/config_true_var.json
```

## 注意事项

- 所有测试文件中的导入路径已更新，可以正确导入父目录中的模块
- 训练脚本会自动创建必要的文件夹（models/, outputs/等）
- 配置文件路径已更新为相对于新的文件夹结构