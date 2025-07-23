#!/usr/bin/env python

import os
import sys

# 设置默认的checkpoint路径
default_checkpoint = 'nn_train/checkpoint_epoch_250.pt'

def main():
    # 检查命令行参数
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = default_checkpoint
        print(f"使用默认checkpoint: {default_checkpoint}")
    
    # 检查checkpoint文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"错误: Checkpoint文件 {checkpoint_path} 不存在!")
        return
    
    # 运行训练脚本并传递checkpoint路径
    cmd = f"python neuronal_network/train_with_count_loss.py {checkpoint_path}"
    print(f"执行命令: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    main()