import os
import sys
import time
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import tifffile
import h5py
from torch.utils.tensorboard import SummaryWriter

# 导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuronal_network.first_level_unets import UNetLevel1
from neuronal_network.second_level_network import SecondLevelNet
from neuronal_network.loss.count_loss import CountLoss, get_emitter_count_per_frame


class EmitterDataset(Dataset):
    """
    数据集类，用于加载TIFF帧和对应的发射体信息
    """
    def __init__(
        self,
        tiff_dir: str,
        emitter_dir: str,
        patch_size: int = 600,
        stride: int = 300,
        frame_window: int = 3,
        transform=None,
        max_frames_per_file: Optional[int] = None,
        train_val_split: float = 0.8,
        is_train: bool = True,
    ):
        """
        初始化数据集
        
        Args:
            tiff_dir: TIFF文件目录
            emitter_dir: 发射体H5文件目录
            patch_size: 图像块大小
            stride: 图像块滑动步长
            frame_window: 连续帧窗口大小
            transform: 数据增强转换
            max_frames_per_file: 每个文件最大帧数
            train_val_split: 训练/验证集分割比例
            is_train: 是否为训练集
        """
        self.tiff_dir = Path(tiff_dir)
        self.emitter_dir = Path(emitter_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.frame_window = frame_window
        self.transform = transform
        self.max_frames_per_file = max_frames_per_file
        self.train_val_split = train_val_split
        self.is_train = is_train
        
        # 获取所有TIFF文件
        self.tiff_files = sorted(list(self.tiff_dir.glob("*.ome.tiff")))
        
        # 构建数据索引
        self.samples = self._build_sample_index()
        
    def _build_sample_index(self) -> List[Dict]:
        """
        构建数据索引，包含所有可能的图像块和帧窗口
        
        Returns:
            包含样本信息的字典列表
        """
        samples = []
        
        for tiff_idx, tiff_file in enumerate(self.tiff_files):
            # 获取对应的发射体文件
            emitter_file = self.emitter_dir / f"emitters_set{tiff_idx}.h5"
            if not emitter_file.exists():
                print(f"Warning: Emitter file {emitter_file} not found, skipping {tiff_file}")
                continue
                
            # 获取TIFF文件信息
            with tifffile.TiffFile(str(tiff_file)) as f:
                n_frames = len(f.pages)
                if self.max_frames_per_file is not None:
                    n_frames = min(n_frames, self.max_frames_per_file)
                
                # 获取图像尺寸
                page = f.pages[0]
                height, width = page.shape
            
            # 计算可能的图像块位置
            x_positions = list(range(0, width - self.patch_size + 1, self.stride))
            y_positions = list(range(0, height - self.patch_size + 1, self.stride))
            
            # 确保最后一个块包含图像边缘
            if width - self.patch_size > 0 and x_positions[-1] < width - self.patch_size:
                x_positions.append(width - self.patch_size)
            if height - self.patch_size > 0 and y_positions[-1] < height - self.patch_size:
                y_positions.append(height - self.patch_size)
            
            # 获取每帧的发射体计数
            try:
                frame_counts = get_emitter_count_per_frame(str(emitter_file))
                # 确保帧数匹配
                frame_counts = frame_counts[:n_frames] if len(frame_counts) > n_frames else frame_counts
            except Exception as e:
                print(f"Error loading emitter counts from {emitter_file}: {e}")
                continue
            
            # 为每个可能的帧窗口和图像块创建样本
            for frame_idx in range(n_frames - self.frame_window + 1):
                for y in y_positions:
                    for x in x_positions:
                        # 计算此图像块中的发射体计数
                        # 这里简化处理，仅使用中间帧的计数
                        middle_frame_idx = frame_idx + self.frame_window // 2
                        count = frame_counts[middle_frame_idx] if middle_frame_idx < len(frame_counts) else 0
                        
                        samples.append({
                            "tiff_file": tiff_file,
                            "emitter_file": emitter_file,
                            "frame_idx": frame_idx,
                            "x": x,
                            "y": y,
                            "count": count,
                            "tiff_idx": tiff_idx
                        })
        
        # 训练/验证集分割
        n_samples = len(samples)
        split_idx = int(n_samples * self.train_val_split)
        
        if self.is_train:
            return samples[:split_idx]
        else:
            return samples[split_idx:]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取指定索引的样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含帧数据和标签的字典
        """
        sample_info = self.samples[idx]
        
        # 加载TIFF文件的指定帧
        frames = []
        with tifffile.TiffFile(str(sample_info["tiff_file"])) as f:
            for i in range(self.frame_window):
                frame_idx = sample_info["frame_idx"] + i
                if frame_idx < len(f.pages):
                    page = f.pages[frame_idx]
                    frame = page.asarray()
                    
                    # 提取图像块
                    x, y = sample_info["x"], sample_info["y"]
                    patch = frame[y:y+self.patch_size, x:x+self.patch_size]
                    
                    # 归一化到[0,1]范围
                    patch = patch.astype(np.float32)
                    if patch.max() > 0:
                        patch = patch / patch.max()
                    
                    frames.append(patch)
        
        # 转换为张量
        frame_tensors = [torch.from_numpy(frame).unsqueeze(0) for frame in frames]  # 添加通道维度
        
        # 获取标签（发射体计数）
        count = torch.tensor(sample_info["count"], dtype=torch.float32)
        
        # 应用数据增强
        if self.transform:
            for i in range(len(frame_tensors)):
                frame_tensors[i] = self.transform(frame_tensors[i])
        
        return {
            "frames": frame_tensors,  # 列表，包含3个形状为[1, patch_size, patch_size]的张量
            "count": count,  # 标量
            "coords": (sample_info["x"], sample_info["y"]),  # 图像块坐标
            "frame_idx": sample_info["frame_idx"],  # 起始帧索引
            "tiff_idx": sample_info["tiff_idx"]  # TIFF文件索引
        }


class MemoryTracker:
    """
    内存跟踪器，用于监控训练过程中的内存使用情况
    """
    def __init__(self, log_dir: str, enabled: bool = True):
        self.enabled = enabled
        if not self.enabled:
            return
            
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.log_file = self.log_dir / "memory_usage.csv"
        
        # 初始化日志文件
        with open(self.log_file, "w") as f:
            f.write("step,gpu_memory_allocated,gpu_memory_reserved\n")
        
        self.step = 0
    
    def update(self):
        """
        更新内存使用情况
        """
        if not self.enabled or not torch.cuda.is_available():
            return
            
        # 获取GPU内存使用情况
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
        
        # 记录到日志文件
        with open(self.log_file, "a") as f:
            f.write(f"{self.step},{allocated:.2f},{reserved:.2f}\n")
        
        self.step += 1


class TrainModel:
    """
    模型训练类
    """
    def __init__(
        self,
        tiff_dir: str,
        emitter_dir: str,
        output_dir: str,
        patch_size: int = 600,
        stride: int = 300,
        batch_size: int = 4,
        num_workers: int = 4,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        use_amp: bool = True,
        memory_tracking: bool = True,
        device: str = "cuda",
        max_frames_per_file: Optional[int] = None,
    ):
        """
        初始化训练类
        
        Args:
            tiff_dir: TIFF文件目录
            emitter_dir: 发射体H5文件目录
            output_dir: 输出目录
            patch_size: 图像块大小
            stride: 图像块滑动步长
            batch_size: 批次大小
            num_workers: 数据加载线程数
            learning_rate: 学习率
            num_epochs: 训练轮数
            use_amp: 是否使用自动混合精度
            memory_tracking: 是否跟踪内存使用
            device: 设备（"cuda"或"cpu"）
            max_frames_per_file: 每个文件最大帧数
        """
        self.tiff_dir = tiff_dir
        self.emitter_dir = emitter_dir
        self.output_dir = Path(output_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.use_amp = use_amp
        self.memory_tracking = memory_tracking
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_frames_per_file = max_frames_per_file
        
        # 创建输出目录
        self.model_dir = self.output_dir / "models"
        self.log_dir = self.output_dir / "tensorboard"
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # 初始化数据集和数据加载器
        self._init_data_loaders()
        
        # 初始化模型
        self._init_models()
        
        # 初始化优化器和损失函数
        self._init_training_components()
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # 初始化内存跟踪器
        self.memory_tracker = MemoryTracker(
            log_dir=str(self.output_dir),
            enabled=self.memory_tracking
        )
    
    def _init_data_loaders(self):
        """
        初始化数据集和数据加载器
        """
        # 创建训练集
        self.train_dataset = EmitterDataset(
            tiff_dir=self.tiff_dir,
            emitter_dir=self.emitter_dir,
            patch_size=self.patch_size,
            stride=self.stride,
            frame_window=3,  # 固定为3帧
            transform=None,  # 可以添加数据增强
            max_frames_per_file=self.max_frames_per_file,
            train_val_split=0.8,
            is_train=True
        )
        
        # 创建验证集
        self.val_dataset = EmitterDataset(
            tiff_dir=self.tiff_dir,
            emitter_dir=self.emitter_dir,
            patch_size=self.patch_size,
            stride=self.stride,
            frame_window=3,
            transform=None,
            max_frames_per_file=self.max_frames_per_file,
            train_val_split=0.8,
            is_train=False
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"训练集样本数: {len(self.train_dataset)}")
        print(f"验证集样本数: {len(self.val_dataset)}")
    
    def _init_models(self):
        """
        初始化模型
        """
        # 初始化三个共享参数的第一层UNet
        self.first_level_unet = UNetLevel1(in_channels=1, base_filters=48, num_classes=48).to(self.device)
        
        # 初始化第二层网络
        self.second_level_net = SecondLevelNet(
            in_frames=3,
            base_filters=48,
            height=self.patch_size,
            width=self.patch_size
        ).to(self.device)
        
        # 打印模型参数数量
        first_level_params = sum(p.numel() for p in self.first_level_unet.parameters())
        second_level_params = sum(p.numel() for p in self.second_level_net.parameters())
        total_params = first_level_params + second_level_params
        
        print(f"第一层UNet参数数量: {first_level_params:,}")
        print(f"第二层网络参数数量: {second_level_params:,}")
        print(f"总参数数量: {total_params:,}")
    
    def _init_training_components(self):
        """
        初始化优化器和损失函数
        """
        # 合并所有参数
        params = list(self.first_level_unet.parameters()) + list(self.second_level_net.parameters())
        
        # 初始化优化器
        self.optimizer = optim.Adam(params, lr=self.learning_rate)
        
        # 初始化学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5
        )
        
        # 初始化损失函数
        self.count_loss_fn = CountLoss()
        
        # 初始化AMP梯度缩放器
        self.scaler = GradScaler(enabled=self.use_amp)
    
    def _process_large_image(self, x: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """
        处理大型图像，通过分块处理避免GPU内存溢出
        
        Args:
            x: 输入图像张量，形状为[B, C, H, W]
            model: 要使用的模型
            
        Returns:
            处理后的图像张量
        """
        B, C, H, W = x.shape
        max_size = 1024  # 最大处理尺寸，可以根据GPU内存调整
        
        # 如果图像尺寸小于最大尺寸，直接处理
        if H <= max_size and W <= max_size:
            with autocast(enabled=self.use_amp):
                return model(x)
        
        # 否则，分块处理
        # 计算块大小和重叠区域
        block_size = max_size
        overlap = 32  # 重叠区域大小
        
        # 计算块数
        n_blocks_h = math.ceil((H - overlap) / (block_size - overlap))
        n_blocks_w = math.ceil((W - overlap) / (block_size - overlap))
        
        # 创建输出张量
        out_channels = 48 if isinstance(model, UNetLevel1) else 1
        output = torch.zeros((B, out_channels, H, W), device=x.device)
        
        # 创建权重张量（用于加权平均重叠区域）
        weights = torch.zeros((B, 1, H, W), device=x.device)
        
        # 处理每个块
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                # 计算块的起始和结束位置
                h_start = i * (block_size - overlap)
                w_start = j * (block_size - overlap)
                h_end = min(h_start + block_size, H)
                w_end = min(w_start + block_size, W)
                
                # 调整起始位置，确保块大小不变
                h_start = max(0, h_end - block_size)
                w_start = max(0, w_end - block_size)
                
                # 提取块
                x_block = x[:, :, h_start:h_end, w_start:w_end]
                
                # 处理块
                with autocast(enabled=self.use_amp):
                    y_block = model(x_block)
                
                # 创建权重块（中心权重高，边缘权重低）
                weight_block = torch.ones((1, 1, h_end-h_start, w_end-w_start), device=x.device)
                
                # 如果不是边缘块，应用边缘衰减
                if i > 0:  # 上边缘
                    for k in range(overlap):
                        weight_block[:, :, k, :] *= (k / overlap)
                if i < n_blocks_h - 1:  # 下边缘
                    for k in range(overlap):
                        weight_block[:, :, -(k+1), :] *= (k / overlap)
                if j > 0:  # 左边缘
                    for k in range(overlap):
                        weight_block[:, :, :, k] *= (k / overlap)
                if j < n_blocks_w - 1:  # 右边缘
                    for k in range(overlap):
                        weight_block[:, :, :, -(k+1)] *= (k / overlap)
                
                # 将块添加到输出中
                output[:, :, h_start:h_end, w_start:w_end] += y_block * weight_block
                weights[:, :, h_start:h_end, w_start:w_end] += weight_block
        
        # 加权平均
        output = output / (weights + 1e-8)
        
        return output
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个轮次
        
        Args:
            epoch: 当前轮次
            
        Returns:
            包含训练指标的字典
        """
        self.first_level_unet.train()
        self.second_level_net.train()
        
        epoch_loss = 0.0
        batch_count = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 获取输入数据
            frames = batch["frames"]  # 列表，包含3个形状为[B, 1, H, W]的张量
            counts = batch["count"].to(self.device)  # [B]
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 更新内存跟踪
            if self.memory_tracking:
                self.memory_tracker.update()
            
            # 处理每一帧，获取特征
            features = []
            for frame in frames:
                frame = frame.to(self.device)  # [B, 1, H, W]
                
                # 使用第一层UNet处理帧
                feat = self._process_large_image(frame, self.first_level_unet)  # [B, 48, H, W]
                features.append(feat)
            
            # 将特征连接起来
            # 方法1：沿通道维度连接
            concat_features = torch.cat(features, dim=1)  # [B, 144, H, W]
            
            # 使用第二层网络处理特征
            with autocast(enabled=self.use_amp):
                logits = self.second_level_net(concat_features)  # [B, 1, H, W]
                probs = torch.sigmoid(logits)  # [B, 1, H, W]
                
                # 计算损失
                loss = self.count_loss_fn(probs, counts)
            
            # 反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 更新统计信息
            epoch_loss += loss.item()
            batch_count += 1
            
            # 打印进度
            if batch_idx % 10 == 0 or batch_idx == len(self.train_loader) - 1:
                print(f"Epoch {epoch+1}/{self.num_epochs} | Batch {batch_idx+1}/{len(self.train_loader)} | Loss: {loss.item():.4f}")
        
        # 计算平均损失
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        
        # 计算轮次时间
        epoch_time = time.time() - start_time
        
        return {"loss": avg_loss, "time": epoch_time}
    
    def validate(self) -> Dict[str, float]:
        """
        在验证集上评估模型
        
        Returns:
            包含验证指标的字典
        """
        self.first_level_unet.eval()
        self.second_level_net.eval()
        
        val_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # 获取输入数据
                frames = batch["frames"]
                counts = batch["count"].to(self.device)
                
                # 处理每一帧，获取特征
                features = []
                for frame in frames:
                    frame = frame.to(self.device)
                    
                    # 使用第一层UNet处理帧
                    with autocast(enabled=self.use_amp):
                        feat = self._process_large_image(frame, self.first_level_unet)
                    features.append(feat)
                
                # 将特征连接起来
                concat_features = torch.cat(features, dim=1)  # [B, 144, H, W]
                
                # 使用第二层网络处理特征
                with autocast(enabled=self.use_amp):
                    logits = self.second_level_net(concat_features)
                    probs = torch.sigmoid(logits)
                    
                    # 计算损失
                    loss = self.count_loss_fn(probs, counts)
                
                # 更新统计信息
                val_loss += loss.item()
                batch_count += 1
                
                # 打印进度
                if batch_idx % 20 == 0 or batch_idx == len(self.val_loader) - 1:
                    print(f"Validation | Batch {batch_idx+1}/{len(self.val_loader)} | Loss: {loss.item():.4f}")
        
        # 计算平均损失
        avg_loss = val_loss / batch_count if batch_count > 0 else 0
        
        return {"loss": avg_loss}
    
    def train(self):
        """
        训练模型
        """
        print(f"开始训练，共{self.num_epochs}轮...")
        
        best_val_loss = float("inf")
        
        for epoch in range(self.num_epochs):
            # 训练一个轮次
            train_metrics = self.train_epoch(epoch)
            
            # 在验证集上评估
            val_metrics = self.validate()
            
            # 更新学习率
            self.scheduler.step(val_metrics["loss"])
            
            # 记录到TensorBoard
            self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            self.writer.add_scalar("Time/epoch", train_metrics["time"], epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)
            
            # 打印轮次摘要
            print(f"Epoch {epoch+1}/{self.num_epochs} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Time: {train_metrics['time']:.2f}s")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self._save_checkpoint(
                    epoch=epoch,
                    is_best=True,
                    val_loss=val_metrics["loss"]
                )
                print(f"  保存最佳模型，验证损失: {best_val_loss:.4f}")
            
            # 每10轮保存一次检查点
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(
                    epoch=epoch,
                    is_best=False,
                    val_loss=val_metrics["loss"]
                )
        
        print("训练完成！")
        self.writer.close()
    
    def _save_checkpoint(self, epoch: int, is_best: bool, val_loss: float):
        """
        保存检查点
        
        Args:
            epoch: 当前轮次
            is_best: 是否为最佳模型
            val_loss: 验证损失
        """
        checkpoint = {
            "epoch": epoch,
            "first_level_state_dict": self.first_level_unet.state_dict(),
            "second_level_state_dict": self.second_level_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "val_loss": val_loss
        }
        
        if is_best:
            checkpoint_path = self.model_dir / "best_model.pth"
        else:
            checkpoint_path = self.model_dir / f"checkpoint_epoch_{epoch+1}.pth"
        
        torch.save(checkpoint, checkpoint_path)


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="训练神经网络模型")
    
    # 数据参数
    parser.add_argument("--tiff_dir", type=str, default="/home/guest/Others/DECODE_rewrite/simulated_data_multi_frames/simulated_multi_frames",
                        help="TIFF文件目录")
    parser.add_argument("--emitter_dir", type=str, default="/home/guest/Others/DECODE_rewrite/simulated_data_multi_frames/emitter_sets",
                        help="发射体H5文件目录")
    parser.add_argument("--output_dir", type=str, default="/home/guest/Others/DECODE_rewrite/nn_train",
                        help="输出目录")
    parser.add_argument("--patch_size", type=int, default=600,
                        help="图像块大小")
    parser.add_argument("--stride", type=int, default=300,
                        help="图像块滑动步长")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="每个文件最大帧数，None表示使用所有帧")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载线程数")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="训练轮数")
    
    # 优化参数
    parser.add_argument("--use_amp", action="store_true",
                        help="是否使用自动混合精度")
    parser.add_argument("--memory_tracking", action="store_true",
                        help="是否跟踪内存使用")
    
    args = parser.parse_args()
    
    # 创建并训练模型
    trainer = TrainModel(
        tiff_dir=args.tiff_dir,
        emitter_dir=args.emitter_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        use_amp=args.use_amp,
        memory_tracking=args.memory_tracking,
        device="cuda",
        max_frames_per_file=args.max_frames
    )
    
    trainer.train()


if __name__ == "__main__":
    main()