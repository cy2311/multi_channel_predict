import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import tifffile
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

# 导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuronal_network.first_level_unets import UNetLevel1
from neuronal_network.second_level_network import SecondLevelNet
from neuronal_network.train_network import TrainModel  # 导入训练模型类以复用部分功能


class EmitterPredictor:
    """
    发射体预测器，用于使用训练好的模型进行预测
    """
    def __init__(
        self,
        checkpoint_path: str,
        patch_size: int = 600,
        stride: int = 300,
        prob_threshold: float = 0.7,
        use_amp: bool = True,
        device: str = "cuda"
    ):
        """
        初始化预测器
        
        Args:
            checkpoint_path: 模型检查点路径
            patch_size: 图像块大小
            stride: 图像块滑动步长
            prob_threshold: 概率阈值，高于此值的像素被视为发射体
            use_amp: 是否使用自动混合精度
            device: 设备（"cuda"或"cpu"）
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.patch_size = patch_size
        self.stride = stride
        self.prob_threshold = prob_threshold
        self.use_amp = use_amp
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 初始化模型
        self._init_models()
        
        print(f"模型已加载，使用设备: {self.device}")
    
    def _init_models(self):
        """
        初始化并加载模型
        """
        # 初始化模型
        self.first_level_unet = UNetLevel1(in_channels=1, base_filters=48, num_classes=48).to(self.device)
        self.second_level_net = SecondLevelNet(in_frames=3, base_filters=48).to(self.device)
        
        # 加载检查点
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # 加载模型权重
        self.first_level_unet.load_state_dict(checkpoint["first_level_state_dict"])
        self.second_level_net.load_state_dict(checkpoint["second_level_state_dict"])
        
        # 设置为评估模式
        self.first_level_unet.eval()
        self.second_level_net.eval()
    
    def _process_large_image(self, x: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """
        处理大型图像，通过分块处理避免GPU内存溢出
        
        Args:
            x: 输入图像张量，形状为[B, C, H, W]
            model: 要使用的模型
            
        Returns:
            处理后的图像张量
        """
        # 复用TrainModel中的方法
        trainer = TrainModel(
            tiff_dir="",
            emitter_dir="",
            output_dir="",
            use_amp=self.use_amp,
            device=self.device.type
        )
        trainer.first_level_unet = self.first_level_unet
        trainer.second_level_net = self.second_level_net
        
        return trainer._process_large_image(x, model)
    
    def predict_tiff(self, tiff_path: str, output_dir: str, max_frames: Optional[int] = None) -> Dict:
        """
        预测TIFF文件中的发射体
        
        Args:
            tiff_path: TIFF文件路径
            output_dir: 输出目录
            max_frames: 最大处理帧数，None表示处理所有帧
            
        Returns:
            包含预测结果的字典
        """
        tiff_path = Path(tiff_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 加载TIFF文件
        print(f"加载TIFF文件: {tiff_path}")
        with tifffile.TiffFile(str(tiff_path)) as f:
            n_frames = len(f.pages)
            if max_frames is not None:
                n_frames = min(n_frames, max_frames)
            
            # 获取图像尺寸
            page = f.pages[0]
            height, width = page.shape
            
            print(f"文件包含 {n_frames} 帧，尺寸为 {height}×{width}")
            
            # 创建输出数组
            prob_maps = np.zeros((n_frames - 2, height, width), dtype=np.float32)
            emitter_counts = np.zeros(n_frames - 2, dtype=int)
            
            # 处理每个可能的帧窗口
            for frame_idx in tqdm(range(n_frames - 2), desc="处理帧"):
                # 加载三个连续帧
                frames = []
                for i in range(3):
                    page = f.pages[frame_idx + i]
                    frame = page.asarray().astype(np.float32)
                    
                    # 归一化到[0,1]范围
                    if frame.max() > 0:
                        frame = frame / frame.max()
                    
                    frames.append(frame)
                
                # 转换为张量
                frame_tensors = [torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(self.device) for frame in frames]
                
                # 处理每一帧，获取特征
                features = []
                for frame in frame_tensors:
                    with torch.no_grad():
                        feat = self._process_large_image(frame, self.first_level_unet)
                    features.append(feat)
                
                # 将特征连接起来
                concat_features = torch.cat(features, dim=1)  # [1, 144, H, W]
                
                # 使用第二层网络处理特征
                with torch.no_grad(), autocast(enabled=self.use_amp):
                    logits = self.second_level_net(concat_features)
                    probs = torch.sigmoid(logits)  # [1, 1, H, W]
                
                # 转换为NumPy数组
                prob_map = probs.squeeze().cpu().numpy()  # [H, W]
                
                # 保存概率图
                prob_maps[frame_idx] = prob_map
                
                # 计算发射体数量（高于阈值的像素数）
                emitter_count = np.sum(prob_map > self.prob_threshold)
                emitter_counts[frame_idx] = emitter_count
            
            # 保存结果
            output_name = tiff_path.stem
            np.save(output_dir / f"{output_name}_prob_maps.npy", prob_maps)
            np.save(output_dir / f"{output_name}_emitter_counts.npy", emitter_counts)
            
            # 创建可视化结果
            self._visualize_results(prob_maps, emitter_counts, output_dir, output_name)
            
            return {
                "prob_maps": prob_maps,
                "emitter_counts": emitter_counts
            }
    
    def _visualize_results(self, prob_maps: np.ndarray, emitter_counts: np.ndarray, output_dir: Path, output_name: str):
        """
        可视化预测结果
        
        Args:
            prob_maps: 概率图数组，形状为[n_frames, H, W]
            emitter_counts: 发射体计数数组，形状为[n_frames]
            output_dir: 输出目录
            output_name: 输出文件名前缀
        """
        # 创建可视化目录
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # 保存一些示例帧的概率图
        n_frames = prob_maps.shape[0]
        sample_indices = np.linspace(0, n_frames - 1, 5, dtype=int)
        
        for i, idx in enumerate(sample_indices):
            if idx >= n_frames:
                continue
                
            plt.figure(figsize=(12, 5))
            
            # 绘制概率图
            plt.subplot(1, 2, 1)
            plt.imshow(prob_maps[idx], cmap="hot", vmin=0, vmax=1)
            plt.colorbar(label="Probability")
            plt.title(f"Frame {idx} Probability Map")
            
            # 绘制二值化结果
            plt.subplot(1, 2, 2)
            binary_map = prob_maps[idx] > self.prob_threshold
            plt.imshow(binary_map, cmap="gray")
            plt.title(f"Thresholded Map (>{self.prob_threshold:.1f}), Count: {emitter_counts[idx]}")
            
            plt.tight_layout()
            plt.savefig(vis_dir / f"{output_name}_frame_{idx}.png", dpi=150)
            plt.close()
        
        # 绘制发射体计数随时间变化的曲线
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(n_frames), emitter_counts)
        plt.xlabel("Frame Index")
        plt.ylabel("Emitter Count")
        plt.title(f"Emitter Count Over Time ({output_name})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(vis_dir / f"{output_name}_counts.png", dpi=150)
        plt.close()


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="使用训练好的模型预测发射体")
    
    # 输入输出参数
    parser.add_argument("--tiff_path", type=str, required=True,
                        help="输入TIFF文件路径")
    parser.add_argument("--output_dir", type=str, default="/home/guest/Others/DECODE_rewrite/nn_train/predictions",
                        help="输出目录")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型检查点路径")
    
    # 预测参数
    parser.add_argument("--patch_size", type=int, default=600,
                        help="图像块大小")
    parser.add_argument("--stride", type=int, default=300,
                        help="图像块滑动步长")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="概率阈值，高于此值的像素被视为发射体")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="最大处理帧数，None表示处理所有帧")
    
    # 优化参数
    parser.add_argument("--use_amp", action="store_true",
                        help="是否使用自动混合精度")
    
    args = parser.parse_args()
    
    # 创建预测器并进行预测
    predictor = EmitterPredictor(
        checkpoint_path=args.checkpoint,
        patch_size=args.patch_size,
        stride=args.stride,
        prob_threshold=args.threshold,
        use_amp=args.use_amp
    )
    
    predictor.predict_tiff(
        tiff_path=args.tiff_path,
        output_dir=args.output_dir,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()