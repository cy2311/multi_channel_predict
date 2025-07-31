from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tifffile as tiff


class ConvBlock(nn.Module):
    """Three consecutive Conv-BN-ReLU layers with 3×3 kernels."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        layers = []
        for idx in range(3):
            layers.append(
                nn.Conv2d(
                    in_channels if idx == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.block(x)


class DownBlock(nn.Module):
    """ConvBlock followed by 2×2 max-pooling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):  # type: ignore
        x = self.conv(x)
        skip = x  # for UNet skip connection
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    """Transposed-conv upsampling followed by ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):  # type: ignore
        x = self.up(x)
        # Ensure same spatial size (in case of rounding)
        if x.size()[2:] != skip.size()[2:]:
            x = nn.functional.interpolate(x, size=skip.size()[2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNetLevel1(nn.Module):
    """UNet with 48 initial filters and 3 Conv layers per stage."""

    def __init__(self, in_channels: int = 1, base_filters: int = 48, num_classes: int = 48):
        super().__init__()
        # Encoder
        self.down1 = DownBlock(in_channels, base_filters)  # 48
        self.down2 = DownBlock(base_filters, base_filters * 2)  # 96
        self.down3 = DownBlock(base_filters * 2, base_filters * 4)  # 192
        # 移除第四层下采样
        # self.down4 = DownBlock(base_filters * 4, base_filters * 8)  # 384

        # Bottleneck (without pooling)
        self.bottleneck = ConvBlock(base_filters * 4, base_filters * 8)  # 192 -> 384

        # Decoder
        # 移除第四层上采样
        # self.up4 = UpBlock(base_filters * 16, base_filters * 8)  # 768 -> 384
        self.up3 = UpBlock(base_filters * 8, base_filters * 4)   # 384 -> 192
        self.up2 = UpBlock(base_filters * 4, base_filters * 2)   # 192 -> 96
        self.up1 = UpBlock(base_filters * 2, base_filters)       # 96 -> 48

        # Final 1×1 convolution to get desired channels.
        self.final_conv = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):  # type: ignore
        # Encoder
        x, skip1 = self.down1(x)  # 48
        x, skip2 = self.down2(x)  # 96
        x, skip3 = self.down3(x)  # 192
        # 移除第四层下采样
        # x, skip4 = self.down4(x)  # 384

        # Bottleneck
        x = self.bottleneck(x)  # 384

        # Decoder
        # 移除第四层上采样
        # x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        return self.final_conv(x)


class ThreeIndependentUNets(nn.Module):
    """三个独立的UNet，不共享参数，用于处理连续的三帧图像"""
    
    def __init__(self, in_channels: int = 1, base_filters: int = 48, num_classes: int = 48):
        super().__init__()
        # 创建三个独立的UNet实例
        self.unet1 = UNetLevel1(in_channels, base_filters, num_classes)
        self.unet2 = UNetLevel1(in_channels, base_filters, num_classes)
        self.unet3 = UNetLevel1(in_channels, base_filters, num_classes)
        
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """处理三个连续帧
        
        Args:
            frames: 形状为 (B, 3, H, W) 的张量，包含三个连续帧
            
        Returns:
            features: 形状为 (B, 3, 48, H, W) 的特征张量
        """
        if frames.dim() == 4 and frames.size(1) == 3:
            # 输入是 (B, 3, H, W)
            frame1 = frames[:, 0:1, :, :]  # (B, 1, H, W)
            frame2 = frames[:, 1:2, :, :]  # (B, 1, H, W)
            frame3 = frames[:, 2:3, :, :]  # (B, 1, H, W)
        else:
            raise ValueError(f"Expected input shape (B, 3, H, W), got {frames.shape}")
            
        # 使用三个独立的UNet处理每一帧
        feat1 = self.unet1(frame1)  # (B, 48, H, W)
        feat2 = self.unet2(frame2)  # (B, 48, H, W)
        feat3 = self.unet3(frame3)  # (B, 48, H, W)
        
        # 堆叠特征 (B, 3, 48, H, W)
        features = torch.stack([feat1, feat2, feat3], dim=1)
        
        return features
    
    def get_parameter_count(self):
        """获取每个UNet的参数数量"""
        unet1_params = sum(p.numel() for p in self.unet1.parameters())
        unet2_params = sum(p.numel() for p in self.unet2.parameters())
        unet3_params = sum(p.numel() for p in self.unet3.parameters())
        total_params = unet1_params + unet2_params + unet3_params
        
        return {
            'unet1': unet1_params,
            'unet2': unet2_params, 
            'unet3': unet3_params,
            'total': total_params
        }


def load_tiff_stack(path: Path) -> np.ndarray:
    """Load OME-TIFF into a NumPy array of shape (num_frames, H, W) using memory mapping.
    
    Memory mapping allows accessing the data without loading the entire file into RAM,
    which is especially useful for large TIFF stacks.
    """

    if not path.exists():
        raise FileNotFoundError(f"TIFF file not found: {path}")
    
    # 使用内存映射加载TIFF堆栈，避免将整个数据加载到RAM中
    with tiff.TiffFile(str(path)) as f:
        data = f.asarray(out="memmap")  # returns (frames, Y, X) as memory-mapped array
    
    return data.astype(np.float32)


def main():
    # Paths
    root = Path(__file__).resolve().parent.parent  # project root assumed one level up
    tiff_path = root / "simulated_data_multi_frames/frames_200f_1200px_camera_photon.ome.tiff"
    output_dir = root / "nn_train"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "features_level1_independent.npy"

    # Load data
    print("Loading TIFF stack…")
    frames_np = load_tiff_stack(tiff_path)
    num_frames, height, width = frames_np.shape
    print(f"Loaded {num_frames} frames of size {height}×{width}.")

    # 确保帧数是3的倍数（用于处理连续三帧）
    if num_frames % 3 != 0:
        # 截断到最接近的3的倍数
        num_frames = (num_frames // 3) * 3
        frames_np = frames_np[:num_frames]
        print(f"Truncated to {num_frames} frames for triplet processing.")

    # 重新组织数据为三帧组
    num_triplets = num_frames // 3
    triplet_frames = frames_np.reshape(num_triplets, 3, height, width)
    
    # Allocate output container for features
    features = np.empty((num_triplets, 3, 48, height, width), dtype=np.float32)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 实例化三个独立的UNet
    net = ThreeIndependentUNets().to(device)
    net.eval()  # inference mode
    
    # 打印参数信息
    param_info = net.get_parameter_count()
    print(f"UNet1 parameters: {param_info['unet1']:,}")
    print(f"UNet2 parameters: {param_info['unet2']:,}")
    print(f"UNet3 parameters: {param_info['unet3']:,}")
    print(f"Total parameters: {param_info['total']:,}")

    # Process triplet frames
    with torch.no_grad():
        for i in range(num_triplets):
            # 获取三帧数据 (3, H, W)
            triplet = torch.from_numpy(triplet_frames[i]).unsqueeze(0)  # (1, 3, H, W)
            triplet = triplet.to(device)

            # 通过三个独立的UNet处理
            feat = net(triplet)  # (1, 3, 48, H, W)
            features[i] = feat.squeeze(0).cpu().numpy()  # (3, 48, H, W)

            if (i + 1) % 10 == 0 or i == num_triplets - 1:
                print(f"Processed {i + 1}/{num_triplets} triplets…")

    # Save features
    print(f"Saving features to {output_path}…")
    np.save(output_path, features)
    print("Done.")


if __name__ == "__main__":
    main()