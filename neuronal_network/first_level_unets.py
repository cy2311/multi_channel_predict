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
    output_path = output_dir / "features_level1.npy"

    # Load data
    print("Loading TIFF stack…")
    frames_np = load_tiff_stack(tiff_path)
    num_frames, height, width = frames_np.shape
    print(f"Loaded {num_frames} frames of size {height}×{width}.")

    # Allocate output container
    features = np.empty((num_frames, 48, height, width), dtype=np.float32)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate a single shared UNet (parameters will be reused for every frame)
    net = UNetLevel1().to(device)
    net.eval()  # inference mode (no training yet)
    print(f"Shared UNet parameters: {sum(p.numel() for p in net.parameters()):,}")

    # Process frames round-robin
    with torch.no_grad():
        for i in range(num_frames):
            frame = torch.from_numpy(frames_np[i]).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            frame = frame.to(device)

            feat = net(frame)  # (1, 48, H, W)
            features[i] = feat.squeeze(0).cpu().numpy()

            if (i + 1) % 10 == 0 or i == num_frames - 1:
                print(f"Processed {i + 1}/{num_frames} frames…")

    # Save features
    print(f"Saving features to {output_path}…")
    np.save(output_path, features)
    print("Done.")


if __name__ == "__main__":
    main()