import math
from typing import Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from first_level_unets import ConvBlock, DownBlock, UpBlock

__all__ = [
    "SecondLevelNet",
    "CountLoss",
]


class SecondLevelNet(nn.Module):
    """Second-level network.

    Pipeline:
      1. Concatenate three consecutive frames along **channel** dimension → 144 ch.
      2. 3×Conv (ConvBlock) to squeeze 144 → 48 ch.
      3. UNet with 48 base filters (3 Conv per stage, 3 down/3 up) identical to first level.
      4. 1×1 Conv → 9 ch output (detection probability + localization parameters)
         - Channel 0: Detection probability (sigmoid is **not** applied; leave to loss/inference)
         - Channel 1-3: Offset prediction (dx, dy, dz)
         - Channel 4: Photon count prediction
         - Channel 5-8: Uncertainty prediction (sigma_x, sigma_y, sigma_z, sigma_photons)
    """

    def __init__(
        self,
        in_frames: int = 3,
        base_filters: int = 48,
        height: int = 1200,
        width: int = 1200,
        with_loc_prediction: bool = True,  # 是否预测定位信息
    ) -> None:
        super().__init__()

        in_channels = in_frames * base_filters  # 3 × 48 = 144
        self.with_loc_prediction = with_loc_prediction

        # Step 1-2: compress to 48 features
        self.compress = ConvBlock(in_channels, base_filters)

        # Encoder
        self.down1 = DownBlock(base_filters, base_filters)  # 48 → 48
        self.down2 = DownBlock(base_filters, base_filters * 2)  # 48 → 96
        self.down3 = DownBlock(base_filters * 2, base_filters * 4)  # 96 → 192
        # 移除第四层下采样
        # self.down4 = DownBlock(base_filters * 4, base_filters * 8)  # 192 → 384

        # Bottleneck
        self.bottleneck = ConvBlock(base_filters * 4, base_filters * 8)  # 192 → 384

        # Decoder
        # 移除第四层上采样
        # self.up4 = UpBlock(base_filters * 16, base_filters * 8)  # 768 → 384
        self.up3 = UpBlock(base_filters * 8, base_filters * 4)  # 384 → 192
        self.up2 = UpBlock(base_filters * 4, base_filters * 2)  # 192 → 96
        self.up1 = UpBlock(base_filters * 2, base_filters)  # 96 → 48

        # Final output layers
        if self.with_loc_prediction:
            # 9通道输出：检测概率 + 定位参数
            self.final_conv = nn.Conv2d(base_filters, 9, kernel_size=1)
            
            # 分离的输出头，用于不同的任务
            self.detection_head = nn.Conv2d(base_filters, 1, kernel_size=1)  # 检测概率
            self.offset_head = nn.Conv2d(base_filters, 3, kernel_size=1)    # dx, dy, dz
            self.photon_head = nn.Conv2d(base_filters, 1, kernel_size=1)    # 光子数
            self.uncertainty_head = nn.Conv2d(base_filters, 4, kernel_size=1)  # sigma_x, sigma_y, sigma_z, sigma_photons
            self.background_head = nn.Conv2d(base_filters, 1, kernel_size=1)  # 背景预测
        else:
            # 仅检测概率
            self.final_conv = nn.Conv2d(base_filters, 1, kernel_size=1)

        # Store spatial dims (optional, may be useful elsewhere)
        self.height = height
        self.width = width

    def forward(self, feats: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            feats: Tensor of shape (B, 144, H, W) **or** (B, 3, 48, H, W).
                   If 5-D tensor, it will be reshaped to (B, 144, H, W).
        Returns:
            If with_loc_prediction=False:
                Prob. map logits of shape (B, 1, H, W)
            If with_loc_prediction=True:
                Dictionary containing:
                - 'prob': Detection probability map (B, 1, H, W)
                - 'offset': Offset predictions (B, 3, H, W) for dx, dy, dz
                - 'photon': Photon count predictions (B, 1, H, W)
                - 'uncertainty': Uncertainty predictions (B, 4, H, W)
        """
        # Handle (B, 3, 48, H, W) input
        if feats.dim() == 5:
            B, F, C, H, W = feats.shape
            assert F == 3 and C == 48, "Expecting (B,3,48,H,W) input."
            feats = feats.view(B, F * C, H, W)
        else:
            assert feats.size(1) == 144, "Expecting 144-channel input."

        x = self.compress(feats)  # (B,48,H,W)

        # Encoder
        x, s1 = self.down1(x)
        x, s2 = self.down2(x)
        x, s3 = self.down3(x)
        # 移除第四层下采样
        # x, s4 = self.down4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        # 移除第四层上采样
        # x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        if not self.with_loc_prediction:
            # 仅返回检测概率图
            logits = self.final_conv(x)
            return logits  # sigmoid left to caller
        else:
            # 使用分离的输出头
            prob = self.detection_head(x)  # 检测概率 (B, 1, H, W)
            offset = self.offset_head(x)   # 偏移量 (B, 3, H, W)
            photon = self.photon_head(x)   # 光子数 (B, 1, H, W)
            uncertainty = self.uncertainty_head(x)  # 不确定性 (B, 4, H, W)
            background = self.background_head(x)  # 背景预测 (B, 1, H, W)
            
            # 返回字典形式的结果
            return {
                'prob': prob,
                'offset': offset,
                'photon': photon,
                'uncertainty': uncertainty,
                'background': background
            }


class CountLoss(nn.Module):
    """Count loss described in DECODE paper (Gaussian approx. to Poisson binomial)."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, prob_map: torch.Tensor, true_count: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Compute count loss.

        Args:
            prob_map: Predicted probabilities, shape (B, 1, H, W) after **sigmoid**.
            true_count: Tensor of shape (B,) with integer ground-truth emitter counts.
        """
        B = prob_map.size(0)
        p = prob_map.view(B, -1)
        mu = p.sum(dim=1)  # (B,)
        var = (p * (1.0 - p)).sum(dim=1) + self.eps  # (B,)

        term1 = 0.5 * (true_count - mu) ** 2 / var
        term2 = torch.log(torch.sqrt(var) * math.sqrt(2 * math.pi))
        loss = term1 + term2
        return loss.mean()