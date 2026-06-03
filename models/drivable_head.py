"""Drivable Area Segmentation Head.

A lightweight decoder that upsamples the temporally-fused features back to
the original image resolution for per-pixel drivable area classification.

Uses a U-Net style decoder with skip connections from the backbone for
fine spatial detail recovery.

Output classes (BDD100K convention):
  0: Background / non-drivable
  1: Direct drivable area (ego-lane)
  2: Alternative drivable area (adjacent lanes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DecoderBlock(nn.Module):
    """Single decoder block: upsample + conv + batch_norm + relu."""

    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = 0):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        total_in = in_channels + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(total_in, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            # Handle size mismatch due to odd dimensions
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DrivableHead(nn.Module):
    """Drivable area segmentation decoder.

    Args:
        in_channels: channels from temporal fusion module
        num_classes: number of drivable area classes
        decoder_channels: channels for each decoder stage (lowest to highest res)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 3,
        decoder_channels: List[int] = [256, 128, 64],
    ):
        super().__init__()
        self.num_classes = num_classes

        # Decoder stages
        channels = [in_channels] + list(decoder_channels)
        self.blocks = nn.ModuleList()
        for i in range(len(decoder_channels)):
            self.blocks.append(
                DecoderBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    skip_channels=0,  # skip connections can be added from backbone
                )
            )

        # Final classification layer
        self.head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Upsample features to produce per-pixel class logits.

        Args:
            features: [B, C, H_f, W_f] fused features (e.g., 1/16 resolution)
        Returns:
            logits: [B, num_classes, H_f*8, W_f*8] (roughly original resolution)
        """
        x = features
        for block in self.blocks:
            x = block(x)

        logits = self.head(x)  # [B, num_classes, H_out, W_out]
        return logits

    def get_drivable_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to hard class predictions.

        Args:
            logits: [B, num_classes, H, W]
        Returns:
            mask: [B, H, W] class indices
        """
        return torch.argmax(logits, dim=1)
