"""Backbone feature extractor — ResNet-18/34 with configurable output stride."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


class ResNetBackbone(nn.Module):
    """ResNet backbone returning multi-scale features.

    Args:
        name: 'resnet18' or 'resnet34'
        pretrained: load ImageNet weights
        output_stride: 16 or 32 — controls dilation in layer4
    Returns:
        features at 1/16 resolution (output_stride=16) or 1/32 (output_stride=32)
    """

    def __init__(
        self,
        name: str = "resnet18",
        pretrained: bool = True,
        output_stride: int = 16,
    ):
        super().__init__()

        weights = "IMAGENET1K_V1" if pretrained else None
        if name == "resnet18":
            resnet = models.resnet18(weights=weights)
            self.out_channels = 512
        elif name == "resnet34":
            resnet = models.resnet34(weights=weights)
            self.out_channels = 512
        elif name == "resnet50":
            resnet = models.resnet50(weights=weights)
            self.out_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {name}")

        # Stage 0: conv1 + bn1 + relu + maxpool
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        # Stage 1-4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32 (or 1/16 with dilation)

        # Adjust dilation for output_stride=16
        if output_stride == 16:
            # Replace stride=2 with stride=1 and add dilation=2 in layer4
            for layer in self.layer4.modules():
                if isinstance(layer, nn.Conv2d) and layer.stride == (2, 2):
                    layer.stride = (1, 1)
                    layer.dilation = (2, 2)
                    layer.padding = (2, 2)

        self._output_stride = output_stride

    @property
    def output_stride(self) -> int:
        return self._output_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features.

        Args:
            x: [B, 3, H, W] input image
        Returns:
            features at output_stride resolution
        """
        x = self.stem(x)       # 1/4
        x = self.layer1(x)     # 1/4
        x = self.layer2(x)     # 1/8
        x = self.layer3(x)     # 1/16
        x = self.layer4(x)     # 1/16 (dilated) or 1/32
        return x


def build_backbone(name: str, pretrained: bool = True, output_stride: int = 16) -> ResNetBackbone:
    """Factory function for backbone creation."""
    return ResNetBackbone(name=name, pretrained=pretrained, output_stride=output_stride)
