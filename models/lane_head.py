"""Lane Head — Line-CNN style ray-anchor based lane detection.

Inspired by:
  "End-to-end Lane Detection through Differentiable Least-Squares Fitting"
  (Line-CNN, Li et al. 2020)

The key idea:
  1. Define a set of ray anchors emanating from the image bottom at various angles.
  2. For each anchor, predict:
     - Classification: does this ray contain a lane? (binary)
     - Regression: lane line parameters relative to the anchor
       (x_offset, y_offset, length, angle_delta)

3. At inference, select top-K anchors by confidence and decode their regressed
   parameters into lane line points.

Unlike the original Line-CNN which uses a line proposal + verification network,
this implementation uses a single-stage anchor-based design with focal loss
for handling the extreme class imbalance (most rays are background).
"""

import math
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LaneHead(nn.Module):
    """Line-CNN style lane detection head.

    Args:
        in_channels: input feature channels from temporal fusion module
        num_anchors: number of ray anchors (angular discretization)
        anchor_angle_range: (min_angle, max_angle) in degrees
        num_reg_params: number of regression parameters per lane
            (default 4: x_offset, y_offset, length, angle_delta)
        hidden_channels: channels in classification/regression branches
        feature_size: (H_f, W_f) of the input feature map (for anchor generation)
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int = 72,
        anchor_angle_range: Tuple[float, float] = (-72.0, 72.0),
        num_reg_params: int = 4,
        hidden_channels: int = 64,
        feature_size: Tuple[int, int] = (18, 50),  # H/16, W/16 for 288×800
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_anchors = num_anchors
        self.num_reg_params = num_reg_params
        self.feature_size = feature_size

        # Generate ray anchor angles
        angles = torch.linspace(
            anchor_angle_range[0], anchor_angle_range[1], num_anchors
        )  # [num_anchors] in degrees
        self.register_buffer("anchor_angles_deg", angles)
        self.register_buffer(
            "anchor_angles_rad", angles * math.pi / 180.0
        )

        # Shared feature refinement
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Global context: aggregate spatial info for anchor classification
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classification branch: per-anchor binary classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * feature_size[0] * feature_size[1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_anchors),  # [num_anchors] logits
        )

        # Regression branch: per-anchor regression parameters
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * feature_size[0] * feature_size[1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_anchors * num_reg_params),
        )

        # Per-anchor feature refinement (lightweight)
        # Processes each anchor's ROI from the feature map
        self.anchor_specific = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> dict:
        """Predict lane lines from fused features.

        Args:
            features: [B, C, H, W] fused feature map
        Returns:
            dict with:
              'lane_logits': [B, num_anchors] classification logits
              'lane_reg': [B, num_anchors, num_reg_params] regression offsets
              'anchor_angles': [num_anchors] anchor angle references
        """
        B = features.shape[0]

        # Shared feature processing
        x = self.shared_conv(features)       # [B, hidden, H, W]
        x = self.anchor_specific(x)          # [B, hidden, H, W]

        # Classification
        lane_logits = self.classifier(x)     # [B, num_anchors]

        # Regression
        reg_flat = self.regressor(x)         # [B, num_anchors * num_reg_params]
        lane_reg = reg_flat.view(B, self.num_anchors, self.num_reg_params)

        return {
            "lane_logits": lane_logits,
            "lane_reg": lane_reg,
            "anchor_angles": self.anchor_angles_rad,
        }

    def decode_lanes(
        self,
        predictions: dict,
        image_size: Tuple[int, int],
        score_threshold: float = 0.5,
        top_k: int = 4,
    ) -> List[List[torch.Tensor]]:
        """Decode predictions into lane line points.

        Args:
            predictions: output dict from forward()
            image_size: (H, W) of the original image
            score_threshold: minimum classification score
            top_k: maximum number of lanes per image
        Returns:
            List (batch) of List (per-lane) of [N, 2] point tensors
        """
        B = predictions["lane_logits"].shape[0]
        all_lanes = []

        H, W = image_size
        # Y positions for sampled points (bottom to top)
        y_samples = torch.linspace(H - 1, 0, steps=20, device=predictions["lane_logits"].device)

        for b in range(B):
            scores = torch.sigmoid(predictions["lane_logits"][b])  # [num_anchors]
            reg = predictions["lane_reg"][b]                      # [num_anchors, 4]
            angles = predictions["anchor_angles"]                  # [num_anchors]

            # Select top anchors
            valid = scores > score_threshold
            if valid.sum() == 0:
                all_lanes.append([])
                continue

            valid_scores = scores[valid]
            valid_reg = reg[valid]
            valid_angles = angles[valid]

            # Sort by score, keep top_k
            sorted_idx = torch.argsort(valid_scores, descending=True)[:top_k]
            top_reg = valid_reg[sorted_idx]        # [K, 4]
            top_angles = valid_angles[sorted_idx]   # [K]

            # Decode regression parameters
            # reg = [x_offset, y_offset, length, angle_delta]
            K = top_reg.shape[0]
            image_lanes = []

            for k in range(K):
                x_off = top_reg[k, 0]   # offset from image center x
                y_off = top_reg[k, 1]   # offset from image bottom y
                length = top_reg[k, 2]  # lane length (fraction of image height)
                a_delta = top_reg[k, 3] # angle delta from anchor

                # Actual angle
                angle = top_angles[k] + a_delta * 0.1  # scaled delta

                # Starting point (bottom of image + offset)
                x_start = W / 2.0 + x_off * W  # x_offset scaled to image width
                y_start = H - 1.0 - y_off * H * 0.1  # near bottom

                # Length
                lane_length = torch.sigmoid(length) * H  # 0..H

                # Generate points along the ray
                # Parametric: x = x_start + t * cos(angle), y = y_start - t * sin(angle)
                t_values = torch.linspace(0, lane_length, steps=20, device=angle.device)
                lane_x = x_start + t_values * torch.cos(angle)
                lane_y = y_start - t_values * torch.sin(angle)

                # Clip to image bounds
                lane_x = torch.clamp(lane_x, 0, W - 1)
                lane_y = torch.clamp(lane_y, 0, H - 1)

                points = torch.stack([lane_x, lane_y], dim=1)  # [20, 2]
                image_lanes.append(points)

            all_lanes.append(image_lanes)

        return all_lanes
