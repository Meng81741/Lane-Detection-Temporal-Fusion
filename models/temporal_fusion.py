"""Temporal Fusion Module — ConvGRU + Lightweight Temporal Attention.

Core innovation: 将遥感变化检测中的多周期特征自适应方法迁移至车载视频流。

The module takes T per-frame feature maps and produces a single temporally-fused
feature map via:

1. ConvGRU: Convolutional Gated Recurrent Unit that maintains a hidden state
   across consecutive frames, capturing temporal dynamics of lane markings
   and road boundaries.

2. Lightweight Temporal Attention: Channel-wise Squeeze-and-Excitation (SE)
   mechanism applied to the concatenated multi-frame features, adaptively
   re-weighting the contribution of each time step.  This helps the model
   focus on high-quality frames and suppress noisy ones (e.g., glare, rain).

Architecture:
    Input: [B, T, C, H, W]
    │
    ├──► ConvGRU: recurrent processing frame-by-frame
    │    hidden state h_t carries temporal context
    │    └──► output: [B, T, C, H, W]
    │
    ├──► Temporal Attention (SE):
    │    Global AvgPool → FC → ReLU → FC → Sigmoid
    │    └──► attention weights: [B, T, 1, 1, 1]
    │
    └──► Weighted sum + final conv → [B, C, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ConvGRUCell(nn.Module):
    """Convolutional GRU Cell.

    Replaces fully-connected operations in standard GRU with convolutions,
    preserving spatial structure across timesteps.

    Equations:
        z_t = σ(W_z * [h_{t-1}, x_t] + b_z)
        r_t = σ(W_r * [h_{t-1}, x_t] + b_r)
        h̃_t = tanh(W_h * [r_t ⊙ h_{t-1}, x_t] + b_h)
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
    """

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # Combined convolution for update gate, reset gate, and candidate
        # input: [x_t, h_{t-1}] concatenated along channel dim
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            3 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Single step of ConvGRU.

        Args:
            x: [B, C_in, H, W] input feature at timestep t
            h: [B, C_h, H, W] hidden state from timestep t-1 (None → zeros)
        Returns:
            h_new: [B, C_h, H, W] updated hidden state
        """
        B, _, H, W = x.shape

        if h is None:
            h = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)

        # Concatenate input and previous hidden state along channel dim
        combined = torch.cat([x, h], dim=1)  # [B, C_in + C_h, H, W]

        # Single convolution produces gates + candidate
        gates = self.conv(combined)  # [B, 3*C_h, H, W]
        z_gate, r_gate, h_candidate = torch.chunk(gates, 3, dim=1)

        z = torch.sigmoid(z_gate)          # update gate
        r = torch.sigmoid(r_gate)          # reset gate
        h_tilde = torch.tanh(h_candidate)  # candidate activation

        # h_t = (1 - z) * h_{t-1} + z * h̃_t
        h_new = (1 - z) * h + z * h_tilde

        return h_new


class TemporalAttention(nn.Module):
    """Lightweight Temporal Channel Attention (SE-style).

    Applies Squeeze-and-Excitation across the time dimension to adaptively
    re-weight each frame's contribution.  Critical for handling:
      - Night scenes (some frames may be too dark)
      - Rain/glare (transient occlusions)
      - Rapid lighting changes (tunnel entry/exit)

    Args:
        channels: feature channels per frame
        num_frames: T — number of timesteps
        reduction: SE reduction ratio
    """

    def __init__(self, channels: int, num_frames: int, reduction: int = 4):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        reduced = max(1, channels // reduction)

        # Global spatial pooling per-frame → channel descriptor
        # We pool over H, W and process each frame independently

        # SE pathway: compress channels → non-linearity → expand → sigmoid
        self.fc1 = nn.Linear(channels, reduced, bias=False)
        self.fc2 = nn.Linear(reduced, channels, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply temporal attention.

        Args:
            features: [B, T, C, H, W] stacked frame features
        Returns:
            attended: [B, T, C, H, W] re-weighted features
        """
        B, T, C, H, W = features.shape

        # Squeeze: Global Average Pool over spatial dims → [B, T, C]
        descriptors = features.mean(dim=[3, 4])  # [B, T, C]

        # Excitation: MLP over channels (shared across timesteps)
        weights = self.fc1(descriptors)           # [B, T, C//r]
        weights = F.relu(weights)
        weights = self.fc2(weights)               # [B, T, C]
        weights = torch.sigmoid(weights)          # [B, T, C]

        # Reshape for broadcasting
        weights = weights.view(B, T, C, 1, 1)     # [B, T, C, 1, 1]

        # Apply attention
        return features * weights


class TemporalFusionModule(nn.Module):
    """Full temporal fusion: ConvGRU + Temporal Attention + Fusion Conv.

    Pipeline:
        features [B, T, C, H, W]
          │
          ├─► ConvGRU (recurrent) → gru_out [B, T, C_h, H, W]
          │
          ├─► Project to C_out + Temporal Attention
          │   → attended [B, T, C_out, H, W]
          │
          └─► Weighted temporal fusion + final conv
              → output [B, C_out, H, W]

    Args:
        in_channels: input feature channels from backbone
        hidden_channels: ConvGRU hidden state channels
        out_channels: output feature channels
        num_frames: T — temporal window size
        use_attention: whether to apply temporal attention
        attention_reduction: SE reduction ratio
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 256,
        num_frames: int = 5,
        use_attention: bool = True,
        attention_reduction: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_frames = num_frames
        self.use_attention = use_attention

        # Input projection: backbone features → ConvGRU input
        self.input_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        # ConvGRU cell
        self.gru_cell = ConvGRUCell(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=3,
        )

        # Temporal attention (applied after GRU)
        if use_attention:
            self.temporal_attention = TemporalAttention(
                channels=hidden_channels,
                num_frames=num_frames,
                reduction=attention_reduction,
            )

        # Output projection & fusion
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, features: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fuse T frames of features into a single temporal representation.

        Args:
            features: [B, T, C_in, H, W] backbone features per frame
            hidden: optional previous hidden state [B, C_h, H, W]
        Returns:
            fused: [B, C_out, H, W] temporally-fused feature map
            hidden: final hidden state (for continued streaming inference)
        """
        B, T, C_in, H, W = features.shape

        # Process each frame through ConvGRU
        gru_outputs = []
        h = hidden
        for t in range(T):
            x_t = features[:, t, :, :, :]          # [B, C_in, H, W]
            x_t = self.input_proj(x_t)              # [B, C_h, H, W]
            h = self.gru_cell(x_t, h)               # [B, C_h, H, W]
            gru_outputs.append(h)

        # Stack GRU outputs: [B, T, C_h, H, W]
        gru_stack = torch.stack(gru_outputs, dim=1)

        # Apply temporal attention
        if self.use_attention:
            gru_stack = self.temporal_attention(gru_stack)

        # Fuse across time: take the last timestep (enriched by GRU)
        # The hidden state already aggregates temporal information;
        # we use the final output + a learned fusion
        fused = gru_stack[:, -1, :, :, :]  # [B, C_h, H, W]

        # Alternative fusion: weighted sum of all timesteps
        if self.use_attention:
            # Softmax over time dimension for normalized weights
            temporal_weights = F.softmax(
                gru_stack.mean(dim=[2, 3, 4]).view(B, T), dim=1
            )  # [B, T]
            temporal_weights = temporal_weights.view(B, T, 1, 1, 1)
            fused_weighted = (gru_stack * temporal_weights).sum(dim=1)  # [B, C_h, H, W]
            # Blend last-timestep + weighted sum
            fused = fused + fused_weighted

        # Output projection
        fused = self.output_proj(fused)  # [B, C_out, H, W]

        return fused, h
