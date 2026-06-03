"""Lane-Drivable Dual-Head Model — full architecture assembly.

Pipelines:
  ┌─────────────────────────────────────────────────┐
  │  Input: [B, T, 3, H, W]  (T consecutive frames) │
  └──────────────────┬──────────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────────┐
  │  Shared Backbone (ResNet-18)                     │
  │  Per-frame feature extraction                    │
  │  Output: [B, T, C_back, H_f, W_f]                │
  └──────────────────┬──────────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────────┐
  │  Temporal Fusion Module                          │
  │  ConvGRU + Lightweight Temporal Attention         │
  │  Output: [B, C_fused, H_f, W_f]                   │
  └──────────────────┬──────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
  ┌───────────────┐   ┌────────────────┐
  │  Lane Head     │   │  Drivable Head │
  │  (Line-CNN)    │   │  (Seg Decoder) │
  │               │   │               │
  │ Output:       │   │ Output:       │
  │  lane_logits  │   │  drivable_logits│
  │  lane_reg     │   │  [B, 3, H, W] │
  └───────────────┘   └────────────────┘
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List

from .backbone import build_backbone
from .temporal_fusion import TemporalFusionModule
from .lane_head import LaneHead
from .drivable_head import DrivableHead


class LaneDrivableDualModel(nn.Module):
    """Full dual-head model for joint lane + drivable area detection.

    Args:
        backbone_name: 'resnet18' | 'resnet34'
        pretrained_backbone: whether to use ImageNet pretrained weights
        temporal_cfg: dict with temporal fusion hyperparams
        lane_cfg: dict with lane head hyperparams
        drivable_cfg: dict with drivable head hyperparams
        image_size: (H, W) input image size
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained_backbone: bool = True,
        temporal_cfg: Optional[dict] = None,
        lane_cfg: Optional[dict] = None,
        drivable_cfg: Optional[dict] = None,
        image_size: Tuple[int, int] = (288, 800),
    ):
        super().__init__()
        self.image_size = image_size

        # Default configs
        if temporal_cfg is None:
            temporal_cfg = {}
        if lane_cfg is None:
            lane_cfg = {}
        if drivable_cfg is None:
            drivable_cfg = {}

        # ---- Backbone ----
        self.backbone = build_backbone(
            name=backbone_name,
            pretrained=pretrained_backbone,
            output_stride=16,
        )
        backbone_channels = self.backbone.out_channels

        # ---- Temporal Fusion ----
        num_frames = temporal_cfg.get("num_frames", 5)
        hidden_channels = temporal_cfg.get("hidden_channels", 128)
        use_attention = temporal_cfg.get("use_attention", True)
        attention_reduction = temporal_cfg.get("attention_reduction", 4)
        fused_channels = hidden_channels * 2  # output channels after fusion

        self.temporal_fusion = TemporalFusionModule(
            in_channels=backbone_channels,
            hidden_channels=hidden_channels,
            out_channels=fused_channels,
            num_frames=num_frames,
            use_attention=use_attention,
            attention_reduction=attention_reduction,
        )
        self.num_frames = num_frames

        # ---- Calculate feature map size ----
        # With output_stride=16: H_f = ceil(H/16), W_f = ceil(W/16)
        H, W = image_size
        H_f = (H + 15) // 16
        W_f = (W + 15) // 16
        self.feature_size = (H_f, W_f)

        # ---- Lane Head ----
        self.lane_head = LaneHead(
            in_channels=fused_channels,
            num_anchors=lane_cfg.get("num_anchors", 72),
            anchor_angle_range=lane_cfg.get("anchor_angles", (-72, 72)),
            num_reg_params=lane_cfg.get("num_regression_parameters", 4),
            hidden_channels=lane_cfg.get("classifier_hidden", 64),
            feature_size=(H_f, W_f),
        )

        # ---- Drivable Head ----
        self.drivable_head = DrivableHead(
            in_channels=fused_channels,
            num_classes=drivable_cfg.get("num_classes", 3),
            decoder_channels=drivable_cfg.get("decoder_channels", [256, 128, 64]),
        )

    def forward(
        self,
        frames: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> dict:
        """Forward pass.

        Args:
            frames: [B, T, 3, H, W] input frame sequence
            hidden: optional ConvGRU hidden state from previous call
            return_hidden: whether to return the final hidden state
        Returns:
            dict with:
              'lane_logits': [B, num_anchors]
              'lane_reg': [B, num_anchors, 4]
              'drivable_logits': [B, num_classes, H_out, W_out]
              'hidden': [B, C_h, H_f, W_f] (if return_hidden=True)
        """
        B, T, C, H, W = frames.shape

        # ---- Per-frame feature extraction ----
        # Reshape to process all frames through backbone
        frames_flat = frames.view(B * T, C, H, W)  # [B*T, 3, H, W]
        features_flat = self.backbone(frames_flat)   # [B*T, C_back, H_f, W_f]
        _, C_back, H_f, W_f = features_flat.shape
        features = features_flat.view(B, T, C_back, H_f, W_f)  # [B, T, C_back, H_f, W_f]

        # ---- Temporal Fusion ----
        fused, new_hidden = self.temporal_fusion(features, hidden)
        # fused: [B, C_fused, H_f, W_f]

        # ---- Dual Head Output ----
        lane_out = self.lane_head(fused)            # lane_logits, lane_reg
        drivable_logits = self.drivable_head(fused)  # [B, num_classes, H_out, W_out]

        output = {
            "lane_logits": lane_out["lane_logits"],
            "lane_reg": lane_out["lane_reg"],
            "anchor_angles": lane_out["anchor_angles"],
            "drivable_logits": drivable_logits,
        }

        if return_hidden:
            output["hidden"] = new_hidden

        return output

    def predict(
        self,
        frames: torch.Tensor,
        lane_score_threshold: float = 0.5,
        top_k_lanes: int = 4,
    ) -> dict:
        """Inference with decoded lane points and drivable mask.

        Args:
            frames: [B, T, 3, H, W]
            lane_score_threshold: min score for lane anchors
            top_k_lanes: max lanes per image
        Returns:
            dict with:
              'lanes': List[List[Tensor]] decoded lane points
              'drivable_mask': [B, H_out, W_out] class predictions
              'drivable_logits': [B, num_classes, H_out, W_out]
        """
        with torch.no_grad():
            output = self.forward(frames)

        # Decode lanes
        lanes = self.lane_head.decode_lanes(
            output,
            image_size=self.image_size,
            score_threshold=lane_score_threshold,
            top_k=top_k_lanes,
        )

        # Drivable mask
        drivable_mask = self.drivable_head.get_drivable_mask(output["drivable_logits"])

        return {
            "lanes": lanes,
            "drivable_mask": drivable_mask,
            "drivable_logits": output["drivable_logits"],
            "lane_logits": output["lane_logits"],
            "lane_reg": output["lane_reg"],
        }

    def streaming_step(
        self,
        frame: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        frame_buffer: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[dict, torch.Tensor, List[torch.Tensor]]:
        """Streaming inference: process one frame at a time.

        Maintains a rolling buffer of recent frames. When the buffer is full,
        the model processes the entire sequence and returns predictions.

        Args:
            frame: [1, 3, H, W] single frame
            hidden: previous ConvGRU hidden state
            frame_buffer: list of previous frames
        Returns:
            (predictions, hidden, updated_buffer)
        """
        if frame_buffer is None:
            frame_buffer = []

        frame_buffer.append(frame.squeeze(0))  # [3, H, W]
        if len(frame_buffer) > self.num_frames:
            frame_buffer = frame_buffer[-self.num_frames:]

        # Pad if buffer not full
        if len(frame_buffer) < self.num_frames:
            # Duplicate first frame
            pad = [frame_buffer[0]] * (self.num_frames - len(frame_buffer))
            frames = pad + frame_buffer
        else:
            frames = frame_buffer

        frames_tensor = torch.stack(frames, dim=0).unsqueeze(0)  # [1, T, 3, H, W]
        output = self.forward(frames_tensor, hidden=hidden, return_hidden=True)
        predictions = self.predict(frames_tensor)

        return predictions, output.get("hidden"), frame_buffer
