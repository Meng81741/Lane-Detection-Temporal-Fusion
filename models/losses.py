"""Joint loss functions for lane detection + drivable area segmentation.

Loss composition:
  L_total = λ_cls * L_lane_cls + λ_reg * L_lane_reg + λ_driv * L_drivable

Where:
  L_lane_cls:  Focal Loss for anchor classification (addresses extreme imbalance)
  L_lane_reg:  Smooth L1 Loss for lane parameter regression
  L_drivable:  CrossEntropy + Dice Loss for drivable area segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss for binary classification with class imbalance.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha: class weight for positive samples
        gamma: focusing parameter (higher = more focus on hard examples)
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: [N, C] raw logits
            targets: [N] binary labels (0 or 1), or [N, C] one-hot
        """
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")

        p_t = torch.exp(-ce_loss)  # p_t = p if y=1 else 1-p
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation — complements CrossEntropy for boundary quality.

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute multi-class Dice loss.

        Args:
            logits: [B, C, H, W] class logits
            targets: [B, H, W] class indices
        """
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets.clamp(0, C - 1), num_classes=C)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()        # [B, C, H, W]

        # Mask out ignore_index
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).unsqueeze(1).float()  # [B, 1, H, W]
            probs = probs * mask
            targets_one_hot = targets_one_hot * mask

        # Dice per class
        intersection = (probs * targets_one_hot).sum(dim=[2, 3])  # [B, C]
        union = probs.sum(dim=[2, 3]) + targets_one_hot.sum(dim=[2, 3])  # [B, C]

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [B, C]
        dice_loss = 1.0 - dice.mean()

        return dice_loss


class JointLoss(nn.Module):
    """Joint loss: Lane detection + Drivable area segmentation.

    L_total = λ_cls * FocalLoss(lane_logits, lane_targets)
            + λ_reg * SmoothL1(lane_reg, lane_reg_targets)
            + λ_driv * (CE(mask_logits, drivable_mask) + Dice(mask_logits, drivable_mask))
    """

    def __init__(
        self,
        lane_cls_weight: float = 2.0,
        lane_reg_weight: float = 1.0,
        drivable_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.lane_cls_weight = lane_cls_weight
        self.lane_reg_weight = lane_reg_weight
        self.drivable_weight = drivable_weight

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.smooth_l1 = nn.SmoothL1Loss(reduction="mean")
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.dice_loss = DiceLoss(ignore_index=-100)

    def forward(
        self,
        predictions: dict,
        targets: dict,
    ) -> dict:
        """Compute all losses.

        Args:
            predictions: dict with keys:
                'lane_logits': [B, num_anchors] classification logits
                'lane_reg': [B, num_anchors, 4] regression parameters
                'drivable_logits': [B, num_classes, H, W] (optional)
            targets: dict with keys:
                'lane_cls_targets': [B, num_anchors] binary labels
                'lane_reg_targets': [B, num_anchors, 4] regression targets
                'drivable_mask': [B, H_d, W_d] class indices (optional)
        Returns:
            dict of losses and total
        """
        losses = {}
        total = 0.0

        # Lane classification loss
        if "lane_logits" in predictions and "lane_cls_targets" in targets:
            lane_cls_loss = self.focal_loss(
                predictions["lane_logits"], targets["lane_cls_targets"]
            )
            losses["lane_cls"] = lane_cls_loss * self.lane_cls_weight
            total = total + losses["lane_cls"]

        # Lane regression loss
        if "lane_reg" in predictions and "lane_reg_targets" in targets:
            # Only compute regression loss for positive anchors
            pos_mask = targets["lane_cls_targets"] > 0  # [B, num_anchors]
            if pos_mask.sum() > 0:
                pred_reg = predictions["lane_reg"][pos_mask]        # [N_pos, 4]
                tgt_reg = targets["lane_reg_targets"][pos_mask]     # [N_pos, 4]
                lane_reg_loss = self.smooth_l1(pred_reg, tgt_reg)
                losses["lane_reg"] = lane_reg_loss * self.lane_reg_weight
                total = total + losses["lane_reg"]
            else:
                losses["lane_reg"] = torch.tensor(0.0, device=predictions["lane_reg"].device)

        # Drivable area loss (CE + Dice)
        if "drivable_logits" in predictions and "drivable_mask" in targets:
            mask_target = targets["drivable_mask"]
            if mask_target is not None:
                drivable_logits = predictions["drivable_logits"]

                # Resize if needed
                if drivable_logits.shape[2:] != mask_target.shape[1:]:
                    drivable_logits = F.interpolate(
                        drivable_logits,
                        size=mask_target.shape[1:],
                        mode="bilinear",
                        align_corners=True,
                    )

                ce = self.ce_loss(drivable_logits, mask_target)
                dice = self.dice_loss(drivable_logits, mask_target)
                losses["drivable_ce"] = ce
                losses["drivable_dice"] = dice
                drivable_loss = (ce + dice) * self.drivable_weight
                losses["drivable"] = drivable_loss
                total = total + drivable_loss

        losses["total"] = total
        return losses
