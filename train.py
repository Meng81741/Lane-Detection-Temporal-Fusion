#!/usr/bin/env python3
"""Training script for Lane-Detection-Temporal-Fusion.

Usage:
    python train.py --config configs/culane.yaml
    python train.py --config configs/bdd100k.yaml --resume checkpoints/last.pth
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data import CULaneDataset, BDD100KDataset
from models import LaneDrivableDualModel, JointLoss
from utils.config_utils import load_config
from utils.metrics import compute_lane_metrics, compute_drivable_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train Lane-Detection-Temporal-Fusion")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--debug", action="store_true", help="Debug mode (single batch)")
    return parser.parse_args()


def build_dataset(cfg: dict, split: str):
    """Build dataset based on config."""
    data_cfg = cfg["data"]
    name = data_cfg["name"]

    if name == "culane":
        return CULaneDataset(
            data_root=data_cfg["data_root"],
            split=split,
            num_frames=data_cfg.get("num_frames", 5),
            frame_interval=data_cfg.get("frame_interval", 1),
            image_size=tuple(data_cfg["image_size"]),
            augment=(split == "train"),
        )
    elif name == "bdd100k":
        return BDD100KDataset(
            data_root=data_cfg["data_root"],
            split=split,
            num_frames=data_cfg.get("num_frames", 5),
            frame_interval=data_cfg.get("frame_interval", 1),
            image_size=tuple(data_cfg["image_size"]),
            augment=(split == "train"),
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")


def build_model(cfg: dict, device: torch.device) -> LaneDrivableDualModel:
    """Build the dual-head model."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    model = LaneDrivableDualModel(
        backbone_name=model_cfg.get("backbone", "resnet18"),
        pretrained_backbone=model_cfg.get("pretrained", True),
        temporal_cfg=model_cfg.get("temporal", {}),
        lane_cfg=model_cfg.get("lane_head", {}),
        drivable_cfg=model_cfg.get("drivable_head", {}),
        image_size=tuple(data_cfg["image_size"]),
    )
    return model.to(device)


def prepare_targets(batch: dict, predictions: dict, device: torch.device) -> dict:
    """Convert dataset annotations to training targets."""
    targets = {}
    B = batch["frames"].shape[0]

    # Lane classification targets
    lane_labels = batch["lane_labels"]
    if "x_coords" in lane_labels:
        # Create binary targets: anchor has lane if any ground truth lane is near
        x_coords = lane_labels["x_coords"].to(device)  # [B, 4, 18]
        # Simplified: mark as positive if at least one lane has valid x-coords
        # Real implementation would match anchors to GT lanes
        has_lane = (x_coords[:, :, 0] > -1).any(dim=1).float()  # [B]
        # For now: create dummy targets
        num_anchors = predictions["lane_logits"].shape[1]
        lane_cls_targets = torch.zeros(B, num_anchors, device=device)
        # Mark some anchors as positive based on GT (simplified)
        for b in range(B):
            valid_lanes = (x_coords[b, :, 0] > -1).sum().item()
            # Assign positive anchors (spread across anchor range)
            if valid_lanes > 0:
                step = num_anchors // max(valid_lanes, 1)
                for l in range(min(valid_lanes, 4)):
                    anchor_idx = int(step * (l + 0.5)) % num_anchors
                    lane_cls_targets[b, anchor_idx] = 1.0

        targets["lane_cls_targets"] = lane_cls_targets

        # Lane regression targets (simplified — zero for negative anchors)
        num_reg = predictions["lane_reg"].shape[2]
        lane_reg_targets = torch.zeros(B, num_anchors, num_reg, device=device)
        targets["lane_reg_targets"] = lane_reg_targets

    # Drivable mask
    if batch.get("drivable_mask") is not None:
        targets["drivable_mask"] = batch["drivable_mask"].to(device)

    return targets


def train_epoch(
    model: LaneDrivableDualModel,
    dataloader: DataLoader,
    criterion: JointLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    cfg: dict,
    writer: Optional[SummaryWriter] = None,
) -> Dict[str, float]:
    """Run one training epoch."""
    model.train()
    epoch_losses = {}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        frames = batch["frames"].to(device)  # [B, T, 3, H, W]

        optimizer.zero_grad()

        # Forward
        predictions = model(frames)

        # Prepare targets
        targets = prepare_targets(batch, predictions, device)

        # Compute loss
        losses = criterion(predictions, targets)

        # Backward
        losses["total"].backward()

        # Gradient clipping
        grad_clip = cfg["train"].get("gradient_clip", 10.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Track losses
        for k, v in losses.items():
            if k not in epoch_losses:
                epoch_losses[k] = []
            epoch_losses[k].append(v.item())

        # Update progress bar
        pbar.set_postfix({
            "loss": losses["total"].item(),
            "cls": losses.get("lane_cls", torch.tensor(0)).item(),
        })

        # TensorBoard
        if writer is not None and batch_idx % 50 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            for k, v in losses.items():
                writer.add_scalar(f"train/{k}", v.item(), global_step)

        if cfg.get("debug") and batch_idx >= 2:
            break

    # Average losses
    return {k: np.mean(v) for k, v in epoch_losses.items()}


@torch.no_grad()
def validate(
    model: LaneDrivableDualModel,
    dataloader: DataLoader,
    criterion: JointLoss,
    device: torch.device,
    epoch: int,
    cfg: dict,
    writer: Optional[SummaryWriter] = None,
) -> Dict[str, float]:
    """Run validation."""
    model.eval()
    val_losses = {}
    all_lane_preds = []
    all_lane_targets = []
    all_drivable_preds = []
    all_drivable_targets = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    for batch_idx, batch in enumerate(pbar):
        frames = batch["frames"].to(device)

        predictions = model(frames)
        targets = prepare_targets(batch, predictions, device)
        losses = criterion(predictions, targets)

        for k, v in losses.items():
            if k not in val_losses:
                val_losses[k] = []
            val_losses[k].append(v.item())

        # Collect predictions for metric computation
        # (simplified — real impl would decode lanes properly)
        if batch.get("drivable_mask") is not None:
            pred_mask = torch.argmax(predictions["drivable_logits"], dim=1)
            all_drivable_preds.append(pred_mask.cpu().numpy())
            all_drivable_targets.append(batch["drivable_mask"].numpy())

        if cfg.get("debug") and batch_idx >= 2:
            break

    metrics = {f"val_{k}": np.mean(v) for k, v in val_losses.items()}

    # Compute drivable mIoU if data available
    if all_drivable_preds:
        preds_flat = [x for batch_x in all_drivable_preds for x in batch_x]
        targets_flat = [x for batch_x in all_drivable_targets for x in batch_x]
        drivable_metrics = compute_drivable_metrics(preds_flat, targets_flat)
        metrics["val_miou"] = drivable_metrics["miou"]
        metrics["val_pixel_acc"] = drivable_metrics["pixel_accuracy"]

    # TensorBoard
    if writer is not None:
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

    return metrics


def main():
    args = parse_args()

    # Load config
    cfg = load_config(args.config)
    if args.debug:
        cfg["debug"] = True

    # Device
    device = torch.device(args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    # Reproducibility
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create directories
    log_dir = Path(cfg["logging"]["log_dir"])
    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir=str(log_dir)) if cfg["logging"].get("tensorboard", True) else None

    # Datasets
    print("[INFO] Loading datasets...")
    train_dataset = build_dataset(cfg, "train")
    val_dataset = build_dataset(cfg, "val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=True,
    )

    print(f"[INFO] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    print("[INFO] Building model...")
    model = build_model(cfg, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Loss
    loss_cfg = cfg["train"].get("loss", {})
    criterion = JointLoss(
        lane_cls_weight=loss_cfg.get("lane_cls_weight", 2.0),
        lane_reg_weight=loss_cfg.get("lane_reg_weight", 1.0),
        drivable_weight=loss_cfg.get("drivable_weight", 1.0),
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )

    # LR Scheduler
    total_epochs = cfg["train"]["epochs"]
    warmup = cfg["train"].get("warmup_epochs", 3)
    scheduler_type = cfg["train"].get("lr_scheduler", "cosine")

    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup)
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"[INFO] Resumed from epoch {start_epoch}")

    # Training loop
    print(f"[INFO] Starting training for {total_epochs} epochs...")
    for epoch in range(start_epoch, total_epochs):
        # Warmup
        if epoch < warmup:
            lr_scale = (epoch + 1) / max(warmup, 1)
            for param_group in optimizer.param_groups:
                param_group["lr"] = cfg["train"]["learning_rate"] * lr_scale

        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, cfg, writer
        )

        # Step scheduler (after warmup)
        if epoch >= warmup:
            if scheduler_type == "plateau":
                scheduler.step(train_losses.get("total", 0))
            else:
                scheduler.step()

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, epoch, cfg, writer
        )

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch:3d}] "
              f"Train Loss: {train_losses.get('total', 0):.4f} | "
              f"Val Loss: {val_metrics.get('val_total', 0):.4f} | "
              f"mIoU: {val_metrics.get('val_miou', 0):.4f} | "
              f"LR: {current_lr:.2e}")

        # Save checkpoint
        val_loss = val_metrics.get("val_total", float("inf"))
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": cfg,
        }

        # Save periodically
        if (epoch + 1) % cfg["logging"].get("save_interval", 5) == 0 or is_best:
            torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pth")
        torch.save(checkpoint, checkpoint_dir / "last.pth")
        if is_best:
            torch.save(checkpoint, checkpoint_dir / "best.pth")

    print(f"[INFO] Training complete. Best val loss: {best_val_loss:.4f}")
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
