#!/usr/bin/env python3
"""Evaluation / testing script for Lane-Detection-Temporal-Fusion.

Usage:
    python test.py --config configs/culane.yaml --checkpoint checkpoints/best.pth
    python test.py --config configs/bdd100k.yaml --checkpoint checkpoints/best.pth --save-predictions
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data import CULaneDataset, BDD100KDataset
from models import LaneDrivableDualModel
from utils.config_utils import load_config
from utils.metrics import compute_lane_metrics, compute_drivable_metrics
from utils.visualization import visualize_predictions, create_comparison_panel


def parse_args():
    parser = argparse.ArgumentParser(description="Test Lane-Detection-Temporal-Fusion")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-predictions", action="store_true", help="Save prediction visualizations")
    parser.add_argument("--output-dir", type=str, default="./outputs/")
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: LaneDrivableDualModel,
    dataloader: DataLoader,
    device: torch.device,
    cfg: dict,
    save_predictions: bool = False,
    output_dir: Path = None,
) -> Dict[str, float]:
    """Run full evaluation."""
    model.eval()

    all_lane_preds = []
    all_lane_targets = []
    all_drivable_preds = []
    all_drivable_targets = []
    sample_count = 0

    image_size = tuple(cfg["data"]["image_size"])  # (H, W)

    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        frames = batch["frames"].to(device)
        B = frames.shape[0]

        # Predict
        preds = model.predict(frames)

        # ---- Collect lane predictions ----
        # Decode lane points and convert to metric format
        decoded_lanes = preds["lanes"]  # List[List[Tensor]]
        for b in range(B):
            # Convert lane points to x-coordinates at fixed y-samples
            # (simplified — real impl uses CULane protocol)
            lane_preds_b = []
            for lane_pts in decoded_lanes[b]:
                # Extract x-coords as 1D array (simplified)
                x_coords = lane_pts[:, 0].cpu().numpy()
                lane_preds_b.append((x_coords, 0.8))  # (coords, confidence)
            all_lane_preds.append(lane_preds_b)

            # Ground truth lanes
            lane_labels = batch["lane_labels"]
            if "x_coords" in lane_labels:
                gt_x = lane_labels["x_coords"][b].numpy()  # [4, 18]
                gt_lanes_b = [gt_x[i] for i in range(4) if gt_x[i, 0] > -1]
            else:
                gt_lanes_b = []
            all_lane_targets.append(gt_lanes_b)

        # ---- Collect drivable predictions ----
        if batch.get("drivable_mask") is not None:
            for b in range(B):
                pred_mask = preds["drivable_mask"][b].cpu().numpy()
                gt_mask = batch["drivable_mask"][b].numpy()
                all_drivable_preds.append(pred_mask)
                all_drivable_targets.append(gt_mask)

        # ---- Save visualizations ----
        if save_predictions and output_dir is not None:
            for b in range(B):
                # Convert frames to numpy
                frame_imgs = [
                    (frames[b, t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    for t in range(frames.shape[1])
                ]
                vis = visualize_predictions(
                    frame_imgs,
                    [decoded_lanes[b]],
                    drivable_mask=preds["drivable_mask"][b].cpu().numpy()
                    if "drivable_mask" in preds else None,
                )
                import cv2
                out_path = output_dir / f"pred_{sample_count:06d}.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                sample_count += 1

    # ---- Compute metrics ----
    metrics = {}

    # Lane metrics
    if all_lane_preds and all_lane_targets:
        lane_metrics = compute_lane_metrics(
            all_lane_preds, all_lane_targets,
            iou_threshold=cfg["eval"].get("lane_iou_threshold", 0.5),
        )
        metrics.update(lane_metrics)
        print(f"\n[Lane Detection Metrics]")
        print(f"  F1:        {lane_metrics['f1']:.4f}")
        print(f"  Precision: {lane_metrics['precision']:.4f}")
        print(f"  Recall:    {lane_metrics['recall']:.4f}")
        print(f"  TP: {lane_metrics['tp']}, FP: {lane_metrics['fp']}, FN: {lane_metrics['fn']}")

    # Drivable metrics
    if all_drivable_preds and all_drivable_targets:
        drivable_metrics = compute_drivable_metrics(
            all_drivable_preds, all_drivable_targets,
            num_classes=cfg["model"]["drivable_head"].get("num_classes", 3),
        )
        metrics.update(drivable_metrics)
        print(f"\n[Drivable Area Metrics]")
        print(f"  mIoU:          {drivable_metrics['miou']:.4f}")
        print(f"  Pixel Accuracy: {drivable_metrics['pixel_accuracy']:.4f}")
        for c, iou in drivable_metrics["per_class_iou"].items():
            print(f"  Class {c} IoU:   {iou:.4f}")

    return metrics


def main():
    args = parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    # Dataset
    data_cfg = cfg["data"]
    if data_cfg["name"] == "culane":
        test_dataset = CULaneDataset(
            data_root=data_cfg["data_root"],
            split="test" if "test" in data_cfg.get("list", "") else "val",
            num_frames=data_cfg.get("num_frames", 5),
            frame_interval=data_cfg.get("frame_interval", 1),
            image_size=tuple(data_cfg["image_size"]),
            augment=False,
        )
    else:
        test_dataset = BDD100KDataset(
            data_root=data_cfg["data_root"],
            split="val",
            num_frames=data_cfg.get("num_frames", 5),
            frame_interval=data_cfg.get("frame_interval", 1),
            image_size=tuple(data_cfg["image_size"]),
            augment=False,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=True,
    )
    print(f"[INFO] Test samples: {len(test_dataset)}")

    # Model
    model = LaneDrivableDualModel(
        backbone_name=cfg["model"].get("backbone", "resnet18"),
        pretrained_backbone=False,  # Not needed for eval
        temporal_cfg=cfg["model"].get("temporal", {}),
        lane_cfg=cfg["model"].get("lane_head", {}),
        drivable_cfg=cfg["model"].get("drivable_head", {}),
        image_size=tuple(data_cfg["image_size"]),
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[INFO] Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Prepare output dir
    output_dir = None
    if args.save_predictions:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate
    metrics = evaluate(
        model, test_loader, device, cfg,
        save_predictions=args.save_predictions,
        output_dir=output_dir,
    )

    # Save metrics
    import json
    metrics_path = Path(args.output_dir) / "metrics.json" if args.save_predictions else Path("./metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in metrics.items()}, f, indent=2)
    print(f"\n[INFO] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
