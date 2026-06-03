#!/usr/bin/env python3
"""Real-time demo script for Lane-Detection-Temporal-Fusion.

Processes a video file frame-by-frame using streaming inference mode
(with ConvGRU hidden state carried across frames).

Usage:
    python demo.py --config configs/bdd100k.yaml \
                   --checkpoint checkpoints/best.pth \
                   --video path/to/video.mp4 \
                   --output demo_output.mp4

    python demo.py --config configs/bdd100k.yaml \
                   --checkpoint checkpoints/best.pth \
                   --camera 0  # webcam
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from models import LaneDrivableDualModel
from utils.config_utils import load_config
from utils.visualization import overlay_drivable, draw_lanes


LANE_COLORS = [
    (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Demo Lane-Detection-Temporal-Fusion")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video", type=str, default=None, help="Video file path")
    parser.add_argument("--camera", type=int, default=None, help="Camera device ID")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def preprocess_frame(frame: np.ndarray, target_size: tuple) -> torch.Tensor:
    """Convert BGR frame to RGB tensor."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (target_size[1], target_size[0]))
    tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
    return tensor


def main():
    args = parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Device: {device}")

    image_size = tuple(cfg["data"]["image_size"])  # (H, W)

    # Build model
    model = LaneDrivableDualModel(
        backbone_name=cfg["model"].get("backbone", "resnet18"),
        pretrained_backbone=False,
        temporal_cfg=cfg["model"].get("temporal", {}),
        lane_cfg=cfg["model"].get("lane_head", {}),
        drivable_cfg=cfg["model"].get("drivable_head", {}),
        image_size=image_size,
    ).to(device)

    # Load weights
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[INFO] Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Video source
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        source_name = f"Camera {args.camera}"
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        source_name = args.video
    else:
        print("[ERROR] Specify --video or --camera")
        sys.exit(1)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source_name}")
        sys.exit(1)

    # Video writer
    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (orig_w, orig_h))
        print(f"[INFO] Output: {args.output} ({orig_w}x{orig_h} @ {fps:.1f}fps)")

    # Streaming state
    hidden = None
    frame_buffer: List[torch.Tensor] = []
    frame_count = 0
    fps_window = []

    print("[INFO] Starting inference... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]
        frame_count += 1
        t_start = time.time()

        # Preprocess
        frame_tensor = preprocess_frame(frame, image_size).unsqueeze(0).to(device)  # [1, 3, H, W]

        # Streaming step
        with torch.no_grad():
            predictions, hidden, frame_buffer = model.streaming_step(
                frame_tensor, hidden=hidden, frame_buffer=frame_buffer
            )

        # FPS
        elapsed = time.time() - t_start
        fps_window.append(1.0 / elapsed if elapsed > 0 else 0)
        if len(fps_window) > 30:
            fps_window.pop(0)
        avg_fps = np.mean(fps_window)

        # Visualization
        vis_frame = frame.copy()

        # Draw drivable area
        if predictions.get("drivable_mask") is not None:
            drivable_mask = predictions["drivable_mask"][0].cpu().numpy()
            drivable_mask_resized = cv2.resize(
                drivable_mask.astype(np.uint8),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )
            alpha = 0.3
            drivable_colors = {
                1: (0, 255, 128),   # direct: green
                2: (255, 128, 0),   # alternative: orange
            }
            for class_id, color in drivable_colors.items():
                region = drivable_mask_resized == class_id
                vis_frame[region] = (
                    vis_frame[region] * (1 - alpha) + np.array(color) * alpha
                ).astype(np.uint8)

        # Draw lane lines
        if predictions.get("lanes"):
            lanes = predictions["lanes"][0]  # lanes for batch index 0
            for i, lane_pts in enumerate(lanes):
                # Rescale points to original image size
                pts = lane_pts.cpu().numpy()
                pts[:, 0] *= orig_w / image_size[1]  # x scale
                pts[:, 1] *= orig_h / image_size[0]  # y scale
                pts = pts.astype(np.int32)

                color = LANE_COLORS[i % len(LANE_COLORS)]
                for j in range(len(pts) - 1):
                    cv2.line(vis_frame, tuple(pts[j]), tuple(pts[j + 1]), color, 3)

        # Overlay info
        cv2.putText(vis_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Frame: {frame_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if len(frame_buffer) < model.num_frames:
            cv2.putText(vis_frame, "Warming up...", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Write output
        if writer is not None:
            writer.write(vis_frame)

        # Display
        if not args.no_display:
            cv2.imshow("Lane-Detection-Temporal-Fusion", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Processed {frame_count} frames. Average FPS: {np.mean(fps_window):.1f}")


if __name__ == "__main__":
    main()
