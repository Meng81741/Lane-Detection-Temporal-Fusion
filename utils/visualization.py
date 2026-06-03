"""Visualization utilities for lane detection and drivable area."""

import colorsys
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch


# Color palette for drivable area classes (BDD100K)
DRIVABLE_COLORS = {
    0: (0, 0, 0),         # background: black
    1: (0, 255, 128),     # direct drivable: green
    2: (255, 128, 0),     # alternative drivable: orange
}

# Distinct colors for lane lines
LANE_COLORS = [
    (255, 0, 0),      # red
    (0, 0, 255),      # blue
    (255, 255, 0),    # yellow
    (0, 255, 255),    # cyan
    (255, 0, 255),    # magenta
    (128, 255, 0),    # lime
    (255, 128, 0),    # orange
    (128, 0, 255),    # purple
]


def overlay_drivable(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay drivable area mask on image.

    Args:
        image: [H, W, 3] RGB image (uint8)
        mask: [H, W] class indices
        alpha: blending factor
    Returns:
        blended image
    """
    H, W = mask.shape
    overlay = image.copy()
    if overlay.shape[:2] != (H, W):
        overlay = cv2.resize(overlay, (W, H))

    for class_id, color in DRIVABLE_COLORS.items():
        if class_id == 0:
            continue
        region = mask == class_id
        overlay[region] = (overlay[region] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)

    return overlay


def draw_lanes(
    image: np.ndarray,
    lanes: List[np.ndarray],
    color: Optional[Tuple[int, int, int]] = None,
    thickness: int = 3,
) -> np.ndarray:
    """Draw lane line points on image.

    Args:
        image: [H, W, 3] RGB image
        lanes: list of [N, 2] point arrays (x, y)
        color: single color or None for per-lane colors
        thickness: line thickness
    Returns:
        image with lanes drawn
    """
    vis = image.copy()
    for i, lane in enumerate(lanes):
        pts = lane.astype(np.int32)
        c = color if color is not None else LANE_COLORS[i % len(LANE_COLORS)]
        for j in range(len(pts) - 1):
            cv2.line(vis, tuple(pts[j]), tuple(pts[j + 1]), c, thickness)
    return vis


def visualize_predictions(
    frames: List[np.ndarray],
    lanes: List[List[np.ndarray]],
    drivable_mask: Optional[np.ndarray] = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """Create a visualization panel with lanes and drivable area.

    Args:
        frames: list of T frames [H, W, 3] uint8
        lanes: list of decoded lane points per image
        drivable_mask: [H, W] drivable class indices or None
        alpha: drivable overlay blending
    Returns:
        visualization image (last frame with overlays)
    """
    # Use last frame
    vis = frames[-1].copy()

    # Overlay drivable area
    if drivable_mask is not None:
        vis = overlay_drivable(vis, drivable_mask, alpha=alpha)

    # Draw lanes
    if lanes:
        vis = draw_lanes(vis, lanes[0] if isinstance(lanes[0], list) else lanes)

    return vis


def create_comparison_panel(
    frame: np.ndarray,
    pred_lanes: List[np.ndarray],
    gt_lanes: List[np.ndarray],
    pred_mask: Optional[np.ndarray] = None,
    gt_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Side-by-side: prediction vs ground truth.

    Args:
        frame: [H, W, 3] image
        pred_lanes: predicted lane points
        gt_lanes: ground truth lane points
        pred_mask: predicted drivable mask
        gt_mask: ground truth drivable mask
    Returns:
        concatenated [H, 2W, 3] comparison image
    """
    H, W = frame.shape[:2]

    # Prediction panel
    pred_vis = frame.copy()
    if pred_mask is not None:
        pred_vis = overlay_drivable(pred_vis, pred_mask)
    pred_vis = draw_lanes(pred_vis, pred_lanes)

    # Ground truth panel
    gt_vis = frame.copy()
    if gt_mask is not None:
        gt_vis = overlay_drivable(gt_vis, gt_mask)
    gt_vis = draw_lanes(gt_vis, gt_lanes, color=(0, 255, 0))

    # Labels
    cv2.putText(pred_vis, "Prediction", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(gt_vis, "Ground Truth", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return np.hstack([pred_vis, gt_vis])
