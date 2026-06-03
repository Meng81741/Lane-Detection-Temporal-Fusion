"""Evaluation metrics for lane detection and drivable area segmentation."""

from typing import List, Tuple, Optional
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_iou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute per-class IoU.

    Args:
        pred: [H, W] predicted class indices
        target: [H, W] ground truth class indices
        num_classes: number of classes
    Returns:
        iou: [num_classes] IoU per class
    """
    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)
    return np.array(ious)


def compute_drivable_metrics(
    pred_masks: List[np.ndarray],
    target_masks: List[np.ndarray],
    num_classes: int = 3,
    ignore_index: int = -100,
) -> dict:
    """Compute drivable area metrics.

    Args:
        pred_masks: list of [H, W] predictions
        target_masks: list of [H, W] targets
        num_classes: number of classes
    Returns:
        dict with miou, per_class_iou, pixel_accuracy
    """
    all_preds = []
    all_targets = []

    for pred, target in zip(pred_masks, target_masks):
        if ignore_index >= 0:
            valid = target != ignore_index
            all_preds.append(pred[valid])
            all_targets.append(target[valid])
        else:
            all_preds.append(pred.flatten())
            all_targets.append(target.flatten())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Per-class IoU
    ious = []
    for c in range(num_classes):
        pred_c = all_preds == c
        target_c = all_targets == c
        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        iou = intersection / union if union > 0 else float("nan")
        ious.append(iou)

    # mIoU (ignore NaN classes)
    valid_ious = [i for i in ious if not np.isnan(i)]
    miou = np.mean(valid_ious) if valid_ious else 0.0

    # Pixel accuracy
    pixel_acc = (all_preds == all_targets).mean()

    return {
        "miou": miou,
        "per_class_iou": {str(c): i for c, i in enumerate(ious)},
        "pixel_accuracy": pixel_acc,
    }


def compute_lane_metrics(
    pred_lanes: List[List[Tuple[np.ndarray, float]]],
    target_lanes: List[List[np.ndarray]],
    iou_threshold: float = 0.5,
    num_y_samples: int = 18,
) -> dict:
    """Compute lane detection metrics (F1, precision, recall).

    Args:
        pred_lanes: per-image list of (lane_x_coords, confidence) tuples
        target_lanes: per-image list of lane x_coords arrays [num_y_samples]
        iou_threshold: IoU threshold for positive match
    Returns:
        dict with f1, precision, recall
    """
    all_tp, all_fp, all_fn = 0, 0, 0

    for preds, targets in zip(pred_lanes, target_lanes):
        if len(targets) == 0:
            all_fp += len(preds)
            continue

        if len(preds) == 0:
            all_fn += len(targets)
            continue

        # Sort predictions by confidence
        preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
        pred_lane_pts = [p[0] for p in preds_sorted]  # x-coordinates arrays

        # Compute pairwise IoU
        matched_targets = set()
        for pred in pred_lane_pts:
            best_iou = 0
            best_t = -1
            for t_idx, target in enumerate(targets):
                if t_idx in matched_targets:
                    continue
                iou = _lane_iou(pred, target)
                if iou > best_iou:
                    best_iou = iou
                    best_t = t_idx

            if best_iou >= iou_threshold and best_t >= 0:
                all_tp += 1
                matched_targets.add(best_t)
            else:
                all_fp += 1

        all_fn += len(targets) - len(matched_targets)

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": all_tp,
        "fp": all_fp,
        "fn": all_fn,
    }


def _lane_iou(pred_x: np.ndarray, target_x: np.ndarray) -> float:
    """Compute IoU between two lane lines at fixed y-positions.

    Args:
        pred_x: [num_y_samples] x-coordinates of predicted lane
        target_x: [num_y_samples] x-coordinates of ground truth lane
    Returns:
        IoU score in [0, 1]
    """
    # Valid positions where both have values (not -2)
    valid = (pred_x > -1) & (target_x > -1)
    if valid.sum() == 0:
        return 0.0

    pred_valid = pred_x[valid]
    target_valid = target_x[valid]

    # Treat each lane as a 30-pixel wide line for intersection
    lane_width = 30.0
    intersection = np.abs(pred_valid - target_valid) < lane_width
    union = np.ones_like(intersection, dtype=bool)  # all valid y-positions count

    return intersection.sum() / union.sum()
