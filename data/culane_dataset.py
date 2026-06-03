"""CULane dataset loader.

CULane format:
  - Images: driver_XX_XXframe/*.jpg (sequential frames)
  - Lane annotations: laneseg_label_w16/*.png (binary lane segmentation)
  - Lists: list/train_gt.txt, list/val_gt.txt, list/test.txt
        Format: /driver_23_30frame/05151649_0422.MP4/00000.jpg  /laneseg... 0 0 0 1 0 ...

Lane label parsing follows the official CULane evaluation protocol:
  Each line in the list file contains the image path and up to 4 lane markers,
  each with x-coordinates at fixed y-anchor points.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2

from .dataset import TemporalFrameDataset


# CULane uses 18 pre-defined y-anchor positions (row indices, normalized to image height)
CULANE_Y_ANCHORS = np.array(
    [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
     0.80, 0.85, 0.90, 0.925, 0.95, 0.965, 0.98, 1.0], dtype=np.float32
)

NUM_CULANE_Y_ANCHORS = len(CULANE_Y_ANCHORS)  # 18


class CULaneDataset(TemporalFrameDataset):
    """CULane lane detection dataset with temporal frame sampling."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_frames: int = 5,
        frame_interval: int = 1,
        image_size: Tuple[int, int] = (288, 800),
        augment: bool = True,
    ):
        # Map split to list file
        split_map = {
            "train": "train_gt.txt",
            "val": "val_gt.txt",
            "test": "test.txt",
        }
        self._list_file = split_map.get(split, "train_gt.txt")
        self._annotations: List[dict] = []  # pre-parsed annotations per frame

        super().__init__(
            data_root=data_root,
            split=split,
            num_frames=num_frames,
            frame_interval=frame_interval,
            image_size=image_size,
            augment=augment,
        )

    def _load_frame_paths(self) -> List[Path]:
        """Parse CULane list file and load frame paths + annotations."""
        list_path = self.data_root / "list" / self._list_file

        if not list_path.exists():
            # Fallback: collect all jpg files recursively
            print(f"[CULaneDataset] List file not found: {list_path}, scanning images...")
            frame_paths = sorted(self.data_root.rglob("*.jpg"))
            self._annotations = [{} for _ in frame_paths]
            return frame_paths

        frame_paths = []
        with open(list_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # First token is image path (relative)
                rel_path = parts[0]
                # Remove leading '/' if present
                if rel_path.startswith('/'):
                    rel_path = rel_path[1:]
                img_path = self.data_root / rel_path
                frame_paths.append(img_path)

                # Parse lane annotations
                # After the image path, there are NUM_CULANE_Y_ANCHORS*4 x-coords
                # representing up to 4 lanes
                lane_xs = []
                if len(parts) > 1:
                    # Each lane: NUM_CULANE_Y_ANCHORS x-coordinates, -2 = missing
                    num_lane_tokens = len(parts) - 1
                    for lane_idx in range(4):
                        start = 1 + lane_idx * NUM_CULANE_Y_ANCHORS
                        end = start + NUM_CULANE_Y_ANCHORS
                        if end <= len(parts):
                            try:
                                xs = [float(x) for x in parts[start:end]]
                                lane_xs.append(xs)
                            except ValueError:
                                lane_xs.append([-2] * NUM_CULANE_Y_ANCHORS)
                        else:
                            lane_xs.append([-2] * NUM_CULANE_Y_ANCHORS)

                self._annotations.append({"lane_xs": lane_xs})

        # Handle case where list file is empty but images exist
        if not frame_paths:
            frame_paths = sorted(self.data_root.rglob("*.jpg"))
            self._annotations = [{} for _ in frame_paths]

        return frame_paths

    def _load_annotation(self, idx: int) -> dict:
        """Load lane annotation for frame at idx."""
        if idx < len(self._annotations):
            ann = self._annotations[idx]
            lane_xs = ann.get("lane_xs", [])
        else:
            lane_xs = []

        # Build lane labels tensor: [4, NUM_Y_ANCHORS] where -2 = missing
        lane_array = np.full((4, NUM_CULANE_Y_ANCHORS), -2.0, dtype=np.float32)
        for i, xs in enumerate(lane_xs[:4]):
            lane_array[i, :len(xs)] = xs

        # Also try to load segmentation label if available
        drivable_mask = None
        # CULane doesn't have drivable area labels by default, leave as None

        return {
            "lane_labels": {
                "x_coords": lane_array,  # [4, 18]
                "y_anchors": CULANE_Y_ANCHORS.copy(),
            },
            "drivable_mask": drivable_mask,
        }

    def _get_video_boundaries(self) -> List[Tuple[int, int]]:
        """Group frames by video clip directory for temporal consistency."""
        if not self.frame_paths:
            return [(0, 0)]

        boundaries = []
        current_clip = None
        clip_start = 0

        for i, path in enumerate(self.frame_paths):
            # CULane path structure: .../driver_XX_XXframe/CLIP_NAME/frame.jpg
            parts = path.parts
            # Find the clip directory (the one containing .jpg files)
            clip_dir = str(path.parent)

            if current_clip is None:
                current_clip = clip_dir
                clip_start = i
            elif clip_dir != current_clip:
                boundaries.append((clip_start, i - 1))
                current_clip = clip_dir
                clip_start = i

        boundaries.append((clip_start, len(self.frame_paths) - 1))
        return boundaries
