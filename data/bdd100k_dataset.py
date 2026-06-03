"""BDD100K dataset loader.

BDD100K format:
  - Images: images/100k/{train,val,test}/*.jpg
  - Drivable labels: labels/drivable/masks/{train,val}/*.png
    (0: background, 1: direct drivable, 2: alternative drivable)
  - Lane labels: labels/lane/masks/{train,val}/*.png
    (binary lane markings)
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2

from .dataset import TemporalFrameDataset


class BDD100KDataset(TemporalFrameDataset):
    """BDD100K dataset with lane + drivable area annotations."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_frames: int = 5,
        frame_interval: int = 1,
        image_size: Tuple[int, int] = (360, 640),
        augment: bool = True,
    ):
        self._drivable_dir: Optional[Path] = None
        self._lane_dir: Optional[Path] = None
        self._use_drivable = True
        self._use_lane = True

        super().__init__(
            data_root=data_root,
            split=split,
            num_frames=num_frames,
            frame_interval=frame_interval,
            image_size=image_size,
            augment=augment,
        )

    def _load_frame_paths(self) -> List[Path]:
        """Find all images for the given split."""
        img_dir = self.data_root / "images" / "100k" / self.split
        if img_dir.exists():
            frame_paths = sorted(img_dir.glob("*.jpg"))
        else:
            # Try alternate structure
            img_dir = self.data_root / "images" / self.split
            if img_dir.exists():
                frame_paths = sorted(img_dir.glob("*.jpg"))
            else:
                frame_paths = sorted(self.data_root.rglob("*.jpg"))
                # Filter to reasonable number for dev
                frame_paths = frame_paths[:1000]

        # Check for label directories
        drivable_dir = self.data_root / "labels" / "drivable" / "masks" / self.split
        if drivable_dir.exists():
            self._drivable_dir = drivable_dir
        else:
            self._use_drivable = False

        lane_dir = self.data_root / "labels" / "lane" / "masks" / self.split
        if lane_dir.exists():
            self._lane_dir = lane_dir
        else:
            self._use_lane = False

        print(f"[BDD100KDataset] {self.split}: {len(frame_paths)} images, "
              f"drivable={'Y' if self._use_drivable else 'N'}, "
              f"lane={'Y' if self._use_lane else 'N'}")

        return frame_paths

    def _load_annotation(self, idx: int) -> dict:
        """Load drivable mask and lane mask for a frame."""
        frame_path = self.frame_paths[idx]

        # Drivable mask
        drivable_mask = None
        if self._use_drivable and self._drivable_dir is not None:
            mask_name = frame_path.stem + ".png"
            mask_path = self._drivable_dir / mask_name
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = cv2.resize(
                        mask, (self.image_size[1], self.image_size[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                    # BDD100K drivable: 0=background, 1=direct, 2=alternative
                    # Some versions use 0-255 scaling
                    mask = np.clip(mask, 0, 2)
                    drivable_mask = mask.astype(np.int64)

        # Lane labels (from mask or json)
        lane_labels = {}
        if self._use_lane:
            # Try lane mask
            if self._lane_dir is not None:
                lane_mask_name = frame_path.stem + ".png"
                lane_mask_path = self._lane_dir / lane_mask_name
                if lane_mask_path.exists():
                    lane_mask = cv2.imread(str(lane_mask_path), cv2.IMREAD_GRAYSCALE)
                    if lane_mask is not None:
                        lane_mask = cv2.resize(
                            lane_mask, (self.image_size[1], self.image_size[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                        lane_labels["mask"] = (lane_mask > 0).astype(np.float32)

        return {
            "lane_labels": lane_labels,
            "drivable_mask": drivable_mask,
        }

    def _get_video_boundaries(self) -> List[Tuple[int, int]]:
        """BDD100K images are per-video; group by filename prefix."""
        if not self.frame_paths:
            return [(0, 0)]

        # BDD100K filenames: videoID-frameID.jpg, e.g. "0000f77c-6257be58-00000001.jpg"
        # Group by the first two parts (video ID)
        boundaries = []
        current_video = None
        clip_start = 0

        for i, path in enumerate(self.frame_paths):
            stem = path.stem
            # Extract video ID = first two segments before '-'
            parts = stem.split("-")
            video_id = "-".join(parts[:2]) if len(parts) >= 2 else stem

            if current_video is None:
                current_video = video_id
                clip_start = i
            elif video_id != current_video:
                boundaries.append((clip_start, i - 1))
                current_video = video_id
                clip_start = i

        boundaries.append((clip_start, len(self.frame_paths) - 1))
        return boundaries
