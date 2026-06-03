"""Abstract temporal frame dataset base class.

Provides frame-sequence sampling logic shared across dataset implementations.
"""

import random
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TemporalFrameDataset(Dataset):
    """Base class for datasets that return T consecutive frames.

    Subclasses must implement:
      - _load_frame_paths() → List[Path]
      - _load_annotation(idx) → dict with 'lane_labels' and/or 'drivable_mask'
      - _get_video_boundaries() → List[Tuple[int, int]]  (start_idx, end_idx)

    Args:
        data_root: root directory of the dataset
        split: 'train' | 'val' | 'test'
        num_frames: T — number of consecutive frames per sample
        frame_interval: step between consecutive frames (1 = consecutive)
        image_size: (H, W) target size
        augment: whether to apply data augmentation
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_frames: int = 5,
        frame_interval: int = 1,
        image_size: Tuple[int, int] = (288, 800),
        augment: bool = True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size  # (H, W)
        self.augment = augment and (split == "train")

        # Load frame paths (subclass responsibility)
        self.frame_paths = self._load_frame_paths()

        # Video boundaries for ensuring temporal continuity
        self._boundaries = self._get_video_boundaries()

        # Build valid start indices (can sample T consecutive frames)
        self._valid_indices = self._build_valid_indices()

    def _load_frame_paths(self) -> List[Path]:
        raise NotImplementedError

    def _load_annotation(self, idx: int) -> dict:
        raise NotImplementedError

    def _get_video_boundaries(self) -> List[Tuple[int, int]]:
        """Return (start, end) inclusive indices for each video clip."""
        return [(0, len(self.frame_paths) - 1)]

    def _build_valid_indices(self) -> List[int]:
        """Build list of indices where we can sample T consecutive frames."""
        valid = []
        stride = self.num_frames * self.frame_interval
        for start, end in self._boundaries:
            for i in range(start, end - stride + 2):
                valid.append(i)
        return valid

    def _load_image(self, path: Path) -> np.ndarray:
        """Load and resize a single image."""
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
        return img

    def _sample_frame_sequence(self, start_idx: int) -> List[np.ndarray]:
        """Sample T frames starting from start_idx."""
        frames = []
        for t in range(self.num_frames):
            idx = start_idx + t * self.frame_interval
            idx = min(idx, len(self.frame_paths) - 1)
            frames.append(self._load_image(self.frame_paths[idx]))
        return frames

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, index: int) -> dict:
        """Return a sample with T frames + annotations for the LAST frame.

        Returns dict with:
            frames: [T, 3, H, W] tensor
            lane_labels: dict (format depends on dataset)
            drivable_mask: [H, W] LongTensor or None
            meta: dict with filenames, indices etc.
        """
        start_idx = self._valid_indices[index]
        # Annotation is for the middle/last frame
        target_idx = start_idx + (self.num_frames - 1) * self.frame_interval
        target_idx = min(target_idx, len(self.frame_paths) - 1)

        # Load frames
        frames = self._sample_frame_sequence(start_idx)

        # Load annotation
        ann = self._load_annotation(target_idx)

        # Convert to tensors
        frame_tensors = []
        for frame in frames:
            frame_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frame_tensors.append(frame_t)
        frames_stacked = torch.stack(frame_tensors, dim=0)  # [T, 3, H, W]

        return {
            "frames": frames_stacked,
            "lane_labels": ann.get("lane_labels", {}),
            "drivable_mask": ann.get("drivable_mask", None),
            "meta": {
                "frame_idx": target_idx,
                "frame_path": str(self.frame_paths[target_idx]),
            },
        }
