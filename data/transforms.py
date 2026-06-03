"""Data augmentation and preprocessing transforms."""

import random
from typing import Tuple, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, frames: List[np.ndarray], mask: Optional[np.ndarray] = None):
        for t in self.transforms:
            frames, mask = t(frames, mask)
        return frames, mask


class RandomHorizontalFlip:
    """Randomly flip all frames and mask horizontally."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, frames, mask):
        if random.random() < self.p:
            frames = [np.fliplr(f) for f in frames]
            if mask is not None:
                mask = np.fliplr(mask)
        return frames, mask


class RandomBrightnessContrast:
    """Random brightness/contrast jitter (applied consistently across frames)."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        p: float = 0.5,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, frames, mask):
        if random.random() < self.p:
            alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
            beta = random.uniform(-self.brightness, self.brightness) * 255

            frames = [
                np.clip(alpha * f.astype(np.float32) + beta, 0, 255).astype(np.uint8)
                for f in frames
            ]
        return frames, mask


class RandomGaussianNoise:
    """Add Gaussian noise (simulates low-light/night conditions)."""

    def __init__(self, sigma: float = 10.0, p: float = 0.3):
        self.sigma = sigma
        self.p = p

    def __call__(self, frames, mask):
        if random.random() < self.p:
            noise_sigma = random.uniform(0, self.sigma)
            frames = [
                np.clip(
                    f.astype(np.float32) + np.random.randn(*f.shape) * noise_sigma,
                    0, 255
                ).astype(np.uint8)
                for f in frames
            ]
        return frames, mask


class Normalize:
    """Normalize frames to [0, 1] float (handled in dataset __getitem__)."""

    def __call__(self, frames, mask):
        return frames, mask


def get_train_transforms() -> Compose:
    """Training augmentations including night/rain simulation."""
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomBrightnessContrast(brightness=0.2, contrast=0.2, p=0.5),
        RandomGaussianNoise(sigma=10.0, p=0.3),
        Normalize(),
    ])


def get_val_transforms() -> Compose:
    """Validation — no augmentation."""
    return Compose([
        Normalize(),
    ])
