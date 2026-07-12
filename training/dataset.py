"""Dataset utilities for the Subway Surfers screenshots.

The screenshots live under ``screen_collector/screens/<speed>/<ACTION>/*.png``
where ``<speed>`` is one of ``slow|medium|fast|legacy`` (just how the data was
collected) and ``<ACTION>`` is the label. We ignore the speed grouping and
train a single classifier over the action labels.

Images are pre-loaded and decoded once into a uint8 tensor cache so that CPU
training does not pay PNG-decode cost every epoch.
"""

from __future__ import annotations

import glob
import os
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from model import (
    CLASSES,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    NORM_MEAN,
    NORM_STD,
)

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def find_samples(screens_root: str) -> List[Tuple[str, int]]:
    """Return a list of (image_path, label_index) for every screenshot."""
    samples: List[Tuple[str, int]] = []
    for path in glob.glob(os.path.join(screens_root, "**", "*.png"), recursive=True):
        action = os.path.basename(os.path.dirname(path))
        if action in CLASS_TO_IDX:
            samples.append((path, CLASS_TO_IDX[action]))
    samples.sort()
    return samples


def _load_resized(path: str) -> np.ndarray:
    """Decode one image to a uint8 HxWx3 array at the model input size."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((INPUT_WIDTH, INPUT_HEIGHT), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


class SubwayDataset(Dataset):
    """In-memory screenshot dataset with optional light augmentation.

    All images are cached as uint8 in RAM (~120 MB for the full set). Per-sample
    normalisation to float happens on access so the cache stays small.
    """

    def __init__(self, samples: List[Tuple[str, int]], augment: bool = False):
        self.augment = augment
        self.mean = np.array(NORM_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(NORM_STD, dtype=np.float32).reshape(3, 1, 1)

        # Decode into RAM, silently skipping any truncated/corrupt PNGs.
        cache = np.empty((len(samples), INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)
        labels = np.empty(len(samples), dtype=np.int64)
        kept = 0
        skipped = []
        for path, lbl in samples:
            try:
                cache[kept] = _load_resized(path)
            except Exception:
                skipped.append(path)
                continue
            labels[kept] = lbl
            kept += 1
        if skipped:
            print(f"  skipped {len(skipped)} unreadable image(s), e.g. {skipped[0]}")
        self.cache = cache[:kept]
        self.labels = labels[:kept]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.cache[idx].astype(np.float32) / 255.0  # HxWx3

        if self.augment:
            # Small brightness jitter. We deliberately do NOT horizontal-flip:
            # LEFT/RIGHT labels are direction-specific, so flipping would corrupt
            # them.
            factor = 1.0 + np.random.uniform(-0.15, 0.15)
            img = np.clip(img * factor, 0.0, 1.0)

        img = np.transpose(img, (2, 0, 1))  # -> 3xHxW
        img = (img - self.mean) / self.std
        return torch.from_numpy(img), int(self.labels[idx])


def stratified_split(
    samples: List[Tuple[str, int]], val_frac: float = 0.15, seed: int = 42
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Split samples into train/val, preserving per-class proportions."""
    rng = np.random.default_rng(seed)
    by_class: dict[int, list] = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)

    train, val = [], []
    for _, items in sorted(by_class.items()):
        idx = rng.permutation(len(items))
        n_val = max(1, int(round(len(items) * val_frac)))
        val_idx = set(idx[:n_val].tolist())
        for j, item in enumerate(items):
            (val if j in val_idx else train).append(item)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def class_weights(samples: List[Tuple[str, int]]) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss to fight imbalance."""
    counts = np.bincount([lbl for _, lbl in samples], minlength=len(CLASSES))
    counts = np.maximum(counts, 1)
    w = counts.sum() / (len(CLASSES) * counts)
    return torch.tensor(w, dtype=torch.float32)
