"""Loads the trained checkpoint and turns a raw frame into an action.

Reuses training/model.py as the single source of truth for the architecture
and preprocessing constants, so inference is guaranteed to match training.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

import config

# training/ isn't a package; import its model.py directly by path so this
# module works whether it's run from live_play/ or the repo root.
_TRAINING_DIR = os.path.join(config.REPO_ROOT, "training")
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)

from model import build_model  # noqa: E402


@dataclass
class Prediction:
    action: str
    confidence: float
    probs: Dict[str, float]


class ActionPredictor:
    """Wraps the checkpoint: raw HxWx3 uint8 RGB frame in, Prediction out."""

    def __init__(self, checkpoint_path: str = config.CHECKPOINT_PATH):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"No checkpoint at {checkpoint_path}. Train the model first "
                "(see training/README.md) or point --checkpoint at an existing .pth."
            )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.classes = ckpt["classes"]
        self.height = ckpt["input"]["height"]
        self.width = ckpt["input"]["width"]
        self.mean = np.array(ckpt["norm"]["mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(ckpt["norm"]["std"], dtype=np.float32).reshape(3, 1, 1)

        self.model = build_model(num_classes=len(self.classes))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        torch.set_num_threads(max(os.cpu_count() or 1, 1))

    def _center_crop_to_aspect(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Crop to the model's width:height aspect, centered. No-op when the
        frame already matches (e.g. native portrait training screenshots)."""
        h, w = frame_rgb.shape[:2]
        target = self.width / self.height
        cur = w / h
        # Tolerance covers the small spread of native screenshot aspects
        # (~0.56-0.57) so they stay a true no-op; only clearly-off (e.g.
        # landscape) frames get cropped.
        if abs(cur - target) <= 0.02 * target + 1e-6:
            return frame_rgb
        if cur > target:  # too wide -> trim width
            new_w = max(1, int(round(h * target)))
            x0 = (w - new_w) // 2
            return frame_rgb[:, x0:x0 + new_w]
        # too tall -> trim height
        new_h = max(1, int(round(w / target)))
        y0 = (h - new_h) // 2
        return frame_rgb[y0:y0 + new_h, :]

    def _preprocess(self, frame_rgb: np.ndarray) -> torch.Tensor:
        # frame_rgb: HxWx3 uint8, RGB, arbitrary size -> resize to model input.
        from PIL import Image

        if getattr(config, "CROP_TO_TRAINING_ASPECT", False):
            frame_rgb = self._center_crop_to_aspect(frame_rgb)

        im = Image.fromarray(frame_rgb).resize((self.width, self.height), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0  # HxWx3
        arr = np.transpose(arr, (2, 0, 1))  # 3xHxW
        arr = (arr - self.mean) / self.std
        return torch.from_numpy(arr).unsqueeze(0).float()

    @torch.no_grad()
    def predict(self, frame_rgb: np.ndarray) -> Prediction:
        x = self._preprocess(frame_rgb)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()
        idx = int(probs.argmax())
        return Prediction(
            action=self.classes[idx],
            confidence=float(probs[idx]),
            probs={c: float(p) for c, p in zip(self.classes, probs)},
        )
