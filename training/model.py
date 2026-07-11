"""CNN model for Subway Surfers action classification.

The network takes a resized game screenshot and predicts which swipe action
the player should perform: UP, DOWN, LEFT, RIGHT, or NONE.

Keep this module dependency-light (only torch) so it can be imported by both
the training script and the live-play inference code.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Canonical class ordering. The saved checkpoint stores this too, but keeping a
# single source of truth here avoids accidental re-ordering.
CLASSES = ["UP", "DOWN", "LEFT", "RIGHT", "NONE"]

# Input geometry (channels, height, width). Portrait phone screens are ~9:16,
# so 128x72 preserves the aspect ratio while staying cheap to train on CPU.
INPUT_CHANNELS = 3
INPUT_HEIGHT = 128
INPUT_WIDTH = 72

# ImageNet-ish per-channel normalisation constants. These are baked into the
# checkpoint config so inference uses the exact same preprocessing.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SubwayCNN(nn.Module):
    """Compact 4-block CNN classifier (~0.5M params)."""

    def __init__(self, num_classes: int = len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(INPUT_CHANNELS, 16),   # 128x72 -> 64x36
            _conv_block(16, 32),               # 64x36  -> 32x18
            _conv_block(32, 64),               # 32x18  -> 16x9
            _conv_block(64, 128),              # 16x9   -> 8x4
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           # -> 128x1x1
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(num_classes: int = len(CLASSES)) -> SubwayCNN:
    return SubwayCNN(num_classes=num_classes)
