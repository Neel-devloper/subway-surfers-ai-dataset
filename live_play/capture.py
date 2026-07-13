"""Screen-region capture for live play.

Grabs a rectangle of the desktop (the browser game area) as an RGB numpy
array, as fast as the display allows, using `mss`. Imports are lazy so this
module can be imported in a headless environment that has no display.

The capture region is stored in region_config.json by the calibration step
(calibrate.py) or supplied explicitly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

import config


@dataclass
class Region:
    left: int
    top: int
    width: int
    height: int

    def as_mss_dict(self) -> dict:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


def load_region(path: str = config.REGION_CONFIG_PATH) -> Optional[Region]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return Region(d["left"], d["top"], d["width"], d["height"])


def save_region(region: Region, path: str = config.REGION_CONFIG_PATH) -> None:
    with open(path, "w") as f:
        json.dump(region.__dict__, f, indent=2)


class ScreenCapture:
    """Repeatedly grabs one screen region as RGB uint8 (HxWx3)."""

    def __init__(self, region: Region):
        import mss  # lazy: needs a display

        self.region = region
        self._sct = mss.mss()
        self._mon = region.as_mss_dict()

    def grab(self) -> np.ndarray:
        raw = self._sct.grab(self._mon)  # BGRA
        arr = np.asarray(raw)[:, :, :3]  # drop alpha -> BGR
        return arr[:, :, ::-1]           # BGR -> RGB

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass
