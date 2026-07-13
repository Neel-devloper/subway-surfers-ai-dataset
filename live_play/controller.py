"""Core decision logic shared by live play and replay.

`PlayController.step(frame)` is the single place that turns a frame into an
action decision: run the model, apply the confidence gate, apply the
per-action cooldown, and (if it fires) send the key through the input backend.

Because both main.py (live) and replay.py (recorded frames) drive the loop
through this same method, anything validated in replay exercises the exact
decision path used during live play — only the frame source and input backend
differ.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

import config
from controls import DryRunBackend, InputBackend
from predictor import ActionPredictor, Prediction


@dataclass
class Decision:
    prediction: Prediction
    action: str            # the chosen action after gating (may be "NONE")
    fired: bool            # whether a key was actually sent
    key: Optional[str]     # the key sent, if any
    reason: str            # why it did / didn't fire (for the overlay + logs)


@dataclass
class PlayController:
    predictor: ActionPredictor
    input_backend: InputBackend = field(default_factory=DryRunBackend)
    confidence_threshold: float = config.CONFIDENCE_THRESHOLD
    cooldowns: Dict[str, float] = field(default_factory=lambda: dict(config.ACTION_COOLDOWN_SEC))
    action_to_key: Dict[str, str] = field(default_factory=lambda: dict(config.ACTION_TO_KEY))
    suppress_static: bool = getattr(config, "SUPPRESS_STATIC_SCENE", False)
    static_diff_threshold: float = getattr(config, "STATIC_SCENE_DIFF_THRESHOLD", 2.0)

    _last_fired_at: Dict[str, float] = field(default_factory=dict, init=False)
    _prev_small: Optional[np.ndarray] = field(default=None, init=False)

    def _is_static(self, frame_rgb: np.ndarray) -> bool:
        """True if this frame is ~identical to the previous one (menu/paused/
        crashed screen). Gameplay scrolls constantly, so it is never static."""
        # Downsample to 32x32 grayscale for a cheap, noise-tolerant comparison.
        small = frame_rgb[::max(1, frame_rgb.shape[0] // 32),
                          ::max(1, frame_rgb.shape[1] // 32)].mean(axis=2)
        prev = self._prev_small
        self._prev_small = small
        if prev is None or prev.shape != small.shape:
            return False
        return float(np.abs(small - prev).mean()) < self.static_diff_threshold

    def step(self, frame_rgb: np.ndarray, now: Optional[float] = None) -> Decision:
        now = time.time() if now is None else now

        if self.suppress_static and self._is_static(frame_rgb):
            pred = self.predictor.predict(frame_rgb)
            return Decision(pred, "NONE", False, None, "static scene (menu/paused)")

        pred = self.predictor.predict(frame_rgb)

        # NONE means "don't act" — the model's way of saying keep running.
        if pred.action == "NONE":
            return Decision(pred, "NONE", False, None, "model chose NONE")

        if pred.confidence < self.confidence_threshold:
            return Decision(
                pred, "NONE", False, None,
                f"low confidence {pred.confidence:.2f} < {self.confidence_threshold:.2f}",
            )

        last = self._last_fired_at.get(pred.action, -1e9)
        cooldown = self.cooldowns.get(pred.action, 0.0)
        if now - last < cooldown:
            return Decision(
                pred, pred.action, False, None,
                f"cooldown ({now - last:.2f}s < {cooldown:.2f}s)",
            )

        key = self.action_to_key.get(pred.action)
        if key is None:
            return Decision(pred, pred.action, False, None, "no key mapping")

        self.input_backend.press(key)
        self._last_fired_at[pred.action] = now
        return Decision(pred, pred.action, True, key, "fired")
