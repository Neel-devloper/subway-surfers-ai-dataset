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

    _last_fired_at: Dict[str, float] = field(default_factory=dict, init=False)

    def step(self, frame_rgb: np.ndarray, now: Optional[float] = None) -> Decision:
        now = time.time() if now is None else now
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
