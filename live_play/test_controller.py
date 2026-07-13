"""Deterministic tests for the decision logic (no model / display needed).

Run: python live_play/test_controller.py
"""

from __future__ import annotations

import numpy as np

from controller import PlayController
from controls import DryRunBackend
from predictor import Prediction


class _FakePredictor:
    """Stand-in predictor returning a fixed prediction, so the gate/cooldown
    logic can be tested without loading the real model."""

    def __init__(self, action: str, conf: float):
        self._p = Prediction(action, conf, {action: conf})

    def predict(self, frame_rgb):
        return self._p


FRAME = np.zeros((10, 10, 3), np.uint8)


# These tests feed identical frames repeatedly, which the static-scene detector
# would (correctly) flag — so disable it here to isolate the gate/cooldown logic.
def test_none_never_fires():
    c = PlayController(_FakePredictor("NONE", 0.99), DryRunBackend(), suppress_static=False)
    assert c.step(FRAME).fired is False


def test_low_confidence_is_gated():
    c = PlayController(_FakePredictor("UP", 0.40), DryRunBackend(),
                       confidence_threshold=0.6, suppress_static=False)
    d = c.step(FRAME)
    assert d.fired is False
    assert "low confidence" in d.reason


def test_fires_then_cooldown_then_fires_again():
    backend = DryRunBackend()
    c = PlayController(
        _FakePredictor("UP", 0.95), backend, confidence_threshold=0.6,
        cooldowns={"UP": 0.35}, suppress_static=False,
    )
    d1 = c.step(FRAME, now=100.0)
    assert d1.fired and d1.key == "up"
    d2 = c.step(FRAME, now=100.1)  # within cooldown
    assert d2.fired is False and "cooldown" in d2.reason
    d3 = c.step(FRAME, now=100.5)  # cooldown elapsed
    assert d3.fired is True
    assert [k for _, k in backend.history] == ["up", "up"]


def test_static_scene_is_suppressed():
    # A confident action on a repeated (static) frame must NOT fire — this is the
    # menu / crash-screen guard. First frame primes the detector; the identical
    # second frame is flagged static.
    backend = DryRunBackend()
    c = PlayController(_FakePredictor("DOWN", 0.99), backend,
                       confidence_threshold=0.6, suppress_static=True,
                       static_diff_threshold=2.0)
    c.step(FRAME, now=0.0)              # primes _prev_small (may fire)
    d = c.step(FRAME, now=1.0)          # identical frame -> static
    assert d.fired is False and "static" in d.reason

    # A clearly different frame is treated as live gameplay again.
    other = FRAME.copy()
    other[:] = 200
    d2 = c.step(other, now=2.0)
    assert "static" not in d2.reason


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
