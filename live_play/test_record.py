"""Tests for the frame-labeling logic in record_browser (no display needed).

Run: python live_play/test_record.py
"""

from __future__ import annotations

import numpy as np

from record_browser import frame_detail, frame_motion, nearest_action, scene_looks_live


def test_no_press_is_none():
    assert nearest_action(10.0, [], window=0.3) is None


def test_press_within_window_labels():
    presses = [(10.1, "LEFT")]
    assert nearest_action(10.0, presses, window=0.3) == "LEFT"


def test_press_outside_window_is_none():
    presses = [(10.5, "LEFT")]
    assert nearest_action(10.0, presses, window=0.3) is None


def test_nearest_press_wins():
    presses = [(10.25, "LEFT"), (10.05, "RIGHT")]
    assert nearest_action(10.0, presses, window=0.3) == "RIGHT"


def test_just_inside_window_labels():
    assert nearest_action(10.0, [(10.29, "UP")], window=0.3) == "UP"


def _noisy_frame(rng: np.random.Generator) -> np.ndarray:
    """Busy, changing scene stand-in (like scrolling gameplay)."""
    return rng.integers(0, 255, size=(64, 48, 3), dtype=np.uint8)


def _gradient_frame() -> np.ndarray:
    """Smooth static gradient — a wallpaper stand-in."""
    col = np.linspace(80, 120, 48, dtype=np.float32)
    img = np.tile(col, (64, 1))
    return np.stack([img, img, img], axis=2).astype(np.uint8)


def test_static_wallpaper_fails_liveness():
    a = _gradient_frame()
    ok, reason = scene_looks_live(a, a.copy())
    assert not ok
    assert frame_motion(a, a.copy()) == 0.0
    assert frame_detail(a) < 1.5


def test_busy_changing_scene_passes_liveness():
    rng = np.random.default_rng(0)
    a, b = _noisy_frame(rng), _noisy_frame(rng)
    ok, reason = scene_looks_live(a, b)
    assert ok, reason


def test_static_but_detailed_scene_fails_liveness():
    # e.g. a crisp screenshot frozen on screen: detail high, motion zero.
    rng = np.random.default_rng(1)
    a = _noisy_frame(rng)
    ok, reason = scene_looks_live(a, a.copy())
    assert not ok
    assert "static" in reason


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
