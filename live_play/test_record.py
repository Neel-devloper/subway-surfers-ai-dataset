"""Tests for the frame-labeling logic in record_browser (no display needed).

Run: python live_play/test_record.py
"""

from __future__ import annotations

from record_browser import nearest_action


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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
