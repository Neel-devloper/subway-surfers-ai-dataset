"""Render what the model sees and decides, for watching live.

Draws the captured frame alongside a panel of per-class confidence bars, the
chosen action, and whether a key fired. Uses OpenCV for a live window; the
pure-drawing helper returns a BGR image so replay.py can reuse it to build a
video without opening a window.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import config
from controller import Decision

# Colours (BGR) per action for the bars / highlight.
_ACTION_COLOR = {
    "UP": (80, 220, 80),
    "DOWN": (80, 180, 240),
    "LEFT": (240, 160, 60),
    "RIGHT": (200, 100, 240),
    "NONE": (170, 170, 170),
}


def render_frame(frame_rgb: np.ndarray, decision: Decision, fps: Optional[float] = None) -> np.ndarray:
    """Return a BGR image visualising the frame + decision. No window needed."""
    import cv2

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ih, iw = bgr.shape[:2]
    # Letterbox into a fixed box (even dims — h264 needs even w/h, and a
    # constant size is required for video encoding across frames).
    box_h = int(config.OVERLAY_GAME_HEIGHT)
    box_w = int(config.OVERLAY_GAME_WIDTH)
    scale = min(box_w / max(iw, 1), box_h / max(ih, 1))
    rw, rh = max(2, int(iw * scale)), max(2, int(ih * scale))
    resized = cv2.resize(bgr, (rw, rh), interpolation=cv2.INTER_AREA)
    game = np.full((box_h, box_w, 3), 12, dtype=np.uint8)
    y0, x0 = (box_h - rh) // 2, (box_w - rw) // 2
    game[y0:y0 + rh, x0:x0 + rw] = resized
    gh, gw = box_h, box_w

    panel_w = int(config.OVERLAY_PANEL_WIDTH)
    panel = np.full((gh, panel_w, 3), 24, dtype=np.uint8)

    pred = decision.prediction
    y = 40
    cv2.putText(panel, "MODEL VIEW", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (240, 240, 240), 2, cv2.LINE_AA)
    y += 40

    # Chosen action, highlighted.
    color = _ACTION_COLOR.get(decision.action, (200, 200, 200))
    label = f"{decision.action}" + ("  [KEY]" if decision.fired else "")
    cv2.putText(panel, label, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    y += 20
    cv2.putText(panel, decision.reason, (16, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (160, 160, 160), 1, cv2.LINE_AA)
    y += 44

    # Per-class confidence bars.
    classes = list(pred.probs.keys())
    bar_x = 120
    bar_w_max = panel_w - bar_x - 16
    for cls in classes:
        p = pred.probs[cls]
        c = _ACTION_COLOR.get(cls, (180, 180, 180))
        cv2.putText(panel, cls, (16, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w_max, y + 16), (60, 60, 60), -1)
        cv2.rectangle(panel, (bar_x, y), (bar_x + int(bar_w_max * p), y + 16), c, -1)
        cv2.putText(panel, f"{p*100:4.0f}%", (bar_x + bar_w_max - 44, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)
        y += 26

    if fps is not None:
        cv2.putText(panel, f"{fps:4.1f} fps", (16, gh - 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (140, 140, 140), 1, cv2.LINE_AA)

    return np.hstack([game, panel])


class LiveOverlay:
    """A live OpenCV window. Call show() each frame; returns False to quit."""

    def __init__(self, window_name: str = config.OVERLAY_WINDOW_NAME):
        import cv2

        self._cv2 = cv2
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    def show(self, frame_rgb: np.ndarray, decision: Decision, fps: Optional[float] = None) -> bool:
        img = render_frame(frame_rgb, decision, fps)
        self._cv2.imshow(self.window_name, img)
        # 'q' or ESC to quit.
        key = self._cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)

    def close(self) -> None:
        try:
            self._cv2.destroyWindow(self.window_name)
        except Exception:
            pass
