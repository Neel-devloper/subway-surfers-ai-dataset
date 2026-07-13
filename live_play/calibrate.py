"""Interactive calibration: tell the controller where the game is on screen.

Run this once (after positioning the Subway Surfers browser window):

    python live_play/calibrate.py

It grabs a full screenshot, lets you drag a box around the *game canvas only*
(exclude the browser toolbar, ads, and page margins so the captured frame
resembles the training screenshots), and saves the rectangle to
region_config.json. Live play then reads that file.

Uses OpenCV's selectROI for the drag-box UI, so it needs a display — this is
meant to be run on your own machine, not in a headless container.
"""

from __future__ import annotations

import sys

import numpy as np

from capture import Region, save_region


def main() -> None:
    try:
        import cv2
        import mss
    except Exception as e:  # pragma: no cover
        print(f"Calibration needs opencv-python + mss on a machine with a display: {e}")
        sys.exit(1)

    with mss.mss() as sct:
        # Monitor 0 is the "all monitors" virtual screen in mss.
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        shot = np.asarray(sct.grab(monitor))[:, :, :3]  # BGR

    print(
        "A window will open with your screen. Drag a box around the GAME CANVAS "
        "only (not the whole browser), then press ENTER or SPACE. Press 'c' to cancel."
    )
    # Downscale for the selector if the screen is large, then scale coords back.
    h, w = shot.shape[:2]
    scale = min(1.0, 1280.0 / max(w, 1), 800.0 / max(h, 1))
    disp = cv2.resize(shot, (int(w * scale), int(h * scale))) if scale < 1.0 else shot

    roi = cv2.selectROI("Calibrate: drag a box around the game canvas", disp,
                        showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, rw, rh = roi
    if rw == 0 or rh == 0:
        print("No region selected — nothing saved.")
        sys.exit(1)

    # Scale back to true screen coordinates, offsetting by the monitor origin.
    region = Region(
        left=int(monitor["left"] + x / scale),
        top=int(monitor["top"] + y / scale),
        width=int(rw / scale),
        height=int(rh / scale),
    )
    save_region(region)
    print(f"Saved capture region: {region.__dict__}")
    print("You can now run: python live_play/main.py")


if __name__ == "__main__":
    main()
