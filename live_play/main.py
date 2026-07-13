"""Live play: watch the trained model play Subway Surfers in your browser.

Run this on your own machine (it drives your real screen + keyboard):

    python live_play/main.py                 # opens Chrome, then plays
    python live_play/main.py --no-open        # you open the game yourself
    python live_play/main.py --dry-run        # decide + show, but send no keys
    python live_play/main.py --url <game-url>

Flow each loop:
    capture game region -> model.predict -> confidence gate + cooldown
    -> press arrow key -> draw overlay.

Prerequisites:
    1. pip install -r live_play/requirements.txt
    2. python live_play/calibrate.py   (once, to mark the game region)

Safety: while it's running, slam your mouse into a screen corner to trigger
pyautogui's fail-safe and abort, or press 'q'/ESC in the overlay window.
"""

from __future__ import annotations

import argparse
import time

import config
from browser import open_chrome_new_window
from capture import ScreenCapture, load_region
from controller import PlayController
from controls import DryRunBackend, PyAutoGuiBackend
from overlay import LiveOverlay
from predictor import ActionPredictor


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch the model play Subway Surfers in a browser.")
    ap.add_argument("--url", default=config.DEFAULT_GAME_URL, help="game URL to open")
    ap.add_argument("--no-open", action="store_true", help="don't auto-open Chrome")
    ap.add_argument("--dry-run", action="store_true", help="decide + overlay but send no keys")
    ap.add_argument("--checkpoint", default=config.CHECKPOINT_PATH)
    ap.add_argument("--threshold", type=float, default=config.CONFIDENCE_THRESHOLD)
    ap.add_argument("--fps", type=float, default=config.TARGET_FPS)
    ap.add_argument("--start-delay", type=float, default=6.0,
                    help="seconds to wait after opening Chrome before playing")
    args = ap.parse_args()

    region = load_region()
    if region is None:
        raise SystemExit(
            "No capture region set. Run `python live_play/calibrate.py` first "
            "(after opening the game so you can box the canvas)."
        )

    predictor = ActionPredictor(args.checkpoint)
    backend = DryRunBackend() if args.dry_run else PyAutoGuiBackend()
    controller = PlayController(predictor, backend, confidence_threshold=args.threshold)

    if not args.no_open:
        print(f"Opening Chrome at {args.url} ...")
        open_chrome_new_window(args.url)
        print(f"Waiting {args.start_delay:.0f}s — switch to the game, start a run, "
              "and click the game so it has keyboard focus.")
        time.sleep(args.start_delay)

    cap = ScreenCapture(region)
    overlay = LiveOverlay()
    frame_interval = 1.0 / max(args.fps, 1.0)

    print("Playing. Press 'q'/ESC in the overlay to stop; mouse-to-corner aborts.")
    last_t = time.time()
    fps_est = args.fps
    try:
        while True:
            loop_start = time.time()
            frame = cap.grab()
            decision = controller.step(frame)

            dt = loop_start - last_t
            last_t = loop_start
            if dt > 0:
                fps_est = 0.9 * fps_est + 0.1 * (1.0 / dt)

            if not overlay.show(frame, decision, fps_est):
                break

            sleep = frame_interval - (time.time() - loop_start)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        cap.close()
        overlay.close()
        print("Stopped.")


if __name__ == "__main__":
    main()
