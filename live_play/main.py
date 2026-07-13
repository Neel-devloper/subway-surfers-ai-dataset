"""Live play: watch the trained model play Subway Surfers in your browser.

Run this on your own machine (it drives your real screen + keyboard):

    python live_play/main.py                      # opens Chrome, then plays
    python live_play/main.py --no-open            # you open the game yourself
    python live_play/main.py --dry-run            # decide + show, but send no keys
    python live_play/main.py --region L,T,W,H     # skip interactive calibration
    python live_play/main.py --duration 60        # auto-stop after 60s
    python live_play/main.py --record run.mp4     # save an annotated recording
    python live_play/main.py --headless           # no live window (just record/log)

Flow each loop:
    capture game region -> model.predict -> confidence gate + cooldown
    -> press arrow key -> draw overlay.

Prerequisites:
    1. pip install -r live_play/requirements.txt
    2. a capture region: either `python live_play/calibrate.py` once, or pass
       --region "left,top,width,height".

Safety: while it's running, slam your mouse into a screen corner to trigger
pyautogui's fail-safe and abort, or press 'q'/ESC in the overlay window.
"""

from __future__ import annotations

import argparse
import time

import config
from browser import open_chrome_new_window
from capture import Region, ScreenCapture, load_region
from controller import PlayController
from controls import DryRunBackend, PyAutoGuiBackend
from predictor import ActionPredictor


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch the model play Subway Surfers in a browser.")
    ap.add_argument("--url", default=config.DEFAULT_GAME_URL, help="game URL to open")
    ap.add_argument("--no-open", action="store_true", help="don't auto-open Chrome")
    ap.add_argument("--dry-run", action="store_true", help="decide + overlay but send no keys")
    ap.add_argument("--region", default=None,
                    help="capture region 'left,top,width,height' (skips calibrate.py)")
    ap.add_argument("--checkpoint", default=config.CHECKPOINT_PATH)
    ap.add_argument("--threshold", type=float, default=config.CONFIDENCE_THRESHOLD)
    ap.add_argument("--fps", type=float, default=config.TARGET_FPS)
    ap.add_argument("--duration", type=float, default=None,
                    help="auto-stop after this many seconds")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="auto-stop after this many frames")
    ap.add_argument("--record", default=None, help="save an annotated .mp4 of the session")
    ap.add_argument("--headless", action="store_true",
                    help="don't open a live window (use with --record/--duration)")
    ap.add_argument("--start-delay", type=float, default=6.0,
                    help="seconds to wait after opening Chrome before playing")
    args = ap.parse_args()

    if args.region:
        region = Region.parse(args.region)
    else:
        region = load_region()
    if region is None:
        raise SystemExit(
            "No capture region set. Either pass --region 'left,top,width,height' "
            "or run `python live_play/calibrate.py` first (after opening the game)."
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

    overlay = None
    if not args.headless:
        from overlay import LiveOverlay
        overlay = LiveOverlay()

    recorder = None
    if args.record:
        import imageio
        recorder = imageio.get_writer(args.record, fps=int(max(args.fps, 1)), macro_block_size=None)

    frame_interval = 1.0 / max(args.fps, 1.0)
    from collections import Counter
    fired = Counter()
    predicted = Counter()

    print("Playing." + ("" if args.headless else " Press 'q'/ESC in the overlay to stop;")
          + " mouse-to-corner aborts.")
    t_start = time.time()
    last_t = t_start
    fps_est = args.fps
    steps = 0
    try:
        while True:
            loop_start = time.time()
            frame = cap.grab()
            decision = controller.step(frame)
            steps += 1
            predicted[decision.prediction.action] += 1
            if decision.fired:
                fired[decision.action] += 1

            dt = loop_start - last_t
            last_t = loop_start
            if dt > 0:
                fps_est = 0.9 * fps_est + 0.1 * (1.0 / dt)

            if overlay is not None:
                if not overlay.show(frame, decision, fps_est):
                    break
            if recorder is not None:
                import cv2
                from overlay import render_frame
                img = render_frame(frame, decision, fps_est)
                recorder.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if args.duration is not None and (time.time() - t_start) >= args.duration:
                break
            if args.max_steps is not None and steps >= args.max_steps:
                break

            sleep = frame_interval - (time.time() - loop_start)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        cap.close()
        if overlay is not None:
            overlay.close()
        if recorder is not None:
            recorder.close()
        elapsed = time.time() - t_start
        print("\n=== session summary ===")
        print(f"frames:            {steps}  ({steps / max(elapsed, 1e-6):.1f} fps over {elapsed:.1f}s)")
        print(f"model predictions: {dict(predicted)}")
        print(f"keys {'that WOULD fire' if args.dry_run else 'fired'}: "
              f"{dict(fired)}  (total {sum(fired.values())})")
        if args.record:
            print(f"recording:         {args.record}")
        print("Stopped.")


if __name__ == "__main__":
    main()
