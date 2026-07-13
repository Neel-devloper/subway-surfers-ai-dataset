"""Grab frame(s) from the capture region and show what the model predicts.

Run this BEFORE a live session to verify two things at once:
  1. the capture region is framed correctly (the saved PNG should show just the
     game canvas, cropped like the training screenshots), and
  2. the model's predictions on real browser frames look sane (this is the cheap
     check for the mobile-vs-browser domain-shift question).

    python live_play/probe.py --region 120,80,540,960 --out probe.png
    python live_play/probe.py --region 120,80,540,960 --frames 5 --interval 0.5

It saves an annotated overlay image (game frame + per-class confidence bars) and
prints the prediction(s) to stdout. Needs a display/screen to capture.
"""

from __future__ import annotations

import argparse
import time

from capture import Region, ScreenCapture, load_region
from controller import PlayController
from controls import DryRunBackend
from predictor import ActionPredictor
import config


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe: one-shot model prediction on the capture region.")
    ap.add_argument("--region", default=None, help="'left,top,width,height' (else uses region_config.json)")
    ap.add_argument("--out", default="probe.png", help="annotated image to write")
    ap.add_argument("--frames", type=int, default=1, help="how many frames to sample")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between sampled frames")
    ap.add_argument("--checkpoint", default=config.CHECKPOINT_PATH)
    ap.add_argument("--threshold", type=float, default=config.CONFIDENCE_THRESHOLD)
    args = ap.parse_args()

    region = Region.parse(args.region) if args.region else load_region()
    if region is None:
        raise SystemExit("No region. Pass --region 'left,top,width,height' or run calibrate.py.")

    predictor = ActionPredictor(args.checkpoint)
    controller = PlayController(predictor, DryRunBackend(), confidence_threshold=args.threshold)
    cap = ScreenCapture(region)

    import cv2
    from overlay import render_frame

    last_img = None
    try:
        for i in range(max(args.frames, 1)):
            frame = cap.grab()
            decision = controller.step(frame)
            p = decision.prediction
            bars = "  ".join(f"{k}:{v*100:4.1f}%" for k, v in p.probs.items())
            print(f"frame {i+1}: top={p.action} ({p.confidence*100:.1f}%)  "
                  f"would_fire={'yes' if decision.fired else 'no'}  [{bars}]")
            last_img = render_frame(frame, decision)
            if i + 1 < args.frames:
                time.sleep(args.interval)
    finally:
        cap.close()

    if last_img is not None:
        cv2.imwrite(args.out, last_img)
        print(f"\nSaved annotated frame -> {args.out}")
        print("Check: does the left panel show just the game canvas, cropped like a "
              "phone screenshot? If not, adjust --region.")


if __name__ == "__main__":
    main()
