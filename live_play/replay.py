"""Dry-run the full decision pipeline over recorded screenshots.

This is the offline validator: it feeds real dataset screenshots through the
exact same `PlayController.step()` used in live play (with a DryRun input
backend so nothing is pressed), renders the model-view overlay for each frame,
and writes an annotated video plus a summary.

It exercises everything except the two links that are inherently local to your
machine — real screen capture and real key injection — so it's the strongest
end-to-end check that can run in a headless container.

    python live_play/replay.py --source medium --limit 300 --out demo.mp4
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import Counter
from typing import List, Tuple

import numpy as np
from PIL import Image

import config
from controller import PlayController
from controls import DryRunBackend
from overlay import render_frame
from predictor import ActionPredictor

SCREENS_ROOT = os.path.join(config.REPO_ROOT, "screen_collector", "screens")
_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{3})")


def collect_stream(source: str) -> List[Tuple[str, str]]:
    """Collect (path, true_label) ordered by capture timestamp, to approximate
    a real gameplay stream. `source` is a speed bucket (slow/medium/fast/
    legacy) or 'all'. Augmentation duplicates (_bright/_dark) are skipped so
    each moment appears once."""
    buckets = [source] if source != "all" else ["slow", "medium", "fast", "legacy"]
    items: List[Tuple[str, str, str]] = []  # (timestamp, path, label)
    for bucket in buckets:
        for label in config.ACTION_TO_KEY.keys() | {"NONE"}:
            for p in glob.glob(os.path.join(SCREENS_ROOT, bucket, label, "*.png")):
                base = os.path.basename(p)
                if "_bright" in base or "_dark" in base:
                    continue
                m = _TS_RE.search(base)
                items.append((m.group(1) if m else base, p, label))
    items.sort(key=lambda t: t[0])
    return [(p, lbl) for _, p, lbl in items]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="medium",
                    help="speed bucket (slow/medium/fast/legacy) or 'all'")
    ap.add_argument("--limit", type=int, default=300, help="max frames to replay")
    ap.add_argument("--out", default=os.path.join(config.REPO_ROOT, "live_play", "demo.mp4"))
    ap.add_argument("--checkpoint", default=config.CHECKPOINT_PATH)
    ap.add_argument("--threshold", type=float, default=config.CONFIDENCE_THRESHOLD)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    stream = collect_stream(args.source)
    if not stream:
        raise SystemExit(f"No frames found for source={args.source}")
    if args.limit:
        stream = stream[: args.limit]
    print(f"Replaying {len(stream)} frames from '{args.source}' -> {args.out}")

    predictor = ActionPredictor(args.checkpoint)
    controller = PlayController(predictor, DryRunBackend(), confidence_threshold=args.threshold)

    import cv2
    import imageio

    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=None)

    fired = Counter()
    pred_counts = Counter()
    # Agreement: does the model's raw top-1 match the folder label? (Useful
    # signal even though NONE frames legitimately shouldn't trigger a key.)
    agree = 0
    # Simulated clock so cooldowns behave as they would at the replay fps.
    t = 0.0
    dt = 1.0 / args.fps
    for i, (path, true_label) in enumerate(stream):
        frame = np.asarray(Image.open(path).convert("RGB"))
        decision = controller.step(frame, now=t)
        t += dt

        pred_counts[decision.prediction.action] += 1
        if decision.fired:
            fired[decision.action] += 1
        if decision.prediction.action == true_label:
            agree += 1

        img = render_frame(frame, decision, fps=float(args.fps))
        # Annotate the ground-truth label from the dataset folder.
        cv2.putText(img, f"dataset label: {true_label}", (16, img.shape[0] - 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 220, 120), 2, cv2.LINE_AA)
        writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(stream)} frames")

    writer.close()

    print("\n=== replay summary ===")
    print(f"frames:            {len(stream)}")
    print(f"top-1 == label:    {agree}/{len(stream)}  ({100*agree/len(stream):.1f}%)")
    print(f"model predictions: {dict(pred_counts)}")
    print(f"keys fired:        {dict(fired)}  (total {sum(fired.values())})")
    print(f"video written:     {args.out}")


if __name__ == "__main__":
    main()
