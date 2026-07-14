"""Record labeled browser gameplay — (frame, key-pressed) pairs for fine-tuning.

You play Subway Surfers in the browser; this watches the game region and your
arrow-key presses and saves labeled frames into
`screen_collector/screens/browser/<ACTION>/`, which `training/train.py` already
picks up. Fine-tuning on this data is what closes the mobile→browser gap.

Labeling: a frame captured within `--label-window` seconds of an arrow-key press
is labeled with that action (the decisive moment); a random `--none-keep`
fraction of the remaining frames are saved as NONE. Every keystroke (with exact
time) and every saved frame is also logged to a per-session `events.csv` /
`manifest.csv`, so the raw data can be re-labeled offline later without replaying.

    python live_play/record_browser.py --region 0,171,466,585 --duration 600

macOS permissions (System Settings → Privacy & Security): the terminal/python
needs **Screen Recording** (to capture) and **Input Monitoring** (for pynput to
read your keystrokes globally while the browser is focused).
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import Counter, deque
from datetime import datetime
from typing import Deque, List, Optional, Tuple

import numpy as np

import config
from capture import Region, ScreenCapture, load_region

ARROW_TO_ACTION = {"up": "UP", "down": "DOWN", "left": "LEFT", "right": "RIGHT"}
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "NONE"]


def nearest_action(
    frame_t: float, presses: List[Tuple[float, str]], window: float
) -> Optional[str]:
    """Return the action of the press closest in time to frame_t within
    +/- window seconds, or None if no press is that close."""
    best_action: Optional[str] = None
    best_dt = window
    for pt, action in presses:
        dt = abs(frame_t - pt)
        if dt <= best_dt:
            best_dt = dt
            best_action = action
    return best_action


class KeyWatcher:
    """Global arrow-key listener (pynput). Records (time, ACTION) on each press.

    Imported lazily so this module stays importable without a display / the
    Input Monitoring permission.
    """

    def __init__(self) -> None:
        from pynput import keyboard

        self._keyboard = keyboard
        self.presses: List[Tuple[float, str]] = []
        self._new: Deque[Tuple[float, str]] = deque()
        self.stop_requested = False
        self._listener = keyboard.Listener(on_press=self._on_press)

    def _on_press(self, key) -> None:
        from pynput import keyboard

        if key == keyboard.Key.esc:
            self.stop_requested = True
            return
        name = getattr(key, "name", None)
        action = ARROW_TO_ACTION.get(name or "")
        if action:
            t = time.time()
            self.presses.append((t, action))
            self._new.append((t, action))

    def drain_new(self) -> List[Tuple[float, str]]:
        out = []
        while self._new:
            out.append(self._new.popleft())
        return out

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()


def main() -> None:
    ap = argparse.ArgumentParser(description="Record labeled browser gameplay for fine-tuning.")
    ap.add_argument("--region", default=None, help="'left,top,width,height' (else region_config.json)")
    ap.add_argument("--out", default=os.path.join(config.REPO_ROOT, "screen_collector", "screens", "browser"),
                    help="dataset root to write <ACTION>/*.png into")
    ap.add_argument("--fps", type=float, default=config.TARGET_FPS)
    ap.add_argument("--duration", type=float, default=None, help="auto-stop after N seconds")
    ap.add_argument("--label-window", type=float, default=0.30,
                    help="seconds around a keypress that a frame inherits its action")
    ap.add_argument("--none-keep", type=float, default=0.15,
                    help="fraction of unlabeled (NONE) frames to keep, to limit imbalance")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    region = Region.parse(args.region) if args.region else load_region()
    if region is None:
        raise SystemExit("No region. Pass --region 'left,top,width,height' or run calibrate.py.")

    rng = np.random.default_rng(args.seed)
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    for a in ACTIONS:
        os.makedirs(os.path.join(args.out, a), exist_ok=True)
    sess_dir = os.path.join(args.out, "_sessions", session_id)
    os.makedirs(sess_dir, exist_ok=True)
    events_path = os.path.join(sess_dir, "events.csv")
    manifest_path = os.path.join(sess_dir, "manifest.csv")

    from PIL import Image

    watcher = KeyWatcher()
    watcher.start()
    cap = ScreenCapture(region)

    print(f"Recording session {session_id}. Play the game now.")
    print("Press ESC (with this listener active) to stop early; Ctrl-C also works.")
    if args.duration:
        print(f"Will auto-stop after {args.duration:.0f}s.")

    # Deferred-finalize buffer: hold frames until they're older than the label
    # window, so a press just AFTER a frame can still label it.
    buffer: Deque[Tuple[float, np.ndarray]] = deque()
    counts: Counter = Counter()
    frame_interval = 1.0 / max(args.fps, 1.0)
    t_start = time.time()
    idx = 0

    events_f = open(events_path, "w", newline="")
    manifest_f = open(manifest_path, "w", newline="")
    events_w = csv.writer(events_f); events_w.writerow(["time", "action"])
    manifest_w = csv.writer(manifest_f); manifest_w.writerow(["time", "action", "path"])
    logged_presses = 0

    def finalize(entry_t: float, entry_frame: np.ndarray) -> None:
        nonlocal idx
        label = nearest_action(entry_t, watcher.presses, args.label_window)
        if label is None:
            if rng.random() >= args.none_keep:
                return
            label = "NONE"
        fname = f"{session_id}_{idx:06d}.png"
        path = os.path.join(args.out, label, fname)
        Image.fromarray(entry_frame).save(path)
        manifest_w.writerow([f"{entry_t:.3f}", label, os.path.relpath(path, config.REPO_ROOT)])
        counts[label] += 1
        idx += 1

    try:
        while True:
            loop_start = time.time()
            frame = cap.grab()
            buffer.append((loop_start, frame))

            for pt, action in watcher.drain_new():
                events_w.writerow([f"{pt:.3f}", action])
                logged_presses += 1

            # Finalize frames older than the label window (future presses seen).
            cutoff = loop_start - args.label_window
            while buffer and buffer[0][0] <= cutoff:
                bt, bf = buffer.popleft()
                finalize(bt, bf)

            if watcher.stop_requested:
                break
            if args.duration is not None and (loop_start - t_start) >= args.duration:
                break

            sleep = frame_interval - (time.time() - loop_start)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        # Flush whatever's left in the buffer.
        while buffer:
            bt, bf = buffer.popleft()
            finalize(bt, bf)
        cap.close()
        watcher.stop()
        events_f.close()
        manifest_f.close()

        elapsed = time.time() - t_start
        print("\n=== recording summary ===")
        print(f"session:        {session_id}  ({elapsed:.0f}s)")
        print(f"key presses:    {logged_presses}")
        print(f"frames saved:   {sum(counts.values())}  -> {args.out}")
        print(f"per class:      {dict(counts)}")
        print(f"events log:     {events_path}")
        print(f"manifest:       {manifest_path}")
        if sum(v for k, v in counts.items() if k != 'NONE') < 40:
            print("\nNote: very few action frames captured. Check Input Monitoring "
                  "permission (keystrokes) and that you were pressing arrow keys.")


if __name__ == "__main__":
    main()
