# Live play — watch the model play Subway Surfers in your browser

This package runs the trained model (`models/subway_surfers_cnn.pth`) as a live
controller: it captures the game area of your screen, predicts a swipe every
frame, and presses the matching arrow key in your browser — while showing you
exactly what it's "seeing" and deciding.

```
  screen region ──capture──▶ model ──▶ confidence gate + cooldown ──▶ arrow key
        ▲                                                                  │
        └───────────────────────── you watch the overlay ◀────────────────┘
```

## Important: where this runs

This runs **on your own computer**, not in the cloud. It needs to see your
screen and drive your keyboard, so it uses local libraries (`mss` for capture,
`pyautogui` for key presses, `opencv` for the overlay window). It was developed
and pipeline-tested in a headless container, but the live loop must be launched
by you locally.

## Setup (once)

```bash
pip install -r live_play/requirements.txt
```

## Run it

1. **Open the game and position its window.** Either let the controller open
   Chrome for you (default) or open it yourself.

2. **Calibrate the capture region** (once, or whenever you move/resize the
   window). Start a run so the game canvas is visible, then:

   ```bash
   python live_play/calibrate.py
   ```

   Drag a box around **just the game canvas** (exclude the browser toolbar,
   page margins, and any ads) and press ENTER. This is saved to
   `region_config.json`. Boxing tightly around the canvas makes the captured
   frame resemble the training screenshots, which matters for accuracy.

3. **Play:**

   ```bash
   python live_play/main.py                 # opens Chrome in a new window, then plays
   python live_play/main.py --no-open       # you already have the game open
   python live_play/main.py --dry-run       # decide + show overlay, but send NO keys
   python live_play/main.py --url <game-url> # use a different Subway Surfers URL
   ```

   After Chrome opens, switch to the game, start a run, and **click the game so
   it has keyboard focus** — key presses go to the focused window.

## Non-interactive / automated runs

For running without the drag-box calibrator or a human at the keyboard (e.g.
driven by an assistant like Claude Cowork), skip `calibrate.py` and pass the
region + stop conditions directly:

```bash
# 1. Sanity-check calibration + predictions on ONE frame (writes probe.png):
python live_play/probe.py --region 120,80,540,960 --out probe.png

# 2. Watch decisions for 30s WITHOUT sending keys, and record it:
python live_play/main.py --no-open --dry-run --region 120,80,540,960 \
    --duration 30 --record dryrun.mp4 --headless

# 3. Actually play for 60s and record:
python live_play/main.py --no-open --region 120,80,540,960 \
    --duration 60 --record play.mp4
```

Useful flags: `--region L,T,W,H` (skip calibration), `--duration SEC` /
`--max-steps N` (auto-stop), `--record out.mp4` (annotated recording),
`--headless` (no live window — just record + a printed summary). `probe.png` /
the recordings are what you share back to review how it's doing.

### macOS permissions (important)

On macOS the app running Python (Terminal, iTerm, or the assistant's shell)
must be granted, in **System Settings → Privacy & Security**:

- **Screen Recording** — otherwise `mss` captures a black/empty frame.
- **Accessibility** — otherwise `pyautogui` key presses are silently dropped.

Grant both, then restart the terminal app. Run `probe.py` first: if the saved
image is black, Screen Recording isn't granted.

### Controls / safety

- Press **`q`** or **ESC** in the overlay window to stop.
- **Slam your mouse into any screen corner** to trigger `pyautogui`'s fail-safe
  and abort instantly — a good panic button while the model drives the keyboard.
- Start with `--dry-run` to watch its decisions without it actually pressing
  keys, until you trust the calibration and predictions.

## How the model plays

Each captured frame is resized to the model's 128×72 input and classified into
`UP / DOWN / LEFT / RIGHT / NONE`. A key is sent only when:

1. the top class isn't `NONE`, **and**
2. its confidence ≥ `CONFIDENCE_THRESHOLD` (default 0.60), **and**
3. that action isn't still in its cooldown window.

`NONE` means "keep running — do nothing." All thresholds, cooldowns, the target
FPS, the key mapping, and the default game URL live in `config.py`.

## Improving the model: record browser data + fine-tune

The model was trained on **mobile** screenshots, so it plays the browser build
below its potential. The fix is to record real browser gameplay and fine-tune.

**1. Record labeled data — you play, it watches your keystrokes:**

```bash
python live_play/record_browser.py --region 0,171,466,585 --duration 600
```

Play normally for ~10 min (arrow keys or WASD). Each frame near a key press is
saved labeled with that action into `screen_collector/screens/browser/<ACTION>/`;
a subsample of the rest are saved as `NONE`. A raw keystroke log + manifest are
written per session (under `browser/_sessions/`) so the data can be re-labeled
offline. On macOS this needs **Input Monitoring** permission (to read your keys)
on top of Screen Recording. Aim for a few hundred+ action frames across several
runs.

The recorder **verifies the scene is live before starting**: it samples the
region and aborts if the content is static or featureless (a desktop wallpaper,
blank area, or paused game — one early session silently recorded 2,488 frames
of wallpaper because the game wasn't visible in the captured region). It also
saves a `preview.png` per session — **always check it shows the game** — skips
frames while the scene is static mid-run (menus/crash screens), and reports the
static percentage in the final summary. `--force` overrides the startup check.

Tip for better data: also record a few runs where you deliberately let the model
play and take over only to correct its mistakes — that captures the recovery
states pure expert play never visits.

**2. Fine-tune, warm-starting from the current model:**

```bash
python training/train.py \
    --init-from models/subway_surfers_cnn.pth \
    --out models/subway_surfers_cnn_browser.pth \
    --browser-oversample 4 --epochs 12 --lr 3e-4
```

`--browser-oversample N` replicates the (initially small) browser frames so they
carry weight against the ~4,400 mobile frames; a lower `--lr` and fewer epochs
avoid wiping out what the model already knows. The original checkpoint is left
untouched.

**3. Play with the fine-tuned model:**

```bash
python live_play/main.py --no-open --region 0,171,466,585 \
    --checkpoint models/subway_surfers_cnn_browser.pth --duration 60 --record play.mp4
```

## Offline replay (no browser needed)

`replay.py` feeds recorded dataset screenshots through the **exact same
decision code** used live (with key presses stubbed out) and renders an
annotated video. It's how the pipeline was validated end-to-end without a real
game attached:

```bash
python live_play/replay.py --source medium --limit 250 --out demo.mp4
```

Note the accuracy it reports is on frames the model largely trained on, so it's
optimistic versus real play — held-out validation accuracy was ~79% (see
`training/README.md`). Replay's job is to prove the plumbing, and to eyeball
the model's behaviour frame by frame.

## The one honest caveat about browser play

The model was trained on **mobile-app** screenshots. The browser build looks
somewhat different (aspect ratio, HUD, surrounding page). The model may
transfer well or may need help. If it plays poorly:

- Re-calibrate so the box hugs the canvas tightly (biggest lever).
- Try lowering/raising `CONFIDENCE_THRESHOLD` in `config.py`.
- If it's still off, collect a small set of browser frames labeled by the right
  action and fine-tune — the training pipeline in `training/` already supports
  this; a few minutes of browser data usually closes the gap.

## Files

| File | Role |
|------|------|
| `config.py` | all tunables (thresholds, cooldowns, URL, overlay size) |
| `predictor.py` | loads the `.pth`, frame → action + confidence |
| `controller.py` | the shared decision logic (gate + cooldown) |
| `controls.py` | input backends (`pyautogui` real, dry-run stub) |
| `capture.py` | `mss` screen-region capture + region config I/O |
| `calibrate.py` | interactive region selector |
| `browser.py` | opens Chrome in a new window, cross-platform |
| `overlay.py` | the live "model view" window / frame renderer |
| `main.py` | live-play entrypoint (the loop) |
| `probe.py` | one-shot: capture a frame, show/save the model's prediction |
| `replay.py` | offline validator over recorded frames → video |
| `test_controller.py` | deterministic tests for the decision logic |
