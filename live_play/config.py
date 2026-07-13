"""Tunables for the live-play controller.

Everything here is safe to edit without touching the other modules — region
calibration, timing, and the game URL all live in one place.
"""

from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

CHECKPOINT_PATH = os.path.join(REPO_ROOT, "models", "subway_surfers_cnn.pth")
REGION_CONFIG_PATH = os.path.join(HERE, "region_config.json")

# Default game URL, used only if the user doesn't pass --url. Poki is a
# well-known, widely-used legitimate host for the browser build of Subway
# Surfers. If it doesn't load for you (region lock, layout change, a site you
# prefer instead), override it with --url <your-url>.
DEFAULT_GAME_URL = "https://poki.com/en/g/subway-surfers"

# Only act on a prediction if the model is at least this confident; otherwise
# treat the frame as NONE. Raise this if the model fires on frames it
# shouldn't; lower it if it feels sluggish to react.
CONFIDENCE_THRESHOLD = 0.60

# The model was trained on PORTRAIT (~9:16) mobile screenshots. A browser
# capture is often LANDSCAPE, and squishing that straight to the portrait model
# input distorts everything horizontally (the main cause of weak browser play).
# When True, the predictor first center-crops each frame to the model's aspect
# ratio, then resizes — so a wide capture contributes its centre column (where
# the lanes/character are) instead of a squished full width. This is a no-op for
# frames that already match the training aspect, so it never hurts native input.
CROP_TO_TRAINING_ASPECT = True

# Subway Surfers gameplay scrolls constantly, so a nearly-static frame means a
# menu / paused / "run over" screen — which is out-of-distribution for the model
# (it has no "not gameplay" class and tends to false-fire on it). When enabled,
# the controller suppresses actions while the scene is static. Threshold is mean
# absolute per-pixel difference (0-255) below which two frames count as "same".
SUPPRESS_STATIC_SCENE = True
STATIC_SCENE_DIFF_THRESHOLD = 2.0

# Minimum seconds between two key presses of the SAME action, so one
# confident frame doesn't fire the same swipe five times in a row.
ACTION_COOLDOWN_SEC = {
    "UP": 0.35,
    "DOWN": 0.35,
    "LEFT": 0.30,
    "RIGHT": 0.30,
}

# Target capture/inference loop rate. The model itself runs in low
# single-digit milliseconds on CPU, so this is capped by capture + display
# overhead, not inference.
TARGET_FPS = 15

# Key sent to the browser for each action label.
ACTION_TO_KEY = {
    "UP": "up",
    "DOWN": "down",
    "LEFT": "left",
    "RIGHT": "right",
}

# Overlay window
OVERLAY_WINDOW_NAME = "Subway Surfers — model view"
# The overlay letterboxes the captured game frame into a fixed box of this
# size (px), preserving aspect ratio, then shows a stats panel beside it. A
# fixed box (rather than a blind scale factor) keeps the window a sane size no
# matter how large or small the capture region is, and keeps every rendered
# frame identical in size (required for video encoding). ~9:16 phone aspect.
OVERLAY_GAME_HEIGHT = 560
OVERLAY_GAME_WIDTH = 316
OVERLAY_PANEL_WIDTH = 360
