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
OVERLAY_SCALE = 3  # upscale factor for the small captured frame in the overlay
