"""Input backends: how a decided action becomes a keypress.

Two implementations behind a common interface:
  - PyAutoGuiBackend: real key presses to the focused window (used on your
    machine during live play).
  - DryRunBackend: records intended presses without touching any keyboard
    (used for the in-container replay validation, and as a safe default).

The abstract interface lets the controller and the rest of the loop stay
identical whether or not a real keyboard is being driven.
"""

from __future__ import annotations

from typing import List, Tuple


class InputBackend:
    def press(self, key: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class DryRunBackend(InputBackend):
    """Logs presses instead of sending them. Nothing touches the OS."""

    def __init__(self) -> None:
        self.history: List[Tuple[float, str]] = []

    def press(self, key: str) -> None:
        import time

        self.history.append((time.time(), key))


class PyAutoGuiBackend(InputBackend):
    """Sends real arrow-key presses to whatever window is focused.

    pyautogui is imported lazily so this module can be imported in a headless
    container (where pyautogui would fail to init without a display).
    """

    def __init__(self) -> None:
        import pyautogui  # noqa: F401 - imported for its side effects/availability

        self._pyautogui = pyautogui
        # Don't let pyautogui insert its own delay between calls; we manage
        # timing ourselves via cooldowns.
        pyautogui.PAUSE = 0.0
        # Keep the fail-safe (slam mouse to a corner to abort) enabled — a
        # sensible panic button while the model is driving your keyboard.
        pyautogui.FAILSAFE = True

    def press(self, key: str) -> None:
        self._pyautogui.press(key)
