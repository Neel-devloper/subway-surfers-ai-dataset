"""Open Google Chrome in a NEW window at the game URL, cross-platform.

This is best-effort convenience: it shells out to the OS to launch Chrome in a
fresh window. If Chrome can't be found it prints the URL so you can open it
yourself. It does not (and can not) reach beyond the machine it runs on.
"""

from __future__ import annotations

import shutil
import subprocess
import sys


def open_chrome_new_window(url: str) -> bool:
    """Launch Chrome in a new window at `url`. Returns True if a launch was
    attempted, False if Chrome wasn't found (URL is printed as a fallback)."""
    platform = sys.platform

    try:
        if platform == "darwin":
            # -n opens a new instance; --args passes flags to Chrome.
            subprocess.Popen(
                ["open", "-na", "Google Chrome", "--args", "--new-window", url]
            )
            return True

        if platform.startswith("win"):
            # `start` needs a shell; empty title arg avoids quoting pitfalls.
            subprocess.Popen(f'start chrome --new-window "{url}"', shell=True)
            return True

        # Linux / other unix
        for exe in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
            if shutil.which(exe):
                subprocess.Popen([exe, "--new-window", url])
                return True

    except Exception as e:
        print(f"Could not launch Chrome automatically ({e}).")

    print(f"Open Chrome yourself and go to: {url}")
    return False
