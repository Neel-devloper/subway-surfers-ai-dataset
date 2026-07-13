# Subway Surfers CNN — Live Play Report

## Setup
- Cloned `Neel-devloper/subway-surfers-ai-dataset` to `~/subway-surfers-ai-dataset`, used the committed model `models/subway_surfers_cnn.pth` (not retrained).
- Python 3.13 venv, `pip install -r live_play/requirements.txt` — succeeded cleanly (torch 2.13, mss, pyautogui, opencv, imageio).
- macOS permissions: **Screen Recording** and **Accessibility** were already granted to the Python binary the venv uses (visible in System Settings → Privacy & Security), so no manual grant was needed. Confirmed empirically — captures were never black, and key presses did register in‑game.
- Game: Poki's Subway Surfers, played in the page's "expanded" (landscape, full-bleed) view.

## Capture region
Final region: `--region 0,78,1372,753`. One wrinkle worth flagging: my screenshot tool's preview image (1372×892) and the real screen mss actually captures (1470×956, per `mss`'s own monitor report) are *not* 1:1 — about a 7% difference, not a clean Retina 2× split. Feeding the preview's raw numbers straight into `--region` still landed correctly (verified with a direct unscaled `mss` grab — clean frame, no black bars, no letterboxing), but it's a good reminder that "guess, then verify with probe.py" is the right workflow rather than trusting either coordinate space blindly.

A more consequential early mistake: my first `probe.png` came out capturing the **desktop**, not the game, even though the game "looked" active through the browser-automation tool. That's because Chrome wasn't actually the frontmost OS window — clicking through the browser extension doesn't raise the real window. Once I explicitly brought Chrome to the front at the OS level, capture worked correctly (`probe2.png`).

## Dry run (30s, no keys) — `dryrun.mp4`
The character was running unattended before the timed capture began and crashed a few seconds in (no auto-restart in this harness), so most of the 30s window shows the static "run over" menu, not gameplay. In the genuine gameplay seconds captured, the model's decisions were plausible but often hesitant right at the moment they mattered — e.g. with a train dead ahead requiring a lane change, it split ~27%/19%/36% across LEFT/RIGHT/NONE instead of committing, meaning it likely would not have fired in time. On the crash/menu screen, it confidently (60–70%) predicted DOWN — a false-positive out-of-distribution failure, since the model has no "not gameplay" class. Harmless here (DOWN does nothing on a menu) but worth knowing about.

## Live run (60s, real keys) — `play.mp4`
First attempt (kept as `play_attempt1_mostlycrashed.mp4`) mostly recorded the crash screen again — the character died during the few seconds it takes to click Resume, focus the canvas, and spin up the Python process, before the recording window even started. Tightened the handoff (fired the script immediately after clicking Resume, no extra delay) and reran: this is the `play.mp4` you're getting.

This run had real, sustained gameplay: predictions were a healthy mix (LEFT 100, RIGHT 180, UP 30, NONE 99, DOWN 466 across 875 frames), and it actually fired 30 real key presses (20 LEFT, 8 RIGHT, 2 DOWN) over roughly the first ~18–20 seconds of live play — visibly dodging trains left/right. Score reached **43** before crashing, versus 13–15 on the earlier uncontrolled/crashed attempts — so the model is providing real, if modest, obstacle avoidance. After the crash, the remaining ~40s of the fixed 60s window is (again) the static menu screen with spurious DOWN firings — harmless no-ops, but padding that makes the raw prediction tally look worse than the live-play portion actually was.

## Verdict: how well did it play, and why
It played *something* — clearly reactive to trains ahead (dodging left/right, occasional jump), and did measurably better than no control. But it's far from good: runs ended in well under 20 seconds each time, and per the repo's own `training/README.md`, held-out validation accuracy is only ~79% on the model's *native* mobile-screenshot format.

Classifying the shortfall:
- **Not permissions/focus** — capture wasn't black, keys did register and visibly affected play (score went from 13–15 idle to 43 controlled).
- **Partly session/timing overhead** — this harness has no crash-detection or auto-restart, so any run-ending collision burns the rest of the fixed `--duration` window on a non-gameplay screen. That's an orchestration gap, not a model problem, but it's the main reason both recordings look worse in aggregate than the live-play segments actually were.
- **Mostly domain shift** — the model was trained on portrait mobile-app screenshots; this is a landscape browser embed with a different aspect ratio, HUD, and surrounding page chrome (exactly the caveat the repo's own README calls out). That mismatch shows up as borderline, split-vote confidence right at decision-critical moments (obstacle dead ahead) rather than as clearly wrong predictions — the model "sees" something but isn't sure what to do about it as often as it should be.

## Files (all in the repo root, copied here too)
- `probe.png` — first probe attempt (accidentally captured the desktop — see note above)
- `probe2.png` — corrected probe, real game frame + prediction
- `dryrun.mp4` — 30s, no keys sent
- `play.mp4` — 60s live control, the good run (score 43)
- `play_attempt1_mostlycrashed.mp4` — 60s live control, first attempt (mostly crash-screen due to setup lag)
- sample frame PNGs (`play2_*.png`, `sample_*.png`, etc.) — individual annotated frames pulled from the videos for inspection
