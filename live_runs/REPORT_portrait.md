# Subway Surfers CNN — Portrait Live-Play Report (session 2)

## What changed since last time
- `git pull` brought in `09b8db5` (Improve browser play: aspect-correct crop + static-scene suppression): the predictor now center-crops each frame to the model's training aspect before resizing, and the controller now suppresses actions on a near-static frame (menu/paused/crashed), which was the source of the spurious "DOWN" spam on the crash screen last time.
- This run additionally rendered the game itself in true portrait, rather than relying on the crop alone.

## Getting to portrait
Chrome's own DevTools device toolbar (Cmd+Opt+I / Cmd+Shift+M) turned out to be unreachable by automation — browser-chrome-level keyboard shortcuts aren't deliverable through either the OS-level automation tool (browsers are intentionally click/type-restricted there) or the page-level extension (DevTools isn't part of the page). Two things stood in for it:

1. Resized the actual Chrome window to a phone-sized portrait rectangle (430×950).
2. Poki's game player has its own in-page "fullscreen" toggle (a button inside `#game-container`) that makes the game iframe fill the viewport responsively instead of staying pinned at a fixed 640×360 landscape size. Triggering that button (found and clicked via the DOM, since it isn't shown in the narrow mobile page layout) gave a genuinely portrait, edge-to-edge game canvas — same practical effect as the device toolbar for our purposes, confirmed visually against `probe_portrait.png`.

One more snag worth naming: bringing the right browser window to the real, OS-visible foreground was unreliable through the normal "activate app" call — it kept raising an unrelated Chrome window with dozens of other tabs. Switching the active tab explicitly (rather than just acting on it via the extension) is what actually made our tab visible on the real screen, which is what matters for screen capture.

## Calibration
Final region: `--region 0,171,466,585`. Verified with both `probe.py`'s annotated output and a raw unscaled `mss` grab (`probe_portrait.png`) — clean phone-shaped frame, no black bars, no browser chrome.

## Dry run (30s, no keys) — `live_runs/dryrun_portrait.mp4`
420 frames, model predicted DOWN on essentially all of them — but only **1** key "would fire." The character died in the few seconds between clicking Resume and the recording actually starting (same idle-crash issue as last session, inherent to a dry run on a fast-paced game with no one at the wheel), so nearly the whole clip is the crash/menu screen. The difference from last time: static-scene suppression correctly recognized it as "static scene (menu/paused)" and held off, instead of confidently misfiring DOWN over and over. That's the fix working as intended.

## Live runs (real keys)
**Run 1 — `live_runs/play_portrait.mp4`** (60s): 884 frames, predictions `{DOWN: 698, UP: 101, LEFT: 53, NONE: 28, RIGHT: 4}`, actually fired `{UP: 16, LEFT: 7, DOWN: 1}` — 24 real key presses. Live play lasted roughly the first ~15 seconds before a crash, reaching **score 43**; the model correctly fired **UP at 91% confidence** on a train blocking the lane directly ahead (see the frame grabs) — a clean, decisive obstacle read, not the borderline 30–40% split-votes seen in the landscape session. After the crash, static-scene suppression correctly held the rest of the 60s silent instead of spamming keys at the menu.

**Run 2 — `live_runs/play_portrait_run2.mp4`** (45s): 662 frames, fired `{UP: 19, LEFT: 10, DOWN: 1}` — 30 real key presses, reaching **score 24**. More key presses than run 1 but a lower score — the model is clearly reactive, but not consistently accurate; some fraction of its confident calls are still wrong or late.

## Does portrait + the new crop make it more decisive?
Yes, noticeably. Compared to the first (landscape) session:
- Confidence at obstacle moments moved from hovering right around the 0.60 threshold (frequently just under it, causing no action when one was needed) to clearly committing above 90% on a correct dodge in run 1.
- The crash-screen false-positive problem (confidently pressing DOWN dozens of times on a static menu) is gone — replaced by correct suppression.
- Scores (43, 24) are in the same range as the best landscape run (43) rather than a dramatic leap — so this is a real, measurable improvement in decision quality at the moments that matter, not a wholesale fix. The model still crashes within 15–20 seconds fairly reliably, and run-to-run variance is high (43 vs 24), meaning it isn't yet reliably good — just less hesitant and less prone to false alarms.

## Files (all in `~/subway-surfers-ai-dataset/live_runs/`, copied here too)
- `probe_portrait.png` — raw unscaled capture used to verify the crop is clean
- `probe_portrait_annotated.png` — probe.py's annotated model-view output
- `dryrun_portrait.mp4` — 30s, no keys
- `play_portrait.mp4` — 60s live control, run 1 (score 43, includes the 91%-confidence correct UP dodge)
- `play_portrait_run2.mp4` — 45s live control, run 2 (score 24)

Last session's landscape artifacts (`probe.png`, `probe2.png`, `dryrun.mp4`, `play.mp4`, etc.) are untouched in the repo root.
