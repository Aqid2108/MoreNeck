# MoreNeck — Next Steps (handoff, 2026-09-25)

State at handoff: neck control layer (pieces 1–2) merged to `master` (`f1d4005`), 105 offline
tests passing. Calibrated, current-limited and verified on the robot: `look()` moves all three
joints the right way, and `rest()` lowers the head and holds it (`rested`) without drooping.

Spec: `docs/superpowers/specs/2026-09-25-neck-control-layer-design.md`
Plan: `docs/superpowers/plans/2026-09-25-neck-control-layer.md` (Task 10 partially done)

## Operating notes

- Start: `.venv/bin/python tools/neck_shell.py`, answer `y` at the prompt *before* typing commands.
- After any fault or `rest()`: `neck.clear_fault()` to resume.
- **Ctrl-D leaves the head held with servos powered** (`limp_after_rest: false`). Cut servo
  power to make it limp.
- Every shell session logs each tick to `logs/shell-*.csv` (file name = save time).

## A. Unfinished neck bring-up (plan Task 10, steps 6–7)

1. **Fault drills from a raised pose** (not neutral):
   - Jam: `neck.look(nod=20)`, then `neck.look(turn=20)` and gently block the turn → expect a
     SOFT fault and hold; `neck.clear_fault()`.
   - SOFT → hold 1 s → rest → `rested`: raise, then
     `neck.cfg = dataclasses.replace(orig, voltage_warn_below=7.8, voltage_soft_below=7.7)`.
   - Ctrl-D mid-move → head rests and stays held.
2. **Rest from several poses** (`look(lean=8, nod=10)`, `look(nod=30, turn=30)`,
   `look(turn=-30)`), `neck.rest()` each → `rested`, no droop.
3. **Current-limit trips at move start**: nod read ~600 mA when starting to lift in two drills,
   but never above 182 mA in the full `log_current` run. Possibly a side effect of lean having
   fallen forward before those starts. Watch the logs; re-run `tools/log_current.py` if it recurs.

## B. Open issues

- **Turn servo voltage dips to 6.5 V at idle** (nod/lean stay at 7.4 V) → reseat or replace the
  turn servo's cable/connectors. At < 6.5 V for 3 ticks this becomes a SOFT fault.
- **Permanent droop fix**: limp is only stable with nod fully on its back stop (~3147) and lean on
  its back stop (~806). Re-run `tools/calibrate.py` (it now sets nod and lean rest to where they
  settle limp), then test `limp_after_rest: true` with drill B. Until then keep holding.
- **Temperature readings glitch**: ~17 single-sample spikes (up to 82 °C) per 40 s. They're
  filtered (3-sample debounce), but the source is unknown (sensor vs. bus).
- **Deferred review minors**: missing servo slows the loop to ~19 Hz; config validation gaps
  (KeyError/TypeError instead of ValueError, voltage threshold order, negative limits); startup
  enables torque one tick before the first safety check; failed `clear_fault()` clears the latch;
  a second Ctrl-C during shutdown; `Reading.error` unused; no raw-bytes parser test.
- Swift toolchain mismatch on this Mac (Command Line Tools vs SDK) — only matters if Swift is
  needed; fix by reinstalling/updating the Command Line Tools.

## C. Face tracking (piece 3) — start with a brainstorm

Camera identified: **"Webcam"**, Sonix Technology, USB `0C45:6366`, UVC, USB 2.0, mounted under
the head (moves with it). macOS also exposes the MacBook camera and an iPhone Continuity camera,
so code must open "Webcam" by name, not index 0. Resolutions/frame rates not yet listed (no
ffmpeg/OpenCV installed); macOS will ask for camera permission on first open.

Questions for the brainstorm:
1. **Camera modes and field of view**: install OpenCV in the venv, list modes, grab a frame,
   confirm it's the head camera and which way is up. FOV is needed to turn pixels into degrees.
2. **Detector**: OpenCV Haar (fast, weak), OpenCV YuNet DNN (fast, good), or MediaPipe
   (good, extra dependency). Target frame rate and latency on this Mac.
3. **Control approach**: because the camera moves with the head, keep the face centred —
   pixel error → small `look()` corrections each frame — vs. estimating absolute angles.
4. **Boundary between piece 3 (perception) and piece 4 (behavior)**: which face to follow when
   there are several, what to do when the face is lost (hold, return to neutral, idle scan),
   and how lean is used (e.g. lean in when someone approaches).
5. **Speed limits**: nod is capped at 30 °/s and lean at 20 °/s (gravity load); tracking must
   stay within them or it will trip tracking-error faults.
6. Are the ESP32 eye displays part of this phase?

Suggested order when resuming: A.1–A.2 (15 min at the robot) → B turn cable → C brainstorm.
