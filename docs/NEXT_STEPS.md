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

## D. Perception (sub-project 3) — status 2026-09-26

Branch `perception` (not merged). Spec `docs/superpowers/specs/2026-09-26-perception-design.md`,
plan `docs/superpowers/plans/2026-09-26-perception.md`, ledger
`.superpowers/sdd/2026-09-26-perception/progress.md`. 174 offline tests pass; independent review
done and its 5 fixable Important findings fixed.

Still to do (needs the user's OK or the hardware):
1. **Install Ultralytics + PyTorch (~1 GB)** and **download `yolo11n-pose.pt` (~6 MB)** → plan Task 7
   Steps 5–7; then add an MPS-vs-CPU keypoint agreement check (review I-7; default to CPU if MPS is off).
2. **Public data** (plan Task 9 Steps 3–5): 2–3 Pexels indoor videos, MOT17 (~5.5 GB) → run
   `tools/eval_perception.py`, record fps / p95 latency / ID switches here.
3. **Camera** (hardware): ✅ `pytest -m camera` passes (2026-09-26): found by USB ID at OpenCV index 1200,
   delivers 1280×720 as requested (native default 1920×1080). Still to do: `tools/perception_view.py`; set
   `rotate` if the image is sideways; calibrate `hfov_deg` with the neck (`geometry.hfov_from_shift`).
   **Camera code must run from macOS Terminal.app**: processes launched by the Claude app are silently
   denied camera access (no prompt, no System Settings entry). Terminal was granted access on 2026-09-26.
   OpenCV gives up ~0.5 s after requesting permission, so a first-ever run needs a retry loop to let the
   macOS prompt appear.
4. **For the behavior spec:** publish timestamped neck poses so perception can (a) interpolate the pose
   to frame capture time and (b) compensate the tracker for head rotation (review I-4/I-6).
5. 14 deferred minor review items are listed in the ledger (`Final: minor (deferred)` lines).

### Live results on the head camera (2026-09-26, Terminal.app)
- Image upright (`rotate: 0`). Live view works; people boxed and labelled.
- **Camera is the fps ceiling: ~18.8 fps at 1280×720**, regardless of requested fps/MJPG (AVFoundation
  ignores both). Pipeline keeps up (view shows 18–19 fps). The spec's ≥ 25 fps target is not reachable
  with this camera; try brighter light (webcams lengthen exposure in dim light) or 640×480 if more is needed.
- Detector: 7 ms/frame on MPS in a Claude-launched process on a still 720p image, but ~31 ms/frame in the
  Terminal.app camera loop (≈ CPU speed). Not limiting (fits in the 53 ms frame period) — investigate later
  (camera thread contention or power management).
- IDs change when a person leaves the frame for > 1 s — by design (re-identification deferred).
- Still to report: facing/away label, approaching/leaving, distance reading at a known distance.
