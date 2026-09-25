# Perception — Design

**Date:** 2026-09-26
**Scope:** Sub-project 3 (perception) of the MoreNeck robot.

## Goal and context

The end goal is a head that looks at people. It decomposes into:

| # | Sub-project | Status |
|---|---|---|
| 1–2 | Neck: servo I/O, safety, motion (`neck/`) | Done, merged |
| **3** | **Perception: camera → tracked people with cues** | **This spec** |
| 4 | Behavior: who to look at, following, lost-face, idle scan, lean-in | Later spec |
| 5 | Eyes: Mac ↔ ESP32 link and eye expressions | Later spec |

Perception knows nothing about behavior or eyes. Behavior consumes only the `Snapshot` contract
below and, optionally, supplies the neck pose.

**Setting:** room / demo table; several people at 1–4 m, often walking past or side-on. A roaming
robot is a possible future, so perception must be portable Python (no macOS-only camera or ML
APIs) with a swappable detector.

**Done means:** from the live camera *or* a video file, perception publishes at ≥ 25 fps a
snapshot of tracked people, each with a stable ID, head angle relative to the camera, distance
estimate, facing score and motion class; a live debug view shows them; an eval tool reports ID
switches, fps and latency on a video with optional ground truth.

## Hardware

- **Compute:** Apple M4 Pro, 12 cores, 24 GB, macOS 26.5.
- **Camera:** USB UVC "Webcam", Sonix Technology, USB `0C45:6366`, USB 2.0, mounted under the head
  (moves with it). Supported resolutions/frame rates not yet known. macOS also exposes
  "MacBook Pro Camera" and an iPhone Continuity camera, so the camera is opened **by name**.

## Architecture

```
perception/
  config.yaml     camera name/index fallback, resolution, rotation, hfov_deg, model, thresholds
  config.py       load/validate config
  sources.py      FrameSource interface; CameraSource(name), VideoFileSource(path)
  detector.py     PersonDetector interface; YoloPoseDetector
  tracker.py      Tracker: detections -> tracks with stable IDs
  geometry.py     pixel <-> camera angle (pinhole model from hfov + resolution)
  features.py     per-track cues: head point/angle, distance, facing, motion
  pipeline.py     Perception: source -> detector -> tracker -> features, background thread
tools/
  perception_view.py   live window: boxes, IDs, keypoints, cue labels, fps, latency
  eval_perception.py   run on a video (+ optional MOT-format ground truth), report metrics
data/                  git-ignored: downloaded datasets and (later) recorded clips
```

**Dependency rule:** behavior imports only `perception.pipeline.Perception` and the `Snapshot`
/ `Person` types. The detector model is used only inside `detector.py`.

### Output contract

```python
@dataclass(frozen=True)
class Person:
    id: int
    head_px: tuple[float, float]           # image coordinates
    head_deg: tuple[float, float]          # (yaw, pitch) relative to camera axis; +yaw = robot's left, +pitch = up
    world_yaw_deg: float | None            # neck turn + head yaw, when a neck pose provider is given
    distance_m: float | None
    distance_source: str | None            # "shoulders" | "body" | "head"
    facing_score: float                    # 0..1
    facing: bool
    motion: str                            # "approaching" | "leaving" | "passing" | "still" | "unknown"
    confidence: float
    visible: bool                          # False while coasting through an occlusion
    first_seen: float
    last_seen: float

@dataclass(frozen=True)
class Snapshot:
    timestamp: float                       # frame capture time (monotonic seconds)
    frame_size: tuple[int, int]
    people: tuple[Person, ...]
    source_ok: bool
    error: str | None
```

Future cues (speaking from a microphone, re-identification, recognition) are added as new
optional `Person` fields without changing existing ones.

### Sources (`sources.py`)

- `FrameSource`: `open()`, `read() -> Frame | None`, `close()`; `Frame = (image, capture_time)`.
- `CameraSource(name, index_fallback, width, height, rotate)`: finds the device index by name
  (via `cv2-enumerate-cameras`; falls back to the configured index), opens it with OpenCV, and
  reads in its own thread keeping **only the latest frame** (stale frames are dropped). Requests
  1280×720, accepts what the camera gives, applies the configured rotation (0/90/180/270).
- `VideoFileSource(path, realtime=False)`: yields frames with timestamps from the file's frame
  rate; `realtime=True` paces playback for the live view.

### Detector (`detector.py`)

- `PersonDetector.detect(image) -> list[Detection]`,
  `Detection = box (x1, y1, x2, y2), score, keypoints: dict[name -> (x, y, conf)]` for `nose`,
  `left_eye`, `right_eye`, `left_ear`, `right_ear`, `left_shoulder`, `right_shoulder`
  (image coordinates, COCO keypoint names).
- `YoloPoseDetector`: Ultralytics YOLO11n-pose, inference size 640, device `mps` when available
  else `cpu`, person score threshold 0.4 (config). Only this class imports `ultralytics`.
- Licence note: Ultralytics is AGPL-3.0 — fine for personal/research use; commercial use needs a
  licence or a swap to another backend (e.g. MediaPipe or an Apache-licensed pose model) behind
  the same interface. MediaPipe is the documented alternative; not built now.

### Tracker (`tracker.py`)

- Constant-velocity prediction per track; association by cost `1 − IoU(predicted box, detection)`
  with optimal assignment (`scipy.optimize.linear_sum_assignment`); pairs with IoU < 0.2 are not
  matched.
- New tracks are **tentative** until matched in 3 consecutive frames; tentative tracks are not
  reported.
- A confirmed track with no match **coasts** on its prediction (`visible=False`) for up to
  `max_coast_s` = 1.0 s, then is deleted. A person reappearing after that gets a new ID.
- IDs are integers, never reused within a run.
- Independent of the detector; tested with synthetic detections.

### Geometry (`geometry.py`)

- `f_px = (width / 2) / tan(hfov / 2)`; vertical uses the same `f_px` (square pixels).
- `yaw = atan((cx − x) / f_px)` (+ = robot's left, i.e. image left), `pitch = atan((cy − y) / f_px)`
  (+ = up). Inverse provided for drawing.
- `hfov_deg` starts at **60** (assumed) in config. Calibration procedure (in bring-up, with the
  neck): turn by a known angle Δ with a static scene; measure the horizontal pixel shift s of a
  fixed feature at the image centre; `hfov = 2·atan((width/2) · tan(Δ) / s)`.

### Features (`features.py`)

- **Head point:** mean of confident (conf ≥ 0.5) nose/eye/ear keypoints; fallback: box top-centre
  at 10% of box height. `head_deg` from geometry.
- **Distance:** first available of, in priority order:
  1. `shoulders`: shoulder width px vs 0.40 m, only if `facing_score ≥ 0.6` and both shoulders
     confident;
  2. `body`: box height vs 1.70 m, only if the box touches neither the top nor the bottom edge;
  3. `head`: eye–ear spread vs 0.15 m.
  `distance = assumed_size · f_px / measured_px`, smoothed per track (EMA, α = 0.3). Expected
  accuracy ±30%; used mainly for trends and a rough "near" threshold.
- **Facing score (0..1):** `0.5 · (both eyes confident) + 0.5 · symmetry`, where symmetry is
  `1 − |d_left − d_right| / (d_left + d_right)` for the horizontal distances from the nose to the
  left/right ear (or eye when ears are missing); 0 if the nose is not confident. `facing` =
  score ≥ 0.6 (config; tuned on recorded clips).
- **Motion** (over a 1 s window, per track): `approaching` if distance rate < −0.3 m/s,
  `leaving` if > +0.3 m/s, else `passing` if |world (or camera, without a pose provider) yaw rate|
  > 15 °/s, else `still`; `unknown` until 0.5 s of history.
- **Neck pose provider (optional):** `pose_provider() -> (turn_deg, nod_deg) | None`, called once
  per frame; `world_yaw_deg = turn_deg + head_yaw`. Without it, `world_yaw_deg` is `None` and
  motion uses camera yaw (fine for fixed-camera video).

### Pipeline (`pipeline.py`)

- `Perception(source, detector, config, pose_provider=None, on_snapshot=None)`;
  `start()`, `latest() -> Snapshot`, `stop()`; context manager.
- Background thread: read frame → detect → track → features → publish `Snapshot` (thread-safe
  swap) → `on_snapshot` callback.
- **Failures:** `read()` returning `None` for > 0.5 s, or a detector exception, publishes a
  snapshot with `source_ok=False`, `people=()`, `error` set, and logs once per failure episode;
  the loop keeps retrying (camera re-open every 2 s).
- **Staleness:** consumers treat a snapshot older than 0.3 s (by `timestamp`) as "nobody in view".
- **No frames are ever written to disk** by the pipeline.

### Tools

- `perception_view.py [--video PATH]`: live OpenCV window (main thread) drawing boxes, IDs,
  keypoints, head point, `distance / facing / motion` labels, fps and capture→publish latency.
- `eval_perception.py VIDEO [--gt MOT_GT_FILE]`: runs the pipeline as fast as possible on a file;
  reports fps, mean/p95 latency, number of tracks; with ground truth (MOT format), ID switches,
  MOTA and IDF1 via `motmetrics`.
- Clip recording (`record_clip.py`) is **out of scope** here; it is added with the hardware, and
  only records on an explicit command into `data/clips/` with consent of the people filmed.

## Data for development (before own recordings)

| Data | Use | Licence |
|---|---|---|
| Synthetic detections (in tests) | Tracker, features, motion | n/a |
| 2–3 free stock videos of people indoors at 1–4 m (e.g. Pexels) | Live view, smoke tests | Pexels licence |
| MOT17, 1–2 sequences kept (~5.5 GB download) | Tracking metrics with ground-truth IDs | CC BY-NC-SA 3.0 (non-commercial) |

All downloads go to git-ignored `data/` and are only fetched with the user's explicit OK
(file, source, size stated). Model weights (`yolo11n-pose.pt`, ~6 MB) are downloaded the same way.
These datasets don't match the robot's viewpoint; facing threshold, distance constants and FOV
are re-tuned on recorded clips once the hardware is back.

## Configuration defaults (`perception/config.yaml`)

`camera_name: Webcam`, `camera_index_fallback: 0`, `width: 1280`, `height: 720`, `rotate: 0`,
`hfov_deg: 60`, `model: yolo11n-pose.pt`, `imgsz: 640`, `person_score_min: 0.4`,
`keypoint_conf_min: 0.5`, `track_iou_min: 0.2`, `track_confirm_frames: 3`, `max_coast_s: 1.0`,
`shoulder_width_m: 0.40`, `body_height_m: 1.70`, `head_width_m: 0.15`, `distance_ema_alpha: 0.3`,
`facing_threshold: 0.6`, `motion_window_s: 1.0`, `approach_rate_mps: 0.3`,
`passing_rate_dps: 15`, `source_timeout_s: 0.5`, `stale_after_s: 0.3`.

## Testing

**Offline (default `pytest`):**
- `geometry`: pixel↔angle round-trip; centre → (0, 0); image-left → +yaw; top → +pitch.
- `features`: head point from keypoints and box fallback; each distance source and its gating
  (e.g. side-on skips shoulders); frontal vs profile facing scores; motion classes from synthetic
  tracks, with and without a pose provider (a tracked person centred in the image while the neck
  turns is `passing`).
- `tracker`: two people crossing keep their IDs; a 0.5 s occlusion keeps the ID (`visible=False`
  meanwhile); a 2 s occlusion yields a new ID; a 1–2-frame false detection is never reported; IDs
  never reused.
- `pipeline`: with a fake source and fake detector — snapshots published, `latest()` thread-safe,
  source failure → `source_ok=False` then recovery, detector exception handled, nothing written to
  disk.
- `sources`: `VideoFileSource` on a tiny generated video (frames, timestamps, end of file).

**Model (`pytest -m model`, needs weights):** 2–3 sample images → expected person counts and
plausible keypoints.

**Eval (manual):** `eval_perception.py` on a MOT17 sequence and stock videos — ID switches,
fps ≥ 25, p95 latency < 60 ms on the M4 Pro. Reported, not gated, until own clips exist.

**Camera (`pytest -m camera`, hardware):** "Webcam" found by name, frames at the configured size,
latest-frame behaviour (no backlog).

## Project setup

- New package `perception/` in the MoreNeck repo, added to `pyproject.toml` packages.
- Dependencies: `opencv-python`, `numpy`, `scipy`, `ultralytics` (pulls in PyTorch),
  `cv2-enumerate-cameras`; dev: `motmetrics`.
- `data/` added to `.gitignore`.

## Out of scope

Behavior (target selection, following, idle, lean-in), eye displays, recognition/re-identification,
speaking detection, clip recording, running on a robot computer other than the Mac.
