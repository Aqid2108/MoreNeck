# Perception Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `perception/` package that turns camera or video-file frames into a published `Snapshot` of tracked people (stable ID, head angle, distance, facing, motion), plus a live debug view and an eval tool.

**Architecture:** A background pipeline reads the latest frame from a `FrameSource`, runs a swappable `PersonDetector` (YOLO11n-pose), associates detections into tracks with our own IoU tracker, derives per-person cues in `features.py`, and publishes an immutable `Snapshot`. Everything except the detector and the live camera is testable offline with synthetic data.

**Tech Stack:** Python 3.12, NumPy, SciPy, OpenCV (`opencv-python`), `cv2-enumerate-cameras`, PyYAML, Ultralytics (YOLO11n-pose, PyTorch), `motmetrics` (dev), pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-perception-design.md`

## Global Constraints

- Portable Python only: no macOS-only camera or ML APIs (OpenCV + Ultralytics only).
- Only `perception/detector.py` imports `ultralytics`/`torch`, and only inside `YoloPoseDetector`.
- **No frames are ever written to disk** by the pipeline.
- Downloads (PyTorch/Ultralytics install ~1 GB, model weights ~6 MB, stock videos, MOT17 ~5.5 GB) happen **only with the user's explicit OK**, stating file, source and size. Datasets and weights live in git-ignored `data/`.
- Sign conventions: `+yaw` = robot's left (image left), `+pitch` = up; `world_yaw_deg = neck_turn_deg + head_yaw`.
- Behavior (later) depends only on `perception.pipeline.Perception`, `perception.pipeline.people_or_empty`, and `perception.types.Person` / `Snapshot`.
- Camera: USB `0C45:6366` ("Webcam"); matched by USB ID, then name, then configured index.
- Tests marked `model` (need weights) and `camera` (need hardware) are skipped by default.

## Review Focus

1. **Camera delivers a different resolution than requested** (e.g. 640×480 instead of 1280×720) → angles and distances use the actual frame size, not the config. Test: Task 4 `test_angles_use_actual_frame_size_not_config`, Task 6 `test_frame_size_comes_from_the_frame`.
2. **A person seen from behind or with all keypoints low-confidence** → head point falls back to the box, facing 0, distance from body or `None`; nothing crashes. Test: Task 4 `test_person_from_behind_uses_box_fallbacks`.
3. **Detector raises mid-run** (e.g. an MPS/driver error) → snapshot `source_ok=False` with the error, loop keeps running and recovers. Test: Task 6 `test_detector_exception_publishes_error_and_continues`.
4. **Camera unplugged mid-run** → after 0.5 s a failure snapshot, reopen attempts every 2 s, recovery when frames return. Test: Task 6 `test_source_timeout_publishes_failure_and_recovers`, `test_reopens_source_while_failing`.
5. **Video file ends** → `VideoFileSource.read()` returns `None` and `eof` becomes `True`; the eval tool stops cleanly. Test: Task 5 `test_video_file_source_reads_frames_then_eof`.

Rulings carried from the spec: `Person` gains a `box` field (needed by the debug view and eval); `perception/types.py` holds `Detection`/`Person`/`Snapshot` so tracker/features tests never import Ultralytics; `camera_usb_id` config added (match by USB ID before name); `on_snapshot` receives `(snapshot, image)` so the debug view can draw on the frame; `people_or_empty()` implements the spec's staleness rule. Task 1 amends the spec with these.

---

## File Structure

```
pyproject.toml                    add perception package, deps, markers
.gitignore                        add data/
perception/
  __init__.py
  config.yaml, config.py          PerceptionConfig + load/validate
  types.py                        Detection, Person, Snapshot, KEYPOINT_NAMES
  geometry.py                     focal_px, pixel_to_angles, angles_to_pixel, hfov_from_shift
  tracker.py                      iou, Track, Tracker
  features.py                     head_point, facing_score, estimate_distance, classify_motion, FeatureExtractor
  sources.py                      Frame, VideoFileSource, CameraSource, find_camera, rotate_image
  pipeline.py                     Perception, people_or_empty
  detector.py                     detections_from_arrays, YoloPoseDetector
tools/
  perception_view.py              live debug window
  eval_perception.py              metrics on a video / MOT sequence
tests/
  conftest.py                     + PBASE, pcfg / make_pcfg fixtures
  test_perception_config.py, test_perception_geometry.py, test_perception_tracker.py,
  test_perception_features.py, test_perception_sources.py, test_perception_pipeline.py,
  test_perception_detector.py, test_perception_tools.py, test_perception_camera.py
```

(Test files are prefixed `test_perception_` because `tests/` is flat and already has `test_config.py`.)

---

### Task 1: Package setup, config and types

**Files:**
- Create: `perception/__init__.py`, `perception/config.yaml`, `perception/config.py`, `perception/types.py`, `tests/test_perception_config.py`
- Modify: `pyproject.toml`, `.gitignore`, `tests/conftest.py`, `docs/superpowers/specs/2026-09-26-perception-design.md`

**Interfaces:**
- Produces: `PerceptionConfig` (fields listed in Step 5), `config_from_dict(d) -> PerceptionConfig`, `load_config(path=DEFAULT_CONFIG)`, `DEFAULT_CONFIG`; `types.KEYPOINT_NAMES`, `Box`, `Detection(box, score, keypoints)`, `Person(...)`, `Snapshot(...)`; fixtures `pcfg`, `make_pcfg(overrides: dict)`.

- [ ] **Step 1: Update `pyproject.toml`**

Replace the `dependencies`, `[project.optional-dependencies]`, `[tool.setuptools]`, `[tool.setuptools.package-data]` and pytest `markers`/`addopts` with:

```toml
dependencies = [
  "st3215 @ git+https://github.com/Mickael-Roger/python-st3215@143cff5e74453a2fd24e7a3d78aeef644364768a",
  "pyserial>=3.5",
  "pyyaml>=6",
  "numpy>=1.26",
  "scipy>=1.11",
  "opencv-python>=4.9",
  "cv2-enumerate-cameras>=1.1",
]

[project.optional-dependencies]
dev = ["pytest>=8", "motmetrics>=1.4"]
vision = ["ultralytics>=8.3"]

[tool.setuptools]
packages = ["neck", "perception"]

[tool.setuptools.package-data]
neck = ["config.yaml"]
perception = ["config.yaml"]
```

and in `[tool.pytest.ini_options]`:

```toml
markers = [
  "hardware: needs the real robot on the bus (run only on request)",
  "model: needs the YOLO pose weights in data/models (run on request)",
  "camera: needs the head camera plugged in (run only on request)",
]
addopts = "-m 'not hardware and not model and not camera'"
```

Append `data/` to `.gitignore`.

- [ ] **Step 2: Install**

Run: `.venv/bin/pip install -q -e '.[dev]'`
Expected: exits 0 (installs NumPy, SciPy, OpenCV, cv2-enumerate-cameras, motmetrics; ~70 MB). This does **not** install Ultralytics/PyTorch (Task 7).

- [ ] **Step 3: Add perception fixtures to `tests/conftest.py`** (append)

```python
PBASE = {
    "camera_usb_id": "0C45:6366", "camera_name": "Webcam", "camera_index_fallback": 0,
    "width": 1280, "height": 720, "rotate": 0, "hfov_deg": 60.0,
    "model": "data/models/yolo11n-pose.pt", "imgsz": 640,
    "person_score_min": 0.4, "keypoint_conf_min": 0.5,
    "track_iou_min": 0.2, "track_confirm_frames": 3, "max_coast_s": 1.0,
    "shoulder_width_m": 0.40, "body_height_m": 1.70, "head_width_m": 0.15,
    "distance_ema_alpha": 0.3, "facing_threshold": 0.6,
    "motion_window_s": 1.0, "approach_rate_mps": 0.3, "passing_rate_dps": 15.0,
    "source_timeout_s": 0.5, "stale_after_s": 0.3, "reopen_every_s": 2.0,
}


def build_pcfg(overrides=None):
    from perception.config import config_from_dict

    d = copy.deepcopy(PBASE)
    d.update(overrides or {})
    return config_from_dict(d)


@pytest.fixture
def pcfg():
    return build_pcfg()


@pytest.fixture
def make_pcfg():
    return build_pcfg
```

- [ ] **Step 4: Write the failing tests**

`tests/test_perception_config.py`:

```python
import copy

import pytest

from perception.config import DEFAULT_CONFIG, config_from_dict, load_config
from perception.types import KEYPOINT_NAMES, Detection, Person, Snapshot
from tests.conftest import PBASE


def test_default_config_file_loads():
    c = load_config(DEFAULT_CONFIG)
    assert c.camera_usb_id == "0C45:6366"
    assert c.camera_name == "Webcam"
    assert c.hfov_deg == 60.0


@pytest.mark.parametrize("key,value,message", [
    ("rotate", 45, "rotate"),
    ("hfov_deg", 200, "hfov"),
    ("width", 0, "width"),
    ("facing_threshold", 1.5, "0..1"),
    ("track_confirm_frames", 0, "track_confirm_frames"),
    ("max_coast_s", 0, "positive"),
    ("camera_usb_id", "0c456366", "camera_usb_id"),
])
def test_validation_errors(key, value, message):
    d = copy.deepcopy(PBASE)
    d[key] = value
    with pytest.raises(ValueError, match=message):
        config_from_dict(d)


def test_missing_and_unknown_keys_are_reported():
    d = copy.deepcopy(PBASE)
    del d["hfov_deg"]
    with pytest.raises(ValueError, match="missing: hfov_deg"):
        config_from_dict(d)
    d = copy.deepcopy(PBASE)
    d["hfvo_deg"] = 60
    with pytest.raises(ValueError, match="unknown.*hfvo_deg"):
        config_from_dict(d)


def test_contract_types_are_immutable():
    det = Detection(box=(0, 0, 10, 20), score=0.9, keypoints={"nose": (5, 2, 0.9)})
    person = Person(id=1, box=(0, 0, 10, 20), head_px=(5, 2), head_deg=(1.0, 2.0), world_yaw_deg=None,
                    distance_m=2.0, distance_source="body", facing_score=0.8, facing=True,
                    motion="still", confidence=0.9, visible=True, first_seen=0.0, last_seen=1.0)
    snap = Snapshot(timestamp=1.0, frame_size=(1280, 720), people=(person,), source_ok=True)
    assert snap.error is None
    assert "nose" in KEYPOINT_NAMES and det.keypoints["nose"][2] == 0.9
    with pytest.raises(AttributeError):
        snap.people = ()
```

- [ ] **Step 5: Run to verify failure**

Run: `.venv/bin/pytest tests/test_perception_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'perception'`.

- [ ] **Step 6: Implement**

`perception/__init__.py`:

```python
"""MoreNeck perception: camera frames -> tracked people with cues."""
```

`perception/config.yaml`:

```yaml
# Perception configuration. See docs/superpowers/specs/2026-09-26-perception-design.md
camera_usb_id: "0C45:6366"     # head camera (Sonix "Webcam"); matched before the name
camera_name: Webcam
camera_index_fallback: 0
width: 1280
height: 720
rotate: 0                      # 0, 90, 180 or 270 degrees clockwise
hfov_deg: 60.0                 # assumed until calibrated with the neck
model: data/models/yolo11n-pose.pt
imgsz: 640
person_score_min: 0.4
keypoint_conf_min: 0.5
track_iou_min: 0.2
track_confirm_frames: 3
max_coast_s: 1.0
shoulder_width_m: 0.40
body_height_m: 1.70
head_width_m: 0.15
distance_ema_alpha: 0.3
facing_threshold: 0.6
motion_window_s: 1.0
approach_rate_mps: 0.3
passing_rate_dps: 15.0
source_timeout_s: 0.5
stale_after_s: 0.3
reopen_every_s: 2.0
```

`perception/config.py`:

```python
"""Perception configuration: loading and validation."""
from __future__ import annotations

import re
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True)
class PerceptionConfig:
    camera_usb_id: str
    camera_name: str
    camera_index_fallback: int
    width: int
    height: int
    rotate: int
    hfov_deg: float
    model: str
    imgsz: int
    person_score_min: float
    keypoint_conf_min: float
    track_iou_min: float
    track_confirm_frames: int
    max_coast_s: float
    shoulder_width_m: float
    body_height_m: float
    head_width_m: float
    distance_ema_alpha: float
    facing_threshold: float
    motion_window_s: float
    approach_rate_mps: float
    passing_rate_dps: float
    source_timeout_s: float
    stale_after_s: float
    reopen_every_s: float


def _validate(c: PerceptionConfig) -> None:
    checks = [
        (re.fullmatch(r"[0-9A-Fa-f]{4}:[0-9A-Fa-f]{4}", c.camera_usb_id) is not None,
         "camera_usb_id must look like 0C45:6366"),
        (c.width > 0 and c.height > 0, "width and height must be positive"),
        (c.rotate in (0, 90, 180, 270), "rotate must be 0, 90, 180 or 270"),
        (10 <= c.hfov_deg <= 170, "hfov_deg must be between 10 and 170"),
        (all(0 <= v <= 1 for v in (c.person_score_min, c.keypoint_conf_min, c.track_iou_min,
                                   c.distance_ema_alpha, c.facing_threshold)),
         "scores, thresholds and alpha must be within 0..1"),
        (c.track_confirm_frames >= 1, "track_confirm_frames must be at least 1"),
        (all(v > 0 for v in (c.max_coast_s, c.shoulder_width_m, c.body_height_m, c.head_width_m,
                             c.motion_window_s, c.approach_rate_mps, c.passing_rate_dps,
                             c.source_timeout_s, c.stale_after_s, c.reopen_every_s, c.imgsz)),
         "times, sizes and rates must be positive"),
    ]
    for ok, message in checks:
        if not ok:
            raise ValueError(message)


def config_from_dict(d: dict) -> PerceptionConfig:
    names = {f.name for f in fields(PerceptionConfig)}
    missing = sorted(names - set(d))
    if missing:
        raise ValueError(f"perception config missing: {', '.join(missing)}")
    unknown = sorted(set(d) - names)
    if unknown:
        raise ValueError(f"unknown perception config keys: {', '.join(unknown)}")
    cfg = PerceptionConfig(**{k: d[k] for k in names})
    _validate(cfg)
    return cfg


def load_config(path: Path | str = DEFAULT_CONFIG) -> PerceptionConfig:
    return config_from_dict(yaml.safe_load(Path(path).read_text()))
```

`perception/types.py`:

```python
"""Perception data contract. Behavior depends only on Person and Snapshot."""
from __future__ import annotations

from dataclasses import dataclass, field

KEYPOINT_NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear",
                  "left_shoulder", "right_shoulder")
Box = tuple[float, float, float, float]   # x1, y1, x2, y2 in image pixels


@dataclass(frozen=True)
class Detection:
    box: Box
    score: float
    keypoints: dict = field(default_factory=dict)   # name -> (x, y, confidence)


@dataclass(frozen=True)
class Person:
    id: int
    box: Box
    head_px: tuple[float, float]
    head_deg: tuple[float, float]       # (yaw, pitch) from the camera axis; +yaw left, +pitch up
    world_yaw_deg: float | None         # neck turn + head yaw, when a neck pose is known
    distance_m: float | None
    distance_source: str | None         # "shoulders" | "body" | "head"
    facing_score: float
    facing: bool
    motion: str                         # "approaching" | "leaving" | "passing" | "still" | "unknown"
    confidence: float
    visible: bool                       # False while coasting through an occlusion
    first_seen: float
    last_seen: float


@dataclass(frozen=True)
class Snapshot:
    timestamp: float                    # capture time of the frame (seconds)
    frame_size: tuple[int, int]         # (width, height)
    people: tuple[Person, ...]
    source_ok: bool
    error: str | None = None
```

- [ ] **Step 7: Amend the spec** — in `docs/superpowers/specs/2026-09-26-perception-design.md`:
  - In the `Person` contract add `box: tuple[float, float, float, float]   # x1, y1, x2, y2` after `id`.
  - Under "Sources", replace "finds the device index by name (via `cv2-enumerate-cameras`; falls back to the configured index)" with "finds the device by USB ID (`camera_usb_id`, `0C45:6366`), then by name, then the configured index (via `cv2-enumerate-cameras`, whose indices encode the backend)".
  - Under "Pipeline", replace `on_snapshot=None` with `on_snapshot=None` *(called as `on_snapshot(snapshot, image)`; image is `None` for failure snapshots)* and add: "`people_or_empty(snapshot, now, stale_after_s)` returns `()` for failed or stale snapshots."
  - Add `camera_usb_id: "0C45:6366"` and `reopen_every_s: 2.0` to the configuration defaults.

- [ ] **Step 8: Run the tests**

Run: `.venv/bin/pytest -q`
Expected: all pass (existing 105 + new).

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml .gitignore perception/__init__.py perception/config.yaml perception/config.py perception/types.py tests/conftest.py tests/test_perception_config.py docs/superpowers/specs/2026-09-26-perception-design.md
git commit -m "feat(perception): package, config and data contract"
```

---

### Task 2: Geometry

**Files:**
- Create: `perception/geometry.py`, `tests/test_perception_geometry.py`

**Interfaces:**
- Produces: `focal_px(width, hfov_deg) -> float`; `pixel_to_angles(x, y, width, height, hfov_deg) -> (yaw_deg, pitch_deg)`; `angles_to_pixel(yaw_deg, pitch_deg, width, height, hfov_deg) -> (x, y)`; `hfov_from_shift(width, turn_deg, shift_px) -> float`.

- [ ] **Step 1: Write the failing tests**

`tests/test_perception_geometry.py`:

```python
import math

import pytest

from perception.geometry import angles_to_pixel, focal_px, hfov_from_shift, pixel_to_angles


def test_focal_length_from_fov():
    assert focal_px(1280, 60.0) == pytest.approx(640 / math.tan(math.radians(30)))


def test_centre_is_straight_ahead():
    assert pixel_to_angles(640, 360, 1280, 720, 60.0) == pytest.approx((0.0, 0.0))


def test_image_left_is_positive_yaw_and_top_is_positive_pitch():
    yaw, pitch = pixel_to_angles(0, 0, 1280, 720, 60.0)
    assert yaw == pytest.approx(30.0)
    assert pitch > 0


def test_round_trip():
    x, y = angles_to_pixel(-12.5, 7.0, 1280, 720, 60.0)
    assert pixel_to_angles(x, y, 1280, 720, 60.0) == pytest.approx((-12.5, 7.0))


def test_hfov_recovered_from_a_known_turn():
    f = focal_px(1280, 62.0)
    shift = f * math.tan(math.radians(20.0))    # a centred feature after turning 20 degrees
    assert hfov_from_shift(1280, 20.0, shift) == pytest.approx(62.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_perception_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'perception.geometry'`.

- [ ] **Step 3: Implement `perception/geometry.py`**

```python
"""Pinhole camera geometry: pixels <-> angles from the camera axis.

+yaw = image left (the robot's left), +pitch = image up.
"""
from __future__ import annotations

import math


def focal_px(width: int, hfov_deg: float) -> float:
    return (width / 2) / math.tan(math.radians(hfov_deg) / 2)


def pixel_to_angles(x: float, y: float, width: int, height: int, hfov_deg: float) -> tuple[float, float]:
    f = focal_px(width, hfov_deg)
    yaw = math.degrees(math.atan((width / 2 - x) / f))
    pitch = math.degrees(math.atan((height / 2 - y) / f))
    return yaw, pitch


def angles_to_pixel(yaw_deg: float, pitch_deg: float, width: int, height: int,
                    hfov_deg: float) -> tuple[float, float]:
    f = focal_px(width, hfov_deg)
    return (width / 2 - f * math.tan(math.radians(yaw_deg)),
            height / 2 - f * math.tan(math.radians(pitch_deg)))


def hfov_from_shift(width: int, turn_deg: float, shift_px: float) -> float:
    """FOV from turning by turn_deg and seeing a centred feature move shift_px."""
    f = shift_px / math.tan(math.radians(turn_deg))
    return math.degrees(2 * math.atan((width / 2) / f))
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_perception_geometry.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add perception/geometry.py tests/test_perception_geometry.py
git commit -m "feat(perception): pinhole geometry and FOV calibration helper"
```

---

### Task 3: Tracker

**Files:**
- Create: `perception/tracker.py`, `tests/test_perception_tracker.py`

**Interfaces:**
- Consumes: `Detection` (Task 1).
- Produces: `iou(a, b) -> float`; `Track` (`id: int | None`, `detection: Detection`, `box: np.ndarray` current estimate, `last_box: np.ndarray` box of the last matched detection, `velocity`, `first_seen`, `last_seen`, `hits`, `visible`, property `confirmed`); `Tracker(iou_min=0.2, confirm_frames=3, max_coast_s=1.0, velocity_alpha=0.5)` with `update(detections: list[Detection], t: float) -> list[Track]` returning confirmed tracks only.

- [ ] **Step 1: Write the failing tests**

`tests/test_perception_tracker.py`:

```python
from perception.tracker import Tracker, iou
from perception.types import Detection

FPS = 30
DT = 1 / FPS


def det(x, y=100, w=100, h=300):
    return Detection(box=(x, y, x + w, y + h), score=0.9)


def ids(tracks):
    return sorted(t.id for t in tracks)


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == 50 / 150


def test_person_is_confirmed_after_three_frames():
    tr = Tracker()
    assert tr.update([det(100)], 0 * DT) == []
    assert tr.update([det(101)], 1 * DT) == []
    tracks = tr.update([det(102)], 2 * DT)
    assert len(tracks) == 1 and tracks[0].id == 1 and tracks[0].visible


def test_one_or_two_frame_flicker_is_never_reported():
    tr = Tracker()
    for i in range(2):
        assert tr.update([det(500)], i * DT) == []
    for i in range(2, 10):
        assert tr.update([], i * DT) == []


def test_two_people_crossing_keep_their_ids():
    tr = Tracker()
    first = None
    for i in range(80):
        a = det(100 + 10 * i, y=100)          # moving right
        b = det(900 - 10 * i, y=140)          # moving left, passes through a around i = 40
        tracks = tr.update([a, b], i * DT)
        if len(tracks) == 2 and first is None:
            first = {t.id: t.box[0] for t in tracks}
    assert first is not None
    right_mover = min(first, key=first.get)   # the one that started on the left
    final = {t.id: t.box[0] for t in tracks}
    assert final[right_mover] > 800           # still the same id, now on the right


def test_short_occlusion_keeps_id_and_coasts():
    tr = Tracker()
    for i in range(30):
        tracks = tr.update([det(100 + 5 * i)], i * DT)
    (before,) = tracks
    for i in range(30, 45):                   # 0.5 s hidden
        (coasting,) = tr.update([], i * DT)
        assert coasting.id == before.id and not coasting.visible
    (after,) = tr.update([det(100 + 5 * 45)], 45 * DT)
    assert after.id == before.id and after.visible


def test_long_occlusion_gets_a_new_id():
    tr = Tracker()
    for i in range(30):
        tracks = tr.update([det(100)], i * DT)
    old_id = tracks[0].id
    for i in range(30, 90):                   # 2 s hidden
        tracks = tr.update([], i * DT)
    assert tracks == []
    for i in range(90, 93):
        tracks = tr.update([det(100)], i * DT)
    assert len(tracks) == 1 and tracks[0].id != old_id


def test_ids_are_never_reused():
    tr = Tracker(max_coast_s=0.1)
    seen = []
    t = 0.0
    for _ in range(3):
        for _ in range(3):
            tracks = tr.update([det(100)], t)
            t += DT
        seen.append(tracks[0].id)
        for _ in range(10):
            tr.update([], t)
            t += DT
    assert seen == sorted(set(seen))
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_perception_tracker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'perception.tracker'`.

- [ ] **Step 3: Implement `perception/tracker.py`**

```python
"""Multi-person tracker: IoU association with constant-velocity prediction.

Independent of the detector, so it can be tested with synthetic detections.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from perception.types import Detection


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    detection: Detection
    box: np.ndarray                  # current estimate (predicted while coasting)
    last_box: np.ndarray             # box of the last matched detection
    first_seen: float
    last_seen: float                 # time of the last matched detection
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))   # box units per second
    hits: int = 1
    id: int | None = None            # assigned on confirmation
    visible: bool = True

    @property
    def confirmed(self) -> bool:
        return self.id is not None

    def predict(self, t: float) -> np.ndarray:
        return self.last_box + self.velocity * (t - self.last_seen)


class Tracker:
    def __init__(self, iou_min: float = 0.2, confirm_frames: int = 3, max_coast_s: float = 1.0,
                 velocity_alpha: float = 0.5):
        self.iou_min = iou_min
        self.confirm_frames = confirm_frames
        self.max_coast_s = max_coast_s
        self.velocity_alpha = velocity_alpha
        self._tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection], t: float) -> list[Track]:
        preds = [tr.predict(t) for tr in self._tracks]
        matches, unmatched_dets = self._associate(preds, detections)
        matched = set()
        for ti, di in matches:
            self._apply(self._tracks[ti], detections[di], t)
            matched.add(ti)
        survivors = []
        for ti, tr in enumerate(self._tracks):
            if ti in matched:
                survivors.append(tr)
            elif tr.confirmed and t - tr.last_seen <= self.max_coast_s:
                tr.visible = False
                tr.box = preds[ti]
                survivors.append(tr)
            # tentative tracks that miss a frame, and expired tracks, are dropped
        for di in unmatched_dets:
            d = detections[di]
            box = np.array(d.box, dtype=float)
            tr = Track(detection=d, box=box, last_box=box.copy(), first_seen=t, last_seen=t)
            self._maybe_confirm(tr)
            survivors.append(tr)
        self._tracks = survivors
        return [tr for tr in self._tracks if tr.confirmed]

    def _associate(self, preds, detections):
        if not preds or not detections:
            return [], list(range(len(detections)))
        cost = np.array([[1.0 - iou(p, d.box) for d in detections] for p in preds])
        rows, cols = linear_sum_assignment(cost)
        matches = [(r, c) for r, c in zip(rows, cols) if 1.0 - cost[r, c] >= self.iou_min]
        used = {c for _, c in matches}
        return matches, [i for i in range(len(detections)) if i not in used]

    def _apply(self, tr: Track, d: Detection, t: float) -> None:
        box = np.array(d.box, dtype=float)
        dt = t - tr.last_seen
        if dt > 0:
            v = (box - tr.last_box) / dt
            tr.velocity = self.velocity_alpha * v + (1 - self.velocity_alpha) * tr.velocity
        tr.detection = d
        tr.box = box
        tr.last_box = box.copy()
        tr.last_seen = t
        tr.visible = True
        tr.hits += 1
        self._maybe_confirm(tr)

    def _maybe_confirm(self, tr: Track) -> None:
        if tr.id is None and tr.hits >= self.confirm_frames:
            tr.id = self._next_id
            self._next_id += 1
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_perception_tracker.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add perception/tracker.py tests/test_perception_tracker.py
git commit -m "feat(perception): IoU tracker with tentative confirmation and occlusion coasting"
```

---

### Task 4: Features

**Files:**
- Create: `perception/features.py`, `tests/test_perception_features.py`

**Interfaces:**
- Consumes: `PerceptionConfig` (Task 1), `Detection`, `Person` (Task 1), `focal_px`, `pixel_to_angles` (Task 2), `Track` (Task 3: `id`, `detection`, `box`, `last_box`, `visible`, `first_seen`, `last_seen`).
- Produces: `head_point(det, conf_min) -> (x, y)`; `facing_score(det, conf_min) -> float`; `estimate_distance(det, frame_size, f_px, facing, cfg) -> (metres | None, source | None)`; `classify_motion(history, cfg) -> str` where `history` is a sequence of `(t, distance_or_None, yaw_deg)`; `FeatureExtractor(cfg).people(tracks, t, frame_size, pose=None) -> tuple[Person, ...]` where `pose = (turn_deg, nod_deg)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_perception_features.py`:

```python
from types import SimpleNamespace

import numpy as np
import pytest

from perception.features import (FeatureExtractor, classify_motion, estimate_distance, facing_score,
                                  head_point)
from perception.geometry import focal_px
from perception.types import Detection

F = focal_px(1280, 60.0)      # ~1108.5 px


def frontal(cx=640, top=200, shoulder_px=110.85):
    kp = {
        "nose": (cx, top + 40, 0.9),
        "left_eye": (cx - 10, top + 30, 0.9), "right_eye": (cx + 10, top + 30, 0.9),
        "left_ear": (cx - 22, top + 35, 0.9), "right_ear": (cx + 22, top + 35, 0.9),
        "left_shoulder": (cx - shoulder_px / 2, top + 120, 0.9),
        "right_shoulder": (cx + shoulder_px / 2, top + 120, 0.9),
    }
    return Detection(box=(cx - 80, top, cx + 80, top + 470), score=0.9, keypoints=kp)


def profile(cx=640, top=200):
    kp = {
        "nose": (cx - 25, top + 40, 0.9),
        "left_eye": (cx - 18, top + 30, 0.9), "right_eye": (cx - 5, top + 30, 0.2),
        "left_ear": (cx + 5, top + 35, 0.9), "right_ear": (cx + 10, top + 35, 0.1),
    }
    return Detection(box=(cx - 60, top, cx + 60, top + 470), score=0.8, keypoints=kp)


def from_behind(box=(600, 0, 700, 719)):
    kp = {n: (650, 100, 0.1) for n in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")}
    return Detection(box=box, score=0.7, keypoints=kp)


def track(det, id_=1, visible=True, t=0.0, box=None):
    b = np.array(det.box, dtype=float)
    return SimpleNamespace(id=id_, detection=det, box=b if box is None else np.array(box, float),
                           last_box=b, visible=visible, first_seen=0.0, last_seen=t)


def test_head_point_is_mean_of_confident_head_keypoints():
    x, y = head_point(frontal(), 0.5)
    assert x == pytest.approx(640)
    assert 230 <= y <= 240


def test_person_from_behind_uses_box_fallbacks(pcfg):
    d = from_behind()
    assert head_point(d, 0.5) == pytest.approx((650, 0 + 0.1 * 719))
    assert facing_score(d, 0.5) == 0.0
    assert estimate_distance(d, (1280, 720), F, 0.0, pcfg) == (None, None)   # box touches both edges


def test_facing_scores_frontal_high_profile_low():
    assert facing_score(frontal(), 0.5) == pytest.approx(1.0)
    assert facing_score(profile(), 0.5) < 0.3


def test_distance_from_shoulders_when_facing(pcfg):
    d, src = estimate_distance(frontal(shoulder_px=F * 0.40 / 4.0), (1280, 720), F, 1.0, pcfg)
    assert src == "shoulders" and d == pytest.approx(4.0)


def test_side_on_skips_shoulders_and_uses_body(pcfg):
    det = frontal()                                       # box 470 px tall, inside the frame
    d, src = estimate_distance(det, (1280, 720), F, 0.2, pcfg)
    assert src == "body" and d == pytest.approx(1.70 * F / 470)


def test_head_spread_is_the_last_resort(pcfg):
    det = Detection(box=(500, 0, 780, 719), score=0.9, keypoints={
        "left_ear": (620, 100, 0.9), "right_ear": (660, 100, 0.9)})
    d, src = estimate_distance(det, (1280, 720), F, 0.0, pcfg)
    assert src == "head" and d == pytest.approx(0.15 * F / 40)


def hist(dist_rate=0.0, yaw_rate=0.0, seconds=1.0, n=31):
    ts = np.linspace(0, seconds, n)
    return [(t, 3.0 + dist_rate * t, 5.0 + yaw_rate * t) for t in ts]


@pytest.mark.parametrize("h,expected", [
    (hist(dist_rate=-0.6), "approaching"),
    (hist(dist_rate=+0.6), "leaving"),
    (hist(yaw_rate=30.0), "passing"),
    (hist(), "still"),
    (hist(seconds=0.3, n=10), "unknown"),
])
def test_motion_classes(pcfg, h, expected):
    assert classify_motion(h, pcfg) == expected


def test_angles_use_actual_frame_size_not_config(pcfg):
    fx = FeatureExtractor(pcfg)                           # config says 1280x720
    det = frontal(cx=320, top=100)
    (p,) = fx.people([track(det)], 0.0, (640, 480))
    assert p.head_deg[0] == pytest.approx(0.0, abs=0.1)   # centred in a 640-wide frame


def test_centred_person_while_neck_turns_is_passing(pcfg):
    fx = FeatureExtractor(pcfg)
    for i in range(31):
        t = i / 30
        (p,) = fx.people([track(frontal(), t=t)], t, (1280, 720), pose=(30.0 * t, 0.0))
    assert p.world_yaw_deg == pytest.approx(30.0, abs=0.5)
    assert p.motion == "passing"


def test_centred_person_without_neck_pose_is_still(pcfg):
    fx = FeatureExtractor(pcfg)
    for i in range(31):
        t = i / 30
        (p,) = fx.people([track(frontal(), t=t)], t, (1280, 720))
    assert p.world_yaw_deg is None and p.motion == "still"


def test_distance_is_smoothed(pcfg):
    fx = FeatureExtractor(pcfg)
    near = frontal(shoulder_px=F * 0.40 / 2.0)            # 2 m
    far = frontal(shoulder_px=F * 0.40 / 4.0)             # 4 m
    fx.people([track(near)], 0.0, (1280, 720))
    (p,) = fx.people([track(far, t=0.1)], 0.1, (1280, 720))
    assert p.distance_m == pytest.approx(0.3 * 4.0 + 0.7 * 2.0)
    assert p.distance_source == "shoulders"


def test_coasting_person_head_follows_predicted_box(pcfg):
    fx = FeatureExtractor(pcfg)
    det = frontal()
    shifted = np.array(det.box) + np.array([50, 0, 50, 0])
    (p,) = fx.people([track(det, visible=False, box=shifted)], 0.0, (1280, 720))
    assert p.head_px[0] == pytest.approx(head_point(det, 0.5)[0] + 50)
    assert p.visible is False


def test_state_is_dropped_for_tracks_that_disappear(pcfg):
    fx = FeatureExtractor(pcfg)
    fx.people([track(frontal(), id_=7)], 0.0, (1280, 720))
    fx.people([], 0.1, (1280, 720))
    assert fx._state == {}
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_perception_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'perception.features'`.

- [ ] **Step 3: Implement `perception/features.py`**

```python
"""Per-person cues from tracks: head point and angle, distance, facing, motion."""
from __future__ import annotations

from collections import deque

import numpy as np

from perception.geometry import focal_px, pixel_to_angles
from perception.types import Detection, Person

HEAD_POINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
MIN_HISTORY_S = 0.5


def _kp(det: Detection, name: str, conf_min: float):
    p = det.keypoints.get(name)
    return p if p is not None and p[2] >= conf_min else None


def head_point(det: Detection, conf_min: float) -> tuple[float, float]:
    pts = [p for n in HEAD_POINTS if (p := _kp(det, n, conf_min))]
    if pts:
        return float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))
    x1, y1, x2, y2 = det.box
    return (x1 + x2) / 2, y1 + 0.1 * (y2 - y1)


def facing_score(det: Detection, conf_min: float) -> float:
    nose = _kp(det, "nose", conf_min)
    if nose is None:
        return 0.0
    le, re = _kp(det, "left_eye", conf_min), _kp(det, "right_eye", conf_min)
    lr, rr = _kp(det, "left_ear", conf_min), _kp(det, "right_ear", conf_min)
    both_eyes = 1.0 if le and re else 0.0
    pair = (lr, rr) if lr and rr else (le, re) if le and re else None
    symmetry = 0.0
    if pair:
        dl, dr = abs(nose[0] - pair[0][0]), abs(nose[0] - pair[1][0])
        if dl + dr > 0:
            symmetry = 1 - abs(dl - dr) / (dl + dr)
    return 0.5 * both_eyes + 0.5 * symmetry


def estimate_distance(det: Detection, frame_size, f_px: float, facing: float, cfg):
    _, height = frame_size
    conf = cfg.keypoint_conf_min
    ls, rs = _kp(det, "left_shoulder", conf), _kp(det, "right_shoulder", conf)
    if ls and rs and facing >= cfg.facing_threshold:
        px = abs(ls[0] - rs[0])
        if px > 1:
            return cfg.shoulder_width_m * f_px / px, "shoulders"
    x1, y1, x2, y2 = det.box
    if y1 > 2 and y2 < height - 2 and y2 - y1 > 1:
        return cfg.body_height_m * f_px / (y2 - y1), "body"
    xs = [p[0] for n in ("left_eye", "right_eye", "left_ear", "right_ear") if (p := _kp(det, n, conf))]
    if len(xs) >= 2 and max(xs) - min(xs) > 1:
        return cfg.head_width_m * f_px / (max(xs) - min(xs)), "head"
    return None, None


def _slope(ts, values) -> float:
    ts = np.asarray(ts, dtype=float)
    return float(np.polyfit(ts - ts[0], np.asarray(values, dtype=float), 1)[0])


def classify_motion(history, cfg) -> str:
    """history: sequence of (t, distance_or_None, yaw_deg), oldest first."""
    if len(history) < 2 or history[-1][0] - history[0][0] < MIN_HISTORY_S:
        return "unknown"
    dist = [(t, d) for t, d, _ in history if d is not None]
    if len(dist) >= 2 and dist[-1][0] - dist[0][0] >= MIN_HISTORY_S:
        rate = _slope([t for t, _ in dist], [d for _, d in dist])
        if rate < -cfg.approach_rate_mps:
            return "approaching"
        if rate > cfg.approach_rate_mps:
            return "leaving"
    yaw_rate = _slope([h[0] for h in history], [h[2] for h in history])
    if abs(yaw_rate) > cfg.passing_rate_dps:
        return "passing"
    return "still"


class FeatureExtractor:
    def __init__(self, cfg):
        self.cfg = cfg
        self._state: dict[int, dict] = {}

    def people(self, tracks, t: float, frame_size, pose=None) -> tuple[Person, ...]:
        width, height = frame_size
        f = focal_px(width, self.cfg.hfov_deg)
        conf = self.cfg.keypoint_conf_min
        out = []
        for tr in tracks:
            det = tr.detection
            hx, hy = head_point(det, conf)
            if not tr.visible:   # coasting: move the head with the predicted box
                hx += ((tr.box[0] + tr.box[2]) - (tr.last_box[0] + tr.last_box[2])) / 2
                hy += ((tr.box[1] + tr.box[3]) - (tr.last_box[1] + tr.last_box[3])) / 2
            yaw, pitch = pixel_to_angles(hx, hy, width, height, self.cfg.hfov_deg)
            world = pose[0] + yaw if pose is not None else None
            score = facing_score(det, conf)
            st = self._state.setdefault(tr.id, {"dist": None, "src": None, "hist": deque()})
            if tr.visible:
                d, src = estimate_distance(det, frame_size, f, score, self.cfg)
                if d is not None:
                    a = self.cfg.distance_ema_alpha
                    st["dist"] = d if st["dist"] is None else a * d + (1 - a) * st["dist"]
                    st["src"] = src
                st["hist"].append((t, st["dist"] if d is not None else None,
                                   world if world is not None else yaw))
                while st["hist"] and st["hist"][0][0] < t - self.cfg.motion_window_s:
                    st["hist"].popleft()
            out.append(Person(
                id=tr.id, box=tuple(float(v) for v in tr.box), head_px=(hx, hy),
                head_deg=(yaw, pitch), world_yaw_deg=world, distance_m=st["dist"],
                distance_source=st["src"], facing_score=score,
                facing=score >= self.cfg.facing_threshold,
                motion=classify_motion(list(st["hist"]), self.cfg), confidence=det.score,
                visible=tr.visible, first_seen=tr.first_seen, last_seen=tr.last_seen))
        live = {tr.id for tr in tracks}
        for gone in set(self._state) - live:
            del self._state[gone]
        return tuple(out)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_perception_features.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add perception/features.py tests/test_perception_features.py
git commit -m "feat(perception): head angle, distance, facing and motion cues"
```

---

### Task 5: Frame sources

**Files:**
- Create: `perception/sources.py`, `tests/test_perception_sources.py`

**Interfaces:**
- Consumes: `PerceptionConfig` (Task 1).
- Produces: `Frame(image: np.ndarray, time: float)`; `CameraError(RuntimeError)`; `rotate_image(img, degrees) -> img`; `find_camera(usb_id, name, enumerate_fn=None) -> int | None`; `VideoFileSource(path, realtime=False, clock=time.monotonic, sleep=time.sleep)` with `open()`, `read() -> Frame | None`, `close()`, `eof: bool`; `CameraSource(usb_id="0C45:6366", name="Webcam", index_fallback=0, width=1280, height=720, rotate=0, capture_factory=cv2.VideoCapture, find=find_camera, clock=time.monotonic)` with the same methods, and `CameraSource.from_config(cfg)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_perception_sources.py`:

```python
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from perception.sources import CameraError, CameraSource, VideoFileSource, find_camera, rotate_image


def make_video(path, n=10, fps=10, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n):
        img = np.full((size[1], size[0], 3), i * 20, dtype=np.uint8)
        writer.write(img)
    writer.release()
    return path


def test_video_file_source_reads_frames_then_eof(tmp_path):
    src = VideoFileSource(make_video(tmp_path / "clip.avi"))
    src.open()
    frames = []
    while (f := src.read()) is not None:
        frames.append(f)
    assert len(frames) == 10
    assert [round(f.time, 2) for f in frames[:3]] == [0.0, 0.1, 0.2]
    assert frames[0].image.shape == (48, 64, 3)
    assert src.eof and src.read() is None
    src.close()


def test_realtime_video_uses_clock_timestamps(tmp_path):
    now = [100.0]
    src = VideoFileSource(make_video(tmp_path / "clip.avi"), realtime=True,
                          clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s))
    src.open()
    a, b = src.read(), src.read()
    assert a.time == pytest.approx(100.0) and b.time == pytest.approx(100.1)


def test_rotate_image():
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    assert rotate_image(img, 0).shape == (48, 64, 3)
    assert rotate_image(img, 90).shape == (64, 48, 3)
    assert rotate_image(img, 180).shape == (48, 64, 3)


def test_find_camera_prefers_usb_id_then_name():
    cams = [SimpleNamespace(index=1200, name="Webcam", vid=0x1234, pid=0x0001),
            SimpleNamespace(index=1201, name="MacBook Pro Camera", vid=None, pid=None),
            SimpleNamespace(index=1202, name="Webcam", vid=0x0C45, pid=0x6366)]
    assert find_camera("0C45:6366", "Webcam", enumerate_fn=lambda: cams) == 1202
    assert find_camera("FFFF:0000", "MacBook Pro Camera", enumerate_fn=lambda: cams) == 1201
    assert find_camera("FFFF:0000", "Nope", enumerate_fn=lambda: cams) is None


class FakeCapture:
    """Produces numbered frames as fast as read() is called (~1 ms each)."""

    def __init__(self, index, opened=True):
        self.index = index
        self.opened = opened
        self.count = 0
        self.props = {}
        self.released = False

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.props[prop] = value

    def read(self):
        time.sleep(0.001)
        self.count += 1
        return True, np.full((48, 64, 3), self.count % 256, dtype=np.uint8)

    def release(self):
        self.released = True


def test_camera_source_returns_only_the_latest_frame():
    caps = []
    src = CameraSource(capture_factory=lambda i: caps.append(FakeCapture(i)) or caps[-1],
                       find=lambda usb, name: 1202)
    src.open()
    try:
        time.sleep(0.1)
        first = src.read()
        assert first is not None
        assert int(first.image[0, 0, 0]) > 20          # skipped the backlog
        assert caps[0].index == 1202
        assert caps[0].props[cv2.CAP_PROP_FRAME_WIDTH] == 1280
        second = src.read()
        assert second is None or second.time > first.time
    finally:
        src.close()
    assert caps[0].released


def test_camera_source_falls_back_to_index_and_reports_failure():
    src = CameraSource(index_fallback=3, capture_factory=lambda i: FakeCapture(i, opened=False),
                       find=lambda usb, name: None)
    with pytest.raises(CameraError, match="index 3"):
        src.open()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_perception_sources.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'perception.sources'`.

- [ ] **Step 3: Implement `perception/sources.py`**

```python
"""Frame sources: the live head camera and video files, behind one interface."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("perception.sources")

_ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    time: float


def rotate_image(img: np.ndarray, degrees: int) -> np.ndarray:
    return img if degrees == 0 else cv2.rotate(img, _ROTATIONS[degrees])


def _default_enumerate():
    from cv2_enumerate_cameras import enumerate_cameras

    return enumerate_cameras()


def find_camera(usb_id: str, name: str, enumerate_fn=None) -> int | None:
    """OpenCV index (backend-encoded) of the camera, by USB ID first, then by name."""
    cams = list((enumerate_fn or _default_enumerate)())
    vid, pid = (int(x, 16) for x in usb_id.split(":"))
    for c in cams:
        if getattr(c, "vid", None) == vid and getattr(c, "pid", None) == pid:
            return c.index
    for c in cams:
        if c.name == name:
            return c.index
    return None


class VideoFileSource:
    def __init__(self, path, realtime: bool = False, clock=time.monotonic, sleep=time.sleep):
        self.path = str(path)
        self.realtime = realtime
        self.clock = clock
        self._sleep = sleep
        self._cap = None
        self._n = 0
        self._fps = 30.0
        self._start = 0.0
        self.eof = False

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise CameraError(f"could not open video {self.path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._n = 0
        self._start = self.clock()
        self.eof = False

    def read(self) -> Frame | None:
        if self.eof or self._cap is None:
            return None
        ok, img = self._cap.read()
        if not ok:
            self.eof = True
            return None
        offset = self._n / self._fps
        self._n += 1
        if not self.realtime:
            return Frame(img, offset)
        due = self._start + offset
        wait = due - self.clock()
        if wait > 0:
            self._sleep(wait)
        return Frame(img, due)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class CameraSource:
    """Live camera. A reader thread keeps only the newest frame, so consumers never see a backlog."""

    def __init__(self, usb_id: str = "0C45:6366", name: str = "Webcam", index_fallback: int = 0,
                 width: int = 1280, height: int = 720, rotate: int = 0,
                 capture_factory=cv2.VideoCapture, find=find_camera, clock=time.monotonic):
        self.usb_id, self.name, self.index_fallback = usb_id, name, index_fallback
        self.width, self.height, self.rotate = width, height, rotate
        self._factory, self._find, self.clock = capture_factory, find, clock
        self._cap = None
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._delivered: Frame | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_config(cls, cfg) -> "CameraSource":
        return cls(cfg.camera_usb_id, cfg.camera_name, cfg.camera_index_fallback, cfg.width,
                   cfg.height, cfg.rotate)

    def open(self) -> None:
        index = self._find(self.usb_id, self.name)
        if index is None:
            log.warning("camera %s / %r not found; trying index %d", self.usb_id, self.name,
                        self.index_fallback)
            index = self.index_fallback
        cap = self._factory(index)
        if not cap.isOpened():
            raise CameraError(f"could not open camera at index {index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, name="camera-reader", daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while not self._stop.is_set():
            ok, img = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = Frame(rotate_image(img, self.rotate), self.clock())
            with self._lock:
                self._latest = frame

    def read(self) -> Frame | None:
        with self._lock:
            frame = self._latest
        if frame is None or frame is self._delivered:
            return None
        self._delivered = frame
        return frame

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._latest = self._delivered = None
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_perception_sources.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add perception/sources.py tests/test_perception_sources.py
git commit -m "feat(perception): video-file and latest-frame camera sources"
```

---

### Task 6: Pipeline

**Files:**
- Create: `perception/pipeline.py`, `tests/test_perception_pipeline.py`

**Interfaces:**
- Consumes: `PerceptionConfig`, `Detection`, `Snapshot` (Task 1); `Tracker` (Task 3); `FeatureExtractor` (Task 4); `Frame` (Task 5); any detector with `detect(image) -> list[Detection]`; any source with `open/read/close`.
- Produces: `Perception(source, detector, cfg, pose_provider=None, on_snapshot=None, clock=time.monotonic, sleep=time.sleep)` with `start()`, `stop()`, `step() -> Snapshot | None`, `latest() -> Snapshot`, context manager; `on_snapshot(snapshot, image_or_None)`; `people_or_empty(snapshot, now, stale_after_s) -> tuple[Person, ...]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_perception_pipeline.py`:

```python
import time

import numpy as np
import pytest

from perception.pipeline import Perception, people_or_empty
from perception.sources import Frame
from perception.types import Detection, Snapshot


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeSource:
    def __init__(self, clock, size=(1280, 720)):
        self.clock = clock
        self.size = size
        self.producing = True
        self.opens = 0
        self.closes = 0

    def open(self):
        self.opens += 1

    def read(self):
        if not self.producing:
            return None
        w, h = self.size
        return Frame(np.zeros((h, w, 3), dtype=np.uint8), self.clock())

    def close(self):
        self.closes += 1


class FakeDetector:
    def __init__(self, dets=None):
        self.dets = dets if dets is not None else [Detection(box=(560, 100, 720, 570), score=0.9)]
        self.raise_next = None

    def detect(self, image):
        if self.raise_next:
            exc, self.raise_next = self.raise_next, None
            raise exc
        return list(self.dets)


def make(pcfg, **kw):
    clock = Clock()
    src = FakeSource(clock, **kw)
    det = FakeDetector()
    seen = []
    p = Perception(src, det, pcfg, clock=clock, sleep=lambda s: None,
                   on_snapshot=lambda snap, img: seen.append((snap, img)))
    src.open()
    p._last_frame_at = clock()
    p._last_reopen = clock()
    return p, src, det, clock, seen


def run(p, clock, n, dt=1 / 30):
    for _ in range(n):
        clock.t += dt
        p.step()


def test_publishes_a_person_after_confirmation(pcfg):
    p, _, _, clock, seen = make(pcfg)
    run(p, clock, 2)
    assert p.latest().people == ()
    run(p, clock, 1)
    snap = p.latest()
    assert snap.source_ok and len(snap.people) == 1 and snap.people[0].id == 1
    assert seen[-1][1] is not None and seen[-1][1].shape == (720, 1280, 3)


def test_frame_size_comes_from_the_frame(make_pcfg):
    p, _, _, clock, _ = make(make_pcfg({"width": 1280, "height": 720}), size=(640, 480))
    run(p, clock, 1)
    assert p.latest().frame_size == (640, 480)


def test_source_timeout_publishes_failure_and_recovers(pcfg):
    p, src, _, clock, seen = make(pcfg)
    run(p, clock, 5)
    src.producing = False
    run(p, clock, 10, dt=0.1)                      # 1.0 s without frames
    snap = p.latest()
    assert not snap.source_ok and snap.people == () and "no frames" in snap.error
    assert seen[-1][1] is None
    src.producing = True
    run(p, clock, 1)
    assert p.latest().source_ok


def test_reopens_source_while_failing(pcfg):
    p, src, _, clock, _ = make(pcfg)
    src.producing = False
    run(p, clock, 50, dt=0.1)                      # 5 s of failure
    assert src.opens >= 3                          # initial open + reopen every 2 s


def test_detector_exception_publishes_error_and_continues(pcfg):
    p, _, det, clock, _ = make(pcfg)
    run(p, clock, 3)
    det.raise_next = RuntimeError("MPS backend failure")
    run(p, clock, 1)
    snap = p.latest()
    assert not snap.source_ok and "MPS backend failure" in snap.error
    run(p, clock, 1)
    assert p.latest().source_ok


def test_pose_provider_feeds_world_yaw(pcfg):
    p, _, _, clock, _ = make(pcfg)
    p.pose_provider = lambda: (10.0, 0.0)
    run(p, clock, 3)
    (person,) = p.latest().people
    assert person.world_yaw_deg == pytest.approx(10.0 + person.head_deg[0])


def test_pose_provider_error_is_ignored(pcfg):
    p, _, _, clock, _ = make(pcfg)

    def broken():
        raise OSError("neck unplugged")

    p.pose_provider = broken
    run(p, clock, 3)
    assert p.latest().source_ok and p.latest().people[0].world_yaw_deg is None


def test_nothing_is_written_to_disk(pcfg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p, _, _, clock, _ = make(pcfg)
    run(p, clock, 20)
    assert list(tmp_path.iterdir()) == []


def test_people_or_empty():
    person_snap = Snapshot(timestamp=10.0, frame_size=(1, 1), people=("someone",), source_ok=True)
    assert people_or_empty(person_snap, now=10.2, stale_after_s=0.3) == ("someone",)
    assert people_or_empty(person_snap, now=10.4, stale_after_s=0.3) == ()
    failed = Snapshot(timestamp=10.0, frame_size=(1, 1), people=(), source_ok=False, error="x")
    assert people_or_empty(failed, now=10.0, stale_after_s=0.3) == ()


def test_threaded_start_and_stop(pcfg):
    src = FakeSource(time.monotonic)
    with Perception(src, FakeDetector(), pcfg) as p:
        time.sleep(0.2)
        assert p.latest().source_ok
    assert src.closes == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_perception_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'perception.pipeline'`.

- [ ] **Step 3: Implement `perception/pipeline.py`**

```python
"""Perception pipeline: source -> detector -> tracker -> features -> Snapshot.

Behavior reads latest() (or uses people_or_empty). No frames are ever written to disk.
"""
from __future__ import annotations

import logging
import threading
import time

from perception.features import FeatureExtractor
from perception.tracker import Tracker
from perception.types import Snapshot

log = logging.getLogger("perception.pipeline")


def people_or_empty(snapshot: Snapshot, now: float, stale_after_s: float) -> tuple:
    """The people in a snapshot, or () if it failed or is older than stale_after_s."""
    if not snapshot.source_ok or now - snapshot.timestamp > stale_after_s:
        return ()
    return snapshot.people


class Perception:
    def __init__(self, source, detector, cfg, pose_provider=None, on_snapshot=None,
                 clock=time.monotonic, sleep=time.sleep):
        self.source = source
        self.detector = detector
        self.cfg = cfg
        self.pose_provider = pose_provider
        self.on_snapshot = on_snapshot
        self.clock = clock
        self._sleep = sleep
        self.tracker = Tracker(cfg.track_iou_min, cfg.track_confirm_frames, cfg.max_coast_s)
        self.features = FeatureExtractor(cfg)
        self._lock = threading.Lock()
        self._latest = Snapshot(0.0, (0, 0), (), False, "not started")
        self._failing = False
        self._last_frame_at = 0.0
        self._last_reopen = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- lifecycle -------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    def start(self) -> None:
        self.source.open()
        self._last_frame_at = self._last_reopen = self.clock()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="perception", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.source.close()

    def latest(self) -> Snapshot:
        with self._lock:
            return self._latest

    # ---- loop ------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if self.step() is None:
                    self._sleep(0.005)
            except Exception:
                log.exception("perception step failed")
                self._sleep(0.05)

    def step(self) -> Snapshot | None:
        now = self.clock()
        frame = self.source.read()
        if frame is None:
            if now - self._last_frame_at > self.cfg.source_timeout_s:
                self._fail(now, "no frames from source")
                if now - self._last_reopen >= self.cfg.reopen_every_s:
                    self._reopen(now)
            return None
        self._last_frame_at = now
        height, width = frame.image.shape[:2]
        try:
            detections = self.detector.detect(frame.image)
        except Exception as exc:
            self._fail(frame.time, f"detector error: {exc!r}")
            return self.latest()
        tracks = self.tracker.update(detections, frame.time)
        pose = None
        if self.pose_provider is not None:
            try:
                pose = self.pose_provider()
            except Exception as exc:
                log.warning("pose provider failed: %r", exc)
        people = self.features.people(tracks, frame.time, (width, height), pose)
        if self._failing:
            log.info("perception recovered")
            self._failing = False
        snap = Snapshot(frame.time, (width, height), people, True, None)
        self._publish(snap, frame.image)
        return snap

    def _fail(self, t: float, message: str) -> None:
        if not self._failing:
            log.error(message)
            self._failing = True
        self._publish(Snapshot(t, self.latest().frame_size, (), False, message), None)

    def _reopen(self, now: float) -> None:
        self._last_reopen = now
        try:
            self.source.close()
            self.source.open()
            log.info("source reopened")
        except Exception as exc:
            log.warning("source reopen failed: %r", exc)

    def _publish(self, snap: Snapshot, image) -> None:
        with self._lock:
            self._latest = snap
        if self.on_snapshot is not None:
            self.on_snapshot(snap, image)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_perception_pipeline.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add perception/pipeline.py tests/test_perception_pipeline.py
git commit -m "feat(perception): threaded pipeline with failure snapshots and staleness helper"
```

---

### Task 7: YOLO pose detector

**Files:**
- Create: `perception/detector.py`, `tests/test_perception_detector.py`

**Interfaces:**
- Consumes: `Detection`, `KEYPOINT_NAMES` (Task 1).
- Produces: `COCO_INDEX: dict[str, int]`; `detections_from_arrays(boxes (n,4), scores (n,), keypoints (n,17,3), score_min) -> list[Detection]`; `YoloPoseDetector(model_path, imgsz=640, score_min=0.4, device=None)` with `detect(image) -> list[Detection]` and `YoloPoseDetector.from_config(cfg, repo_root)`.

**Gate:** Step 5 installs Ultralytics + PyTorch (~1 GB from PyPI) and Step 6 downloads `yolo11n-pose.pt` (~6 MB from the Ultralytics GitHub releases). **Ask the user before each**, stating source and size. If declined, complete Steps 1–4 and 8 (offline tests) and leave the `model` tests for later.

- [ ] **Step 1: Write the failing tests**

`tests/test_perception_detector.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from perception.detector import COCO_INDEX, detections_from_arrays

REPO = Path(__file__).resolve().parent.parent
WEIGHTS = REPO / "data" / "models" / "yolo11n-pose.pt"


def test_coco_indices_for_used_keypoints():
    assert COCO_INDEX == {"nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
                          "left_shoulder": 5, "right_shoulder": 6}


def test_detections_from_arrays_maps_keypoints_and_filters_scores():
    boxes = np.array([[10, 20, 110, 320], [0, 0, 5, 5]], dtype=float)
    scores = np.array([0.9, 0.2])
    kps = np.zeros((2, 17, 3))
    kps[0, 0] = (60, 40, 0.95)       # nose
    kps[0, 6] = (90, 120, 0.8)       # right shoulder
    dets = detections_from_arrays(boxes, scores, kps, score_min=0.4)
    assert len(dets) == 1
    d = dets[0]
    assert d.box == (10.0, 20.0, 110.0, 320.0) and d.score == pytest.approx(0.9)
    assert d.keypoints["nose"] == pytest.approx((60, 40, 0.95))
    assert d.keypoints["right_shoulder"] == pytest.approx((90, 120, 0.8))
    assert set(d.keypoints) == set(COCO_INDEX)


def test_missing_weights_give_a_clear_error(tmp_path):
    pytest.importorskip("ultralytics")
    from perception.detector import YoloPoseDetector

    with pytest.raises(FileNotFoundError, match="yolo11n-pose.pt"):
        YoloPoseDetector(tmp_path / "yolo11n-pose.pt")


@pytest.mark.model
@pytest.mark.parametrize("image,min_people", [("bus.jpg", 3), ("zidane.jpg", 2)])
def test_finds_people_in_sample_images(image, min_people):
    import cv2
    from ultralytics.utils import ASSETS

    from perception.detector import YoloPoseDetector

    det = YoloPoseDetector(WEIGHTS)
    dets = det.detect(cv2.imread(str(ASSETS / image)))
    assert len(dets) >= min_people
    for d in dets:
        x1, y1, x2, y2 = d.box
        nx, ny, nc = d.keypoints["nose"]
        if nc >= 0.5:
            assert x1 - 5 <= nx <= x2 + 5 and y1 - 5 <= ny <= y2 + 5
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_perception_detector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'perception.detector'`.

- [ ] **Step 3: Implement `perception/detector.py`**

```python
"""Person detection. Only YoloPoseDetector touches Ultralytics/PyTorch.

Ultralytics is AGPL-3.0: fine for personal/research use; commercial use needs a licence or a
different backend behind the same detect() interface (MediaPipe is the documented alternative).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from perception.types import KEYPOINT_NAMES, Detection

COCO_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}   # first 7 COCO keypoints


def detections_from_arrays(boxes, scores, keypoints, score_min: float) -> list[Detection]:
    out = []
    for box, score, kps in zip(np.asarray(boxes), np.asarray(scores), np.asarray(keypoints)):
        if score < score_min:
            continue
        kp = {name: (float(kps[i][0]), float(kps[i][1]), float(kps[i][2]))
              for name, i in COCO_INDEX.items()}
        out.append(Detection(box=tuple(float(v) for v in box), score=float(score), keypoints=kp))
    return out


class YoloPoseDetector:
    def __init__(self, model_path, imgsz: int = 640, score_min: float = 0.4, device: str | None = None):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"pose model weights not found: {path} "
                                    "(download yolo11n-pose.pt into data/models/)")
        import torch
        from ultralytics import YOLO

        self.model = YOLO(str(path))
        self.imgsz = imgsz
        self.score_min = score_min
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

    @classmethod
    def from_config(cls, cfg, repo_root) -> "YoloPoseDetector":
        return cls(Path(repo_root) / cfg.model, cfg.imgsz, cfg.person_score_min)

    def detect(self, image) -> list[Detection]:
        r = self.model.predict(image, imgsz=self.imgsz, conf=self.score_min, device=self.device,
                               classes=[0], verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        return detections_from_arrays(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                      r.keypoints.data.cpu().numpy(), self.score_min)
```

- [ ] **Step 4: Run the offline tests**

Run: `.venv/bin/pytest tests/test_perception_detector.py -v`
Expected: 2 passed; `test_missing_weights_give_a_clear_error` skipped (Ultralytics not installed yet); model tests deselected.

- [ ] **Step 5: Install Ultralytics (after the user's OK)**

Run: `.venv/bin/pip install -q -e '.[dev,vision]'`
Expected: exits 0. Then `.venv/bin/pytest tests/test_perception_detector.py -v` → 3 passed.

- [ ] **Step 6: Download the weights (after the user's OK)**

```bash
mkdir -p data/models && curl -L -o data/models/yolo11n-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt
```
Expected: a ~6 MB file.

- [ ] **Step 7: Run the model tests**

Run: `.venv/bin/pytest -m model -v`
Expected: 2 passed.

- [ ] **Step 8: Commit**

```bash
git add perception/detector.py tests/test_perception_detector.py
git commit -m "feat(perception): YOLO11n-pose detector behind the Detection interface"
```

---

### Task 8: Tools — debug view and eval

**Files:**
- Create: `tools/perception_view.py`, `tools/eval_perception.py`, `tests/test_perception_tools.py`

**Interfaces:**
- Consumes: everything above; `CameraSource.from_config`, `VideoFileSource`, `YoloPoseDetector.from_config`, `Perception`, `perception.config.load_config`.
- Produces: `tools.perception_view.draw_overlay(image, snapshot, fps, latency_s) -> image`; `tools.eval_perception.parse_mot_gt(path) -> dict[int, list[(id, box)]]`, `mot_summary(gt_by_frame, hyp_by_frame) -> dict` with keys `num_switches`, `mota`, `idf1`.

- [ ] **Step 1: Write the failing tests**

`tests/test_perception_tools.py`:

```python
import subprocess
import sys

import numpy as np
import pytest

from perception.types import Person, Snapshot


def person(id_=1, box=(100, 100, 200, 400)):
    return Person(id=id_, box=box, head_px=(150, 130), head_deg=(5.0, 2.0), world_yaw_deg=None,
                  distance_m=2.5, distance_source="body", facing_score=0.9, facing=True,
                  motion="still", confidence=0.9, visible=True, first_seen=0.0, last_seen=1.0)


def test_draw_overlay_draws_people_and_keeps_shape():
    from tools.perception_view import draw_overlay

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    snap = Snapshot(timestamp=1.0, frame_size=(640, 480), people=(person(),), source_ok=True)
    out = draw_overlay(img.copy(), snap, fps=29.7, latency_s=0.041)
    assert out.shape == img.shape and out.any()


def test_draw_overlay_shows_failures():
    from tools.perception_view import draw_overlay

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    snap = Snapshot(timestamp=1.0, frame_size=(640, 480), people=(), source_ok=False, error="no frames")
    assert draw_overlay(img.copy(), snap, fps=0.0, latency_s=0.0).any()


def test_parse_mot_gt_keeps_pedestrians_only(tmp_path):
    from tools.eval_perception import parse_mot_gt

    gt = tmp_path / "gt.txt"
    gt.write_text("1,1,10,20,30,40,1,1,1.0\n"
                  "1,2,50,60,30,40,0,1,1.0\n"       # ignored (conf 0)
                  "2,1,12,20,30,40,1,1,0.9\n"
                  "2,3,90,90,10,10,1,7,1.0\n")      # ignored (class 7, not pedestrian)
    parsed = parse_mot_gt(gt)
    assert parsed == {1: [(1, (10.0, 20.0, 40.0, 60.0))], 2: [(1, (12.0, 20.0, 42.0, 60.0))]}


def test_mot_summary_counts_id_switches():
    from tools.eval_perception import mot_summary

    gt = {f: [(1, (0.0, 0.0, 10.0, 10.0))] for f in range(1, 11)}
    perfect = {f: [(7, (0.0, 0.0, 10.0, 10.0))] for f in range(1, 11)}
    swapped = {f: [(7 if f <= 5 else 8, (0.0, 0.0, 10.0, 10.0))] for f in range(1, 11)}
    assert mot_summary(gt, perfect)["num_switches"] == 0
    assert mot_summary(gt, perfect)["idf1"] == pytest.approx(1.0)
    assert mot_summary(gt, swapped)["num_switches"] == 1


@pytest.mark.parametrize("script", ["tools/perception_view.py", "tools/eval_perception.py"])
def test_tool_scripts_show_help(script):
    result = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_perception_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.perception_view'`.

- [ ] **Step 3: Implement `tools/perception_view.py`**

```python
"""Live perception debug view. Nothing is saved.

Usage:
  .venv/bin/python tools/perception_view.py              # head camera
  .venv/bin/python tools/perception_view.py --video F    # a video file, played in real time
Press q or Esc to quit.
"""
import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # repo root, for `perception`/`tools` when run as a script

import cv2  # noqa: E402
import numpy as np  # noqa: E402

GREEN, GREY, RED, WHITE = (0, 200, 0), (150, 150, 150), (0, 0, 255), (255, 255, 255)


def draw_overlay(image: np.ndarray, snapshot, fps: float, latency_s: float) -> np.ndarray:
    for p in snapshot.people:
        colour = GREEN if p.visible else GREY
        x1, y1, x2, y2 = (int(v) for v in p.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        cv2.circle(image, (int(p.head_px[0]), int(p.head_px[1])), 5, colour, -1)
        dist = f"{p.distance_m:.1f}m" if p.distance_m is not None else "?m"
        label = f"#{p.id} {dist} {'facing' if p.facing else 'away'} {p.motion}"
        cv2.putText(image, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    status = f"{fps:4.1f} fps  {latency_s * 1000:4.0f} ms  {len(snapshot.people)} people"
    cv2.putText(image, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    if not snapshot.source_ok:
        cv2.putText(image, f"SOURCE FAILED: {snapshot.error}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, RED, 2)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", help="play a video file instead of the camera")
    args = parser.parse_args()

    from perception.config import load_config
    from perception.detector import YoloPoseDetector
    from perception.pipeline import Perception
    from perception.sources import CameraSource, VideoFileSource

    cfg = load_config()
    source = VideoFileSource(args.video, realtime=True) if args.video else CameraSource.from_config(cfg)
    latest = {"snap": None, "img": None, "times": []}

    def on_snapshot(snap, img):
        latest["snap"], latest["img"] = snap, img
        latest["times"] = (latest["times"] + [time.monotonic()])[-30:]

    with Perception(source, YoloPoseDetector.from_config(cfg, REPO), cfg, on_snapshot=on_snapshot):
        while True:
            snap, img, times = latest["snap"], latest["img"], latest["times"]
            if snap is not None:
                frame = img.copy() if img is not None else np.zeros((480, 640, 3), np.uint8)
                fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
                latency = time.monotonic() - snap.timestamp if not args.video else 0.0
                cv2.imshow("perception", draw_overlay(frame, snap, fps, latency))
            if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement `tools/eval_perception.py`**

```python
"""Evaluate perception on a video or image sequence, optionally against MOT ground truth.

Usage:
  .venv/bin/python tools/eval_perception.py VIDEO [--gt gt.txt] [--fps 30]
VIDEO can be a file or an image-sequence pattern such as data/MOT17/.../img1/%06d.jpg
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # repo root, for `perception`/`tools` when run as a script


def parse_mot_gt(path):
    """MOT gt.txt -> {frame: [(id, (x1, y1, x2, y2))]}, pedestrians (class 1) with conf 1 only."""
    out = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        f = line.split(",")
        frame, tid, x, y, w, h, conf = int(f[0]), int(f[1]), *map(float, f[2:7])
        cls = int(float(f[7])) if len(f) > 7 else 1
        if conf == 1 and cls == 1:
            out[frame].append((tid, (x, y, x + w, y + h)))
    return dict(out)


def mot_summary(gt_by_frame, hyp_by_frame) -> dict:
    import motmetrics as mm

    def xywh(box):
        return [box[0], box[1], box[2] - box[0], box[3] - box[1]]

    acc = mm.MOTAccumulator(auto_id=False)
    for frame in sorted(set(gt_by_frame) | set(hyp_by_frame)):
        gt = gt_by_frame.get(frame, [])
        hyp = hyp_by_frame.get(frame, [])
        dist = mm.distances.iou_matrix([xywh(b) for _, b in gt], [xywh(b) for _, b in hyp], max_iou=0.5)
        acc.update([i for i, _ in gt], [i for i, _ in hyp], dist, frameid=frame)
    s = mm.metrics.create().compute(acc, metrics=["num_switches", "mota", "idf1"], name="run")
    return {k: float(s[k].iloc[0]) for k in ("num_switches", "mota", "idf1")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--gt", help="MOT-format ground truth (gt.txt)")
    args = parser.parse_args()

    from perception.config import load_config
    from perception.detector import YoloPoseDetector
    from perception.pipeline import Perception
    from perception.sources import VideoFileSource

    cfg = load_config()
    source = VideoFileSource(args.video)
    p = Perception(source, YoloPoseDetector.from_config(cfg, REPO), cfg)
    source.open()
    hyp, step_times, tracks_seen, frame_no = {}, [], set(), 0
    start = time.monotonic()
    while True:
        t0 = time.monotonic()
        snap = p.step()
        if snap is None:
            if source.eof:
                break
            continue
        frame_no += 1
        step_times.append(time.monotonic() - t0)
        hyp[frame_no] = [(q.id, q.box) for q in snap.people if q.visible]
        tracks_seen.update(q.id for q in snap.people)
    source.close()
    elapsed = time.monotonic() - start
    step_times.sort()
    print(f"frames {frame_no}  fps {frame_no / elapsed:.1f}  "
          f"latency mean {1000 * sum(step_times) / len(step_times):.0f} ms  "
          f"p95 {1000 * step_times[int(0.95 * (len(step_times) - 1))]:.0f} ms  tracks {len(tracks_seen)}")
    if args.gt:
        s = mot_summary(parse_mot_gt(args.gt), hyp)
        print(f"ID switches {s['num_switches']:.0f}  MOTA {s['mota']:.3f}  IDF1 {s['idf1']:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/pytest tests/test_perception_tools.py -v`
Expected: all pass (the `--help` tests import nothing heavy before argument parsing).
If `mot_summary` fails with `AttributeError: module 'numpy' has no attribute 'asfarray'`
(motmetrics 1.4.0 vs NumPy 2), add `"numpy<2"` to `dependencies`, reinstall with
`.venv/bin/pip install -q -e '.[dev]'`, re-run, and ledger it as a ruling.

Run: `.venv/bin/pytest -q`
Expected: whole offline suite passes.

- [ ] **Step 6: Commit**

```bash
git add tools/perception_view.py tools/eval_perception.py tests/test_perception_tools.py
git commit -m "feat(tools): perception debug view and MOT eval"
```

---

### Task 9: Public data runs and camera tests (gated; camera part needs hardware)

**Files:**
- Create: `tests/test_perception_camera.py`
- Modify: `docs/NEXT_STEPS.md`

**Interfaces:**
- Consumes: `CameraSource`, `find_camera`, `load_config`, the tools from Task 8.
- Produces: measured fps/latency/ID-switch numbers recorded in `docs/NEXT_STEPS.md`; camera tests to run when the head camera is connected.

**Gate:** each download below needs the user's explicit OK (file, source, size). Nothing goes into git.

- [ ] **Step 1: Write the camera tests** (skipped by default; run with `-m camera`)

`tests/test_perception_camera.py`:

```python
"""Head-camera tests. Run only with the camera plugged in: .venv/bin/pytest -m camera -v"""
import time

import pytest

from perception.config import load_config
from perception.sources import CameraSource, find_camera

pytestmark = pytest.mark.camera


def test_head_camera_is_found_by_usb_id_or_name():
    cfg = load_config()
    assert find_camera(cfg.camera_usb_id, cfg.camera_name) is not None


def test_head_camera_delivers_fresh_frames():
    src = CameraSource.from_config(load_config())
    src.open()
    try:
        time.sleep(1.0)
        a = src.read()
        time.sleep(0.2)
        b = src.read()
        assert a is not None and b is not None and b.time > a.time
        print("frame size:", a.image.shape[1], "x", a.image.shape[0])
    finally:
        src.close()
```

Run: `.venv/bin/pytest -q`
Expected: offline suite still passes; camera tests deselected.

- [ ] **Step 2: Commit the camera tests**

```bash
git add tests/test_perception_camera.py
git commit -m "test(perception): head camera tests (run with -m camera)"
```

- [ ] **Step 3: Stock videos (after the user's OK)** — the user picks 2–3 free Pexels videos of small groups indoors at 1–4 m; download each into `data/videos/` (state file name and size first). Run:

```bash
.venv/bin/python tools/eval_perception.py data/videos/<file>.mp4
.venv/bin/python tools/perception_view.py --video data/videos/<file>.mp4
```
Expected: fps ≥ 25, p95 latency < 60 ms; in the view, people boxed with stable IDs, side-on people still tracked.

- [ ] **Step 4: MOT17 (after the user's OK; ~5.5 GB from motchallenge.net, CC BY-NC-SA 3.0)** — download `MOT17.zip` into `data/`, unzip, keep `train/MOT17-09-FRCNN` and `train/MOT17-04-FRCNN`, delete the rest. Run:

```bash
.venv/bin/python tools/eval_perception.py "data/MOT17/train/MOT17-09-FRCNN/img1/%06d.jpg" --gt data/MOT17/train/MOT17-09-FRCNN/gt/gt.txt
```
Expected: prints fps, latency, ID switches, MOTA, IDF1 (reported, not gated).

- [ ] **Step 5: Record the numbers** — add a "Perception results (public data)" section to `docs/NEXT_STEPS.md` with the fps, p95 latency, ID switches, MOTA and IDF1 per video, and move "run `pytest -m camera` and `perception_view.py` on the head camera; calibrate hfov with the neck; record own clips" into the hardware to-do list.

```bash
git add docs/NEXT_STEPS.md
git commit -m "docs: perception results on public data; hardware to-dos"
```

- [ ] **Step 6: With the hardware (later session, user present)** — run `.venv/bin/pytest -m camera -v`, then `.venv/bin/python tools/perception_view.py` (macOS will ask for camera permission). Check people are boxed and IDs stay stable while walking past; note the reported frame size. FOV calibration and clip recording follow in the behavior phase.
