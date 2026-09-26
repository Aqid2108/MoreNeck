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

    def update(self, detections: list[Detection], t: float,
               shift_px: tuple[float, float] = (0.0, 0.0)) -> list[Track]:
        dx, dy = shift_px
        if dx or dy:   # the head turned: move every track with the image
            shift = np.array([dx, dy, dx, dy], dtype=float)
            for tr in self._tracks:
                tr.last_box = tr.last_box + shift
                tr.box = tr.box + shift
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
