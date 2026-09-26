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


def has_head_keypoints(det: Detection, conf_min: float) -> bool:
    return any(_kp(det, n, conf_min) for n in HEAD_POINTS)


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
        # hfov is the sensor's horizontal FOV; after a 90/270 rotation the sensor width is the image height.
        sensor_width = height if self.cfg.rotate in (90, 270) else width
        f = focal_px(sensor_width, self.cfg.hfov_deg)
        conf = self.cfg.keypoint_conf_min
        out = []
        for tr in tracks:
            det = tr.detection
            hx, hy = head_point(det, conf)
            if not tr.visible:   # coasting: move the head with the predicted box
                hx += ((tr.box[0] + tr.box[2]) - (tr.last_box[0] + tr.last_box[2])) / 2
                hy += ((tr.box[1] + tr.box[3]) - (tr.last_box[1] + tr.last_box[3])) / 2
            yaw, pitch = pixel_to_angles(hx, hy, width, height, self.cfg.hfov_deg, f_px=f)
            world = pose[0] + yaw if pose is not None else None
            score = facing_score(det, conf)
            st = self._state.setdefault(tr.id, {"dist": None, "src": None, "hist": deque(), "kind": None})
            kind = "world" if world is not None else "camera"
            if st["kind"] != kind:          # never regress world yaw against camera yaw
                st["hist"].clear()
                st["kind"] = kind
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
                visible=tr.visible, first_seen=tr.first_seen, last_seen=tr.last_seen,
                head_from_keypoints=has_head_keypoints(det, conf)))
        live = {tr.id for tr in tracks}
        for gone in set(self._state) - live:
            del self._state[gone]
        return tuple(out)
