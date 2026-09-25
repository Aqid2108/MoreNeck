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
    camera_index_fallback: int | None
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
        (c.camera_index_fallback is None or c.camera_index_fallback >= 0,
         "camera_index_fallback must be null or a non-negative index"),
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
