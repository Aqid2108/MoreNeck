"""Behavior presets: loading and validation."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

PRESETS_FILE = Path(__file__).with_name("presets.yaml")


@dataclass(frozen=True)
class Preset:
    name: str
    w_facing: float
    w_near: float
    w_approach: float
    stickiness: float
    switch_hold_s: float
    follow_gain: float
    pursuit_hz: float
    min_move_speed_dps: float
    hop_start_deg: float
    max_nod_deg: float
    min_confidence: float
    low_confidence_drop_s: float
    acquire_max_nod_deg: float
    glance_newcomer: bool
    glance_movement: bool
    glance_s: float
    glance_cooldown_s: float
    lost_hold_s: float
    idle_scan: bool
    scan_after_s: float
    scan_amplitude_deg: float
    scan_speed_dps: float
    scan_nod_deg: float
    scan_pause_s: float
    lean_deg: float
    avert_after_s: float | None
    avert_deg: float
    avert_s: float
    deadband_deg: float
    lean_near_m: float
    lean_release_m: float
    lean_release_s: float
    coast_factor: float
    near_full_m: float
    near_zero_m: float
    newcomer_window_s: float
    loop_hz: float


def _validate(p: Preset) -> None:
    checks = [
        (0 < p.follow_gain <= 1, "follow_gain must be in (0, 1]"),
        (min(p.w_facing, p.w_near, p.w_approach, p.stickiness) >= 0,
         "weights and stickiness must be non-negative"),
        (p.near_full_m < p.near_zero_m, "near_full_m must be less than near_zero_m"),
        (p.lean_near_m < p.lean_release_m, "lean_near_m must be less than lean_release_m"),
        (p.lean_deg >= 0, "lean_deg must be non-negative"),
        (p.avert_after_s is None or p.avert_after_s > 0, "avert_after_s must be null or positive"),
        (0 <= p.coast_factor <= 1, "coast_factor must be within 0..1"),
        (0 < p.max_nod_deg <= 90, "max_nod_deg must be in (0, 90]"),
        (p.min_move_speed_dps > 0 and p.hop_start_deg >= 0, "min_move_speed_dps > 0, hop_start_deg >= 0"),
        (0 < p.min_confidence <= 1, "min_confidence must be in (0, 1]"),
        (p.low_confidence_drop_s > 0, "low_confidence_drop_s must be positive"),
        (0 <= p.acquire_max_nod_deg <= p.max_nod_deg, "acquire_max_nod_deg must be within 0..max_nod_deg"),
        (all(v > 0 for v in (p.switch_hold_s, p.glance_s, p.glance_cooldown_s, p.lost_hold_s,
                             p.scan_after_s, p.scan_amplitude_deg, p.scan_speed_dps, p.scan_pause_s,
                             p.avert_s, p.deadband_deg, p.lean_release_s, p.newcomer_window_s,
                             p.loop_hz, p.pursuit_hz)),
         "times, speeds and amplitudes must be positive"),
    ]
    for ok, message in checks:
        if not ok:
            raise ValueError(f"preset {p.name}: {message}")


def preset_from_dict(name: str, d: dict) -> Preset:
    names = {f.name for f in fields(Preset)} - {"name"}
    missing = sorted(names - set(d))
    if missing:
        raise ValueError(f"preset {name} missing: {', '.join(missing)}")
    unknown = sorted(set(d) - names)
    if unknown:
        raise ValueError(f"preset {name} has unknown keys: {', '.join(unknown)}")
    preset = Preset(name=name, **{k: d[k] for k in names})
    _validate(preset)
    return preset


def load_presets(path: Path | str = PRESETS_FILE) -> dict[str, Preset]:
    raw = yaml.safe_load(Path(path).read_text())
    shared = raw.get("shared", {})
    return {name: preset_from_dict(name, {**shared, **values})
            for name, values in raw["presets"].items()}


def load_preset(name: str, path: Path | str = PRESETS_FILE) -> Preset:
    presets = load_presets(path)
    if name not in presets:
        raise ValueError(f"unknown preset {name!r}; choose from {', '.join(sorted(presets))}")
    return presets[name]
