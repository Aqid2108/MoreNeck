"""Neck configuration: loading, validation and degree <-> raw conversion."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

COUNTS_PER_REV = 4096
DEG_PER_COUNT = 360.0 / COUNTS_PER_REV
JOINT_NAMES = ("turn", "nod", "lean")
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True)
class JointConfig:
    name: str
    id: int
    raw_min: int
    raw_max: int
    neutral: int
    rest: int
    direction: int
    max_speed: float
    max_accel: float
    current_limit_ma: float
    hold_current_limit_ma: float
    temp_warn_c: float
    temp_soft_c: float
    temp_hard_c: float

    def to_raw(self, deg: float) -> float:
        return self.neutral + self.direction * deg / DEG_PER_COUNT

    def to_deg(self, raw: float) -> float:
        return (raw - self.neutral) * self.direction * DEG_PER_COUNT

    def clamp_raw(self, raw: float) -> float:
        return min(max(raw, self.raw_min), self.raw_max)


@dataclass(frozen=True)
class NeckConfig:
    loop_hz: float
    port_vid: int
    baudrate: int
    voltage_warn_below: float
    voltage_soft_below: float
    voltage_hard_below: float
    voltage_hard_above: float
    tracking_max_error_deg: float
    tracking_max_duration_s: float
    spike_ticks: int
    hold_window_s: float
    hard_misses: int
    soft_limit_margin_deg: float
    soft_hold_timeout_s: float
    rest_speed_deg_s: float
    rest_torque_fraction: float
    rest_done_within_deg: float
    near_rest_deg: float
    joints: dict[str, JointConfig]


def _validate_joint(j: JointConfig) -> None:
    def fail(msg):
        raise ValueError(f"joint {j.name}: {msg}")

    if not (0 <= j.raw_min < j.raw_max <= 4095):
        fail(f"need 0 <= raw_min < raw_max <= 4095, got {j.raw_min}..{j.raw_max}")
    if not (j.raw_min <= j.neutral <= j.raw_max):
        fail(f"neutral {j.neutral} outside {j.raw_min}..{j.raw_max}")
    if not (j.raw_min <= j.rest <= j.raw_max):
        fail(f"rest {j.rest} outside {j.raw_min}..{j.raw_max}")
    if j.direction not in (1, -1):
        fail(f"direction must be 1 or -1, got {j.direction}")
    if not (j.temp_warn_c < j.temp_soft_c < j.temp_hard_c):
        fail("need temp_warn_c < temp_soft_c < temp_hard_c")
    if j.max_speed <= 0 or j.max_accel <= 0:
        fail("max_speed and max_accel must be positive")


def config_from_dict(d: dict) -> NeckConfig:
    raw_joints = d.get("joints", {})
    missing = [n for n in JOINT_NAMES if n not in raw_joints]
    if missing:
        raise ValueError(f"config missing joints: {', '.join(missing)}")
    joints = {n: JointConfig(name=n, **raw_joints[n]) for n in JOINT_NAMES}
    for j in joints.values():
        _validate_joint(j)
    ids = [j.id for j in joints.values()]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate servo id in joints: {ids}")
    top = {f.name for f in fields(NeckConfig)} - {"joints"}
    return NeckConfig(**{k: d[k] for k in top}, joints=joints)


def load_config(path: Path | str = DEFAULT_CONFIG) -> NeckConfig:
    return config_from_dict(yaml.safe_load(Path(path).read_text()))


def update_joint_config(path: Path | str, updates: dict[str, dict]) -> None:
    """Merge per-joint field updates into a config file and validate the result."""
    path = Path(path)
    d = yaml.safe_load(path.read_text())
    for joint, values in updates.items():
        d["joints"][joint].update(values)
    config_from_dict(d)
    path.write_text(yaml.safe_dump(d, sort_keys=False))
