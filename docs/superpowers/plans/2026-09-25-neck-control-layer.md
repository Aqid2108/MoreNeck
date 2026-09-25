# Neck Control Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python package that drives the 3-servo MoreNeck (lean, nod, turn) through `NeckController.look(turn, nod, lean)` with smooth coordinated motion, soft limits, continuous safety monitoring and a tiered fault response.

**Architecture:** One 50 Hz control loop (`NeckController`) owns the servo bus. Each tick: one group-read of all servos → pure `safety.check()` → trapezoidal setpoint step toward target (`motion.step_axis`) → one group-write of goals. `ServoBus` is a thin wrapper over python-st3215's *low-level* group read/write classes and is the only code that touches the serial port. Everything above `bus.py` is testable offline against a simulated `FakeBus`.

**Tech Stack:** Python 3.12, python-st3215 (pinned git commit, low-level classes only), pyserial, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-neck-control-layer-design.md`

## Global Constraints

- Python ≥ 3.12; virtualenv at `.venv/` in the repo root.
- `python-st3215` pinned to commit `143cff5e74453a2fd24e7a3d78aeef644364768a`; use only `PortHandler`, `protocol_packet_handler`, `GroupSyncRead`, `GroupSyncWrite` and constants from `st3215.values`. Never call its high-level `ST3215` class methods (`ReadCurrent` reads only the low byte).
- **No EEPROM writes anywhere.** Only RAM registers are written: torque enable (40), acceleration (41), goal position (42–43), goal speed (46–47), torque limit (48–49).
- Bus: USB VID `0x1A86` (WCH CH343), 1,000,000 baud; port found by VID, never hard-coded.
- Joint IDs: lean = 1, nod = 3, turn = 4.
- Only `neck/controller.py` imports `neck/bus.py`'s `ServoBus` at runtime (tools construct one and hand it to the controller; `tools/calibrate.py` is the single torque-off, read-only exception).
- Tests marked `hardware` are skipped by default and are **only run when the user asks, with the user at the robot and a hand on servo power.**
- Units above `bus.py`: joint angles in degrees relative to neutral; setpoints/targets internally in raw counts (4096 per revolution).

## Review Focus

1. **A servo is unpowered or unplugged when `start()` is called** → `start()` raises `BusError` naming the missing joint and torque is never enabled on the others. Test: Task 6 `test_start_with_missing_servo_raises_and_never_enables_torque`.
2. **Perception code passes `NaN`/`inf` to `look()`** → returns `False`, targets unchanged, nothing crashes. Test: Task 6 `test_look_rejects_non_finite_angles`.
3. **USB cable pulled mid-run (serial exception inside a tick)** → controller latches a HARD fault, attempts torque-off, and the loop thread survives instead of dying silently. Tests: Task 3 `test_read_all_returns_none_on_serial_exception`, Task 8 `test_tick_exception_latches_hard_fault_and_goes_limp`.
4. **Hand-edited `config.yaml` with an inverted range, neutral outside limits, or duplicate IDs** → `load_config` raises `ValueError` naming the joint. Tests: Task 2 `test_validation_errors`.
5. **`close()` called twice, or before `start()`** (e.g. Ctrl-C during startup) → no error, no bus access. Test: Task 8 `test_close_is_idempotent_and_safe_before_start`.

---

## File Structure

```
pyproject.toml              package metadata, deps, pytest config (hardware marker)
neck/
  __init__.py               package marker
  config.yaml               per-joint limits/tuning (edited by tools/calibrate.py)
  config.py                 JointConfig/NeckConfig, load/validate, deg<->raw, update_joint_config
  bus.py                    Reading, decode_reading, goal_bytes, ServoBus (only serial I/O)
  motion.py                 step_axis: trapezoidal setpoint stepping (pure)
  safety.py                 Level, Verdict, JointHistory, new_history, check (pure)
  controller.py             Mode, Fault, NeckController (loop, look/rest/clear_fault/close)
tools/
  probe.py                  read-only bus scan (standalone pyserial)
  calibrate.py              torque-off hand-move calibration -> config.yaml
  log_current.py            scripted motion + CSV log + recommended current limits
  neck_shell.py             interactive REPL with a started NeckController
tests/
  conftest.py               BASE config dict, cfg / make_cfg fixtures
  fake_bus.py               FakeClock, FakeBus (simulated servos)
  test_imports.py
  test_config.py
  test_bus.py
  test_motion.py
  test_safety.py
  test_controller.py        startup, look, motion
  test_controller_faults.py
  test_controller_lifecycle.py
  test_tools.py             calibrate.derive_joint, log_current.summarize
  test_hardware.py          marked hardware
```

---

### Task 1: Project scaffold and probe tool

**Files:**
- Create: `pyproject.toml`, `neck/__init__.py`, `tests/test_imports.py`, `tools/probe.py`

**Interfaces:**
- Consumes: nothing.
- Produces: installable package `neck`; `pytest` runs with `hardware` tests deselected by default.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "moreneck"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "st3215 @ git+https://github.com/Mickael-Roger/python-st3215@143cff5e74453a2fd24e7a3d78aeef644364768a",
  "pyserial>=3.5",
  "pyyaml>=6",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.setuptools]
packages = ["neck"]

[tool.setuptools.package-data]
neck = ["config.yaml"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
markers = ["hardware: needs the real robot on the bus (run only on request)"]
addopts = "-m 'not hardware'"
```

- [ ] **Step 2: Create the package marker**

`neck/__init__.py`:

```python
"""MoreNeck neck control layer."""
```

- [ ] **Step 3: Create the venv and install**

Run:
```bash
python3 -m venv .venv && .venv/bin/pip install -q -e '.[dev]'
```
Expected: exits 0.

- [ ] **Step 4: Write the import smoke test**

`tests/test_imports.py`:

```python
def test_library_low_level_classes_import():
    from st3215.group_sync_read import GroupSyncRead
    from st3215.group_sync_write import GroupSyncWrite
    from st3215.port_handler import PortHandler
    from st3215.protocol_packet_handler import protocol_packet_handler
    from st3215.values import COMM_SUCCESS, STS_PRESENT_POSITION_L

    assert STS_PRESENT_POSITION_L == 0x38
    assert COMM_SUCCESS == 0
    assert all([GroupSyncRead, GroupSyncWrite, PortHandler, protocol_packet_handler])


def test_package_imports():
    import neck

    assert neck.__doc__
```

- [ ] **Step 5: Run it**

Run: `.venv/bin/pytest -v`
Expected: 2 passed.

- [ ] **Step 6: Add the read-only probe tool**

`tools/probe.py`:

```python
"""Read-only probe of the servo bus: PING + READ only, never writes.

Usage: .venv/bin/python tools/probe.py
"""
import time

import serial
from serial.tools import list_ports

VID = 0x1A86


def find_port():
    for p in list_ports.comports():
        if p.vid == VID:
            return p.device
    raise SystemExit(f"No USB serial device with VID {VID:#06x}. Is the bus adapter plugged in?")


def packet(sid, instr, params=b""):
    body = bytes([sid, len(params) + 2, instr]) + params
    return b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])


def txrx(ser, pkt, timeout=0.03):
    ser.reset_input_buffer()
    ser.write(pkt)
    ser.flush()
    deadline = time.time() + timeout
    buf = b""
    while time.time() < deadline:
        buf += ser.read(ser.in_waiting or 1)
        i = buf.find(b"\xff\xff")
        if i >= 0 and len(buf) >= i + 4:
            n = buf[i + 3]
            if len(buf) >= i + 4 + n:
                r = buf[i:i + 4 + n]
                if r[2] == pkt[2]:
                    return r
                buf = buf[i + 4 + n:]
    return None


def read(ser, sid, addr, n):
    r = txrx(ser, packet(sid, 0x02, bytes([addr, n])))
    if not r:
        return None, None
    return r[4], r[5:5 + n]


def u16(b, i):
    return b[i] | (b[i + 1] << 8)


def signed15(v):
    return -(v & 0x7FFF) if v & 0x8000 else v


def main():
    port = find_port()
    for baud in (1_000_000, 500_000, 115_200):
        with serial.Serial(port, baud, timeout=0.01) as ser:
            found = [sid for sid in range(0, 21) if txrx(ser, packet(sid, 0x01))]
            print(f"{port} @ {baud}: responding IDs {found}")
            if not found:
                continue
            for sid in found:
                _, head = read(ser, sid, 0x00, 7)
                _, ee = read(ser, sid, 0x09, 28)
                err, rt = read(ser, sid, 0x38, 15)
                _, te = read(ser, sid, 0x28, 1)
                _, tl = read(ser, sid, 0x30, 2)
                print(f"\n== ID {sid} (packet error {err}) ==")
                if head:
                    print(f"  firmware {head[0]}.{head[1]}  model {u16(head, 3):#06x}")
                if ee:
                    print(f"  angle limits : {u16(ee, 0)} .. {u16(ee, 2)}")
                    print(f"  max temp     : {ee[4]} C")
                    print(f"  volt limits  : {ee[6] / 10:.1f} .. {ee[5] / 10:.1f} V")
                    print(f"  max torque   : {u16(ee, 7)}")
                    print(f"  prot current : {u16(ee, 0x1C - 9) * 6.5:.0f} mA")
                    print(f"  mode         : {ee[0x21 - 9]}")
                if rt:
                    print(f"  position     : {u16(rt, 0)}")
                    print(f"  speed        : {signed15(u16(rt, 2))}")
                    print(f"  load         : {u16(rt, 4)}")
                    print(f"  voltage      : {rt[6] / 10:.1f} V")
                    print(f"  temperature  : {rt[7]} C")
                    print(f"  status byte  : {rt[9]:#04x}")
                    print(f"  moving       : {rt[10]}")
                    print(f"  current      : {abs(signed15(u16(rt, 13))) * 6.5:.0f} mA")
                if te:
                    print(f"  torque enable: {te[0]}")
                if tl:
                    print(f"  torque limit : {u16(tl, 0)} (RAM reg 48)")
            return


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml neck/__init__.py tests/test_imports.py tools/probe.py
git commit -m "chore: project scaffold, pinned python-st3215, read-only probe tool"
```

---

### Task 2: Configuration

**Files:**
- Create: `neck/config.yaml`, `neck/config.py`, `tests/conftest.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `COUNTS_PER_REV = 4096`, `DEG_PER_COUNT = 360 / 4096`, `JOINT_NAMES = ("turn", "nod", "lean")`, `DEFAULT_CONFIG: Path`
  - `JointConfig` (frozen dataclass): `name, id, raw_min, raw_max, neutral, rest, direction, max_speed, max_accel, current_limit_ma, hold_current_limit_ma, temp_warn_c, temp_soft_c, temp_hard_c`; methods `to_raw(deg) -> float`, `to_deg(raw) -> float`, `clamp_raw(raw) -> float`
  - `NeckConfig` (frozen dataclass): `loop_hz, port_vid, baudrate, voltage_warn_below, voltage_soft_below, voltage_hard_below, voltage_hard_above, tracking_max_error_deg, tracking_max_duration_s, spike_ticks, hold_window_s, hard_misses, soft_limit_margin_deg, soft_hold_timeout_s, rest_speed_deg_s, rest_torque_fraction, rest_done_within_deg, near_rest_deg, joints: dict[str, JointConfig]`
  - `config_from_dict(d: dict) -> NeckConfig`, `load_config(path=DEFAULT_CONFIG) -> NeckConfig`, `update_joint_config(path, updates: dict[str, dict]) -> None`
  - Test fixtures: `cfg` (NeckConfig from `BASE`), `make_cfg(overrides: dict) -> NeckConfig` where keys are top-level field names or `"joint.field"`.

- [ ] **Step 1: Write `neck/config.yaml`**

```yaml
# MoreNeck control configuration. Raw positions are servo counts (0-4095, 4096/rev).
# Joint limits/neutral/rest/direction are rewritten by tools/calibrate.py.
loop_hz: 50
port_vid: 0x1A86
baudrate: 1000000

voltage_warn_below: 6.8
voltage_soft_below: 6.5
voltage_hard_below: 6.0
voltage_hard_above: 8.0

tracking_max_error_deg: 8.0
tracking_max_duration_s: 0.3
spike_ticks: 3
hold_window_s: 2.0
hard_misses: 5
soft_limit_margin_deg: 2.0
soft_hold_timeout_s: 1.0

rest_speed_deg_s: 15.0
rest_torque_fraction: 0.5
rest_done_within_deg: 3.0
near_rest_deg: 5.0

joints:
  turn:
    id: 4
    raw_min: 1873
    raw_max: 2918
    neutral: 2286
    rest: 2286
    direction: 1
    max_speed: 120.0
    max_accel: 400.0
    current_limit_ma: 800
    hold_current_limit_ma: 800
    temp_warn_c: 55
    temp_soft_c: 60
    temp_hard_c: 65
  nod:
    id: 3
    raw_min: 2744
    raw_max: 3133
    neutral: 2939
    rest: 3133
    direction: 1
    max_speed: 90.0
    max_accel: 300.0
    current_limit_ma: 800
    hold_current_limit_ma: 800
    temp_warn_c: 55
    temp_soft_c: 60
    temp_hard_c: 65
  lean:
    id: 1
    raw_min: 824
    raw_max: 1043
    neutral: 934
    rest: 824
    direction: 1
    max_speed: 45.0
    max_accel: 150.0
    current_limit_ma: 800
    hold_current_limit_ma: 800
    temp_warn_c: 55
    temp_soft_c: 60
    temp_hard_c: 65
```

- [ ] **Step 2: Write the test fixtures**

`tests/conftest.py` (frozen copy of spec defaults — tests must not depend on the live, calibratable `neck/config.yaml`):

```python
import copy

import pytest

from neck.config import config_from_dict


def _joint(id_, raw_min, raw_max, neutral, rest, max_speed, max_accel):
    return {
        "id": id_, "raw_min": raw_min, "raw_max": raw_max, "neutral": neutral, "rest": rest,
        "direction": 1, "max_speed": max_speed, "max_accel": max_accel,
        "current_limit_ma": 800, "hold_current_limit_ma": 800,
        "temp_warn_c": 55, "temp_soft_c": 60, "temp_hard_c": 65,
    }


BASE = {
    "loop_hz": 50, "port_vid": 0x1A86, "baudrate": 1_000_000,
    "voltage_warn_below": 6.8, "voltage_soft_below": 6.5,
    "voltage_hard_below": 6.0, "voltage_hard_above": 8.0,
    "tracking_max_error_deg": 8.0, "tracking_max_duration_s": 0.3,
    "spike_ticks": 3, "hold_window_s": 2.0, "hard_misses": 5,
    "soft_limit_margin_deg": 2.0, "soft_hold_timeout_s": 1.0,
    "rest_speed_deg_s": 15.0, "rest_torque_fraction": 0.5,
    "rest_done_within_deg": 3.0, "near_rest_deg": 5.0,
    "joints": {
        "turn": _joint(4, 1873, 2918, 2286, 2286, 120.0, 400.0),
        "nod": _joint(3, 2744, 3133, 2939, 3133, 90.0, 300.0),
        "lean": _joint(1, 824, 1043, 934, 824, 45.0, 150.0),
    },
}


def build_config(overrides=None):
    d = copy.deepcopy(BASE)
    for key, value in (overrides or {}).items():
        if "." in key:
            joint, field = key.split(".", 1)
            d["joints"][joint][field] = value
        else:
            d[key] = value
    return config_from_dict(d)


@pytest.fixture
def cfg():
    return build_config()


@pytest.fixture
def make_cfg():
    return build_config
```

- [ ] **Step 3: Write the failing tests**

`tests/test_config.py`:

```python
import copy

import pytest
import yaml

from neck.config import (
    DEFAULT_CONFIG, DEG_PER_COUNT, config_from_dict, load_config, update_joint_config,
)
from tests.conftest import BASE


def test_default_config_file_loads_with_measured_values():
    c = load_config(DEFAULT_CONFIG)
    assert {n: j.id for n, j in c.joints.items()} == {"turn": 4, "nod": 3, "lean": 1}
    assert (c.joints["lean"].raw_min, c.joints["lean"].raw_max) == (824, 1043)
    assert (c.joints["nod"].raw_min, c.joints["nod"].raw_max) == (2744, 3133)
    assert (c.joints["turn"].raw_min, c.joints["turn"].raw_max) == (1873, 2918)
    assert c.port_vid == 0x1A86
    assert c.loop_hz == 50


def test_deg_raw_roundtrip_respects_direction(make_cfg):
    c = make_cfg({"turn.direction": -1})
    turn = c.joints["turn"]
    assert turn.to_raw(0.0) == 2286
    assert turn.to_raw(10.0) == pytest.approx(2286 - 10.0 / DEG_PER_COUNT)
    assert turn.to_deg(turn.to_raw(-17.5)) == pytest.approx(-17.5)


def test_clamp_raw(cfg):
    lean = cfg.joints["lean"]
    assert lean.clamp_raw(0) == 824
    assert lean.clamp_raw(5000) == 1043
    assert lean.clamp_raw(900.5) == 900.5


@pytest.mark.parametrize(
    "joint,field,value,message",
    [
        ("lean", "raw_min", 1100, "lean"),          # raw_min >= raw_max
        ("nod", "neutral", 3500, "nod"),            # neutral outside limits
        ("turn", "rest", 100, "turn"),              # rest outside limits
        ("turn", "direction", 0, "turn"),           # direction must be +/-1
        ("nod", "id", 1, "id"),                     # duplicate id with lean
        ("lean", "temp_soft_c", 70, "lean"),        # warn < soft < hard violated
        ("turn", "raw_max", 5000, "turn"),          # beyond 4095
    ],
)
def test_validation_errors(joint, field, value, message):
    d = copy.deepcopy(BASE)
    d["joints"][joint][field] = value
    with pytest.raises(ValueError, match=message):
        config_from_dict(d)


def test_missing_joint_is_rejected():
    d = copy.deepcopy(BASE)
    del d["joints"]["nod"]
    with pytest.raises(ValueError, match="nod"):
        config_from_dict(d)


def test_update_joint_config_rewrites_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(copy.deepcopy(BASE)))
    update_joint_config(path, {"turn": {"neutral": 2300, "direction": -1}})
    c = load_config(path)
    assert c.joints["turn"].neutral == 2300
    assert c.joints["turn"].direction == -1
    assert c.joints["nod"].neutral == 2939
```

- [ ] **Step 4: Run to verify failure**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'neck.config'`.

- [ ] **Step 5: Implement `neck/config.py`**

```python
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
```

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add neck/config.yaml neck/config.py tests/conftest.py tests/test_config.py
git commit -m "feat: neck configuration with validation and deg/raw conversion"
```

---

### Task 3: Servo bus

**Files:**
- Create: `neck/bus.py`, `tests/test_bus.py`, `tests/test_hardware.py`

**Interfaces:**
- Consumes: `NeckConfig` (Task 2).
- Produces:
  - `Reading` (frozen dataclass): `position: int, speed: int, load: int, voltage: float, temperature: int, status: int, moving: bool, current_ma: float, error: int`
  - `BusError(RuntimeError)`
  - `decode_reading(packet: list[int]) -> Reading` — `packet[0]` is the reply error byte, `packet[1:16]` are registers 0x38–0x46
  - `goal_bytes(raw: float) -> list[int]` — `[lo, hi]`, rounded and clamped to 0..4095
  - `find_port(vid: int) -> str`
  - `ServoBus(ids: dict[str, int], vid: int = 0x1A86, baudrate: int = 1_000_000, port: str | None = None)` and `ServoBus.from_config(cfg)`; methods `open()`, `close()`, `ping() -> set[str]`, `read_all() -> dict[str, Reading | None]`, `write_goals(goals: dict[str, float]) -> bool`, `set_torque(on: bool) -> bool`, `set_torque_limit(fraction: float) -> bool`, `configure_for_streaming() -> bool`, `read_u16(name: str, address: int) -> int | None`

Library quirks this task must handle (verified in the pinned source):
- `GroupSyncRead.rxPacket` leaves `data_dict` untouched when nothing is received → clear every entry to `[]` before each read, or stale data is returned as fresh.
- `protocol_packet_handler.txPacket` sets `portHandler.is_using = True` and a serial exception mid-transaction leaves it set forever (every later call returns `COMM_PORT_BUSY`) → reset `is_using = False` before each transaction (safe: the controller is the only caller, single-threaded under its lock).
- Sync read waits up to `LATENCY_TIMER` (50 ms) when a servo does not answer, so a missing servo slows that tick to ~50 ms. Safety counts misses, not time.
- Present current is sign-magnitude (bit 15) in registers 69–70, 6.5 mA/count; speed is sign-magnitude bit 15; load is sign-magnitude bit 10.

- [ ] **Step 1: Write the failing tests**

`tests/test_bus.py`:

```python
import serial

from neck.bus import Reading, ServoBus, decode_reading, goal_bytes


def packet(position=805, speed=0, load=0, voltage=74, temp=27, status=0, moving=0,
           current=0, error=0):
    d = [0] * 15
    d[0], d[1] = position & 0xFF, position >> 8
    d[2], d[3] = speed & 0xFF, speed >> 8
    d[4], d[5] = load & 0xFF, load >> 8
    d[6] = voltage
    d[7] = temp
    d[9] = status
    d[10] = moving
    d[13], d[14] = current & 0xFF, current >> 8
    return [error] + d


def test_decode_basic_fields():
    r = decode_reading(packet(position=3132, voltage=74, temp=26, status=0x20, moving=1, error=0x20))
    assert r == Reading(position=3132, speed=0, load=0, voltage=7.4, temperature=26,
                        status=0x20, moving=True, current_ma=0.0, error=0x20)


def test_decode_current_uses_both_bytes():
    # 384 counts = 0x0180: a low-byte-only reader would see 128 (832 mA)
    assert decode_reading(packet(current=384)).current_ma == 384 * 6.5


def test_decode_current_is_magnitude_of_sign_magnitude():
    assert decode_reading(packet(current=0x8000 | 100)).current_ma == 650.0


def test_decode_signed_speed_and_load():
    r = decode_reading(packet(speed=0x8000 | 100, load=0x0400 | 500))
    assert r.speed == -100
    assert r.load == -500


def test_goal_bytes_rounds_and_clamps():
    assert goal_bytes(805.4) == [0x25, 0x03]
    assert goal_bytes(-3) == [0, 0]
    assert goal_bytes(9999) == [0xFF, 0x0F]


class StubPort:
    is_using = True  # simulate a flag left set by an earlier failed transaction


class StubReader:
    def __init__(self):
        self.data_dict = {1: [], 3: []}
        self.replies = {}
        self.raise_exc = None

    def txRxPacket(self):
        if self.raise_exc:
            raise self.raise_exc
        for sid, data in self.replies.items():
            self.data_dict[sid] = data
        return 0


def stub_bus():
    bus = ServoBus({"lean": 1, "nod": 3})
    bus._port_handler = StubPort()
    bus._reader = StubReader()
    return bus


def test_read_all_decodes_and_resets_busy_flag():
    bus = stub_bus()
    bus._reader.replies = {1: packet(position=900), 3: packet(position=3000)}
    out = bus.read_all()
    assert out["lean"].position == 900
    assert out["nod"].position == 3000
    assert bus._port_handler.is_using is False


def test_read_all_never_returns_stale_data():
    bus = stub_bus()
    bus._reader.replies = {1: packet(position=900), 3: packet(position=3000)}
    bus.read_all()
    bus._reader.replies = {3: packet(position=3001)}  # lean stops answering
    out = bus.read_all()
    assert out["lean"] is None
    assert out["nod"].position == 3001


def test_read_all_returns_none_on_serial_exception():
    bus = stub_bus()
    bus._reader.raise_exc = serial.SerialException("device disconnected")
    assert bus.read_all() == {"lean": None, "nod": None}
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_bus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'neck.bus'`.

- [ ] **Step 3: Implement `neck/bus.py`**

```python
"""ServoBus: the only code that touches the serial port.

Wraps python-st3215's low-level group read/write so all servos are read in one
transaction and written in one transaction per tick. Never writes EEPROM.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import serial
from serial.tools import list_ports
from st3215.group_sync_read import GroupSyncRead
from st3215.group_sync_write import GroupSyncWrite
from st3215.port_handler import PortHandler
from st3215.protocol_packet_handler import protocol_packet_handler
from st3215.values import (
    COMM_SUCCESS, STS_ACC, STS_GOAL_POSITION_L, STS_GOAL_SPEED_L, STS_PRESENT_POSITION_L,
    STS_TORQUE_ENABLE,
)

log = logging.getLogger("neck.bus")

REG_TORQUE_LIMIT = 48          # RAM, 2 bytes, 0-1000 (not defined in st3215.values)
READ_START = STS_PRESENT_POSITION_L   # 0x38
READ_LEN = 15                  # registers 0x38..0x46
CURRENT_MA_PER_COUNT = 6.5
STREAM_ACC = 254               # servo-side ramp effectively off; controller shapes motion
STREAM_SPEED = 3400            # servo-side speed cap effectively off


class BusError(RuntimeError):
    pass


@dataclass(frozen=True)
class Reading:
    position: int
    speed: int
    load: int
    voltage: float
    temperature: int
    status: int
    moving: bool
    current_ma: float
    error: int


def _sign_magnitude(value: int, sign_bit: int) -> int:
    magnitude = value & ((1 << sign_bit) - 1)
    return -magnitude if value & (1 << sign_bit) else magnitude


def decode_reading(packet: list[int]) -> Reading:
    """packet[0] = reply error byte; packet[1:16] = registers 0x38..0x46."""
    d = packet[1:]

    def u16(i):
        return d[i] | (d[i + 1] << 8)

    return Reading(
        position=u16(0),
        speed=_sign_magnitude(u16(2), 15),
        load=_sign_magnitude(u16(4), 10),
        voltage=d[6] / 10,
        temperature=d[7],
        status=d[9],
        moving=bool(d[10]),
        current_ma=abs(_sign_magnitude(u16(13), 15)) * CURRENT_MA_PER_COUNT,
        error=packet[0],
    )


def goal_bytes(raw: float) -> list[int]:
    v = min(max(int(round(raw)), 0), 4095)
    return [v & 0xFF, v >> 8]


def find_port(vid: int) -> str:
    for p in list_ports.comports():
        if p.vid == vid:
            return p.device
    raise BusError(f"no USB serial device with VID {vid:#06x}; is the bus adapter plugged in?")


class ServoBus:
    def __init__(self, ids: dict[str, int], vid: int = 0x1A86, baudrate: int = 1_000_000,
                 port: str | None = None):
        self.ids = dict(ids)
        self._vid = vid
        self._baudrate = baudrate
        self._port = port
        self._port_handler = None
        self._ph = None
        self._reader = None

    @classmethod
    def from_config(cls, cfg) -> "ServoBus":
        return cls({n: j.id for n, j in cfg.joints.items()}, vid=cfg.port_vid, baudrate=cfg.baudrate)

    def open(self) -> None:
        port = self._port or find_port(self._vid)
        self._port_handler = PortHandler(port)
        self._port_handler.baudrate = self._baudrate
        if not self._port_handler.openPort():
            raise BusError(f"could not open {port}")
        self._ph = protocol_packet_handler(self._port_handler)
        self._reader = GroupSyncRead(self._ph, READ_START, READ_LEN)
        for sid in self.ids.values():
            self._reader.addParam(sid)
        log.info("bus open on %s @ %d", port, self._baudrate)

    def close(self) -> None:
        if self._port_handler is not None and self._port_handler.is_open:
            self._port_handler.closePort()

    def _begin(self) -> None:
        self._port_handler.is_using = False

    def ping(self) -> set[str]:
        found = set()
        for name, sid in self.ids.items():
            self._begin()
            _, comm, _ = self._ph.ping(sid)
            if comm == COMM_SUCCESS:
                found.add(name)
        return found

    def read_all(self) -> dict[str, Reading | None]:
        for sid in self.ids.values():
            self._reader.data_dict[sid] = []
        self._begin()
        try:
            self._reader.txRxPacket()
        except (serial.SerialException, OSError) as exc:
            log.warning("group read failed: %r", exc)
            return {name: None for name in self.ids}
        out = {}
        for name, sid in self.ids.items():
            data = self._reader.data_dict.get(sid)
            out[name] = decode_reading(data) if data and len(data) == READ_LEN + 1 else None
        return out

    def _sync_write(self, address: int, length: int, per_id: dict[int, list[int]]) -> bool:
        writer = GroupSyncWrite(self._ph, address, length)
        for sid, data in per_id.items():
            writer.addParam(sid, data)
        self._begin()
        try:
            return writer.txPacket() == COMM_SUCCESS
        except (serial.SerialException, OSError) as exc:
            log.warning("sync write to %d failed: %r", address, exc)
            return False

    def write_goals(self, goals: dict[str, float]) -> bool:
        return self._sync_write(STS_GOAL_POSITION_L, 2,
                                {self.ids[n]: goal_bytes(raw) for n, raw in goals.items()})

    def set_torque(self, on: bool) -> bool:
        return self._sync_write(STS_TORQUE_ENABLE, 1, {sid: [1 if on else 0] for sid in self.ids.values()})

    def set_torque_limit(self, fraction: float) -> bool:
        v = int(round(min(max(fraction, 0.0), 1.0) * 1000))
        return self._sync_write(REG_TORQUE_LIMIT, 2, {sid: [v & 0xFF, v >> 8] for sid in self.ids.values()})

    def configure_for_streaming(self) -> bool:
        acc = self._sync_write(STS_ACC, 1, {sid: [STREAM_ACC] for sid in self.ids.values()})
        spd = self._sync_write(STS_GOAL_SPEED_L, 2,
                               {sid: [STREAM_SPEED & 0xFF, STREAM_SPEED >> 8] for sid in self.ids.values()})
        return acc and spd

    def read_u16(self, name: str, address: int) -> int | None:
        self._begin()
        value, comm, _ = self._ph.read2ByteTxRx(self.ids[name], address)
        return value if comm == COMM_SUCCESS else None
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_bus.py -v`
Expected: all pass.

- [ ] **Step 5: Write the read-only hardware test**

`tests/test_hardware.py`:

```python
"""Hardware tests. Run ONLY on request, user at the robot: .venv/bin/pytest -m hardware -v"""
import pytest

from neck.bus import REG_TORQUE_LIMIT, ServoBus
from neck.config import load_config

pytestmark = pytest.mark.hardware


@pytest.fixture
def bus():
    b = ServoBus.from_config(load_config())
    b.open()
    yield b
    b.close()


def test_all_joints_answer_ping(bus):
    assert bus.ping() == {"turn", "nod", "lean"}


def test_read_all_returns_sane_readings(bus):
    readings = bus.read_all()
    assert set(readings) == {"turn", "nod", "lean"}
    for name, r in readings.items():
        assert r is not None, name
        assert 0 <= r.position <= 4095
        assert 6.5 <= r.voltage <= 8.0
        assert 10 <= r.temperature <= 60
        assert r.status == 0


def test_torque_limit_register_address_is_correct(bus):
    # At power-up RAM torque limit (48) is loaded from EEPROM max torque, which read 1000 on all servos.
    # If this fails, STOP: REG_TORQUE_LIMIT is wrong and rest() would write the wrong register.
    for name in ("turn", "nod", "lean"):
        assert bus.read_u16(name, REG_TORQUE_LIMIT) == 1000, name
```

- [ ] **Step 6: Confirm the hardware tests are skipped by default**

Run: `.venv/bin/pytest -v`
Expected: all non-hardware tests pass; `test_hardware.py` tests reported as deselected.

- [ ] **Step 7: Commit**

```bash
git add neck/bus.py tests/test_bus.py tests/test_hardware.py
git commit -m "feat: ServoBus over python-st3215 group read/write with 2-byte current"
```

---

### Task 4: Motion stepping

**Files:**
- Create: `neck/motion.py`, `tests/test_motion.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `step_axis(pos: float, vel: float, target: float, vmax: float, amax: float, dt: float) -> tuple[float, float]` — any consistent units (the controller uses raw counts and counts/s). Returns exactly `(target, 0.0)` on arrival.

- [ ] **Step 1: Write the failing tests**

`tests/test_motion.py`:

```python
import pytest

from neck.motion import step_axis

DT = 0.02


def run_to(pos, vel, target, vmax, amax, max_steps=2000):
    traj = [(pos, vel)]
    for _ in range(max_steps):
        pos, vel = step_axis(pos, vel, target, vmax, amax, DT)
        traj.append((pos, vel))
        if pos == target and vel == 0.0:
            break
    return traj


def test_reaches_target_and_stops():
    assert run_to(0.0, 0.0, 1000.0, 500.0, 2000.0)[-1] == (1000.0, 0.0)


def test_respects_speed_and_accel_limits():
    traj = run_to(0.0, 0.0, 1000.0, 500.0, 2000.0)
    for (_, v0), (p1, v1) in zip(traj, traj[1:]):
        assert abs(v1) <= 500.0 + 1e-9
        if (p1, v1) != (1000.0, 0.0):
            assert abs(v1 - v0) <= 2000.0 * DT + 1e-9


def test_does_not_overshoot():
    assert all(p <= 1000.0 for p, _ in run_to(0.0, 0.0, 1000.0, 500.0, 2000.0))


def test_works_in_negative_direction():
    traj = run_to(1000.0, 0.0, 0.0, 500.0, 2000.0)
    assert traj[-1] == (0.0, 0.0)
    assert all(p >= 0.0 for p, _ in traj)


def test_decelerates_before_reversing_when_target_jumps_behind():
    pos, vel = step_axis(0.0, 500.0, -1000.0, 500.0, 2000.0, DT)
    assert vel == pytest.approx(460.0)
    assert pos == pytest.approx(9.2)


def test_slows_down_gradually_when_vmax_is_lowered():
    _, vel = step_axis(0.0, 500.0, 1000.0, 100.0, 2000.0, DT)
    assert vel == pytest.approx(460.0)


def test_stays_put_at_target():
    assert step_axis(5.0, 0.0, 5.0, 500.0, 2000.0, DT) == (5.0, 0.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_motion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'neck.motion'`.

- [ ] **Step 3: Implement `neck/motion.py`**

```python
"""Trapezoidal setpoint stepping, recomputed every tick so moving targets are followed smoothly."""
from __future__ import annotations

import math


def step_axis(pos: float, vel: float, target: float, vmax: float, amax: float,
              dt: float) -> tuple[float, float]:
    err = target - pos
    if err == 0.0 and vel == 0.0:
        return target, 0.0
    # Fastest speed from which we can still brake to a stop at the target.
    v_brake = math.sqrt(2.0 * amax * abs(err))
    v_desired = math.copysign(min(vmax, v_brake), err)
    dv = max(-amax * dt, min(amax * dt, v_desired - vel))
    vel += dv
    new_pos = pos + vel * dt
    if (target - pos) * (target - new_pos) <= 0.0:   # reached or crossed the target
        return target, 0.0
    return new_pos, vel
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_motion.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add neck/motion.py tests/test_motion.py
git commit -m "feat: trapezoidal setpoint stepping"
```

---

### Task 5: Safety checks

**Files:**
- Create: `neck/safety.py`, `tests/test_safety.py`

**Interfaces:**
- Consumes: `NeckConfig`, `JointConfig`, `DEG_PER_COUNT` (Task 2); `Reading` (Task 3).
- Produces:
  - `Level(IntEnum)`: `OK=0, WARN=1, SOFT=2, HARD=3`
  - `Verdict` dataclass: `level: Level = Level.OK`, `reasons: list[str]`; method `add(level, reason)` (keeps the max level; appends `"<LEVEL>: <reason>"`)
  - `JointHistory` dataclass: `misses, over_current_ticks, tracking_since, holding_since, currents: deque[(t, mA)], limits_armed`
  - `new_history(cfg) -> dict[str, JointHistory]`
  - `check(readings: dict[str, Reading | None], setpoints: dict[str, float], holding: dict[str, bool], cfg, history, now: float) -> Verdict` — mutates `history`.

- [ ] **Step 1: Write the failing tests**

`tests/test_safety.py`:

```python
from neck.bus import Reading
from neck.safety import Level, check, new_history


def reading(position, **kw):
    base = dict(position=position, speed=0, load=0, voltage=7.4, temperature=30, status=0,
                moving=False, current_ma=100.0, error=0)
    base.update(kw)
    return Reading(**base)


def nominal(cfg, **per_joint):
    return {n: reading(j.neutral, **per_joint.get(n, {})) for n, j in cfg.joints.items()}


def chk(cfg, hist, readings, now=0.0, setpoints=None, holding=None):
    setpoints = setpoints or {n: float(j.neutral) for n, j in cfg.joints.items()}
    holding = holding or {n: False for n in cfg.joints}
    return check(readings, setpoints, holding, cfg, hist, now)


def test_nominal_is_ok(cfg):
    v = chk(cfg, new_history(cfg), nominal(cfg))
    assert v.level == Level.OK
    assert v.reasons == []


def test_missed_reads_warn_then_hard_on_fifth(cfg):
    hist = new_history(cfg)
    r = nominal(cfg)
    r["nod"] = None
    for i in range(4):
        assert chk(cfg, hist, r, now=i).level == Level.WARN
    assert chk(cfg, hist, r, now=5).level == Level.HARD


def test_miss_counter_resets_on_reply(cfg):
    hist = new_history(cfg)
    missing = nominal(cfg)
    missing["nod"] = None
    for _ in range(4):
        chk(cfg, hist, missing)
    chk(cfg, hist, nominal(cfg))
    assert chk(cfg, hist, missing).level == Level.WARN


def test_status_bits_are_hard(cfg):
    assert chk(cfg, new_history(cfg), nominal(cfg, nod={"status": 0x20})).level == Level.HARD


def test_voltage_thresholds(cfg):
    for volts, level in [(6.7, Level.WARN), (6.4, Level.SOFT), (5.9, Level.HARD), (8.1, Level.HARD),
                         (7.4, Level.OK)]:
        r = {n: reading(j.neutral, voltage=volts) for n, j in cfg.joints.items()}
        assert chk(cfg, new_history(cfg), r).level == level, volts


def test_temperature_thresholds_are_per_joint(make_cfg):
    cfg = make_cfg({"nod.temp_warn_c": 50, "nod.temp_soft_c": 55, "nod.temp_hard_c": 60})
    assert chk(cfg, new_history(cfg), nominal(cfg, nod={"temperature": 52})).level == Level.WARN
    assert chk(cfg, new_history(cfg), nominal(cfg, nod={"temperature": 56})).level == Level.SOFT
    assert chk(cfg, new_history(cfg), nominal(cfg, nod={"temperature": 60})).level == Level.HARD
    assert chk(cfg, new_history(cfg), nominal(cfg, turn={"temperature": 56})).level == Level.WARN


def test_current_spike_needs_three_consecutive_ticks(cfg):
    hist = new_history(cfg)
    hot = nominal(cfg, nod={"current_ma": 900.0})
    assert chk(cfg, hist, hot, now=0.00).level == Level.OK
    assert chk(cfg, hist, hot, now=0.02).level == Level.OK
    assert chk(cfg, hist, hot, now=0.04).level == Level.SOFT


def test_current_spike_counter_resets_on_dip(cfg):
    hist = new_history(cfg)
    hot = nominal(cfg, nod={"current_ma": 900.0})
    chk(cfg, hist, hot)
    chk(cfg, hist, hot)
    chk(cfg, hist, nominal(cfg))
    assert chk(cfg, hist, hot).level == Level.OK


def test_holding_mean_current_over_window(make_cfg):
    cfg = make_cfg({"nod.hold_current_limit_ma": 500})
    hist = new_history(cfg)
    r = nominal(cfg, nod={"current_ma": 600.0})
    holding = {n: True for n in cfg.joints}
    for i in range(100):                       # t = 0.00 .. 1.98 s
        assert chk(cfg, hist, r, now=i / 50, holding=holding).level == Level.OK
    assert chk(cfg, hist, r, now=100 / 50, holding=holding).level == Level.SOFT


def test_holding_window_restarts_after_motion(make_cfg):
    cfg = make_cfg({"nod.hold_current_limit_ma": 500})
    hist = new_history(cfg)
    r = nominal(cfg, nod={"current_ma": 600.0})
    hold = {n: True for n in cfg.joints}
    move = {n: False for n in cfg.joints}
    for i in range(90):
        chk(cfg, hist, r, now=i / 50, holding=hold)
    chk(cfg, hist, r, now=90 / 50, holding=move)
    for i in range(91, 140):                   # less than 2 s of continuous holding again
        assert chk(cfg, hist, r, now=i / 50, holding=hold).level == Level.OK


def test_tracking_error_must_persist(cfg):
    hist = new_history(cfg)
    turn = cfg.joints["turn"]
    r = nominal(cfg, turn={"position": turn.neutral + 100})   # ~8.8 degrees off
    assert chk(cfg, hist, r, now=0.00).level == Level.OK
    assert chk(cfg, hist, r, now=0.20).level == Level.OK
    assert chk(cfg, hist, r, now=0.32).level == Level.SOFT


def test_soft_limit_check_arms_only_after_entering_range(cfg):
    hist = new_history(cfg)
    lean = cfg.joints["lean"]
    at = lambda pos: {**nominal(cfg), "lean": reading(pos)}
    sp = lambda pos: {**{n: float(j.neutral) for n, j in cfg.joints.items()}, "lean": float(pos)}
    assert chk(cfg, hist, at(lean.raw_min - 34), setpoints=sp(lean.raw_min - 34)).level == Level.OK
    chk(cfg, hist, at(900), setpoints=sp(900))
    assert chk(cfg, hist, at(lean.raw_min - 34), setpoints=sp(lean.raw_min - 34)).level == Level.SOFT


def test_most_severe_wins_and_all_reasons_kept(cfg):
    r = nominal(cfg, nod={"temperature": 66, "voltage": 6.4})
    v = chk(cfg, new_history(cfg), r)
    assert v.level == Level.HARD
    assert any("temperature" in s for s in v.reasons)
    assert any("voltage" in s for s in v.reasons)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_safety.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'neck.safety'`.

- [ ] **Step 3: Implement `neck/safety.py`**

```python
"""Safety checks: pure function over one tick of readings plus a small rolling history."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

from neck.config import DEG_PER_COUNT


class Level(IntEnum):
    OK = 0
    WARN = 1
    SOFT = 2
    HARD = 3


@dataclass
class Verdict:
    level: Level = Level.OK
    reasons: list[str] = field(default_factory=list)

    def add(self, level: Level, reason: str) -> None:
        self.level = max(self.level, level)
        self.reasons.append(f"{level.name}: {reason}")


@dataclass
class JointHistory:
    misses: int = 0
    over_current_ticks: int = 0
    tracking_since: float | None = None
    holding_since: float | None = None
    currents: deque = field(default_factory=deque)
    limits_armed: bool = False


def new_history(cfg) -> dict[str, JointHistory]:
    return {name: JointHistory() for name in cfg.joints}


def check(readings, setpoints, holding, cfg, history, now) -> Verdict:
    v = Verdict()
    for name, jc in cfg.joints.items():
        r = readings.get(name)
        h = history[name]

        if r is None:
            h.misses += 1
            if h.misses >= cfg.hard_misses:
                v.add(Level.HARD, f"{name}: no reply for {h.misses} reads")
            else:
                v.add(Level.WARN, f"{name}: missed read")
            continue
        h.misses = 0

        if r.status:
            v.add(Level.HARD, f"{name}: servo status fault bits {r.status:#04x}")

        if r.voltage > cfg.voltage_hard_above or r.voltage < cfg.voltage_hard_below:
            v.add(Level.HARD, f"{name}: voltage {r.voltage:.1f} V")
        elif r.voltage < cfg.voltage_soft_below:
            v.add(Level.SOFT, f"{name}: voltage {r.voltage:.1f} V")
        elif r.voltage < cfg.voltage_warn_below:
            v.add(Level.WARN, f"{name}: voltage {r.voltage:.1f} V")

        if r.temperature >= jc.temp_hard_c:
            v.add(Level.HARD, f"{name}: temperature {r.temperature} C")
        elif r.temperature >= jc.temp_soft_c:
            v.add(Level.SOFT, f"{name}: temperature {r.temperature} C")
        elif r.temperature >= jc.temp_warn_c:
            v.add(Level.WARN, f"{name}: temperature {r.temperature} C")

        h.over_current_ticks = h.over_current_ticks + 1 if r.current_ma > jc.current_limit_ma else 0
        if h.over_current_ticks >= cfg.spike_ticks:
            v.add(Level.SOFT, f"{name}: current {r.current_ma:.0f} mA over {jc.current_limit_ma} mA "
                              f"for {h.over_current_ticks} ticks")

        if holding.get(name):
            if h.holding_since is None:
                h.holding_since = now
                h.currents.clear()
            h.currents.append((now, r.current_ma))
            while h.currents and h.currents[0][0] < now - cfg.hold_window_s:
                h.currents.popleft()
            if now - h.holding_since >= cfg.hold_window_s:
                mean = sum(c for _, c in h.currents) / len(h.currents)
                if mean > jc.hold_current_limit_ma:
                    v.add(Level.SOFT, f"{name}: holding current mean {mean:.0f} mA over "
                                      f"{jc.hold_current_limit_ma} mA")
        else:
            h.holding_since = None
            h.currents.clear()

        err_deg = abs(r.position - setpoints[name]) * DEG_PER_COUNT
        if err_deg > cfg.tracking_max_error_deg:
            if h.tracking_since is None:
                h.tracking_since = now
            elif now - h.tracking_since > cfg.tracking_max_duration_s:
                v.add(Level.SOFT, f"{name}: tracking error {err_deg:.1f} deg for "
                                  f"{now - h.tracking_since:.2f} s")
        else:
            h.tracking_since = None

        if jc.raw_min <= r.position <= jc.raw_max:
            h.limits_armed = True
        margin = cfg.soft_limit_margin_deg / DEG_PER_COUNT
        if h.limits_armed and not (jc.raw_min - margin <= r.position <= jc.raw_max + margin):
            v.add(Level.SOFT, f"{name}: position {r.position} outside soft limits "
                              f"{jc.raw_min}..{jc.raw_max}")
    return v
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_safety.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add neck/safety.py tests/test_safety.py
git commit -m "feat: safety checks with debounce, holding mean, tracking and soft limits"
```

---

### Task 6: FakeBus and controller startup/motion

**Files:**
- Create: `tests/fake_bus.py`, `neck/controller.py`, `tests/test_controller.py`

**Interfaces:**
- Consumes: `NeckConfig` (Task 2); `Reading`, `BusError` (Task 3); `step_axis` (Task 4); `Level`, `Verdict`, `new_history`, `check` (Task 5).
- Produces:
  - `tests/fake_bus.py`: `FakeClock` (callable; `.t`; `advance(dt)`), `FakeServo`, `FakeBus(cfg, clock, start_positions=None, speed_deg_s=300.0)` implementing the `ServoBus` method set; inspection attributes `opened, torque_log: list[bool], torque_limit_log: list[float], writes: list[dict], streaming_configured: bool, jump_risk: bool, raise_on_read: Exception | None`; `servos[name]` has `position, goal, torque, temperature, voltage, status, current_ma, jammed, responding`.
  - `neck/controller.py`: `Mode(str, Enum)`: `IDLE, RUNNING, RESTING, LIMP`; `Fault` dataclass `time, level, reasons, readings`; `NeckController(cfg, bus, clock=time.monotonic, sleep=time.sleep, on_tick=None)` with `start(run_thread=True)`, `look(turn=0.0, nod=0.0, lean=0.0) -> bool`, `tick()`, attributes `mode, latched, faults, readings, setpoints, targets, velocities`, property `state -> dict`. `on_tick(now, readings, setpoints, holding)`. (Fault handling, `rest`, `clear_fault`, `close` and threading are completed in Tasks 7–8; this task creates the file with their final signatures.)

- [ ] **Step 1: Write the FakeBus**

`tests/fake_bus.py`:

```python
"""Simulated servo bus for offline controller tests."""
from __future__ import annotations

from dataclasses import dataclass

from neck.bus import Reading
from neck.config import DEG_PER_COUNT


class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@dataclass
class FakeServo:
    position: float
    goal: int | None = None
    torque: bool = False
    temperature: int = 30
    voltage: float = 7.4
    status: int = 0
    current_ma: float = 100.0
    jammed: bool = False
    responding: bool = True


class FakeBus:
    def __init__(self, cfg, clock, start_positions=None, speed_deg_s: float = 300.0):
        start_positions = start_positions or {n: j.rest for n, j in cfg.joints.items()}
        self.servos = {n: FakeServo(position=float(start_positions[n])) for n in cfg.joints}
        self.clock = clock
        self.speed_counts = speed_deg_s / DEG_PER_COUNT
        self._last = clock()
        self.opened = False
        self.torque_log: list[bool] = []
        self.torque_limit_log: list[float] = []
        self.writes: list[dict] = []
        self.streaming_configured = False
        self.jump_risk = False
        self.raise_on_read: Exception | None = None

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def ping(self) -> set[str]:
        return {n for n, s in self.servos.items() if s.responding}

    def _advance(self) -> None:
        now = self.clock()
        max_step = self.speed_counts * max(now - self._last, 0.0)
        self._last = now
        for s in self.servos.values():
            if s.torque and not s.jammed and s.goal is not None:
                delta = s.goal - s.position
                s.position += max(-max_step, min(max_step, delta))

    def read_all(self):
        if self.raise_on_read:
            raise self.raise_on_read
        self._advance()
        return {
            n: Reading(position=int(round(s.position)), speed=0, load=0, voltage=s.voltage,
                       temperature=s.temperature, status=s.status, moving=False,
                       current_ma=s.current_ma, error=0) if s.responding else None
            for n, s in self.servos.items()
        }

    def write_goals(self, goals) -> bool:
        for n, raw in goals.items():
            assert isinstance(raw, int) and 0 <= raw <= 4095, (n, raw)
            self.servos[n].goal = raw
        self.writes.append(dict(goals))
        return True

    def set_torque(self, on: bool) -> bool:
        if on:
            for s in self.servos.values():
                if s.goal is None or abs(s.goal - s.position) > 1:
                    self.jump_risk = True
        for s in self.servos.values():
            s.torque = on
        self.torque_log.append(on)
        return True

    def set_torque_limit(self, fraction: float) -> bool:
        self.torque_limit_log.append(fraction)
        return True

    def configure_for_streaming(self) -> bool:
        self.streaming_configured = True
        return True
```

- [ ] **Step 2: Write the failing tests**

`tests/test_controller.py`:

```python
import math

import pytest

from neck.bus import BusError
from neck.config import DEG_PER_COUNT
from neck.controller import Mode, NeckController
from tests.fake_bus import FakeBus, FakeClock


def make(cfg, start="neutral"):
    clock = FakeClock()
    starts = {n: (j.neutral if start == "neutral" else j.rest) for n, j in cfg.joints.items()}
    bus = FakeBus(cfg, clock, start_positions=starts)
    ctrl = NeckController(cfg, bus, clock=clock, sleep=clock.advance)
    return ctrl, bus, clock


def run(ctrl, clock, seconds):
    for _ in range(round(seconds * ctrl.cfg.loop_hz)):
        clock.advance(1 / ctrl.cfg.loop_hz)
        ctrl.tick()


def test_startup_has_no_jump_and_enables_torque(cfg):
    ctrl, bus, _ = make(cfg, start="rest")
    ctrl.start(run_thread=False)
    assert ctrl.mode == Mode.RUNNING
    assert not bus.jump_risk
    assert bus.torque_log == [True]
    assert bus.streaming_configured
    assert bus.torque_limit_log == [1.0]


def test_start_with_missing_servo_raises_and_never_enables_torque(cfg):
    ctrl, bus, _ = make(cfg)
    bus.servos["nod"].responding = False
    with pytest.raises(BusError, match="nod"):
        ctrl.start(run_thread=False)
    assert bus.torque_log == []
    assert ctrl.mode == Mode.IDLE


def test_moves_from_rest_to_neutral_after_start(cfg):
    ctrl, bus, clock = make(cfg, start="rest")
    ctrl.start(run_thread=False)
    run(ctrl, clock, 3.0)
    for n, j in cfg.joints.items():
        assert bus.servos[n].position == pytest.approx(j.neutral, abs=2), n


def test_look_moves_joint_to_target(cfg):
    ctrl, bus, clock = make(cfg)
    ctrl.start(run_thread=False)
    assert ctrl.look(turn=20.0) is True
    run(ctrl, clock, 2.0)
    expected = cfg.joints["turn"].to_raw(20.0)
    assert bus.servos["turn"].position == pytest.approx(expected, abs=2)


def test_look_clamps_to_soft_limits(cfg):
    ctrl, bus, clock = make(cfg)
    ctrl.start(run_thread=False)
    ctrl.look(turn=500.0, nod=-500.0)
    assert ctrl.targets["turn"] == cfg.joints["turn"].raw_max
    assert ctrl.targets["nod"] == cfg.joints["nod"].raw_min
    run(ctrl, clock, 4.0)
    for w in bus.writes:
        for n, raw in w.items():
            assert cfg.joints[n].raw_min <= raw <= cfg.joints[n].raw_max, (n, raw)


def test_writes_respect_max_speed(cfg):
    ctrl, bus, clock = make(cfg)
    ctrl.start(run_thread=False)
    ctrl.look(turn=40.0)
    run(ctrl, clock, 2.0)
    per_tick = cfg.joints["turn"].max_speed / DEG_PER_COUNT / cfg.loop_hz
    turn = [w["turn"] for w in bus.writes]
    assert max(abs(b - a) for a, b in zip(turn, turn[1:])) <= per_tick + 1


def test_look_rejects_non_finite_angles(cfg):
    ctrl, _, _ = make(cfg)
    ctrl.start(run_thread=False)
    before = dict(ctrl.targets)
    assert ctrl.look(turn=math.nan) is False
    assert ctrl.look(nod=math.inf) is False
    assert ctrl.targets == before


def test_look_before_start_is_rejected(cfg):
    ctrl, _, _ = make(cfg)
    assert ctrl.look(turn=10.0) is False


def test_on_tick_receives_readings_and_setpoints(cfg):
    seen = []
    clock = FakeClock()
    bus = FakeBus(cfg, clock, start_positions={n: j.neutral for n, j in cfg.joints.items()})
    ctrl = NeckController(cfg, bus, clock=clock, sleep=clock.advance,
                          on_tick=lambda now, r, sp, hold: seen.append((now, r, sp, hold)))
    ctrl.start(run_thread=False)
    run(ctrl, clock, 0.1)
    assert len(seen) == 5
    now, readings, setpoints, holding = seen[-1]
    assert set(readings) == set(setpoints) == set(holding) == {"turn", "nod", "lean"}
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_controller.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'neck.controller'`.

- [ ] **Step 4: Implement `neck/controller.py`**

This is the complete controller; Tasks 7 and 8 add tests that exercise the fault and lifecycle parts of it.

```python
"""NeckController: the 50 Hz loop that owns the servo bus.

Only this module drives ServoBus. Callers (perception, behavior, tools) use
look(), rest(), clear_fault(), close(), state and faults.
"""
from __future__ import annotations

import atexit
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

from neck.bus import BusError
from neck.config import DEG_PER_COUNT
from neck.motion import step_axis
from neck.safety import Level, Verdict, check, new_history

log = logging.getLogger("neck.controller")


class Mode(str, Enum):
    IDLE = "idle"        # not started, or closed
    RUNNING = "running"  # torque on, following targets
    RESTING = "resting"  # lowering to the rest pose at reduced speed/torque
    LIMP = "limp"        # torque off


@dataclass
class Fault:
    time: float
    level: Level
    reasons: list[str]
    readings: dict = field(repr=False)


class NeckController:
    def __init__(self, cfg, bus, clock=time.monotonic, sleep=time.sleep, on_tick=None):
        self.cfg = cfg
        self.bus = bus
        self.clock = clock
        self._sleep = sleep
        self.on_tick = on_tick
        self._lock = threading.RLock()
        self.mode = Mode.IDLE
        self.latched: Fault | None = None
        self.faults: list[Fault] = []
        self.readings: dict = {}
        self.setpoints: dict[str, float] = {}
        self.velocities: dict[str, float] = {}
        self.targets: dict[str, float] = {}
        self._history = new_history(cfg)
        self._soft_since: float | None = None
        self._rest_speed: float | None = None
        self._last_warn: list[str] | None = None
        self._opened = False
        self._closed = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ---- lifecycle -------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def start(self, run_thread: bool = True) -> None:
        self.bus.open()
        self._opened = True
        self._startup()
        atexit.register(self.close)
        if run_thread:
            self._thread = threading.Thread(target=self._run, name="neck-loop", daemon=True)
            self._thread.start()

    def _startup(self) -> None:
        with self._lock:
            readings = self.bus.read_all()
            missing = sorted(n for n, r in readings.items() if r is None)
            if missing:
                raise BusError(f"servos not responding: {', '.join(missing)}")
            present = {n: r.position for n, r in readings.items()}
            self.bus.write_goals(present)          # goal = present BEFORE torque on: no jump
            self.bus.configure_for_streaming()
            self.bus.set_torque_limit(1.0)
            self.bus.set_torque(True)
            self.readings = readings
            self.setpoints = {n: float(p) for n, p in present.items()}
            self.velocities = {n: 0.0 for n in present}
            self.targets = {n: float(j.neutral) for n, j in self.cfg.joints.items()}
            self._history = new_history(self.cfg)
            self._soft_since = None
            self._rest_speed = None
            self.mode = Mode.RUNNING

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        try:
            if self.mode in (Mode.RUNNING, Mode.RESTING):
                self.rest()
        finally:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            if self._opened:
                self.bus.close()

    # ---- public commands -------------------------------------------------

    def look(self, turn: float = 0.0, nod: float = 0.0, lean: float = 0.0) -> bool:
        angles = {"turn": turn, "nod": nod, "lean": lean}
        if not all(math.isfinite(a) for a in angles.values()):
            log.warning("look() ignored non-finite angles %s", angles)
            return False
        with self._lock:
            if self.mode != Mode.RUNNING or self.latched is not None:
                return False
            for name, deg in angles.items():
                jc = self.cfg.joints[name]
                self.targets[name] = jc.clamp_raw(jc.to_raw(deg))
        return True

    def rest(self, timeout: float = 10.0) -> None:
        with self._lock:
            if self.mode in (Mode.IDLE, Mode.LIMP):
                return
            if self.mode == Mode.RUNNING:
                self._begin_rest()
        start = self.clock()
        threaded = self._thread is not None and self._thread.is_alive()
        while self.mode == Mode.RESTING:
            if self.clock() - start > timeout:
                log.error("rest did not finish within %.1f s; going limp", timeout)
                with self._lock:
                    self._go_limp()
                break
            self._sleep(1 / self.cfg.loop_hz)
            if not threaded:
                self.tick()

    def clear_fault(self) -> None:
        with self._lock:
            self.latched = None
            self._startup()

    @property
    def state(self) -> dict:
        with self._lock:
            return {
                "mode": self.mode,
                "latched": self.latched,
                "readings": dict(self.readings),
                "setpoints": dict(self.setpoints),
                "targets": dict(self.targets),
            }

    # ---- loop ------------------------------------------------------------

    def _run(self) -> None:
        period = 1 / self.cfg.loop_hz
        next_t = time.monotonic()
        while not self._stop.is_set():
            self._safe_tick()
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_t = time.monotonic()

    def _safe_tick(self) -> None:
        try:
            self.tick()
        except Exception as exc:
            log.exception("tick failed")
            with self._lock:
                self._fault(Verdict(Level.HARD, [f"HARD: controller exception: {exc!r}"]),
                            self.readings, self.clock())
                self._go_limp()

    def tick(self) -> None:
        with self._lock:
            if self.mode == Mode.IDLE:
                return
            now = self.clock()
            readings = self.bus.read_all()
            self.readings = readings
            holding = {n: self.velocities.get(n, 0.0) == 0.0
                       and self.setpoints.get(n) == self.targets.get(n) for n in self.cfg.joints}
            if self.mode != Mode.LIMP:
                verdict = check(readings, self.setpoints, holding, self.cfg, self._history, now)
                self._handle(verdict, readings, now)
            if self.mode in (Mode.RUNNING, Mode.RESTING):
                self._step_motion(1 / self.cfg.loop_hz)
                self.bus.write_goals({n: int(round(sp)) for n, sp in self.setpoints.items()})
                if self.mode == Mode.RESTING and self._rest_done(readings):
                    self._go_limp()
            if self.on_tick:
                self.on_tick(now, readings, dict(self.setpoints), holding)

    # ---- internals (call with lock held) -----------------------------------

    def _handle(self, verdict: Verdict, readings: dict, now: float) -> None:
        if verdict.level == Level.WARN and verdict.reasons != self._last_warn:
            log.warning("; ".join(verdict.reasons))
        self._last_warn = verdict.reasons if verdict.level == Level.WARN else None

        if self.mode == Mode.RESTING:
            if verdict.level == Level.HARD:
                self._fault(verdict, readings, now)
                self._go_limp()
            return

        if verdict.level == Level.HARD or (verdict.level == Level.SOFT and self._near_rest(readings)):
            self._fault(verdict, readings, now)
            self._go_limp()
        elif verdict.level == Level.SOFT:
            if self.latched is None:
                self._fault(verdict, readings, now)
                self._hold(readings)
                self._soft_since = now
            elif self._soft_since is None:
                self._soft_since = now
            elif now - self._soft_since >= self.cfg.soft_hold_timeout_s:
                log.error("SOFT fault persisted %.1f s; resting", now - self._soft_since)
                self._begin_rest()
        else:
            self._soft_since = None

    def _fault(self, verdict: Verdict, readings: dict, now: float) -> None:
        fault = Fault(time=now, level=verdict.level, reasons=list(verdict.reasons), readings=dict(readings))
        self.faults.append(fault)
        if self.latched is None:
            self.latched = fault
        log.error("fault %s: %s", verdict.level.name, "; ".join(verdict.reasons))

    def _hold(self, readings: dict) -> None:
        for name, r in readings.items():
            if r is not None:
                self.setpoints[name] = float(r.position)
                self.targets[name] = float(r.position)
            self.velocities[name] = 0.0

    def _near_rest(self, readings: dict) -> bool:
        for name in ("lean", "nod"):
            r = readings.get(name)
            if r is None:
                return False
            if abs(r.position - self.cfg.joints[name].rest) * DEG_PER_COUNT > self.cfg.near_rest_deg:
                return False
        return True

    def _rest_done(self, readings: dict) -> bool:
        for name in ("lean", "nod"):
            r = readings.get(name)
            if r is None:
                return False
            if abs(r.position - self.cfg.joints[name].rest) * DEG_PER_COUNT > self.cfg.rest_done_within_deg:
                return False
        return True

    def _begin_rest(self) -> None:
        self.targets = {n: float(j.rest) for n, j in self.cfg.joints.items()}
        self._rest_speed = self.cfg.rest_speed_deg_s
        self.bus.set_torque_limit(self.cfg.rest_torque_fraction)
        self.mode = Mode.RESTING

    def _go_limp(self) -> None:
        try:
            self.bus.set_torque(False)
        except Exception:
            log.exception("torque off failed")
        self.velocities = {n: 0.0 for n in self.cfg.joints}
        self.mode = Mode.LIMP

    def _step_motion(self, dt: float) -> None:
        for name, jc in self.cfg.joints.items():
            vmax = jc.max_speed if self._rest_speed is None else min(jc.max_speed, self._rest_speed)
            self.setpoints[name], self.velocities[name] = step_axis(
                self.setpoints[name], self.velocities[name], self.targets[name],
                vmax / DEG_PER_COUNT, jc.max_accel / DEG_PER_COUNT, dt)
```

Note on timing: motion uses a fixed `dt = 1 / loop_hz`. If ticks run slow (e.g. a servo not answering makes a group read take ~50 ms), the head moves more slowly than configured — the safe direction.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/pytest tests/test_controller.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tests/fake_bus.py neck/controller.py tests/test_controller.py
git commit -m "feat: NeckController loop with no-jump startup and smooth look()"
```

---

### Task 7: Controller fault handling

**Files:**
- Create: `tests/test_controller_faults.py`
- Modify (only if a test exposes a bug): `neck/controller.py`

**Interfaces:**
- Consumes: `NeckController`, `Mode`, `Fault` (Task 6); `Level` (Task 5); `FakeBus`, `FakeClock` (Task 6).
- Produces: verified behavior of SOFT hold, SOFT→rest, HARD→limp, near-rest shortcut, latching, `clear_fault()`, HARD during rest.

- [ ] **Step 1: Write the tests**

`tests/test_controller_faults.py`:

```python
import pytest

from neck.controller import Mode, NeckController
from neck.safety import Level
from tests.fake_bus import FakeBus, FakeClock


def make(cfg, start="neutral"):
    clock = FakeClock()
    starts = {n: (j.neutral if start == "neutral" else j.rest) for n, j in cfg.joints.items()}
    bus = FakeBus(cfg, clock, start_positions=starts)
    ctrl = NeckController(cfg, bus, clock=clock, sleep=clock.advance)
    ctrl.start(run_thread=False)
    return ctrl, bus, clock


def run(ctrl, clock, seconds):
    for _ in range(round(seconds * ctrl.cfg.loop_hz)):
        clock.advance(1 / ctrl.cfg.loop_hz)
        ctrl.tick()


def test_hard_fault_goes_limp_immediately(cfg):
    ctrl, bus, clock = make(cfg)
    bus.servos["nod"].temperature = 66
    run(ctrl, clock, 0.02)
    assert ctrl.mode == Mode.LIMP
    assert bus.torque_log[-1] is False
    assert ctrl.latched.level == Level.HARD
    assert any("temperature" in r for r in ctrl.latched.reasons)


def test_status_bits_go_limp(cfg):
    ctrl, bus, clock = make(cfg)
    bus.servos["turn"].status = 0x20
    run(ctrl, clock, 0.02)
    assert ctrl.mode == Mode.LIMP


def test_lost_servo_goes_limp_after_five_misses(cfg):
    ctrl, bus, clock = make(cfg)
    bus.servos["nod"].responding = False
    run(ctrl, clock, 4 / 50)
    assert ctrl.mode == Mode.RUNNING
    run(ctrl, clock, 1 / 50)
    assert ctrl.mode == Mode.LIMP


def test_jam_holds_and_latches_without_resting(cfg):
    ctrl, bus, clock = make(cfg)
    ctrl.look(turn=30.0)
    run(ctrl, clock, 0.1)
    bus.servos["turn"].jammed = True
    jammed_at = bus.servos["turn"].position
    run(ctrl, clock, 1.0)
    assert ctrl.latched is not None and ctrl.latched.level == Level.SOFT
    assert any("tracking" in r for r in ctrl.latched.reasons)
    assert ctrl.mode == Mode.RUNNING                     # holding, not resting
    assert ctrl.setpoints["turn"] == pytest.approx(jammed_at, abs=1)
    assert ctrl.look(turn=0.0) is False                  # latched: commands ignored
    run(ctrl, clock, 3.0)
    assert ctrl.mode == Mode.RUNNING


def test_persistent_soft_fault_rests_then_goes_limp(cfg):
    ctrl, bus, clock = make(cfg)
    bus.servos["nod"].current_ma = 2000.0
    run(ctrl, clock, 0.1)
    assert ctrl.latched.level == Level.SOFT
    assert ctrl.mode == Mode.RUNNING
    run(ctrl, clock, 1.0)
    assert ctrl.mode == Mode.RESTING
    assert bus.torque_limit_log[-1] == 0.5
    run(ctrl, clock, 4.0)
    assert ctrl.mode == Mode.LIMP
    assert bus.torque_log[-1] is False
    for name in ("lean", "nod"):
        assert abs(bus.servos[name].position - cfg.joints[name].rest) <= 35   # within 3 degrees


def test_rest_moves_no_faster_than_rest_speed(cfg):
    ctrl, bus, clock = make(cfg)
    bus.servos["nod"].current_ma = 2000.0
    run(ctrl, clock, 1.2)
    assert ctrl.mode == Mode.RESTING
    n_before = len(bus.writes)
    run(ctrl, clock, 1.0)
    nod = [w["nod"] for w in bus.writes[n_before:]]
    per_tick = cfg.rest_speed_deg_s / (360 / 4096) / cfg.loop_hz
    assert max(abs(b - a) for a, b in zip(nod, nod[1:])) <= per_tick + 1


def test_soft_fault_near_rest_goes_limp_without_resting(cfg):
    ctrl, bus, clock = make(cfg, start="rest")
    bus.servos["lean"].current_ma = 2000.0
    run(ctrl, clock, 0.06)
    assert ctrl.mode == Mode.LIMP
    assert 0.5 not in bus.torque_limit_log


def test_hard_fault_during_rest_goes_limp(cfg):
    ctrl, bus, clock = make(cfg)
    bus.servos["nod"].current_ma = 2000.0
    run(ctrl, clock, 1.2)
    assert ctrl.mode == Mode.RESTING
    bus.servos["turn"].temperature = 70
    run(ctrl, clock, 0.02)
    assert ctrl.mode == Mode.LIMP
    assert ctrl.faults[-1].level == Level.HARD


def test_clear_fault_restarts_without_jump(cfg):
    ctrl, bus, clock = make(cfg)
    bus.servos["nod"].temperature = 66
    run(ctrl, clock, 0.02)
    assert ctrl.mode == Mode.LIMP
    bus.servos["nod"].temperature = 30
    ctrl.clear_fault()
    assert ctrl.mode == Mode.RUNNING
    assert ctrl.latched is None
    assert len(ctrl.faults) == 1
    assert not bus.jump_risk
    assert ctrl.look(turn=10.0) is True


def test_fault_record_keeps_readings(cfg):
    ctrl, bus, clock = make(cfg)
    bus.servos["nod"].temperature = 66
    run(ctrl, clock, 0.02)
    fault = ctrl.faults[0]
    assert fault.readings["nod"].temperature == 66
    assert fault.time == pytest.approx(0.02)
```

- [ ] **Step 2: Run the tests**

Run: `.venv/bin/pytest tests/test_controller_faults.py -v`
Expected: all pass. If any fail, fix `neck/controller.py` so the behavior matches the spec's "Fault handling" section (not the test), re-run, and note the fix in the commit message.

- [ ] **Step 3: Commit**

```bash
git add tests/test_controller_faults.py neck/controller.py
git commit -m "test: controller fault tiers (hold, rest, limp, near-rest, latch, clear)"
```

---

### Task 8: Controller lifecycle and threading

**Files:**
- Create: `tests/test_controller_lifecycle.py`
- Modify (only if a test exposes a bug): `neck/controller.py`

**Interfaces:**
- Consumes: `NeckController`, `Mode` (Task 6); `FakeBus`, `FakeClock` (Task 6).
- Produces: verified `rest()`, `close()`, context-manager, threaded loop and exception handling.

- [ ] **Step 1: Write the tests**

`tests/test_controller_lifecycle.py`:

```python
import time

import pytest

from neck.controller import Mode, NeckController
from neck.safety import Level
from tests.fake_bus import FakeBus, FakeClock


def make(cfg):
    clock = FakeClock()
    bus = FakeBus(cfg, clock, start_positions={n: j.neutral for n, j in cfg.joints.items()})
    ctrl = NeckController(cfg, bus, clock=clock, sleep=clock.advance)
    return ctrl, bus, clock


def test_rest_without_thread_drives_ticks_until_limp(cfg):
    ctrl, bus, _ = make(cfg)
    ctrl.start(run_thread=False)
    ctrl.rest()
    assert ctrl.mode == Mode.LIMP
    assert bus.torque_log[-1] is False
    for name in ("lean", "nod"):
        assert abs(bus.servos[name].position - cfg.joints[name].rest) <= 35


def test_rest_times_out_to_limp_when_joint_is_stuck(cfg):
    ctrl, bus, _ = make(cfg)
    ctrl.start(run_thread=False)
    bus.servos["lean"].jammed = True
    ctrl.rest(timeout=3.0)
    assert ctrl.mode == Mode.LIMP


def test_close_rests_and_closes_bus(cfg):
    ctrl, bus, _ = make(cfg)
    ctrl.start(run_thread=False)
    ctrl.close()
    assert ctrl.mode == Mode.LIMP
    assert not bus.opened


def test_close_is_idempotent_and_safe_before_start(cfg):
    ctrl, bus, _ = make(cfg)
    ctrl.close()
    ctrl.close()
    assert bus.torque_log == []
    assert ctrl.mode == Mode.IDLE


def test_context_manager_rests_on_keyboard_interrupt(cfg):
    ctrl, bus, _ = make(cfg)
    with pytest.raises(KeyboardInterrupt):
        with ctrl:
            ctrl.start(run_thread=False)
            ctrl.look(turn=15.0)
            raise KeyboardInterrupt
    assert ctrl.mode == Mode.LIMP
    assert not bus.opened


def test_tick_exception_latches_hard_fault_and_goes_limp(cfg):
    ctrl, bus, _ = make(cfg)
    ctrl.start(run_thread=False)
    bus.raise_on_read = OSError("device disconnected")
    ctrl._safe_tick()
    assert ctrl.mode == Mode.LIMP
    assert ctrl.latched.level == Level.HARD
    assert "controller exception" in ctrl.latched.reasons[0]
    assert bus.torque_log[-1] is False


def test_threaded_loop_runs_and_closes_cleanly(cfg):
    bus = FakeBus(cfg, time.monotonic, start_positions={n: j.neutral for n, j in cfg.joints.items()})
    ctrl = NeckController(cfg, bus)
    ctrl.start()
    assert ctrl.look(turn=10.0)
    time.sleep(0.3)
    assert len(bus.writes) >= 10
    ctrl.close()
    assert ctrl.mode == Mode.LIMP
    assert not bus.opened
    assert not ctrl._thread.is_alive()
```

- [ ] **Step 2: Run the tests**

Run: `.venv/bin/pytest tests/test_controller_lifecycle.py -v`
Expected: all pass (the threaded test takes ~2 s). If any fail, fix `neck/controller.py` to match the spec, re-run, and note the fix in the commit message.

- [ ] **Step 3: Run the whole suite**

Run: `.venv/bin/pytest -v`
Expected: all pass; hardware tests deselected.

- [ ] **Step 4: Commit**

```bash
git add tests/test_controller_lifecycle.py neck/controller.py
git commit -m "test: controller rest/close/threading and exception safety"
```

---

### Task 9: Tools — calibrate, log_current, neck_shell

**Files:**
- Create: `tools/__init__.py`, `tools/calibrate.py`, `tools/log_current.py`, `tools/neck_shell.py`, `tests/test_tools.py`

**Interfaces:**
- Consumes: `load_config`, `update_joint_config`, `DEFAULT_CONFIG`, `DEG_PER_COUNT` (Task 2); `ServoBus` (Task 3); `NeckController` (Task 6).
- Produces:
  - `tools.calibrate.derive_joint(name: str, neutral_raw: int, positive_raw: int, seen_min: int, seen_max: int, limp_raw: int, margin_deg: float = 2.0) -> dict` with keys `raw_min, raw_max, neutral, rest, direction`
  - `tools.log_current.summarize(rows: list[dict], hold_window_s: float) -> dict[str, dict]` with per-joint `peak_ma, hold_mean_ma, recommended_current_limit_ma, recommended_hold_limit_ma`

- [ ] **Step 1: Write the failing tests**

`tests/test_tools.py`:

```python
import pytest

from tools.calibrate import derive_joint
from tools.log_current import summarize


def test_derive_joint_insets_limits_and_picks_rest_nearest_limp_end():
    j = derive_joint("lean", neutral_raw=934, positive_raw=1000, seen_min=801, seen_max=1066,
                     limp_raw=808)
    assert j == {"raw_min": 824, "raw_max": 1043, "neutral": 934, "rest": 824, "direction": 1}


def test_derive_joint_negative_direction_and_high_rest():
    j = derive_joint("nod", neutral_raw=2939, positive_raw=2850, seen_min=2721, seen_max=3156,
                     limp_raw=3154)
    assert j["direction"] == -1
    assert j["rest"] == 3133


def test_derive_joint_turn_rests_at_neutral():
    j = derive_joint("turn", neutral_raw=2286, positive_raw=2400, seen_min=1850, seen_max=2941,
                     limp_raw=2290)
    assert j["rest"] == 2286


def test_derive_joint_rejects_no_movement():
    with pytest.raises(ValueError, match="did not move"):
        derive_joint("turn", 2286, 2290, 1850, 2941, 2286)


def test_derive_joint_rejects_neutral_outside_range():
    with pytest.raises(ValueError, match="neutral"):
        derive_joint("lean", 1070, 1000, 801, 1066, 808)


def test_derive_joint_rejects_wrapping_range():
    with pytest.raises(ValueError, match="wrap"):
        derive_joint("turn", 2286, 2400, 100, 4000, 2286)


def test_summarize_peak_and_holding_mean():
    rows = []
    for i in range(200):                          # 4 s at 50 Hz
        t = i / 50
        holding = t >= 1.0
        rows.append({"t": t, "joint": "nod", "current_ma": 900.0 if t < 1.0 else 300.0,
                     "holding": holding})
    s = summarize(rows, hold_window_s=2.0)["nod"]
    assert s["peak_ma"] == 900.0
    assert s["hold_mean_ma"] == pytest.approx(300.0)
    assert s["recommended_current_limit_ma"] == 1200      # ceil(1.3 * 900 / 50) * 50
    assert s["recommended_hold_limit_ma"] == 400          # ceil(1.3 * 300 / 50) * 50
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools'`.

- [ ] **Step 3: Create `tools/__init__.py`**

```python
"""Command-line tools for MoreNeck bring-up."""
```

- [ ] **Step 4: Implement `tools/calibrate.py`**

```python
"""Torque-off, hand-move calibration. Writes limits/neutral/rest/direction into neck/config.yaml.

Usage: .venv/bin/python tools/calibrate.py
The only bus write is torque OFF (makes the neck limp). HOLD THE HEAD before starting.
"""
from __future__ import annotations

import threading
import time

from neck.bus import ServoBus
from neck.config import DEFAULT_CONFIG, DEG_PER_COUNT, load_config, update_joint_config

MIN_MOVE_COUNTS = 20


def derive_joint(name, neutral_raw, positive_raw, seen_min, seen_max, limp_raw, margin_deg=2.0):
    if seen_max - seen_min > 2048:
        raise ValueError(f"{name}: range {seen_min}..{seen_max} looks like it crosses the 0/4095 wrap; "
                         "not supported")
    if not seen_min <= neutral_raw <= seen_max:
        raise ValueError(f"{name}: neutral {neutral_raw} outside recorded range {seen_min}..{seen_max}")
    if abs(positive_raw - neutral_raw) < MIN_MOVE_COUNTS:
        raise ValueError(f"{name}: joint did not move when asked for the positive direction")
    margin = round(margin_deg / DEG_PER_COUNT)
    raw_min, raw_max = seen_min + margin, seen_max - margin
    neutral = min(max(neutral_raw, raw_min), raw_max)
    if name == "turn":
        rest = neutral
    else:
        rest = raw_min if abs(limp_raw - raw_min) <= abs(limp_raw - raw_max) else raw_max
    return {"raw_min": raw_min, "raw_max": raw_max, "neutral": neutral, "rest": rest,
            "direction": 1 if positive_raw > neutral_raw else -1}


def positions(bus):
    readings = bus.read_all()
    missing = [n for n, r in readings.items() if r is None]
    if missing:
        raise SystemExit(f"no reply from: {missing}")
    return {n: r.position for n, r in readings.items()}


def main():
    cfg = load_config()
    bus = ServoBus.from_config(cfg)
    bus.open()
    try:
        input("HOLD THE HEAD. Press Enter to turn torque OFF (the neck will go limp)... ")
        bus.set_torque(False)

        input("\n1) Hold the head LEVEL and FACING STRAIGHT AHEAD, then press Enter... ")
        neutral = positions(bus)

        positive = {}
        prompts = {"turn": "turn the head to ITS LEFT", "nod": "tilt the head UP",
                   "lean": "lean the neck FORWARD"}
        for name, what in prompts.items():
            input(f"\n2) From neutral, {what} by ~10 degrees and hold it, then press Enter... ")
            positive[name] = positions(bus)[name]
            input("   Return to neutral, then press Enter... ")

        print("\n3) Move each joint gently to BOTH ends of its SAFE travel (stop before anything binds).")
        print("   Press Enter when done.")
        done = threading.Event()
        threading.Thread(target=lambda: (input(), done.set()), daemon=True).start()
        seen_min = dict(neutral)
        seen_max = dict(neutral)
        while not done.is_set():
            for n, p in positions(bus).items():
                seen_min[n] = min(seen_min[n], p)
                seen_max[n] = max(seen_max[n], p)
            print("   " + "  ".join(f"{n}: {seen_min[n]}..{seen_max[n]}" for n in neutral), end="\r")
            time.sleep(0.05)

        input("\n\n4) Gently let the head settle into its limp REST position, then press Enter... ")
        limp = positions(bus)

        updates = {n: derive_joint(n, neutral[n], positive[n], seen_min[n], seen_max[n], limp[n])
                   for n in cfg.joints}
        print("\nNew joint settings:")
        for n, u in updates.items():
            print(f"  {n}: {u}")
        if input(f"\nWrite these to {DEFAULT_CONFIG}? [y/N] ").strip().lower() == "y":
            update_joint_config(DEFAULT_CONFIG, updates)
            print("Saved.")
        else:
            print("Not saved.")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Implement `tools/log_current.py`**

```python
"""Run a scripted motion sequence, log every tick to CSV, and recommend current limits.

Usage: .venv/bin/python tools/log_current.py
MOVES THE NECK. Be at the robot with a hand on servo power.
"""
from __future__ import annotations

import csv
import logging
import math
import time
from collections import deque
from pathlib import Path

from neck.bus import ServoBus
from neck.config import load_config
from neck.controller import NeckController


def summarize(rows, hold_window_s):
    out = {}
    for joint in sorted({r["joint"] for r in rows}):
        jr = sorted((r for r in rows if r["joint"] == joint), key=lambda r: r["t"])
        peak = max(r["current_ma"] for r in jr)
        best_hold = 0.0
        window = deque()
        hold_start = None
        for r in jr:
            if not r["holding"]:
                hold_start = None
                window.clear()
                continue
            if hold_start is None:
                hold_start = r["t"]
            window.append((r["t"], r["current_ma"]))
            while window[0][0] < r["t"] - hold_window_s:
                window.popleft()
            if r["t"] - hold_start >= hold_window_s:
                best_hold = max(best_hold, sum(c for _, c in window) / len(window))

        def recommend(x):
            return int(math.ceil(1.3 * x / 50) * 50)

        out[joint] = {"peak_ma": peak, "hold_mean_ma": best_hold,
                      "recommended_current_limit_ma": recommend(peak),
                      "recommended_hold_limit_ma": recommend(best_hold)}
    return out


def sequence(cfg):
    """(turn, nod, lean, seconds) steps reaching 80% of each side of every joint's range."""
    def ends(name):
        j = cfg.joints[name]
        a, b = j.to_deg(j.raw_min), j.to_deg(j.raw_max)
        return 0.8 * min(a, b), 0.8 * max(a, b)

    steps = [(0, 0, 0, 3.0)]
    for name in ("turn", "nod", "lean"):
        lo, hi = ends(name)
        for deg in (lo, hi, 0.0):
            pose = {"turn": 0.0, "nod": 0.0, "lean": 0.0, name: deg}
            steps.append((pose["turn"], pose["nod"], pose["lean"], 3.0))
    t_lo, t_hi = ends("turn")
    n_lo, n_hi = ends("nod")
    l_lo, l_hi = ends("lean")
    steps += [(t_lo, n_hi, l_hi, 3.0), (t_hi, n_lo, l_lo, 3.0), (0, 0, 0, 3.0)]
    return steps


def main():
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    if input("This MOVES the neck. Hand on servo power? [y/N] ").strip().lower() != "y":
        return
    Path("logs").mkdir(exist_ok=True)
    path = Path("logs") / time.strftime("current-%Y%m%d-%H%M%S.csv")
    rows = []

    def on_tick(now, readings, setpoints, holding):
        for n, r in readings.items():
            if r is not None:
                rows.append({"t": now, "joint": n, "position": r.position, "setpoint": setpoints[n],
                             "current_ma": r.current_ma, "load": r.load, "temperature": r.temperature,
                             "voltage": r.voltage, "holding": holding[n]})

    with NeckController(cfg, ServoBus.from_config(cfg), on_tick=on_tick) as neck:
        neck.start()
        for turn, nod, lean, secs in sequence(cfg):
            if not neck.look(turn=turn, nod=nod, lean=lean):
                print("look() refused (fault latched):", neck.latched)
                break
            time.sleep(secs)

    if not rows:
        print("No readings were logged.")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nLogged {len(rows)} rows to {path}\n")
    for joint, s in summarize(rows, cfg.hold_window_s).items():
        print(f"{joint:5s} peak {s['peak_ma']:6.0f} mA  hold mean {s['hold_mean_ma']:6.0f} mA  "
              f"-> current_limit_ma {s['recommended_current_limit_ma']}, "
              f"hold_current_limit_ma {s['recommended_hold_limit_ma']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Implement `tools/neck_shell.py`**

```python
"""Interactive shell with a running NeckController bound to `neck`.

Usage: .venv/bin/python tools/neck_shell.py
Try: neck.look(turn=10), neck.state, neck.faults, neck.clear_fault(), neck.rest()
MOVES THE NECK. Be at the robot with a hand on servo power.
"""
import code
import logging

from neck.bus import ServoBus
from neck.config import load_config
from neck.controller import NeckController


def main():
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    if input("This enables torque and MOVES the neck. Hand on servo power? [y/N] ").strip().lower() != "y":
        return
    with NeckController(cfg, ServoBus.from_config(cfg)) as neck:
        neck.start()
        code.interact(banner="`neck` is running. Ctrl-D to rest and exit.", local={"neck": neck})


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the tests**

Run: `.venv/bin/pytest tests/test_tools.py -v`
Expected: all pass.

Run: `.venv/bin/pytest -v`
Expected: whole suite passes.

- [ ] **Step 8: Commit**

```bash
git add tools/__init__.py tools/calibrate.py tools/log_current.py tools/neck_shell.py tests/test_tools.py
git commit -m "feat: calibrate, log_current and neck_shell bring-up tools"
```

---

### Task 10: Hardware bring-up (user at the robot)

**Files:**
- Modify: `neck/config.yaml` (via `tools/calibrate.py` and the current-limit update in step 6)

**Interfaces:**
- Consumes: everything above.
- Produces: a calibrated `neck/config.yaml` with measured current limits; confirmation that every fault path works on the real neck.

**Gate:** Do not start any step without the user confirming they are at the robot, the head is supported or clear, and a hand is on servo power. Stop at the first unexpected result and report it to the user before continuing. Each step must pass before the next.

- [ ] **Step 1: Read-only checks**

Run: `.venv/bin/python tools/probe.py`
Expected: IDs `[1, 3, 4]` at 1,000,000 baud, ~7.4 V, status `0x00`, torque limit `1000`.

Run: `.venv/bin/pytest -m hardware -v`
Expected: 3 passed. **If `test_torque_limit_register_address_is_correct` fails, stop** — register 48 is wrong and rest would write the wrong register.

- [ ] **Step 2: Calibrate (torque off)**

Run: `.venv/bin/python tools/calibrate.py` and follow the prompts with the user.
Expected: new ranges within ~2° of the spec's measured values; direction signs recorded; saved.

Commit:
```bash
git add neck/config.yaml
git commit -m "chore: calibrated joint limits, neutral, rest and direction"
```

- [ ] **Step 3: Hold test (torque on, no commanded motion)**

Run: `.venv/bin/python tools/neck_shell.py`, then in the shell:
```python
neck.state["mode"], neck.faults
```
Expected: no lurch when torque engages; head moves slowly to neutral; mode `running`, no faults after 30 s. Exit with Ctrl-D (head rests, goes limp).

- [ ] **Step 4: One joint at a time**

In `neck_shell.py`, in this order, waiting for each move to finish and checking `neck.faults` is empty:
```python
neck.look(turn=10); neck.look(turn=-10); neck.look(turn=0)
neck.look(nod=5); neck.look(nod=-5); neck.look(nod=0)
neck.look(lean=5); neck.look(lean=-5); neck.look(lean=0)
```
Expected: each joint moves the way its name says (turn +: left, nod +: up, lean +: forward). If any is reversed, stop and re-run calibration.

- [ ] **Step 5: Measure current and set limits**

Run: `.venv/bin/python tools/log_current.py`
Expected: sequence completes without faults; per-joint recommendations printed.

Note: the 800 mA placeholder limits may trip during this run. If a SOFT current fault occurs, temporarily raise that joint's `current_limit_ma`/`hold_current_limit_ma` to 1500 in `neck/config.yaml`, re-run, then set the final values from the printed recommendations.

Edit `neck/config.yaml` with the recommended `current_limit_ma` and `hold_current_limit_ma` for each joint, then:
```bash
git add neck/config.yaml
git commit -m "chore: per-joint current limits from measured motion"
```

- [ ] **Step 6: Fault drills**

In `neck_shell.py`:
1. `neck.look(turn=20)` and gently hold the head so it can't turn → expect a SOFT tracking fault, head holds; `neck.faults[-1].reasons` mentions tracking. `neck.clear_fault()` → back to running.
2. `neck.look(nod=8)`, wait for the move, then `neck.rest()` → head lowers slowly and goes limp. `neck.clear_fault()` → back to running at neutral.
3. Temporarily set `voltage_warn_below: 7.6` and `voltage_soft_below: 7.5` in `neck/config.yaml`, start the shell → expect SOFT voltage fault, hold, then rest after 1 s. Restore the values (`git checkout neck/config.yaml`).
4. `neck.look(turn=-20)` and immediately press Ctrl-D (exit the shell) mid-move → head rests and goes limp before the program exits.

Expected: every drill behaves as described; no drops from a raised pose.

- [ ] **Step 7: Rest from several poses**

In `neck_shell.py`, for each of `look(lean=8, nod=10)`, `look(lean=-8, nod=-10, turn=30)`, `look(turn=-30)`: wait for the move, then `neck.rest()`, then `neck.clear_fault()` to continue.
Expected: each rest lowers slowly to the stops and goes limp without a hard drop.

- [ ] **Step 8: Final suite and wrap-up**

Run: `.venv/bin/pytest -v`
Expected: all offline tests pass.

Report to the user: calibrated values, measured current limits, and any drill that behaved unexpectedly.
