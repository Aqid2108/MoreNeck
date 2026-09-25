# Neck Control Layer — Design

**Date:** 2026-09-25
**Scope:** Pieces 1 (servo I/O + safety) and 2 (neck motion) of the MoreNeck robot.

## Goal and context

The end goal is a robot head that looks at people. That decomposes into four pieces:

| # | Piece | Status |
|---|---|---|
| 1 | Servo I/O + safety | **This spec** |
| 2 | Neck motion (`look(turn, nod, lean)`) | **This spec** |
| 3 | Perception (camera → face position) | Later spec |
| 4 | Behavior (what to look at, idle, re-acquire) | Later spec |

Pieces 3 and 4 will only ever interact with the neck through `NeckController.look()`,
`rest()`, `state` and `faults`.

**Done means:** code can command `look(turn, nod, lean)` in degrees; the three joints move
together smoothly within soft limits; current, load, temperature and voltage are monitored
continuously while moving *and* holding; the tiered fault response below works and every
fault is logged.

## Hardware (measured 2026-09-25)

- **Host:** Mac, Python 3.12. Everything runs on the Mac; the ESP32 only drives the eye displays.
- **Bus adapter:** WCH CH343 (USB `1A86:55D3`), macOS port `/dev/cu.usbmodem*`. Located by
  USB VID **and** PID (`port_vid`, `port_pid`), never a hard-coded path.
- **Bus:** 1,000,000 baud. Servos: Feetech STS3215 family, model word `0x0309`, firmware 3.10.
- **Power:** 2S pack through a regulated power module; measured 7.4 V at every servo.
- **Mechanism:** serial chain. Head is heavy and cantilevered forward; lean and nod carry a
  gravity load at all times. Camera is mounted on the head.

| Joint | ID | Measured travel (raw) | Travel | EEPROM profile | Resting (limp) position |
|---|---|---|---|---|---|
| Lean (base) | 1 | 801 – 1066 | 23° | 80 °C, 4–12 V, prot. current 1950 mA | 808 — on end stop |
| Nod (middle) | 3 | 2721 – 3156 | 38° | 70 °C, 4–8 V, prot. current 3250 mA | 3154 — on end stop |
| Turn (top) | 4 | 1850 – 2941 | 96° | 80 °C, 4–12 V, prot. current 1950 mA | 2286 — user-set straight ahead |

- EEPROM angle limits are unset (0–4095) on all servos. No travel range crosses the 0/4095 wrap.
- IDs 1 and 4 have a different EEPROM profile from ID 3; whether they are 12 V variants is
  unconfirmed. All limits in this design are therefore **per joint**.
- Limp rest (lean and nod on their end stops) was judged "somewhat safe": acceptable to drop
  onto from nearby, not from a raised pose.

## Library

Use `python-st3215` pinned to commit `143cff5e74453a2fd24e7a3d78aeef644364768a`, **low-level
layer only**: `PortHandler`, `protocol_packet_handler`, `GroupSyncRead`, `GroupSyncWrite`.

Do not use its high-level per-servo helpers in the control loop:

- `ReadCurrent` reads only the low byte of the 2-byte current register (wraps above ~1.66 A).
- Every high-level call is a separate per-servo transaction; the class constructs a
  `GroupSyncWrite` but never uses it.

## Architecture

```
neck/
  config.yaml      per-joint: id, name, raw_min, raw_max, neutral, rest, direction (+1/-1),
                   max_speed, max_accel, current_limit_ma, hold_current_limit_ma,
                   temp_warn_c, temp_soft_c, temp_hard_c; global: voltage thresholds, loop_hz
  config.py        loads/validates config.yaml; deg <-> raw conversion (4096 counts / 360°)
  bus.py           ServoBus — the only code that touches the serial port
  safety.py        check(...) -> Verdict — pure, no I/O
  controller.py    NeckController — owns the loop and the bus
tools/
  probe.py         read-only bus scan (ping, EEPROM + runtime registers)
  calibrate.py     torque-off hand-move readout; records range, neutral, direction into config.yaml
  log_current.py   CSV log of per-tick readings during motion, for tuning current limits
tests/
  fake_bus.py      simulated bus implementing ServoBus's interface
  test_safety.py, test_controller.py, test_bus_parsing.py
  test_hardware.py (marked `hardware`, skipped by default)
```

**Dependency rule:** only `controller.py` uses `bus.py`. Pieces 3 and 4 use only
`NeckController`. This gives one owner of the servos and one place safety is enforced.

### `ServoBus` (bus.py)

- `open()` / `close()` — find port by VID/PID, open at 1 Mbaud.
- `ping(ids) -> set[int]`
- `read_all() -> dict[joint, Reading | None]` — one `GroupSyncRead` of registers
  `0x38`–`0x46` (15 bytes). `Reading` = position, speed, load, voltage (V), temperature (°C),
  status byte, moving flag, current (mA, **both bytes**, sign-magnitude decoded). `None` for a
  servo that did not reply.
- `write_goals(dict[joint, raw])` — one `GroupSyncWrite` of goal position.
- `set_torque(joints, on: bool)`
- `set_torque_limit(joints, fraction)` — RAM register, not EEPROM.
- `configure_for_streaming(joints)` — set the servo's own goal speed/acceleration so it tracks
  streamed setpoints directly (maximum speed, no internal acceleration ramp).

No EEPROM writes anywhere in this spec.

### `safety.check` (safety.py)

`check(readings, setpoints, config, history) -> Verdict`, where `Verdict` is `OK`, `WARN(reasons)`,
`SOFT(reasons)` or `HARD(reasons)`; the most severe condition wins. `history` is a small
rolling buffer owned by the controller (last ~2 s of readings per joint, consecutive-miss and
consecutive-over counters).

| Check | WARN | SOFT | HARD |
|---|---|---|---|
| Communication | 1 missed read | — | 5 consecutive misses (100 ms) |
| Servo status byte | — | — | any bit set |
| Voltage | < 6.8 V | < 6.5 V | < 6.0 V or > 8.0 V |
| Temperature (per joint; defaults) | ≥ 55 °C | ≥ 60 °C | ≥ 65 °C |
| Current spike | — | > `current_limit_ma` for 3 consecutive ticks | — |
| Current while holding | — | 2 s mean > `hold_current_limit_ma` | — |
| Tracking error | — | \|present − setpoint\| > 8° for > 0.3 s | — |
| Outside soft limits | — | present position beyond `raw_min`/`raw_max` by > 2° | — |

The limp rest positions sit ~1.5–2° outside the soft limits (the head rests on its stops), so
the "outside soft limits" check is not armed for a joint until that joint has first entered
its soft range after startup or `clear_fault()`.

Initial `current_limit_ma` and `hold_current_limit_ma` are 800 mA for every joint. They are
replaced with measured values in bring-up step 5.

### `NeckController` (controller.py)

**Public API**

- `start()` — open bus, run startup sequence, launch loop thread.
- `look(turn, nod, lean)` — degrees relative to neutral; sets the target, clamped to soft
  limits. Ignored (returns `False`) while a fault is latched.
- `rest()` — run the rest sequence, then go limp.
- `clear_fault()` — unlatch after a fault; re-runs the startup sequence.
- `close()` — `rest()` then close the bus. Also runs on context-manager exit, `KeyboardInterrupt`
  and `atexit`.
- `state` — latest readings, setpoints, targets. `faults` — list of fault records.

**Startup sequence (no jump):** read present positions → write them as goal positions →
`configure_for_streaming` → torque on → setpoint = present → target = neutral pose.

**Tick (loop_hz = 50):**

1. `read_all()`
2. `safety.check()` → handle verdict (below)
3. Step each joint's setpoint toward its target, limited by `max_speed` and `max_accel`
   (trapezoidal velocity profile, recomputed every tick so moving targets are followed
   smoothly).
4. `write_goals()`

**Motion defaults** (tunable in config.yaml)

| Joint | max_speed | max_accel |
|---|---|---|
| Turn | 120 °/s | 400 °/s² |
| Nod | 90 °/s | 300 °/s² |
| Lean | 45 °/s | 150 °/s² |

**Fault handling**

- **WARN:** log only.
- **SOFT:** latch fault; target := present position (hold). If the condition persists for 1 s,
  run the rest sequence. If it clears, keep holding (fault stays latched until `clear_fault()`).
- **HARD:** latch fault; torque off on all joints immediately.
- **Near-rest shortcut:** if lean and nod are both within 5° of their rest positions, any
  SOFT or HARD fault → torque off immediately.
- **Rest sequence:** torque limit → 50%; targets := rest pose; max speed 15 °/s on all joints;
  when lean and nod are within 3° of rest, torque off. If rest has not finished within
  `rest_timeout_s` (10 s) — e.g. something is under the head — record a fault and torque off.
- **While limp:** torque-off is re-sent every tick (it is an unacknowledged broadcast).
- **Startup goal check:** after writing goal = present, read the goal registers back and only
  enable torque once they match (up to 3 attempts, else `BusError`).
- **Rest pose vs. end stops:** the rest *targets* for lean and nod are the soft-limit ends
  nearest their stops (not the stops themselves), so the servo never stalls against a stop
  under torque; going limp lets the last ~2° settle onto the stop.
- Every fault record: timestamp, tier, reasons, full readings for all joints.

**Known gap:** the STS3215 has no communication timeout. If the Python process dies without
running `close()` or USB disconnects, servos hold their last goal with torque on. The head
stays put rather than falling; recovery is re-running the program or cutting servo power.

## Initial config.yaml values

Soft limits are the measured travel pulled in by 2° (23 counts).

| Joint | ID | raw_min | raw_max | rest (raw) | neutral (raw) | direction |
|---|---|---|---|---|---|---|
| Lean | 1 | 824 | 1043 | 824 | 934 (midpoint) | +1 |
| Nod | 3 | 2744 | 3133 | 3133 | 2939 (midpoint) | +1 |
| Turn | 4 | 1873 | 2918 | 2286 | 2286 | +1 |

Neutral for lean and nod starts at the range midpoint, and every direction starts at +1.
`calibrate.py` replaces these during bring-up step 1: the user holds the head level and
facing forward to capture neutral, then moves each joint in its positive direction
(turn = left, nod = up, lean = forward) to capture direction. The turn range is asymmetric
about 2286 (−36° / +56°); calibration confirms or corrects that neutral.

## Testing

**Offline (pytest, default):**

- `test_safety.py` — every threshold row in the table, debounce counts, 2 s holding mean,
  tracking-error timing, most-severe-wins.
- `test_controller.py` against `FakeBus` (simulated servos follow setpoints with first-order
  lag; injectable jam, dropped replies, temperature rise, voltage sag, status fault):
  no-jump startup; speed/accel limits respected; targets clamped to soft limits;
  SOFT → hold → rest → limp; HARD → immediate limp; near-rest shortcut; latch and
  `clear_fault()`; `close()` and `KeyboardInterrupt` always rest.
- `test_bus_parsing.py` — decode recorded group-read packets, including 2-byte current,
  sign-magnitude speed/current, and missing replies.

**Hardware (`pytest -m hardware`, run only on request, user at the robot with a hand on
servo power; each step passes before the next):**

1. Read-only: `bus.read_all()` matches `tools/probe.py`; run `calibrate.py`.
2. Hold: torque on at present position, no motion — verify no jump; log holding current 30 s.
3. Turn only: ±10°, then full range at half speed; log current.
4. Nod only, then lean only, same procedure.
5. Combined `look()` motion under `log_current.py`; set per-joint `current_limit_ma` and
   `hold_current_limit_ma` above the measured peaks and 2 s means.
6. Fault drills: gently obstruct the head (tracking error → hold → rest); temporarily set the
   voltage WARN/SOFT thresholds above 7.4 V; Ctrl-C mid-move.
7. Rest and shutdown from several poses.

## Project setup

- Git repository rooted at `MoreNeck/` (separate from the home-directory repo).
- `.venv` (Python 3.12); `pyproject.toml` with `python-st3215` (git dependency pinned to the
  commit above), `pyyaml`, `pytest`.

## Out of scope

Camera and perception, behavior logic, ESP32 eye displays, EEPROM changes (including angle
limits and IDs), ROS integration.
