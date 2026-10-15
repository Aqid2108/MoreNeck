# MoreNeck

A robot head that looks at people. Three Feetech STS3215 bus servos (lean, nod, turn) move a
head with a USB camera; a pose model finds people and a behavior engine decides who to look at
and how to move.

## Layout

- `neck/` — 50 Hz servo controller: trapezoidal motion, verified servo settings, tiered faults, rest mode.
- `perception/` — camera capture, YOLO11 pose detection, tracking, facing / distance / motion cues.
- `behavior/` — attention state machine with presets (`calm`, `curious`, `shy`), head kinematics, smooth pursuit.
- `tools/` — `neck_shell.py`, `calibrate.py`, `perception_view.py`, `behave.py` and diagnostics.
- `docs/` — design specs, implementation plans and `NEXT_STEPS.md`.

## Setup

Python 3.12.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev,vision]'
.venv/bin/pytest
```

Offline tests run by default. Tests marked `hardware`, `camera` or `model` need the robot, the head
camera or the YOLO pose weights (`data/models/yolo11n-pose.pt`, not included).

## Running

```bash
.venv/bin/python tools/behave.py --preset calm --view
```

This moves the neck: keep a hand on servo power. `--dry-run` runs without the servos, `--log`
writes per-tick logs to `logs/`.

## Status

Neck control and perception are done; behavior is in hardware tuning on the `behavior` branch.
Eye displays (ESP32) are planned.

## License

MIT — see [LICENSE](LICENSE). The optional `vision` extra installs Ultralytics, which is
licensed separately under AGPL-3.0.
