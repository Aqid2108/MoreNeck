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

## Windows (any laptop)

Runs on the CPU, so no GPU or GPU drivers are needed.

1. Install **Python 3.13** (64-bit) from python.org or `winget install Python.Python.3.13`.
   (3.13+ is required on Windows: older versions have a 15.6 ms clock, too coarse for the 50 Hz loop.)
2. Plug in the servo board and the head camera. The board must show in Device Manager under
   *Ports* as **USB-Enhanced-SERIAL CH343 (COMx)**. If it shows as an unknown device, install the
   WCH CH343 driver (CH343SER) first.
3. Double-click **`setup.bat`**. It creates `.venv`, installs CPU-only PyTorch and the project,
   downloads `data/models/yolo11n-pose.pt`, then runs `tools/check_devices.py`, which reports
   the COM port, each servo (read-only ping), the head camera and the model.
4. Double-click **`run.bat`** to start the head (calm preset, camera window). In the window,
   `q` quits and `c` clears a neck fault. Other modes: `run.bat --preset curious`,
   `run.bat --dry-run` (no servos), `run.bat --preset shy --view --log`.

Hardware is found by USB ID, so no COM port or camera index needs setting: servo board
`1A86:55D3`, head camera `0C45:6366` (the laptop's built-in camera is ignored). To force a port,
set `port: COM6` in `neck/config.yaml`.

Other tools (from this folder): `.venv\Scripts\python tools\check_devices.py`,
`.venv\Scripts\python tools\neck_shell.py` (Ctrl-Z then Enter to rest and exit),
`.venv\Scripts\python tools\perception_view.py`, `.venv\Scripts\pytest`.

If the camera won't open, close other apps using it (Teams, Zoom, Camera) and check
*Settings > Privacy & security > Camera > Let desktop apps access your camera*.

Speed: the camera delivers 30 fps at 1280×720 on Windows; the CPU pose model is the limit. Plug
the laptop in and use the *Balanced* or *Best performance* power mode: a plan whose *Maximum
processor state* is below 100% turns Intel Turbo Boost off and roughly halves the frame rate
(measured 8 fps on a Core Ultra 7 155H capped at 95%). The detector uses up to 6 CPU threads;
override with the `MORENECK_CPU_THREADS` environment variable.

## Setup (macOS / Linux)

Python 3.12.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev,vision]'
.venv/bin/pytest
```

Offline tests run by default. Tests marked `hardware`, `camera` or `model` need the robot, the head
camera or the YOLO pose weights (`data/models/yolo11n-pose.pt`, not included; `setup.bat`
downloads it on Windows, elsewhere fetch it from the Ultralytics assets release v8.3.0).

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
