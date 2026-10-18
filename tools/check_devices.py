"""Read-only hardware check: servo bus port + servo pings, head camera, pose model. Never enables torque.

Usage: python tools/check_devices.py      (Windows: .venv\\Scripts\\python tools\\check_devices.py)
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # repo root, for package imports when run as a script


def check_servos() -> bool:
    from serial.tools import list_ports

    from neck.bus import BusError, ServoBus, find_port
    from neck.config import load_config

    cfg = load_config()
    print("Serial ports:")
    for p in list_ports.comports():
        mark = "  <- servo board" if (p.vid, p.pid) == (cfg.port_vid, cfg.port_pid) else ""
        vidpid = f"{p.vid:04X}:{p.pid:04X}" if p.vid is not None else "-"
        print(f"  {p.device:<14} {vidpid:<10} {p.description}{mark}")
    try:
        port = cfg.port or find_port(cfg.port_vid, cfg.port_pid)
    except BusError as exc:
        print("  FAIL:", exc)
        return False
    bus = ServoBus.from_config(cfg)
    try:
        bus.open()
        found = bus.ping()
        readings = bus.read_all()
    except Exception as exc:  # report, don't crash the check
        print(f"  FAIL: could not talk to the servos on {port}: {exc!r}")
        return False
    finally:
        bus.close()
    ok = True
    for name in ("turn", "nod", "lean"):
        r = readings.get(name)
        if name in found and r is not None:
            print(f"  {name:<5} id {cfg.joints[name].id}: OK  position {r.position}  "
                  f"{r.voltage:.1f} V  {r.temperature} C")
        else:
            print(f"  {name:<5} id {cfg.joints[name].id}: NOT RESPONDING (servo power on? cable?)")
            ok = False
    return ok


def check_camera() -> bool:
    import time

    from perception.config import load_config
    from perception.sources import CameraError, CameraSource, list_cameras

    cfg = load_config()
    print("Cameras:")
    for c in list_cameras():
        vidpid = f"{c.vid:04X}:{c.pid:04X}" if getattr(c, "vid", None) is not None else "-"
        print(f"  index {c.index:<6} {vidpid:<10} {c.name}")
    src = CameraSource.from_config(cfg)
    try:
        src.open()
    except CameraError as exc:
        print("  FAIL:", exc)
        return False
    try:
        deadline = time.monotonic() + 5.0
        frame = None
        while frame is None and time.monotonic() < deadline:
            time.sleep(0.05)
            frame = src.read()
    finally:
        src.close()
    if frame is None:
        print("  FAIL: head camera opened but delivered no frames (in use by another app? privacy setting?)")
        return False
    print(f"  head camera OK at index {src.opened_index}: {frame.image.shape[1]}x{frame.image.shape[0]}")
    return True


def check_model() -> bool:
    from perception.config import load_config

    path = REPO / load_config().model
    if not path.exists():
        print(f"Pose model: MISSING {path} (run setup.bat)")
        return False
    import torch

    print(f"Pose model: OK {path.name}  (torch {torch.__version__}, running on CPU)")
    return True


def main():
    results = {}
    for name, fn in (("servos", check_servos), ("camera", check_camera), ("model", check_model)):
        print(f"\n--- {name} ---")
        try:
            results[name] = fn()
        except Exception as exc:
            print(f"  FAIL: {exc!r}")
            results[name] = False
    print("\nSummary:", "  ".join(f"{k} {'OK' if v else 'FAIL'}" for k, v in results.items()))
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
