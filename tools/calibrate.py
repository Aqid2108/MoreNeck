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
MIN_RANGE_DEG = 15.0


def derive_joint(name, neutral_raw, positive_raw, seen_min, seen_max, limp_raw, margin_deg=2.0):
    if seen_max - seen_min > 2048:
        raise ValueError(f"{name}: range {seen_min}..{seen_max} looks like it crosses the 0/4095 wrap; "
                         "not supported")
    if not seen_min <= neutral_raw <= seen_max:
        raise ValueError(f"{name}: neutral {neutral_raw} outside recorded range {seen_min}..{seen_max}")
    if abs(positive_raw - neutral_raw) < MIN_MOVE_COUNTS:
        raise ValueError(f"{name}: joint did not move when asked for the positive direction")
    if (seen_max - seen_min) * DEG_PER_COUNT < MIN_RANGE_DEG:
        raise ValueError(f"{name}: only moved {(seen_max - seen_min) * DEG_PER_COUNT:.0f} deg in the range "
                         f"step; move it to both ends of its safe travel")
    margin = round(margin_deg / DEG_PER_COUNT)
    raw_min, raw_max = seen_min + margin, seen_max - margin
    neutral = min(max(neutral_raw, raw_min), raw_max)
    if name == "turn":
        rest = neutral
    else:
        rest = limp_raw   # lean and nod are only stable limp where they settle, on their back stops
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

        try:
            updates = {n: derive_joint(n, neutral[n], positive[n], seen_min[n], seen_max[n], limp[n])
                       for n in cfg.joints}
        except ValueError as exc:
            print(f"\nCalibration rejected: {exc}\nNothing was saved; please run it again.")
            return
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
