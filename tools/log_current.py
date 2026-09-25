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
