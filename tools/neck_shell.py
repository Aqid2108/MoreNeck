"""Interactive shell with a running NeckController bound to `neck`. Every tick is logged to logs/shell-*.csv.

Usage: .venv/bin/python tools/neck_shell.py
Try: neck.look(turn=10), neck.state, neck.faults, neck.clear_fault(), neck.rest()
MOVES THE NECK. Be at the robot with a hand on servo power.
"""
import code
import logging
import time

from neck.bus import ServoBus
from neck.config import load_config
from neck.controller import NeckController
from tools.ticklog import TickLog


def main():
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    if input("This enables torque and MOVES the neck. Hand on servo power? [y/N] ").strip().lower() != "y":
        return
    holder = {}
    log = TickLog(extra=lambda: {"mode": holder["neck"].mode.value})
    try:
        with NeckController(cfg, ServoBus.from_config(cfg), on_tick=log) as neck:
            holder["neck"] = neck
            neck.start()
            code.interact(banner="`neck` is running. Ctrl-D to rest and exit.", local={"neck": neck})
    finally:
        if log.rows:
            print("tick log saved to", log.save(time.strftime("logs/shell-%Y%m%d-%H%M%S.csv")))


if __name__ == "__main__":
    main()
