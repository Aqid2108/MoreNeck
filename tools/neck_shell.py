"""Interactive shell with a running NeckController bound to `neck`. Every tick is logged to logs/shell-*.csv.

Usage: python tools/neck_shell.py
Try: neck.look(turn=10), neck.state, neck.faults, neck.clear_fault(), neck.rest()
MOVES THE NECK. Be at the robot with a hand on servo power.
"""
import code
import logging
import time

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `tools.*` when run as a script

from neck.bus import ServoBus
from neck.config import load_config
from neck.controller import NeckController
from tools.ticklog import TickLog  # noqa: E402


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
            eof_key = "Ctrl-Z then Enter" if sys.platform == "win32" else "Ctrl-D"
            code.interact(banner=f"`neck` is running. {eof_key} to rest and exit.", local={"neck": neck})
    finally:
        if log.rows:
            repo = Path(__file__).resolve().parent.parent   # logs/ in the repo, whatever the cwd
            print("tick log saved to", log.save(repo / time.strftime("logs/shell-%Y%m%d-%H%M%S.csv")))


if __name__ == "__main__":
    main()
