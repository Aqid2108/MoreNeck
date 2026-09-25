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
