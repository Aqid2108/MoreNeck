"""20 Hz loop: perception snapshot -> engine -> neck (or printed commands in dry-run)."""
from __future__ import annotations

import logging
import threading
import time

from behavior.engine import NeckStatus
from perception.pipeline import people_or_empty

log = logging.getLogger("behavior.runner")


class Runner:
    def __init__(self, engine, perception, neck=None, stale_after_s: float = 0.3,
                 clock=time.monotonic, sleep=time.sleep, printer=print, on_tick=None):
        self.engine = engine
        self.perception = perception
        self.neck = neck
        self.stale_after_s = stale_after_s
        self.clock = clock
        self._sleep = sleep
        self.printer = printer
        self._last_print = float("-inf")
        self.on_tick = on_tick          # diagnostic record per tick (see tools/behave.py --log)

    def neck_status(self) -> NeckStatus:
        if self.neck is None:                       # dry-run
            return NeckStatus(running=True, pose=None)
        from neck.controller import Mode

        st = self.neck.state
        running = st["mode"] == Mode.RUNNING and st["latched"] is None
        return NeckStatus(running=running, pose=self.neck.pose())

    def status_line(self) -> str:
        c = self.engine.command
        cmd = f"turn {c.turn:+6.1f} nod {c.nod:+5.1f} lean {c.lean:4.1f}" if c else "no command"
        return f"{self.engine.state.value:9s} target {self.engine.attention.target_id}  {cmd}"

    def tick(self):
        now = self.clock()
        snapshot = self.perception.latest()
        people = people_or_empty(snapshot, now, self.stale_after_s)
        status = self.neck_status()
        cmd = self.engine.tick(people, snapshot if snapshot.source_ok else None, status, now)
        if self.on_tick is not None:
            c = self.engine.command
            self.on_tick({
                "t": now, "state": self.engine.state.value, "target": self.engine.attention.target_id,
                "glance": self.engine.glance_id, "sent": cmd is not None,
                "command": [c.turn, c.nod, c.lean] if c else None,
                "neck_pose": list(status.pose) if status.pose else None,
                "snapshot_ts": snapshot.timestamp, "source_ok": snapshot.source_ok,
                "snapshot_pose": list(snapshot.neck_pose) if snapshot.neck_pose else None,
                "people": [{"id": p.id, "yaw": p.head_deg[0], "pitch": p.head_deg[1],
                            "conf": p.confidence, "visible": p.visible, "facing": p.facing_score,
                            "dist": p.distance_m, "motion": p.motion} for p in people],
            })
        if cmd is not None:
            if self.neck is not None:
                if not self.neck.look(cmd.turn, cmd.nod, cmd.lean):
                    log.warning("neck refused look(); pausing until it runs again")
            elif now - self._last_print >= 0.5:
                self.printer(self.status_line())
                self._last_print = now
        return cmd

    def run(self, stop: threading.Event) -> None:
        period = 1.0 / self.engine.p.loop_hz
        while not stop.is_set():
            start = self.clock()
            try:
                self.tick()
            except Exception:
                log.exception("behavior tick failed")
            self._sleep(max(0.0, period - (self.clock() - start)))

    def clear_fault(self) -> None:
        if self.neck is not None:
            self.neck.clear_fault()
