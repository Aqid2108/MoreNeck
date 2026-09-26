"""NeckController: the 50 Hz loop that owns the servo bus.

Only this module drives ServoBus. Callers (perception, behavior, tools) use
look(), rest(), clear_fault(), close(), state and faults.
"""
from __future__ import annotations

import atexit
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

from neck.bus import BusError
from neck.config import DEG_PER_COUNT
from neck.motion import step_axis
from neck.safety import Level, Verdict, check, new_history

log = logging.getLogger("neck.controller")


class Mode(str, Enum):
    IDLE = "idle"        # not started, or closed
    RUNNING = "running"  # torque on, following targets
    RESTING = "resting"  # lowering to the rest pose at reduced speed/torque
    RESTED = "rested"    # holding the rest pose with torque on (limp_after_rest: false)
    LIMP = "limp"        # torque off


@dataclass
class Fault:
    time: float
    level: Level
    reasons: list[str]
    readings: dict = field(repr=False)


def _reason_keys(reasons: list[str]) -> frozenset:
    """'SOFT: nod: current 2000 mA ...' -> ('SOFT:', 'nod:', 'current'): identity without the numbers."""
    return frozenset(tuple(r.split()[:3]) for r in reasons)


class NeckController:
    def __init__(self, cfg, bus, clock=time.monotonic, sleep=time.sleep, on_tick=None):
        self.cfg = cfg
        self.bus = bus
        self.clock = clock
        self._sleep = sleep
        self.on_tick = on_tick
        self._lock = threading.RLock()
        self.mode = Mode.IDLE
        self.latched: Fault | None = None
        self.faults: list[Fault] = []
        self.readings: dict = {}
        self.setpoints: dict[str, float] = {}
        self.velocities: dict[str, float] = {}
        self.targets: dict[str, float] = {}
        self._history = new_history(cfg)
        self._soft_since: float | None = None
        self._rest_speed: float | None = None
        self._rest_started = 0.0
        self._last_warn: list[str] | None = None
        self._opened = False
        self._closed = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ---- lifecycle -------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def start(self, run_thread: bool = True) -> None:
        self.bus.open()
        self._opened = True
        self._startup()
        atexit.register(self.close)
        if run_thread:
            self._thread = threading.Thread(target=self._run, name="neck-loop", daemon=True)
            self._thread.start()

    def _startup(self) -> None:
        with self._lock:
            readings = self.bus.read_all()
            missing = sorted(n for n, r in readings.items() if r is None)
            if missing:
                raise BusError(f"servos not responding: {', '.join(missing)}")
            present = {n: r.position for n, r in readings.items()}
            self._write_verified_goals(present)    # goal = present BEFORE torque on: no jump
            self.bus.configure_for_streaming()
            self.bus.set_torque_limit(1.0)
            self.bus.set_torque(True)
            self.readings = readings
            self.setpoints = {n: float(p) for n, p in present.items()}
            self.velocities = {n: 0.0 for n in present}
            self.targets = {n: float(j.neutral) for n, j in self.cfg.joints.items()}
            self._history = new_history(self.cfg)
            self._soft_since = None
            self._rest_speed = None
            self.mode = Mode.RUNNING

    def _write_verified_goals(self, present: dict[str, int], attempts: int = 3) -> None:
        # Goal writes are unacknowledged broadcasts; a lost one would make torque-on drive the
        # head back to a stale goal at full speed. Read the registers back before trusting them.
        goals = {}
        for _ in range(attempts):
            self.bus.write_goals(present)
            goals = self.bus.read_goals()
            if all(g is not None and abs(g - present[n]) <= 1 for n, g in goals.items()):
                return
        raise BusError(f"goal positions did not verify before torque-on: wrote {present}, read {goals}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        try:
            if self.mode in (Mode.RUNNING, Mode.RESTING):
                self.rest()
        finally:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            if self._opened:
                self.bus.close()

    # ---- public commands -------------------------------------------------

    def look(self, turn: float = 0.0, nod: float = 0.0, lean: float = 0.0) -> bool:
        angles = {"turn": turn, "nod": nod, "lean": lean}
        if not all(math.isfinite(a) for a in angles.values()):
            log.warning("look() ignored non-finite angles %s", angles)
            return False
        with self._lock:
            if self.mode != Mode.RUNNING or self.latched is not None:
                return False
            for name, deg in angles.items():
                jc = self.cfg.joints[name]
                self.targets[name] = jc.clamp_raw(jc.to_raw(deg))
        return True

    def rest(self, timeout: float = 10.0) -> None:
        with self._lock:
            if self.mode in (Mode.IDLE, Mode.LIMP, Mode.RESTED):
                return
            if self.mode == Mode.RUNNING:
                self._begin_rest()
        start = self.clock()
        threaded = self._thread is not None and self._thread.is_alive()
        while self.mode == Mode.RESTING:
            if self.clock() - start > timeout:
                log.error("rest did not finish within %.1f s; going limp", timeout)
                with self._lock:
                    self._go_limp()
                break
            self._sleep(1 / self.cfg.loop_hz)
            if not threaded:
                self.tick()

    def clear_fault(self) -> None:
        with self._lock:
            self.latched = None
            self._startup()

    @property
    def state(self) -> dict:
        with self._lock:
            return {
                "mode": self.mode,
                "latched": self.latched,
                "readings": dict(self.readings),
                "setpoints": dict(self.setpoints),
                "targets": dict(self.targets),
            }


    def pose(self) -> tuple[float, float, float] | None:
        """Current (turn, nod, lean) in degrees from neutral, from measured positions."""
        with self._lock:
            try:
                return tuple(self.cfg.joints[n].to_deg(self.readings[n].position)
                             for n in ("turn", "nod", "lean"))
            except (KeyError, AttributeError):   # not started yet, or a servo didn't answer
                return None

    # ---- loop ------------------------------------------------------------

    def _run(self) -> None:
        period = 1 / self.cfg.loop_hz
        next_t = time.monotonic()
        while not self._stop.is_set():
            self._safe_tick()
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_t = time.monotonic()

    def _safe_tick(self) -> None:
        try:
            self.tick()
        except Exception as exc:
            with self._lock:
                if self.mode == Mode.LIMP and self.latched is not None:
                    log.debug("tick failed while limp: %r", exc)
                    return
            log.exception("tick failed")
            with self._lock:
                self._fault(Verdict(Level.HARD, [f"HARD: controller exception: {exc!r}"], limp=True),
                            self.readings, self.clock())
                self._go_limp()

    def tick(self) -> None:
        with self._lock:
            if self.mode == Mode.IDLE:
                return
            now = self.clock()
            readings = self.bus.read_all()
            self.readings = readings
            holding = {n: self.velocities.get(n, 0.0) == 0.0
                       and self.setpoints.get(n) == self.targets.get(n) for n in self.cfg.joints}
            if self.mode == Mode.LIMP:
                # Torque-off is an unacknowledged broadcast: keep re-sending it while limp.
                self.bus.set_torque(False)
            else:
                verdict = check(readings, self.setpoints, holding, self.cfg, self._history, now)
                self._handle(verdict, readings, now)
            if self.mode == Mode.RESTING and now - self._rest_started > self.cfg.rest_timeout_s:
                self._fault(Verdict(Level.SOFT, [f"SOFT: rest did not finish within "
                                                 f"{self.cfg.rest_timeout_s:.0f} s"]), readings, now)
                self._go_limp()
            if self.mode in (Mode.RUNNING, Mode.RESTING, Mode.RESTED):
                self._step_motion(1 / self.cfg.loop_hz)
                # Goal + matching goal speed: the servo moves steadily between our updates instead
                # of dashing to each goal and waiting (measured +/-7 deg/s speed ripple before).
                speeds = {n: abs(self.velocities[n]) * self.cfg.servo_speed_factor
                          + self.cfg.servo_speed_margin for n in self.setpoints}
                self.bus.write_goals({n: int(round(sp)) for n, sp in self.setpoints.items()},
                                     speeds=speeds, acc=self.cfg.servo_acc)
                if self.mode == Mode.RESTING and self._rest_done(readings):
                    if self.cfg.limp_after_rest:
                        self._go_limp()
                    else:
                        # Keep holding: limp, the head can tip forward on lean (hardware).
                        self.mode = Mode.RESTED
            if self.on_tick:
                self.on_tick(now, readings, dict(self.setpoints), holding)

    # ---- internals (call with lock held) -----------------------------------

    def _handle(self, verdict: Verdict, readings: dict, now: float) -> None:
        if verdict.level == Level.WARN and verdict.reasons != self._last_warn:
            log.warning("; ".join(verdict.reasons))
        self._last_warn = verdict.reasons if verdict.level == Level.WARN else None

        if self.mode == Mode.RESTED:
            if verdict.limp:
                self._fault(verdict, readings, now)
                self._go_limp()
            return

        if self.mode == Mode.RESTING:
            if verdict.limp:
                self._fault(verdict, readings, now)
                self._go_limp()
            elif verdict.level == Level.HARD and \
                    _reason_keys(verdict.reasons) != _reason_keys(self.faults[-1].reasons if self.faults else []):
                self._fault(verdict, readings, now)   # record it; keep lowering the head
            return

        if verdict.limp:
            # Servos can't be trusted to lower the head. (No near-rest shortcut: going limp with
            # lean even 2 deg off its back stop lets the head fall 70 deg forward.)
            self._fault(verdict, readings, now)
            self._go_limp()
        elif verdict.level == Level.HARD:
            # Servos still respond: lower the head under control, then go limp. Dropping it from a
            # raised pose swings it hard enough to knock lean off its rest stop (seen on hardware).
            self._fault(verdict, readings, now)
            self._begin_rest()
        elif verdict.level == Level.SOFT:
            if self.latched is None:
                self._fault(verdict, readings, now)
                self._hold(readings)
                self._soft_since = now
            elif _reason_keys(verdict.reasons) != _reason_keys(self.faults[-1].reasons):
                self._fault(verdict, readings, now)
            if self._soft_since is None:
                self._soft_since = now
            if now - self._soft_since >= self.cfg.soft_hold_timeout_s:
                log.error("SOFT fault persisted %.1f s; resting", now - self._soft_since)
                self._begin_rest()
        else:
            self._soft_since = None

    def _fault(self, verdict: Verdict, readings: dict, now: float) -> None:
        fault = Fault(time=now, level=verdict.level, reasons=list(verdict.reasons), readings=dict(readings))
        self.faults.append(fault)
        if self.latched is None:
            self.latched = fault
        log.error("fault %s: %s", verdict.level.name, "; ".join(verdict.reasons))

    def _hold(self, readings: dict) -> None:
        for name, r in readings.items():
            if r is not None:
                self.setpoints[name] = float(r.position)
                self.targets[name] = float(r.position)
            self.velocities[name] = 0.0

    def _rest_done(self, readings: dict) -> bool:
        for name in ("lean", "nod"):
            r = readings.get(name)
            if r is None:
                return False
            jc = self.cfg.joints[name]
            if abs(r.position - jc.rest) * DEG_PER_COUNT > jc.rest_tolerance_deg:
                return False
        return True

    def _begin_rest(self) -> None:
        self.targets = {n: float(j.rest) for n, j in self.cfg.joints.items()}
        self._rest_speed = self.cfg.rest_speed_deg_s
        self._rest_started = self.clock()
        self.bus.set_torque_limit(self.cfg.rest_torque_fraction)
        self.mode = Mode.RESTING

    def _go_limp(self) -> None:
        try:
            self.bus.set_torque(False)
        except Exception:
            log.exception("torque off failed")
        self.velocities = {n: 0.0 for n in self.cfg.joints}
        self.mode = Mode.LIMP

    def _step_motion(self, dt: float) -> None:
        for name, jc in self.cfg.joints.items():
            vmax = jc.max_speed if self._rest_speed is None else min(jc.max_speed, self._rest_speed)
            self.setpoints[name], self.velocities[name] = step_axis(
                self.setpoints[name], self.velocities[name], self.targets[name],
                vmax / DEG_PER_COUNT, jc.max_accel / DEG_PER_COUNT, dt)
