"""Simulated servo bus for offline controller tests."""
from __future__ import annotations

from dataclasses import dataclass

from neck.bus import Reading
from neck.config import DEG_PER_COUNT


class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@dataclass
class FakeServo:
    position: float
    goal: int | None = None
    torque: bool = False
    temperature: int = 30
    voltage: float = 7.4
    status: int = 0
    current_ma: float = 100.0
    jammed: bool = False
    responding: bool = True


class FakeBus:
    def __init__(self, cfg, clock, start_positions=None, speed_deg_s: float = 300.0):
        start_positions = start_positions or {n: j.rest for n, j in cfg.joints.items()}
        self.servos = {n: FakeServo(position=float(start_positions[n])) for n in cfg.joints}
        self.clock = clock
        self.speed_counts = speed_deg_s / DEG_PER_COUNT
        self._last = clock()
        self.opened = False
        self.torque_log: list[bool] = []
        self.torque_limit_log: list[float] = []
        self.writes: list[dict] = []
        self.streaming_configured = False
        self.jump_risk = False
        self.raise_on_read: Exception | None = None
        self.drop_goal_writes = 0      # next N write_goals calls are silently lost (unacked broadcast)
        self.drop_torque_off = 0       # next N set_torque(False) calls are silently lost

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def ping(self) -> set[str]:
        return {n for n, s in self.servos.items() if s.responding}

    def _advance(self) -> None:
        now = self.clock()
        max_step = self.speed_counts * max(now - self._last, 0.0)
        self._last = now
        for s in self.servos.values():
            if s.torque and not s.jammed and s.goal is not None:
                delta = s.goal - s.position
                s.position += max(-max_step, min(max_step, delta))

    def read_all(self):
        if self.raise_on_read:
            raise self.raise_on_read
        self._advance()
        return {
            n: Reading(position=int(round(s.position)), speed=0, load=0, voltage=s.voltage,
                       temperature=s.temperature, status=s.status, moving=False,
                       current_ma=s.current_ma, error=0) if s.responding else None
            for n, s in self.servos.items()
        }

    def read_goals(self):
        return {n: (s.goal if s.responding else None) for n, s in self.servos.items()}

    def write_goals(self, goals) -> bool:
        if self.drop_goal_writes:
            self.drop_goal_writes -= 1
            return True
        for n, raw in goals.items():
            assert isinstance(raw, int) and 0 <= raw <= 4095, (n, raw)
            self.servos[n].goal = raw
        self.writes.append(dict(goals))
        return True

    def set_torque(self, on: bool) -> bool:
        self.torque_log.append(on)
        if not on and self.drop_torque_off:
            self.drop_torque_off -= 1
            return True
        if on:
            for s in self.servos.values():
                if s.goal is None or abs(s.goal - s.position) > 1:
                    self.jump_risk = True
        for s in self.servos.values():
            s.torque = on
        return True

    def set_torque_limit(self, fraction: float) -> bool:
        self.torque_limit_log.append(fraction)
        return True

    def configure_for_streaming(self) -> bool:
        self.streaming_configured = True
        return True
