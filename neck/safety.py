"""Safety checks: pure function over one tick of readings plus a small rolling history."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

from neck.config import DEG_PER_COUNT


class Level(IntEnum):
    OK = 0
    WARN = 1
    SOFT = 2
    HARD = 3


@dataclass
class Verdict:
    level: Level = Level.OK
    reasons: list[str] = field(default_factory=list)

    def add(self, level: Level, reason: str) -> None:
        self.level = max(self.level, level)
        self.reasons.append(f"{level.name}: {reason}")


@dataclass
class JointHistory:
    misses: int = 0
    over_current_ticks: int = 0
    tracking_since: float | None = None
    holding_since: float | None = None
    currents: deque = field(default_factory=deque)
    limits_armed: bool = False


def new_history(cfg) -> dict[str, JointHistory]:
    return {name: JointHistory() for name in cfg.joints}


def check(readings, setpoints, holding, cfg, history, now) -> Verdict:
    v = Verdict()
    for name, jc in cfg.joints.items():
        r = readings.get(name)
        h = history[name]

        if r is None:
            h.misses += 1
            if h.misses >= cfg.hard_misses:
                v.add(Level.HARD, f"{name}: no reply for {h.misses} reads")
            else:
                v.add(Level.WARN, f"{name}: missed read")
            continue
        h.misses = 0

        if r.status:
            v.add(Level.HARD, f"{name}: servo status fault bits {r.status:#04x}")

        if r.voltage > cfg.voltage_hard_above or r.voltage < cfg.voltage_hard_below:
            v.add(Level.HARD, f"{name}: voltage {r.voltage:.1f} V")
        elif r.voltage < cfg.voltage_soft_below:
            v.add(Level.SOFT, f"{name}: voltage {r.voltage:.1f} V")
        elif r.voltage < cfg.voltage_warn_below:
            v.add(Level.WARN, f"{name}: voltage {r.voltage:.1f} V")

        if r.temperature >= jc.temp_hard_c:
            v.add(Level.HARD, f"{name}: temperature {r.temperature} C")
        elif r.temperature >= jc.temp_soft_c:
            v.add(Level.SOFT, f"{name}: temperature {r.temperature} C")
        elif r.temperature >= jc.temp_warn_c:
            v.add(Level.WARN, f"{name}: temperature {r.temperature} C")

        h.over_current_ticks = h.over_current_ticks + 1 if r.current_ma > jc.current_limit_ma else 0
        if h.over_current_ticks >= cfg.spike_ticks:
            v.add(Level.SOFT, f"{name}: current {r.current_ma:.0f} mA over {jc.current_limit_ma} mA "
                              f"for {h.over_current_ticks} ticks")

        if holding.get(name):
            if h.holding_since is None:
                h.holding_since = now
                h.currents.clear()
            h.currents.append((now, r.current_ma))
            while h.currents and h.currents[0][0] < now - cfg.hold_window_s:
                h.currents.popleft()
            if now - h.holding_since >= cfg.hold_window_s:
                mean = sum(c for _, c in h.currents) / len(h.currents)
                if mean > jc.hold_current_limit_ma:
                    v.add(Level.SOFT, f"{name}: holding current mean {mean:.0f} mA over "
                                      f"{jc.hold_current_limit_ma} mA")
        else:
            h.holding_since = None
            h.currents.clear()

        err_deg = abs(r.position - setpoints[name]) * DEG_PER_COUNT
        if err_deg > cfg.tracking_max_error_deg:
            if h.tracking_since is None:
                h.tracking_since = now
            elif now - h.tracking_since > cfg.tracking_max_duration_s:
                v.add(Level.SOFT, f"{name}: tracking error {err_deg:.1f} deg for "
                                  f"{now - h.tracking_since:.2f} s")
        else:
            h.tracking_since = None

        if jc.raw_min <= r.position <= jc.raw_max:
            h.limits_armed = True
        margin = cfg.soft_limit_margin_deg / DEG_PER_COUNT
        if h.limits_armed and not (jc.raw_min - margin <= r.position <= jc.raw_max + margin):
            v.add(Level.SOFT, f"{name}: position {r.position} outside soft limits "
                              f"{jc.raw_min}..{jc.raw_max}")
    return v
