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
    limp: bool = False   # True only for HARD faults where the servos can't be trusted to lower the head

    def add(self, level: Level, reason: str, limp: bool = False) -> None:
        if level == Level.OK:
            return
        self.level = max(self.level, level)
        self.limp = self.limp or limp
        self.reasons.append(f"{level.name}: {reason}")


@dataclass
class JointHistory:
    misses: int = 0
    over_current_ticks: int = 0
    tracking_since: float | None = None
    holding_since: float | None = None
    currents: deque = field(default_factory=deque)
    limits_armed: bool = False
    recent: dict = field(default_factory=dict)   # check name -> deque of recent levels


def _debounced(h: JointHistory, key: str, level: Level, ticks: int) -> Level:
    """Level sustained over the last `ticks` samples: one corrupted reading can't raise a fault."""
    window = h.recent.setdefault(key, deque(maxlen=ticks))
    window.append(level)
    return min(window) if len(window) == ticks else Level.OK


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
                v.add(Level.HARD, f"{name}: no reply for {h.misses} reads", limp=True)
            else:
                v.add(Level.WARN, f"{name}: missed read")
            continue
        h.misses = 0

        n = cfg.fault_debounce_ticks
        status = _debounced(h, "status", Level.HARD if r.status else Level.OK, n)
        v.add(status, f"{name}: servo status fault bits {r.status:#04x}")

        over = r.voltage > cfg.voltage_hard_above
        if over or r.voltage < cfg.voltage_hard_below:
            volt = Level.HARD
        elif r.voltage < cfg.voltage_soft_below:
            volt = Level.SOFT
        elif r.voltage < cfg.voltage_warn_below:
            volt = Level.WARN
        else:
            volt = Level.OK
        volt = _debounced(h, "voltage", volt, n)
        v.add(volt, f"{name}: voltage {r.voltage:.1f} V", limp=over and volt == Level.HARD)

        if r.temperature >= jc.temp_hard_c:
            temp = Level.HARD
        elif r.temperature >= jc.temp_soft_c:
            temp = Level.SOFT
        elif r.temperature >= jc.temp_warn_c:
            temp = Level.WARN
        else:
            temp = Level.OK
        v.add(_debounced(h, "temperature", temp, n), f"{name}: temperature {r.temperature} C")

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
