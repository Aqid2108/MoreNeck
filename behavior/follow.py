"""Closed-loop follow law and lean-in policy."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    turn: float
    nod: float
    lean: float


def follow_axis(base: float, offset: float, gain: float, deadband: float, previous) -> float:
    """Move part of the way from where the head was at capture toward the target."""
    if previous is not None and abs(offset) < deadband:
        return previous
    return base + gain * offset


class Lean:
    """Lean in when the target is close and facing; release with hysteresis."""

    def __init__(self, preset):
        self.p = preset
        self.active = False
        self._not_facing_since: float | None = None

    def reset(self) -> None:
        self.active = False
        self._not_facing_since = None

    def update(self, person, now: float) -> float:
        d = person.distance_m
        if not self.active:
            if d is not None and d <= self.p.lean_near_m and person.facing:
                self.active = True
                self._not_facing_since = None
        elif d is None or d > self.p.lean_release_m:
            self.reset()
        elif not person.facing:
            if self._not_facing_since is None:
                self._not_facing_since = now
            elif now - self._not_facing_since >= self.p.lean_release_s:
                self.reset()
        else:
            self._not_facing_since = None
        return self.p.lean_deg if self.active else 0.0
