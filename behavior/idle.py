"""Idle scanning: sweep turn back and forth, pausing at each end."""
from __future__ import annotations


class Scanner:
    def __init__(self, preset):
        self.p = preset
        self.angle = 0.0
        self._direction = 1.0          # start sweeping left
        self._pause_until = float("-inf")

    def reset(self, start_deg: float = 0.0) -> None:
        self.angle = max(-self.p.scan_amplitude_deg, min(self.p.scan_amplitude_deg, start_deg))
        self._direction = 1.0
        self._pause_until = float("-inf")

    def step(self, dt: float, now: float) -> float:
        if now < self._pause_until:
            return self.angle
        self.angle += self._direction * self.p.scan_speed_dps * dt
        if abs(self.angle) >= self.p.scan_amplitude_deg:
            self.angle = self._direction * self.p.scan_amplitude_deg
            self._direction = -self._direction
            self._pause_until = now + self.p.scan_pause_s
        return self.angle
