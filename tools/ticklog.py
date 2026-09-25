"""Per-tick CSV logging via NeckController's on_tick hook."""
from __future__ import annotations

import csv
from pathlib import Path


class TickLog:
    def __init__(self, extra=None):
        self.rows: list[dict] = []
        self._extra = extra or (lambda: {})

    def __call__(self, now, readings, setpoints, holding) -> None:
        extra = self._extra()
        for name, r in readings.items():
            if r is not None:
                self.rows.append({"t": now, "joint": name, "position": r.position,
                                  "setpoint": setpoints[name], "current_ma": r.current_ma,
                                  "load": r.load, "temperature": r.temperature,
                                  "voltage": r.voltage, "holding": holding[name], **extra})

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]) if self.rows else ["t"])
            writer.writeheader()
            writer.writerows(self.rows)
        return path
