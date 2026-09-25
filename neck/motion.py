"""Trapezoidal setpoint stepping, recomputed every tick so moving targets are followed smoothly."""
from __future__ import annotations

import math


def step_axis(pos: float, vel: float, target: float, vmax: float, amax: float,
              dt: float) -> tuple[float, float]:
    err = target - pos
    if err == 0.0 and vel == 0.0:
        return target, 0.0
    # Fastest speed from which we can still brake to a stop at the target.
    v_brake = math.sqrt(2.0 * amax * abs(err))
    v_desired = math.copysign(min(vmax, v_brake), err)
    dv = max(-amax * dt, min(amax * dt, v_desired - vel))
    vel += dv
    new_pos = pos + vel * dt
    if (target - pos) * (target - new_pos) <= 0.0:   # reached or crossed the target
        return target, 0.0
    return new_pos, vel
