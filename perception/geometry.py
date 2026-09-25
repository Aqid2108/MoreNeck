"""Pinhole camera geometry: pixels <-> angles from the camera axis.

+yaw = image left (the robot's left), +pitch = image up.
"""
from __future__ import annotations

import math


def focal_px(width: int, hfov_deg: float) -> float:
    return (width / 2) / math.tan(math.radians(hfov_deg) / 2)


def pixel_to_angles(x: float, y: float, width: int, height: int, hfov_deg: float) -> tuple[float, float]:
    f = focal_px(width, hfov_deg)
    yaw = math.degrees(math.atan((width / 2 - x) / f))
    pitch = math.degrees(math.atan((height / 2 - y) / f))
    return yaw, pitch


def angles_to_pixel(yaw_deg: float, pitch_deg: float, width: int, height: int,
                    hfov_deg: float) -> tuple[float, float]:
    f = focal_px(width, hfov_deg)
    return (width / 2 - f * math.tan(math.radians(yaw_deg)),
            height / 2 - f * math.tan(math.radians(pitch_deg)))


def hfov_from_shift(width: int, turn_deg: float, shift_px: float) -> float:
    """FOV from turning by turn_deg and seeing a centred feature move shift_px."""
    f = shift_px / math.tan(math.radians(turn_deg))
    return math.degrees(2 * math.atan((width / 2) / f))
