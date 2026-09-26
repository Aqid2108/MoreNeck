"""Head kinematics for MoreNeck's joint order: lean and nod tilt the base (parallel pitch axes),
turn rotates on top of them. Camera frame: x forward, y left, z up. Angles in degrees:
+turn / +yaw = left, +nod / +lean / +pitch = up. Lean's translation of the head is ignored.

Adding camera offsets to joint angles is only right when the head faces forward: turned far,
nodding mostly rolls the view, and a naive sum creeps the nod upward (seen on the robot).
"""
from __future__ import annotations

import math


def _pitch_up(v, deg):
    a = math.radians(deg)
    x, y, z = v
    return (x * math.cos(a) - z * math.sin(a), y, x * math.sin(a) + z * math.cos(a))


def _turn_left(v, deg):
    a = math.radians(deg)
    x, y, z = v
    return (x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a), z)


def world_vector(azimuth_deg: float, elevation_deg: float):
    """Unit vector for a direction given as azimuth (+left) and elevation (+up) in the base frame."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))


def camera_to_world(turn, nod, lean, yaw, pitch):
    """Base-frame unit vector of a point seen at camera (yaw, pitch) from pose (turn, nod, lean)."""
    return _pitch_up(_turn_left(world_vector(yaw, pitch), turn), nod + lean)


def world_to_camera(turn, nod, lean, v):
    """Camera (yaw, pitch) at which base-frame direction v appears from pose (turn, nod, lean)."""
    x, y, z = _turn_left(_pitch_up(v, -(nod + lean)), -turn)
    return math.degrees(math.atan2(y, x)), math.degrees(math.asin(max(-1.0, min(1.0, z))))


def aim_for(turn, nod, lean, yaw, pitch):
    """(turn, nod) that centres a point seen at camera (yaw, pitch), keeping the same lean."""
    x, y, z = camera_to_world(turn, nod, lean, yaw, pitch)
    new_turn = math.degrees(math.asin(max(-1.0, min(1.0, y))))
    new_nod = math.degrees(math.atan2(z, x)) - lean
    return new_turn, new_nod
