"""Perception data contract. Behavior depends only on Person and Snapshot."""
from __future__ import annotations

from dataclasses import dataclass, field

KEYPOINT_NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear",
                  "left_shoulder", "right_shoulder")
Box = tuple[float, float, float, float]   # x1, y1, x2, y2 in image pixels


@dataclass(frozen=True)
class Detection:
    box: Box
    score: float
    keypoints: dict = field(default_factory=dict)   # name -> (x, y, confidence)


@dataclass(frozen=True)
class Person:
    id: int
    box: Box
    head_px: tuple[float, float]
    head_deg: tuple[float, float]       # (yaw, pitch) from the camera axis; +yaw left, +pitch up
    world_yaw_deg: float | None         # neck turn + head yaw, when a neck pose is known
    distance_m: float | None
    distance_source: str | None         # "shoulders" | "body" | "head"
    facing_score: float
    facing: bool
    motion: str                         # "approaching" | "leaving" | "passing" | "still" | "unknown"
    confidence: float
    visible: bool                       # False while coasting through an occlusion
    first_seen: float
    last_seen: float
    head_from_keypoints: bool = True    # False: head point guessed from the box top (don't aim at it)


@dataclass(frozen=True)
class Snapshot:
    timestamp: float                    # capture time of the frame (seconds)
    frame_size: tuple[int, int]         # (width, height)
    people: tuple[Person, ...]
    source_ok: bool
    error: str | None = None
    neck_pose: tuple[float, float, float] | None = None   # (turn, nod, lean) when the frame was read
