"""Closed-loop simulation: people in the room, a camera on the moving head, a speed-limited neck."""
from __future__ import annotations

import random
from dataclasses import dataclass, replace

from behavior.engine import Engine, NeckStatus
from neck.motion import step_axis
from perception.pipeline import people_or_empty
from perception.types import Person, Snapshot

HFOV, VFOV = 60.0, 34.0


@dataclass
class SimPerson:
    id: int
    yaw: float                   # room direction in degrees, +left
    pitch: float = 10.0
    distance: float = 2.0
    facing_score: float = 0.9
    motion: str = "still"
    first_seen: float = -10.0
    yaw_rate: float = 0.0


class SimNeck:
    """Speed- and acceleration-limited like the real NeckController (same step_axis profile)."""
    LIMITS = {"turn": (-60.0, 60.0), "nod": (0.0, 60.0), "lean": (0.0, 30.0)}
    SPEED = {"turn": 120.0, "nod": 30.0, "lean": 20.0}
    ACCEL = {"turn": 200.0, "nod": 300.0, "lean": 150.0}
    SUBSTEPS = 5

    def __init__(self):
        self.pos = {"turn": 0.0, "nod": 0.0, "lean": 0.0}
        self.vel = {k: 0.0 for k in self.pos}
        self.target = dict(self.pos)
        self.running = True
        self.looks = []
        self.turn_speeds = []            # setpoint speed of turn, every substep

    def look(self, turn, nod, lean) -> bool:
        if not self.running:
            return False
        for k, v in (("turn", turn), ("nod", nod), ("lean", lean)):
            lo, hi = self.LIMITS[k]
            self.target[k] = min(max(v, lo), hi)
        self.looks.append((turn, nod, lean))
        return True

    def step(self, dt) -> None:
        h = dt / self.SUBSTEPS
        for _ in range(self.SUBSTEPS):
            for k in self.pos:
                self.pos[k], self.vel[k] = step_axis(self.pos[k], self.vel[k], self.target[k],
                                                     self.SPEED[k], self.ACCEL[k], h)
            self.turn_speeds.append((h, self.vel["turn"]))

    def stop_go_rate(self) -> float:
        """Stop-and-go cycles per second of turning (the jerk measured on the robot: 5.1/s)."""
        cycles, moving, going = 0, 0.0, False
        for h, v in self.turn_speeds:
            if abs(v) > 5:
                moving += h
            if not going and abs(v) > 8:
                going = True
            elif going and abs(v) < 1:
                going, cycles = False, cycles + 1
        return cycles / moving if moving > 0 else 0.0

    def pose(self):
        return (self.pos["turn"], self.pos["nod"], self.pos["lean"])


class SimCamera:
    def __init__(self, fps=19.0, latency=0.06, noise_deg=0.0, seed=1):
        self.period, self.latency = 1.0 / fps, latency
        self.noise_deg = noise_deg          # head-angle measurement noise (robot: ~1.9 deg)
        self._rng = random.Random(seed)
        self._next = 0.0
        self._pending = []
        self.failed = False
        self.latest = Snapshot(0.0, (1280, 720), (), False, "not started")

    def step(self, now, world, neck) -> None:
        if now >= self._next:
            pose = neck.pose()
            if self.failed:
                shot = Snapshot(now, (1280, 720), (), False, "no frames from source")
            else:
                seen = tuple(self._noisy(p) for p in (self._see(w, pose, now) for w in world)
                             if p is not None)
                shot = Snapshot(now, (1280, 720), seen, True, None, pose)
            self._pending.append((now + self.latency, shot))
            self._next += self.period
        while self._pending and self._pending[0][0] <= now:
            self.latest = self._pending.pop(0)[1]

    def _noisy(self, person):
        if not self.noise_deg:
            return person
        yaw, pitch = person.head_deg
        return replace(person, head_deg=(yaw + self._rng.gauss(0, self.noise_deg),
                                         pitch + self._rng.gauss(0, self.noise_deg)))

    @staticmethod
    def _see(w, pose, now):
        yaw, pitch = w.yaw - pose[0], w.pitch - pose[1]
        if abs(yaw) > HFOV / 2 or abs(pitch) > VFOV / 2:
            return None
        return Person(id=w.id, box=(0.0, 0.0, 1.0, 1.0), head_px=(0.0, 0.0), head_deg=(yaw, pitch),
                      world_yaw_deg=w.yaw, distance_m=w.distance, distance_source="body",
                      facing_score=w.facing_score, facing=w.facing_score >= 0.6, motion=w.motion,
                      confidence=0.9, visible=True, first_seen=w.first_seen, last_seen=now)


class Sim:
    def __init__(self, preset, people=(), dt=0.05):
        self.engine = Engine(preset)
        self.neck = SimNeck()
        self.camera = SimCamera()
        self.people = list(people)
        self.dt = dt
        self.t = 0.0
        self.log = []

    def run(self, seconds: float) -> None:
        for _ in range(round(seconds / self.dt)):
            self.t = round(self.t + self.dt, 9)
            for p in self.people:
                p.yaw += p.yaw_rate * self.dt
            self.camera.step(self.t, self.people, self.neck)
            shot = self.camera.latest
            people = people_or_empty(shot, self.t, 0.3)
            cmd = self.engine.tick(people, shot if shot.source_ok else None,
                                   NeckStatus(self.neck.running, self.neck.pose()), self.t)
            if cmd is not None:
                self.neck.look(cmd.turn, cmd.nod, cmd.lean)
            self.neck.step(self.dt)
            self.log.append((self.t, self.engine.state, self.neck.pose(), self.engine.attention.target_id))

    def turn(self) -> float:
        return self.neck.pos["turn"]
