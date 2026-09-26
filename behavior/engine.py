"""Behavior state machine. Pure: no I/O, time is passed in."""
from __future__ import annotations

from dataclasses import dataclass
import math
from enum import Enum

from behavior.attention import Attention
from behavior.follow import Command, Lean
from behavior.idle import Scanner
from behavior.kinematics import aim_for


class State(str, Enum):
    TRACKING = "tracking"
    GLANCING = "glancing"
    AVERTING = "averting"
    LOST = "lost"
    IDLE = "idle"
    SCANNING = "scanning"
    PAUSED = "paused"


@dataclass(frozen=True)
class NeckStatus:
    running: bool                                  # neck RUNNING with no latched fault
    pose: tuple[float, float, float] | None


NEUTRAL = Command(0.0, 0.0, 0.0)


class Engine:
    def __init__(self, preset, turn_range: tuple[float, float] = (-60.0, 60.0)):
        self.p = preset
        self.turn_range = turn_range                      # neck soft limits for turn, degrees
        self.attention = Attention(preset)
        self.lean = Lean(preset)
        self.scanner = Scanner(preset)
        self.state = State.IDLE
        self.command: Command | None = None
        self.glance_id: int | None = None
        self._since = 0.0
        self._last_tick: float | None = None
        self._last_snapshot_ts: float | None = None
        self._facing_since: float | None = None
        self._avert_command: Command | None = None
        self._aim: tuple[float, float] | None = None     # smoothed room direction of who we follow
        self._aim_id: int | None = None
        self._aim_cmd: Command | None = None              # last aim (before pursuit smoothing)
        self._x: list[float] | None = None                # pursuit filter state: position ...
        self._v = [0.0, 0.0]                              # ... and velocity (turn, nod)

    def _enter(self, state: State, now: float) -> None:
        self.state = state
        self._since = now

    def _base(self, snapshot, neck):
        if snapshot is not None and snapshot.neck_pose is not None:
            return snapshot.neck_pose
        if neck.pose is not None:
            return neck.pose
        # No pose (dry run): the head is really at neutral, so never build on our own commands.
        return (0.0, 0.0, 0.0)

    def _follow(self, person, base, new_frame: bool, lean: float) -> Command:
        """Aim at the person's room direction (pose at capture + camera offset), smoothed.

        Commanding the whole remaining angle lets the neck glide with its own speed and
        acceleration limits. Stepping a fraction per frame made it stop and restart ~5 times a
        second on the robot. A coasting (predicted) target is never chased.
        """
        # A held aim belongs to one person; never carry it over to someone else (on the robot a
        # stale aim made the head swing back to where the previous target had been).
        prev = self._aim_cmd if self._aim_id == person.id else None
        # Only aim at a head that was actually seen: a coasting prediction or a head point guessed
        # from the box top (pinned at the image edge when the face is out of frame) runs away.
        seen = person.visible and person.head_from_keypoints
        if prev is not None and (not new_frame or not seen):
            return Command(prev.turn, prev.nod, lean)
        if not seen:                                # nothing of theirs to hold: stay where we are
            return Command(base[0], base[1], lean)
        base_lean = base[2] if len(base) > 2 else 0.0      # measured lean, for the geometry only
        turn, nod = aim_for(base[0], base[1], base_lean, *person.head_deg)
        lo, hi = self.turn_range
        if not lo <= turn <= hi:
            # Out of turn's reach: look as far as possible, and don't tilt to compensate (turned far,
            # the tilt that would line the person up is large - nod climbed to 43-72 deg on the robot).
            turn = min(max(turn, lo), hi)
            nod = self._aim[1] if self._aim is not None and self._aim_id == person.id else base[1]
        nod = min(nod, self.p.max_nod_deg)     # ceiling: looking up at the ceiling breeds ghosts
        world = (turn, nod)
        if self._aim is None or self._aim_id != person.id:
            aim = world
        else:
            a = self.p.follow_gain
            aim = tuple(old if abs(new - old) < self.p.deadband_deg else old + a * (new - old)
                        for old, new in zip(self._aim, world))
        self._aim, self._aim_id = aim, person.id
        return Command(aim[0], aim[1], lean)

    def _pursue(self, cmd: Command, dt: float) -> Command:
        """Critically damped spring from the current command toward the aim, every tick.

        The neck then receives a target that glides (velocity eases in and out) instead of one
        that jumps each frame, which its accelerate-brake-stop profile turned into shudder.
        """
        if self._x is None:
            start = self.command or cmd
            self._x, self._v = [start.turn, start.nod], [0.0, 0.0]
        w = 2 * math.pi * self.p.pursuit_hz
        dt = min(dt, 0.1)
        for i, goal in enumerate((cmd.turn, cmd.nod)):
            accel = w * w * (goal - self._x[i]) - 2 * w * self._v[i]
            self._v[i] += accel * dt
            self._x[i] += self._v[i] * dt
        return Command(self._x[0], self._x[1], cmd.lean)

    def tick(self, people, snapshot, neck: NeckStatus, now: float) -> Command | None:
        dt = 0.0 if self._last_tick is None else now - self._last_tick
        self._last_tick = now

        if not neck.running:
            if self.state != State.PAUSED:
                self._enter(State.PAUSED, now)
            return None
        if self.state == State.PAUSED:          # resume from scratch
            self._enter(State.IDLE, now)
            self.command = None
            self.lean.reset()

        new_frame = snapshot is not None and snapshot.timestamp != self._last_snapshot_ts
        if new_frame:
            self._last_snapshot_ts = snapshot.timestamp
        base = self._base(snapshot, neck)
        by_id = {x.id: x for x in people}
        target_id = self.attention.update(people, now)

        # --- transitions ---
        if self.state in (State.TRACKING, State.GLANCING, State.AVERTING) and target_id is None:
            self._enter(State.LOST, now)
            self.lean.reset()
        if self.state in (State.LOST, State.IDLE, State.SCANNING) and target_id is not None:
            self._enter(State.TRACKING, now)
            self._facing_since = None
        if self.state == State.TRACKING:
            target = by_id[target_id]
            if self.p.avert_after_s is not None:
                if target.facing:
                    self._facing_since = now if self._facing_since is None else self._facing_since
                    if now - self._facing_since >= self.p.avert_after_s:
                        cur = self.command.turn if self.command else base[0]
                        away = self.p.avert_deg if cur <= 0 else -self.p.avert_deg
                        self._avert_command = Command(cur + away, self.command.nod if self.command else base[1], 0.0)
                        self._enter(State.AVERTING, now)
                else:
                    self._facing_since = None
            if self.state == State.TRACKING:
                glance = self.attention.glance_candidate(people, now)
                if glance is not None:
                    self.glance_id = glance
                    self._enter(State.GLANCING, now)
        if self.state == State.GLANCING and (self.glance_id not in by_id
                                             or now - self._since >= self.p.glance_s):
            self._enter(State.TRACKING, now)
        if self.state == State.AVERTING and now - self._since >= self.p.avert_s:
            self._enter(State.TRACKING, now)
            self._facing_since = None
        if self.state == State.LOST and now - self._since >= self.p.lost_hold_s:
            self._enter(State.IDLE, now)
        if (self.state == State.IDLE and self.p.idle_scan
                and now - self._since >= self.p.scan_after_s):
            self.scanner.reset(self.command.turn if self.command else 0.0)
            self._enter(State.SCANNING, now)

        # --- output ---
        if self.state == State.TRACKING:
            target = by_id[target_id]
            aim = self._follow(target, base, new_frame, self.lean.update(target, now))
            self._aim_cmd = aim
            cmd = self._pursue(aim, dt)
        elif self.state == State.GLANCING:
            lean = self.command.lean if self.command else 0.0
            aim = self._follow(by_id[self.glance_id], base, new_frame, lean)
            self._aim_cmd = aim
            cmd = self._pursue(aim, dt)
        elif self.state == State.AVERTING:
            cmd = self._avert_command
        elif self.state == State.LOST:
            cmd = self.command or NEUTRAL
        elif self.state == State.SCANNING:
            cmd = Command(self.scanner.step(dt, now), self.p.scan_nod_deg, 0.0)
        else:                                            # IDLE
            cmd = NEUTRAL

        if self.state not in (State.TRACKING, State.GLANCING):
            self._x = None                                # restart pursuit from wherever we are
            self._aim = self._aim_id = self._aim_cmd = None   # and forget the old target's aim
        changed = cmd != self.command
        self.command = cmd
        return cmd if changed else None
