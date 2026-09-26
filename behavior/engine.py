"""Behavior state machine. Pure: no I/O, time is passed in."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from behavior.attention import Attention
from behavior.follow import Command, Lean, follow_axis
from behavior.idle import Scanner


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
    def __init__(self, preset):
        self.p = preset
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
        prev = self.command
        if not new_frame and prev is not None:
            return Command(prev.turn, prev.nod, lean)
        yaw, pitch = person.head_deg
        turn = follow_axis(base[0], yaw, self.p.follow_gain, self.p.deadband_deg,
                           prev.turn if prev else None)
        nod = follow_axis(base[1], pitch, self.p.follow_gain, self.p.deadband_deg,
                          prev.nod if prev else None)
        return Command(turn, nod, lean)

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
            cmd = self._follow(target, base, new_frame, self.lean.update(target, now))
        elif self.state == State.GLANCING:
            lean = self.command.lean if self.command else 0.0
            cmd = self._follow(by_id[self.glance_id], base, new_frame, lean)
        elif self.state == State.AVERTING:
            cmd = self._avert_command
        elif self.state == State.LOST:
            cmd = self.command or NEUTRAL
        elif self.state == State.SCANNING:
            cmd = Command(self.scanner.step(dt, now), self.p.scan_nod_deg, 0.0)
        else:                                            # IDLE
            cmd = NEUTRAL

        changed = cmd != self.command
        self.command = cmd
        return cmd if changed else None
