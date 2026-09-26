import pytest

from behavior.engine import Engine, NeckStatus, State
from perception.types import Snapshot
from tests.behavior_helpers import person

RUNNING = NeckStatus(running=True, pose=(0.0, 0.0, 0.0))


def snap(t, people, pose=(0.0, 0.0, 0.0)):
    return Snapshot(timestamp=t, frame_size=(1280, 720), people=tuple(people), source_ok=True,
                    neck_pose=pose)


def test_idle_at_start_commands_neutral(preset):
    e = Engine(preset)
    cmd = e.tick([], None, RUNNING, 0.0)
    assert e.state == State.IDLE and (cmd.turn, cmd.nod, cmd.lean) == (0.0, 0.0, 0.0)


def test_tracking_follows_from_the_pose_at_capture(preset):
    e = Engine(preset)
    p = [person(1, yaw=20.0, pitch=5.0)]
    cmd = e.tick(p, snap(1.0, p, pose=(10.0, 2.0, 0.0)), RUNNING, 1.0)
    assert e.state == State.TRACKING
    assert cmd.turn == pytest.approx(10.0 + 0.4 * 20.0) and cmd.nod == pytest.approx(2.0 + 0.4 * 5.0)


def test_each_snapshot_is_used_once(preset):
    e = Engine(preset)
    p = [person(1, yaw=20.0)]
    s = snap(1.0, p)
    first = e.tick(p, s, RUNNING, 1.0)
    assert e.tick(p, s, RUNNING, 1.05) is None           # same frame: no new command
    assert e.command == first


def test_without_any_pose_the_base_is_neutral_so_dry_run_does_not_wind_up(preset):
    # In a dry run the head stays at neutral, so the camera offset never shrinks.
    e = Engine(preset)
    p = [person(1, yaw=20.0)]
    dry = NeckStatus(running=True, pose=None)
    for i in range(10):
        e.tick(p, snap(1.0 + i * 0.1, p, pose=None), dry, 1.0 + i * 0.1)
    assert e.command.turn == pytest.approx(0.4 * 20)


def test_neck_fault_pauses_and_sends_nothing(preset):
    e = Engine(preset)
    p = [person(1, yaw=20.0)]
    e.tick(p, snap(1.0, p), RUNNING, 1.0)
    stopped = NeckStatus(running=False, pose=(0.0, 0.0, 0.0))
    assert e.tick(p, snap(1.1, p), stopped, 1.1) is None
    assert e.state == State.PAUSED
    assert e.tick(p, snap(1.2, p), stopped, 1.2) is None
    e.tick(p, snap(1.3, p), RUNNING, 1.3)
    assert e.state in (State.IDLE, State.TRACKING)
