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
    from behavior.kinematics import aim_for

    turn, nod = aim_for(10.0, 2.0, 0.0, 20.0, 5.0)                        # exact head geometry
    assert (cmd.turn, cmd.nod) == pytest.approx((turn, nod))
    assert turn == pytest.approx(30.0, abs=1.0) and nod == pytest.approx(7.0, abs=1.0)


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
    assert e.command.turn == pytest.approx(20.0)


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


def test_aim_is_smoothed_across_frames(preset):
    e = Engine(preset)
    first = [person(1, yaw=20.0)]
    e.tick(first, snap(1.0, first, pose=(10.0, 0.0, 0.0)), RUNNING, 1.0)        # aim 30
    moved = [person(1, yaw=30.0)]
    cmd = e.tick(moved, snap(1.1, moved, pose=(10.0, 0.0, 0.0)), RUNNING, 1.1)  # measured 40
    assert e._aim_cmd.turn == pytest.approx(30.0 + 0.25 * 10.0)                 # aim moves part way
    assert 30.0 < cmd.turn < e._aim_cmd.turn                                     # command glides


def test_command_glides_to_the_aim_and_never_past_it(preset):
    e = Engine(preset)
    first = [person(1, yaw=0.0)]
    e.tick(first, snap(1.0, first), RUNNING, 1.0)                                # aim 0, command 0
    far = [person(1, yaw=20.0)]
    e.tick(far, snap(1.05, far), RUNNING, 1.05)                                  # aim jumps to 20? no: 5
    turns = []
    for i in range(2, 60):                                                        # same frame: aim held
        e.tick(far, snap(1.05, far), RUNNING, 1.0 + i * 0.05)
        turns.append(e.command.turn)
    assert all(b >= a for a, b in zip(turns, turns[1:]))                          # monotonic, no bounce
    assert turns[-1] == pytest.approx(e._aim_cmd.turn, abs=0.05)
    assert max(turns) <= e._aim_cmd.turn + 1e-6


def test_small_changes_inside_the_dead_zone_do_not_move_the_aim(preset):
    e = Engine(preset)
    p = [person(1, yaw=20.0)]
    e.tick(p, snap(1.0, p), RUNNING, 1.0)
    jitter = [person(1, yaw=21.0)]
    assert e.tick(jitter, snap(1.1, jitter), RUNNING, 1.1) is None
    assert e.command.turn == pytest.approx(20.0)


def test_coasting_target_is_never_chased(preset):
    # On the robot a coasting prediction drifted off the image and dragged the head to its limit.
    e = Engine(preset)
    seen = [person(1, yaw=10.0)]
    e.tick(seen, snap(1.0, seen), RUNNING, 1.0)
    coasting = [person(1, yaw=45.0, visible=False)]
    for i in range(1, 10):
        e.tick(coasting, snap(1.0 + i * 0.05, coasting), RUNNING, 1.0 + i * 0.05)
    assert e.command.turn == pytest.approx(10.0)


def test_a_head_point_guessed_from_the_box_is_never_chased(preset):
    # On the robot: face out of frame at 0.8 m, box-top fallback pinned at the image top,
    # nod ran away to +82 deg.
    e = Engine(preset)
    seen = [person(1, pitch=5.0)]
    e.tick(seen, snap(1.0, seen), RUNNING, 1.0)
    guessed = [person(1, pitch=15.0, head_from_keypoints=False)]
    for i in range(1, 40):
        e.tick(guessed, snap(1.0 + i * 0.05, guessed, pose=(0.0, 5.0, 0.0)), RUNNING, 1.0 + i * 0.05)
    assert e.command.nod == pytest.approx(5.0, abs=0.1)


def test_close_facing_target_gets_the_preset_lean_whatever_the_current_lean(preset):
    # Regression: the measured lean shadowed the lean decision, so lean just held still.
    e = Engine(preset)
    close = [person(1, yaw=5.0, distance=1.0, facing_score=0.9)]
    cmd = None
    for i in range(20):
        c = e.tick(close, snap(1.0 + i * 0.05, close, pose=(0.0, 0.0, 0.5)), RUNNING, 1.0 + i * 0.05)
        cmd = c or cmd
    assert e.command.lean == pytest.approx(5.0)
    far = [person(1, yaw=5.0, distance=3.0, facing_score=0.9)]
    for i in range(20, 40):
        e.tick(far, snap(1.0 + i * 0.05, far, pose=(0.0, 0.0, 5.0)), RUNNING, 1.0 + i * 0.05)
    assert e.command.lean == pytest.approx(0.0)
