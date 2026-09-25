import pytest

from neck.motion import step_axis

DT = 0.02


def run_to(pos, vel, target, vmax, amax, max_steps=2000):
    traj = [(pos, vel)]
    for _ in range(max_steps):
        pos, vel = step_axis(pos, vel, target, vmax, amax, DT)
        traj.append((pos, vel))
        if pos == target and vel == 0.0:
            break
    return traj


def test_reaches_target_and_stops():
    assert run_to(0.0, 0.0, 1000.0, 500.0, 2000.0)[-1] == (1000.0, 0.0)


def test_respects_speed_and_accel_limits():
    traj = run_to(0.0, 0.0, 1000.0, 500.0, 2000.0)
    for (_, v0), (p1, v1) in zip(traj, traj[1:]):
        assert abs(v1) <= 500.0 + 1e-9
        if (p1, v1) != (1000.0, 0.0):
            assert abs(v1 - v0) <= 2000.0 * DT + 1e-9


def test_does_not_overshoot():
    assert all(p <= 1000.0 for p, _ in run_to(0.0, 0.0, 1000.0, 500.0, 2000.0))


def test_works_in_negative_direction():
    traj = run_to(1000.0, 0.0, 0.0, 500.0, 2000.0)
    assert traj[-1] == (0.0, 0.0)
    assert all(p >= 0.0 for p, _ in traj)


def test_decelerates_before_reversing_when_target_jumps_behind():
    pos, vel = step_axis(0.0, 500.0, -1000.0, 500.0, 2000.0, DT)
    assert vel == pytest.approx(460.0)
    assert pos == pytest.approx(9.2)


def test_slows_down_gradually_when_vmax_is_lowered():
    _, vel = step_axis(0.0, 500.0, 1000.0, 100.0, 2000.0, DT)
    assert vel == pytest.approx(460.0)


def test_stays_put_at_target():
    assert step_axis(5.0, 0.0, 5.0, 500.0, 2000.0, DT) == (5.0, 0.0)
