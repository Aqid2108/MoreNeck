import math

import pytest

from neck.bus import BusError
from neck.config import DEG_PER_COUNT
from neck.controller import Mode, NeckController
from tests.fake_bus import FakeBus, FakeClock


def make(cfg, start="neutral"):
    clock = FakeClock()
    starts = {n: (j.neutral if start == "neutral" else j.rest) for n, j in cfg.joints.items()}
    bus = FakeBus(cfg, clock, start_positions=starts)
    ctrl = NeckController(cfg, bus, clock=clock, sleep=clock.advance)
    return ctrl, bus, clock


def run(ctrl, clock, seconds):
    for _ in range(round(seconds * ctrl.cfg.loop_hz)):
        clock.advance(1 / ctrl.cfg.loop_hz)
        ctrl.tick()


def test_startup_has_no_jump_and_enables_torque(cfg):
    ctrl, bus, _ = make(cfg, start="rest")
    ctrl.start(run_thread=False)
    assert ctrl.mode == Mode.RUNNING
    assert not bus.jump_risk
    assert bus.torque_log == [True]
    assert bus.streaming_configured
    assert bus.torque_limit_log == [1.0]


def test_start_with_missing_servo_raises_and_never_enables_torque(cfg):
    ctrl, bus, _ = make(cfg)
    bus.servos["nod"].responding = False
    with pytest.raises(BusError, match="nod"):
        ctrl.start(run_thread=False)
    assert bus.torque_log == []
    assert ctrl.mode == Mode.IDLE


def test_moves_from_rest_to_neutral_after_start(cfg):
    ctrl, bus, clock = make(cfg, start="rest")
    ctrl.start(run_thread=False)
    run(ctrl, clock, 3.0)
    for n, j in cfg.joints.items():
        assert bus.servos[n].position == pytest.approx(j.neutral, abs=2), n


def test_look_moves_joint_to_target(cfg):
    ctrl, bus, clock = make(cfg)
    ctrl.start(run_thread=False)
    assert ctrl.look(turn=20.0) is True
    run(ctrl, clock, 2.0)
    expected = cfg.joints["turn"].to_raw(20.0)
    assert bus.servos["turn"].position == pytest.approx(expected, abs=2)


def test_look_clamps_to_soft_limits(cfg):
    ctrl, bus, clock = make(cfg)
    ctrl.start(run_thread=False)
    ctrl.look(turn=500.0, nod=-500.0)
    assert ctrl.targets["turn"] == cfg.joints["turn"].raw_max
    assert ctrl.targets["nod"] == cfg.joints["nod"].raw_min
    run(ctrl, clock, 4.0)
    for w in bus.writes:
        for n, raw in w.items():
            assert cfg.joints[n].raw_min <= raw <= cfg.joints[n].raw_max, (n, raw)


def test_writes_respect_max_speed(cfg):
    ctrl, bus, clock = make(cfg)
    ctrl.start(run_thread=False)
    ctrl.look(turn=40.0)
    run(ctrl, clock, 2.0)
    per_tick = cfg.joints["turn"].max_speed / DEG_PER_COUNT / cfg.loop_hz
    turn = [w["turn"] for w in bus.writes]
    assert max(abs(b - a) for a, b in zip(turn, turn[1:])) <= per_tick + 1


def test_look_rejects_non_finite_angles(cfg):
    ctrl, _, _ = make(cfg)
    ctrl.start(run_thread=False)
    before = dict(ctrl.targets)
    assert ctrl.look(turn=math.nan) is False
    assert ctrl.look(nod=math.inf) is False
    assert ctrl.targets == before


def test_look_before_start_is_rejected(cfg):
    ctrl, _, _ = make(cfg)
    assert ctrl.look(turn=10.0) is False


def test_on_tick_receives_readings_and_setpoints(cfg):
    seen = []
    clock = FakeClock()
    bus = FakeBus(cfg, clock, start_positions={n: j.neutral for n, j in cfg.joints.items()})
    ctrl = NeckController(cfg, bus, clock=clock, sleep=clock.advance,
                          on_tick=lambda now, r, sp, hold: seen.append((now, r, sp, hold)))
    ctrl.start(run_thread=False)
    run(ctrl, clock, 0.1)
    assert len(seen) == 5
    now, readings, setpoints, holding = seen[-1]
    assert set(readings) == set(setpoints) == set(holding) == {"turn", "nod", "lean"}


def test_startup_retries_a_lost_goal_write(cfg):
    ctrl, bus, _ = make(cfg, start="rest")
    bus.drop_goal_writes = 1
    ctrl.start(run_thread=False)
    assert ctrl.mode == Mode.RUNNING
    assert not bus.jump_risk


def test_startup_refuses_torque_when_goal_cannot_be_verified(cfg):
    ctrl, bus, _ = make(cfg, start="rest")
    bus.drop_goal_writes = 100
    with pytest.raises(BusError, match="goal"):
        ctrl.start(run_thread=False)
    assert bus.torque_log == []


def test_pose_is_none_before_start(cfg):
    ctrl, _, _ = make(cfg)
    assert ctrl.pose() is None


def test_pose_reports_measured_angles_in_degrees(cfg):
    ctrl, _, clock = make(cfg)
    ctrl.start(run_thread=False)
    ctrl.look(turn=20.0)
    run(ctrl, clock, 2.0)
    turn, nod, lean = ctrl.pose()
    assert turn == pytest.approx(20.0, abs=0.3)
    assert nod == pytest.approx(0.0, abs=0.3) and lean == pytest.approx(0.0, abs=0.3)


def test_pose_is_none_when_a_servo_is_silent(cfg):
    ctrl, bus, clock = make(cfg)
    ctrl.start(run_thread=False)
    bus.servos["nod"].responding = False
    run(ctrl, clock, 0.02)
    assert ctrl.pose() is None


def test_goal_speed_follows_the_planned_speed_and_rests_at_the_margin(cfg):
    ctrl, bus, clock = make(cfg)
    ctrl.start(run_thread=False)
    ctrl.look(turn=40.0)
    run(ctrl, clock, 0.3)                                     # mid-move
    v = abs(ctrl.velocities["turn"])
    assert v > 100
    assert bus.speed_writes[-1]["turn"] == pytest.approx(v * 1.2 + 100.0)
    run(ctrl, clock, 3.0)                                     # arrived, holding
    assert bus.speed_writes[-1]["turn"] == pytest.approx(100.0)
    assert bus.acc_writes[-1] == 50
