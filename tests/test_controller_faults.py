import pytest

from neck.controller import Mode, NeckController
from neck.safety import Level
from tests.fake_bus import FakeBus, FakeClock


def make(cfg, start="neutral"):
    clock = FakeClock()
    starts = {n: (j.neutral if start == "neutral" else j.rest) for n, j in cfg.joints.items()}
    bus = FakeBus(cfg, clock, start_positions=starts)
    ctrl = NeckController(cfg, bus, clock=clock, sleep=clock.advance)
    ctrl.start(run_thread=False)
    return ctrl, bus, clock


def run(ctrl, clock, seconds):
    for _ in range(round(seconds * ctrl.cfg.loop_hz)):
        clock.advance(1 / ctrl.cfg.loop_hz)
        ctrl.tick()


def test_overtemperature_rests_before_going_limp(cfg):
    ctrl, bus, clock = make(cfg)
    ctrl.look(nod=15.0)                                  # raised pose, like the hardware incident
    run(ctrl, clock, 1.0)
    bus.servos["nod"].temperature = 66
    run(ctrl, clock, 0.06)
    assert ctrl.mode == Mode.RESTING                     # lowered under control, not dropped
    assert ctrl.latched.level == Level.HARD
    assert any("temperature" in r for r in ctrl.latched.reasons)
    assert bus.torque_limit_log[-1] == 0.5
    run(ctrl, clock, 4.0)
    assert ctrl.mode == Mode.LIMP
    for name in ("lean", "nod"):
        assert abs(bus.servos[name].position - cfg.joints[name].rest) <= 35


def test_single_bad_temperature_reading_does_not_fault(cfg):
    ctrl, bus, clock = make(cfg)
    bus.servos["nod"].temperature = 82
    run(ctrl, clock, 0.02)
    bus.servos["nod"].temperature = 28
    run(ctrl, clock, 0.5)
    assert ctrl.mode == Mode.RUNNING
    assert ctrl.faults == []


def test_status_bits_rest_then_limp(cfg):
    ctrl, bus, clock = make(cfg)
    bus.servos["turn"].status = 0x20
    run(ctrl, clock, 0.06)
    assert ctrl.mode == Mode.RESTING
    run(ctrl, clock, 4.0)
    assert ctrl.mode == Mode.LIMP


def test_overvoltage_goes_limp_immediately(cfg):
    ctrl, bus, clock = make(cfg)
    for s in bus.servos.values():
        s.voltage = 8.2
    run(ctrl, clock, 0.06)
    assert ctrl.mode == Mode.LIMP
    assert 0.5 not in bus.torque_limit_log


def test_lost_servo_goes_limp_after_five_misses(cfg):
    ctrl, bus, clock = make(cfg)
    bus.servos["nod"].responding = False
    run(ctrl, clock, 4 / 50)
    assert ctrl.mode == Mode.RUNNING
    run(ctrl, clock, 1 / 50)
    assert ctrl.mode == Mode.LIMP


def test_jam_holds_and_latches_without_resting(cfg):
    ctrl, bus, clock = make(cfg)
    ctrl.look(turn=30.0)
    run(ctrl, clock, 0.1)
    bus.servos["turn"].jammed = True
    jammed_at = bus.servos["turn"].position
    run(ctrl, clock, 1.0)
    assert ctrl.latched is not None and ctrl.latched.level == Level.SOFT
    assert any("tracking" in r for r in ctrl.latched.reasons)
    assert ctrl.mode == Mode.RUNNING                     # holding, not resting
    assert ctrl.setpoints["turn"] == pytest.approx(jammed_at, abs=1)
    assert ctrl.look(turn=0.0) is False                  # latched: commands ignored
    run(ctrl, clock, 3.0)
    assert ctrl.mode == Mode.RUNNING


def test_persistent_soft_fault_rests_then_goes_limp(cfg):
    ctrl, bus, clock = make(cfg)
    bus.servos["nod"].current_ma = 2000.0
    run(ctrl, clock, 0.1)
    assert ctrl.latched.level == Level.SOFT
    assert ctrl.mode == Mode.RUNNING
    run(ctrl, clock, 1.0)
    assert ctrl.mode == Mode.RESTING
    assert bus.torque_limit_log[-1] == 0.5
    run(ctrl, clock, 4.0)
    assert ctrl.mode == Mode.LIMP
    assert bus.torque_log[-1] is False
    for name in ("lean", "nod"):
        assert abs(bus.servos[name].position - cfg.joints[name].rest) <= 35   # within 3 degrees


def test_rest_moves_no_faster_than_rest_speed(cfg):
    ctrl, bus, clock = make(cfg)
    bus.servos["nod"].current_ma = 2000.0
    run(ctrl, clock, 1.2)
    assert ctrl.mode == Mode.RESTING
    n_before = len(bus.writes)
    run(ctrl, clock, 1.0)
    nod = [w["nod"] for w in bus.writes[n_before:]]
    per_tick = cfg.rest_speed_deg_s / (360 / 4096) / cfg.loop_hz
    assert max(abs(b - a) for a, b in zip(nod, nod[1:])) <= per_tick + 1


def test_fault_near_rest_still_rests_before_going_limp(cfg):
    # Going limp with lean even slightly off its back stop lets the head fall forward (hardware).
    ctrl, bus, clock = make(cfg, start="rest")
    bus.servos["lean"].current_ma = 2000.0
    run(ctrl, clock, 0.06)
    assert ctrl.mode == Mode.RUNNING                     # holding, not limp
    run(ctrl, clock, 3.0)
    assert 0.5 in bus.torque_limit_log                   # went through the rest sequence
    assert ctrl.mode == Mode.LIMP
    assert abs(bus.servos["lean"].position - cfg.joints["lean"].rest) <= 6


def test_rest_is_not_done_until_lean_is_on_its_stop(cfg):
    # Drill B: rest ended with lean 23 counts (2 deg) off its stop; limp then let it fall 72 deg.
    ctrl, bus, clock = make(cfg)
    bus.servos["nod"].current_ma = 2000.0
    run(ctrl, clock, 1.2)
    assert ctrl.mode == Mode.RESTING
    bus.servos["lean"].position = cfg.joints["lean"].rest + 23
    bus.servos["lean"].jammed = True
    run(ctrl, clock, 3.0)                                # nod reaches its rest well within this
    assert abs(bus.servos["nod"].position - cfg.joints["nod"].rest) <= 2
    assert ctrl.mode == Mode.RESTING                     # still trying to get lean onto its stop


def test_lost_servo_during_rest_goes_limp(cfg):
    ctrl, bus, clock = make(cfg)
    bus.servos["nod"].current_ma = 2000.0
    run(ctrl, clock, 1.2)
    assert ctrl.mode == Mode.RESTING
    bus.servos["turn"].responding = False
    run(ctrl, clock, 0.1)
    assert ctrl.mode == Mode.LIMP
    assert ctrl.faults[-1].level == Level.HARD


def test_overtemperature_during_rest_keeps_resting_and_is_recorded(cfg):
    ctrl, bus, clock = make(cfg)
    bus.servos["nod"].current_ma = 2000.0
    run(ctrl, clock, 1.2)
    assert ctrl.mode == Mode.RESTING
    bus.servos["turn"].temperature = 70
    run(ctrl, clock, 0.1)
    assert ctrl.mode == Mode.RESTING
    assert ctrl.faults[-1].level == Level.HARD


def test_clear_fault_restarts_without_jump(cfg):
    ctrl, bus, clock = make(cfg)
    bus.servos["nod"].temperature = 66
    run(ctrl, clock, 4.0)
    assert ctrl.mode == Mode.LIMP
    bus.servos["nod"].temperature = 30
    ctrl.clear_fault()
    assert ctrl.mode == Mode.RUNNING
    assert ctrl.latched is None
    assert len(ctrl.faults) == 1
    assert not bus.jump_risk
    assert ctrl.look(turn=10.0) is True


def test_fault_record_keeps_readings(cfg):
    ctrl, bus, clock = make(cfg)
    bus.servos["nod"].temperature = 66
    run(ctrl, clock, 0.06)
    fault = ctrl.faults[0]
    assert fault.readings["nod"].temperature == 66
    assert fault.time == pytest.approx(0.06)


def test_automatic_rest_times_out_when_obstructed(cfg):
    ctrl, bus, clock = make(cfg)
    bus.servos["nod"].current_ma = 2000.0
    run(ctrl, clock, 1.2)
    assert ctrl.mode == Mode.RESTING
    bus.servos["nod"].jammed = True
    run(ctrl, clock, cfg.rest_timeout_s + 0.5)
    assert ctrl.mode == Mode.LIMP
    assert all(not s.torque for s in bus.servos.values())
    assert any("rest did not finish" in r for f in ctrl.faults for r in f.reasons)


def test_limp_resends_torque_off_until_it_sticks(cfg):
    ctrl, bus, clock = make(cfg)
    bus.drop_torque_off = 1
    for s in bus.servos.values():
        s.voltage = 8.2                              # over-voltage: immediate limp
    run(ctrl, clock, 0.06)
    assert ctrl.mode == Mode.LIMP
    assert bus.servos["nod"].torque is True          # the first torque-off packet was lost
    run(ctrl, clock, 0.02)
    assert all(not s.torque for s in bus.servos.values())


def test_new_soft_fault_after_latch_is_recorded_once(cfg):
    ctrl, bus, clock = make(cfg)
    bus.servos["nod"].current_ma = 2000.0
    run(ctrl, clock, 0.1)
    bus.servos["nod"].current_ma = 100.0
    run(ctrl, clock, 0.2)
    assert len(ctrl.faults) == 1
    bus.servos["turn"].temperature = 61
    run(ctrl, clock, 0.5)
    assert len(ctrl.faults) == 2
    assert any("temperature" in r for r in ctrl.faults[-1].reasons)


def test_rest_is_not_done_until_nod_is_fully_back(cfg):
    # Drill B again: lean was on its stop but nod stopped ~2 deg short of its back stop; limp let
    # the head's weight sit in front of the lean axis and it tipped forward 75 deg.
    ctrl, bus, clock = make(cfg)
    bus.servos["nod"].current_ma = 2000.0
    run(ctrl, clock, 1.2)
    assert ctrl.mode == Mode.RESTING
    bus.servos["nod"].position = cfg.joints["nod"].rest - 23
    bus.servos["nod"].jammed = True
    run(ctrl, clock, 3.0)
    assert abs(bus.servos["lean"].position - cfg.joints["lean"].rest) <= 2
    assert ctrl.mode == Mode.RESTING


def make_holding(make_cfg):
    cfg = make_cfg({"limp_after_rest": False})
    return (cfg, *make(cfg))


def test_rest_can_end_holding_the_rest_pose_instead_of_limp(make_cfg):
    cfg, ctrl, bus, clock = make_holding(make_cfg)
    bus.servos["nod"].current_ma = 2000.0
    run(ctrl, clock, 5.0)
    assert ctrl.mode == Mode.RESTED
    assert all(s.torque for s in bus.servos.values())    # held: the head cannot droop forward
    assert bus.torque_limit_log[-1] == 0.5
    for name in ("lean", "nod"):
        assert abs(bus.servos[name].position - cfg.joints[name].rest) <= 6
    assert ctrl.look(turn=10) is False


def test_uncontrollable_fault_while_rested_goes_limp(make_cfg):
    cfg, ctrl, bus, clock = make_holding(make_cfg)
    bus.servos["nod"].current_ma = 2000.0
    run(ctrl, clock, 5.0)
    assert ctrl.mode == Mode.RESTED
    bus.servos["turn"].responding = False
    run(ctrl, clock, 0.2)
    assert ctrl.mode == Mode.LIMP


def test_clear_fault_from_rested_resumes(make_cfg):
    cfg, ctrl, bus, clock = make_holding(make_cfg)
    bus.servos["nod"].current_ma = 2000.0
    run(ctrl, clock, 5.0)
    bus.servos["nod"].current_ma = 100.0
    ctrl.clear_fault()
    assert ctrl.mode == Mode.RUNNING
    assert ctrl.look(turn=10) is True
