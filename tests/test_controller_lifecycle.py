import time

import pytest

from neck.controller import Mode, NeckController
from neck.safety import Level
from tests.fake_bus import FakeBus, FakeClock


def make(cfg):
    clock = FakeClock()
    bus = FakeBus(cfg, clock, start_positions={n: j.neutral for n, j in cfg.joints.items()})
    ctrl = NeckController(cfg, bus, clock=clock, sleep=clock.advance)
    return ctrl, bus, clock


def test_rest_without_thread_drives_ticks_until_limp(cfg):
    ctrl, bus, _ = make(cfg)
    ctrl.start(run_thread=False)
    ctrl.rest()
    assert ctrl.mode == Mode.LIMP
    assert bus.torque_log[-1] is False
    for name in ("lean", "nod"):
        assert abs(bus.servos[name].position - cfg.joints[name].rest) <= 35


def test_rest_times_out_to_limp_when_joint_is_stuck(cfg):
    ctrl, bus, _ = make(cfg)
    ctrl.start(run_thread=False)
    bus.servos["lean"].jammed = True
    ctrl.rest(timeout=3.0)
    assert ctrl.mode == Mode.LIMP


def test_close_rests_and_closes_bus(cfg):
    ctrl, bus, _ = make(cfg)
    ctrl.start(run_thread=False)
    ctrl.close()
    assert ctrl.mode == Mode.LIMP
    assert not bus.opened


def test_close_is_idempotent_and_safe_before_start(cfg):
    ctrl, bus, _ = make(cfg)
    ctrl.close()
    ctrl.close()
    assert bus.torque_log == []
    assert ctrl.mode == Mode.IDLE


def test_context_manager_rests_on_keyboard_interrupt(cfg):
    ctrl, bus, _ = make(cfg)
    with pytest.raises(KeyboardInterrupt):
        with ctrl:
            ctrl.start(run_thread=False)
            ctrl.look(turn=15.0)
            raise KeyboardInterrupt
    assert ctrl.mode == Mode.LIMP
    assert not bus.opened


def test_tick_exception_latches_hard_fault_and_goes_limp(cfg):
    ctrl, bus, _ = make(cfg)
    ctrl.start(run_thread=False)
    bus.raise_on_read = OSError("device disconnected")
    ctrl._safe_tick()
    assert ctrl.mode == Mode.LIMP
    assert ctrl.latched.level == Level.HARD
    assert "controller exception" in ctrl.latched.reasons[0]
    assert bus.torque_log[-1] is False


def test_threaded_loop_runs_and_closes_cleanly(cfg):
    bus = FakeBus(cfg, time.monotonic, start_positions={n: j.neutral for n, j in cfg.joints.items()})
    ctrl = NeckController(cfg, bus)
    ctrl.start()
    assert ctrl.look(turn=10.0)
    time.sleep(0.3)
    assert len(bus.writes) >= 10
    ctrl.close()
    assert ctrl.mode == Mode.LIMP
    assert not bus.opened
    assert not ctrl._thread.is_alive()


def test_repeated_tick_exceptions_record_a_single_fault(cfg):
    ctrl, bus, _ = make(cfg)
    ctrl.start(run_thread=False)
    bus.raise_on_read = OSError("device disconnected")
    for _ in range(10):
        ctrl._safe_tick()
    assert ctrl.mode == Mode.LIMP
    assert len(ctrl.faults) == 1


def test_close_leaves_head_held_at_rest_when_configured(make_cfg):
    cfg = make_cfg({"limp_after_rest": False})
    ctrl, bus, _ = make(cfg)
    ctrl.start(run_thread=False)
    ctrl.close()
    assert ctrl.mode == Mode.RESTED
    assert bus.torque_log[-1] is True
    assert not bus.opened
