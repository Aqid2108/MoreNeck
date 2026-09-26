import subprocess
import sys
import threading

from behavior.engine import Engine, State
from behavior.runner import Runner
from neck.controller import Mode
from perception.types import Snapshot
from tests.behavior_helpers import person


class Clock:
    def __init__(self):
        self.t = 10.0

    def __call__(self):
        return self.t


class FakePerception:
    def __init__(self, snap):
        self.snap = snap

    def latest(self):
        return self.snap


class FakeNeck:
    def __init__(self):
        self.mode, self.latched, self.looks, self.cleared = Mode.RUNNING, None, [], 0

    @property
    def state(self):
        return {"mode": self.mode, "latched": self.latched}

    def pose(self):
        return (0.0, 0.0, 0.0)

    def look(self, turn, nod, lean):
        self.looks.append((turn, nod, lean))
        return True

    def clear_fault(self):
        self.cleared += 1


def snap(t, people):
    return Snapshot(t, (1280, 720), tuple(people), True, None, (0.0, 0.0, 0.0))


def test_sends_look_commands_for_a_person(preset):
    clock = Clock()
    neck = FakeNeck()
    r = Runner(Engine(preset), FakePerception(snap(10.0, [person(1, yaw=20.0)])), neck, clock=clock)
    r.tick()
    assert neck.looks and neck.looks[-1][0] > 0


def test_stale_snapshots_mean_nobody_is_there(preset):
    clock = Clock()
    neck = FakeNeck()
    r = Runner(Engine(preset), FakePerception(snap(9.0, [person(1, yaw=20.0)])), neck, clock=clock)
    r.tick()
    assert r.engine.state == State.IDLE and neck.looks == [(0.0, 0.0, 0.0)]


def test_neck_fault_sends_nothing(preset):
    clock = Clock()
    neck = FakeNeck()
    neck.latched = "fault"
    r = Runner(Engine(preset), FakePerception(snap(10.0, [person(1, yaw=20.0)])), neck, clock=clock)
    r.tick()
    assert neck.looks == [] and r.engine.state == State.PAUSED


def test_dry_run_prints_at_most_twice_per_second(preset):
    clock = Clock()
    printed = []
    per = FakePerception(snap(10.0, [person(1, yaw=20.0)]))
    r = Runner(Engine(preset), per, None, clock=clock, printer=printed.append)
    for i in range(20):                               # 1 s of changing commands
        clock.t = 10.0 + i * 0.05
        per.snap = snap(clock.t, [person(1, yaw=20.0 - i)])
        r.tick()
    assert 1 <= len(printed) <= 3
    assert "tracking" in printed[0]


def test_clear_fault_is_explicit(preset):
    neck = FakeNeck()
    r = Runner(Engine(preset), FakePerception(snap(10.0, [])), neck, clock=Clock())
    r.tick()
    assert neck.cleared == 0
    r.clear_fault()
    assert neck.cleared == 1


def test_run_loop_survives_a_failing_tick(preset):
    class Broken:
        calls = 0

        def latest(self):
            Broken.calls += 1
            raise RuntimeError("camera exploded")

    stop = threading.Event()
    r = Runner(Engine(preset), Broken(), FakeNeck(), sleep=lambda s: stop.set() if Broken.calls >= 3 else None)
    r.run(stop)
    assert Broken.calls >= 3


def test_behave_help():
    result = subprocess.run([sys.executable, "tools/behave.py", "--help"], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "--preset" in result.stdout and "--dry-run" in result.stdout


def test_on_tick_records_what_each_layer_saw_and_decided(preset):
    clock = Clock()
    records = []
    r = Runner(Engine(preset), FakePerception(snap(10.0, [person(3, yaw=12.0, pitch=4.0)])), FakeNeck(),
               clock=clock, on_tick=records.append)
    r.tick()
    rec = records[0]
    assert rec["t"] == 10.0 and rec["state"] == "tracking" and rec["target"] == 3
    assert rec["snapshot_ts"] == 10.0 and rec["snapshot_pose"] == [0.0, 0.0, 0.0]
    assert rec["neck_pose"] == [0.0, 0.0, 0.0] and rec["sent"] is True
    assert rec["command"] == [rec["command"][0], rec["command"][1], 0.0] and rec["command"][0] > 0
    assert rec["people"] == [{"id": 3, "yaw": 12.0, "pitch": 4.0, "conf": 0.9, "visible": True,
                              "facing": 0.8, "dist": 2.0, "motion": "still", "head_kp": True}]
