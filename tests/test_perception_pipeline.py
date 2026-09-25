import time

import numpy as np
import pytest

from perception.pipeline import Perception, people_or_empty
from perception.sources import Frame
from perception.types import Detection, Snapshot


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeSource:
    def __init__(self, clock, size=(1280, 720)):
        self.clock = clock
        self.size = size
        self.producing = True
        self.opens = 0
        self.closes = 0

    def open(self):
        self.opens += 1

    def read(self):
        if not self.producing:
            return None
        w, h = self.size
        return Frame(np.zeros((h, w, 3), dtype=np.uint8), self.clock())

    def close(self):
        self.closes += 1


class FakeDetector:
    def __init__(self, dets=None):
        self.dets = dets if dets is not None else [Detection(box=(560, 100, 720, 570), score=0.9)]
        self.raise_next = None

    def detect(self, image):
        if self.raise_next:
            exc, self.raise_next = self.raise_next, None
            raise exc
        return list(self.dets)


def make(pcfg, **kw):
    clock = Clock()
    src = FakeSource(clock, **kw)
    det = FakeDetector()
    seen = []
    p = Perception(src, det, pcfg, clock=clock, sleep=lambda s: None,
                   on_snapshot=lambda snap, img: seen.append((snap, img)))
    src.open()
    p._last_frame_at = clock()
    p._last_reopen = clock()
    return p, src, det, clock, seen


def run(p, clock, n, dt=1 / 30):
    for _ in range(n):
        clock.t += dt
        p.step()


def test_publishes_a_person_after_confirmation(pcfg):
    p, _, _, clock, seen = make(pcfg)
    run(p, clock, 2)
    assert p.latest().people == ()
    run(p, clock, 1)
    snap = p.latest()
    assert snap.source_ok and len(snap.people) == 1 and snap.people[0].id == 1
    assert seen[-1][1] is not None and seen[-1][1].shape == (720, 1280, 3)


def test_frame_size_comes_from_the_frame(make_pcfg):
    p, _, _, clock, _ = make(make_pcfg({"width": 1280, "height": 720}), size=(640, 480))
    run(p, clock, 1)
    assert p.latest().frame_size == (640, 480)


def test_source_timeout_publishes_failure_and_recovers(pcfg):
    p, src, _, clock, seen = make(pcfg)
    run(p, clock, 5)
    src.producing = False
    run(p, clock, 10, dt=0.1)                      # 1.0 s without frames
    snap = p.latest()
    assert not snap.source_ok and snap.people == () and "no frames" in snap.error
    assert seen[-1][1] is None
    src.producing = True
    run(p, clock, 1)
    assert p.latest().source_ok


def test_reopens_source_while_failing(pcfg):
    p, src, _, clock, _ = make(pcfg)
    src.producing = False
    run(p, clock, 50, dt=0.1)                      # 5 s of failure
    assert src.opens >= 3                          # initial open + reopen every 2 s


def test_detector_exception_publishes_error_and_continues(pcfg):
    p, _, det, clock, _ = make(pcfg)
    run(p, clock, 3)
    det.raise_next = RuntimeError("MPS backend failure")
    run(p, clock, 1)
    snap = p.latest()
    assert not snap.source_ok and "MPS backend failure" in snap.error
    run(p, clock, 1)
    assert p.latest().source_ok


def test_pose_provider_feeds_world_yaw(pcfg):
    p, _, _, clock, _ = make(pcfg)
    p.pose_provider = lambda: (10.0, 0.0)
    run(p, clock, 3)
    (person,) = p.latest().people
    assert person.world_yaw_deg == pytest.approx(10.0 + person.head_deg[0])


def test_pose_provider_error_is_ignored(pcfg):
    p, _, _, clock, _ = make(pcfg)

    def broken():
        raise OSError("neck unplugged")

    p.pose_provider = broken
    run(p, clock, 3)
    assert p.latest().source_ok and p.latest().people[0].world_yaw_deg is None


def test_nothing_is_written_to_disk(pcfg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p, _, _, clock, _ = make(pcfg)
    run(p, clock, 20)
    assert list(tmp_path.iterdir()) == []


def test_people_or_empty():
    person_snap = Snapshot(timestamp=10.0, frame_size=(1, 1), people=("someone",), source_ok=True)
    assert people_or_empty(person_snap, now=10.2, stale_after_s=0.3) == ("someone",)
    assert people_or_empty(person_snap, now=10.4, stale_after_s=0.3) == ()
    failed = Snapshot(timestamp=10.0, frame_size=(1, 1), people=(), source_ok=False, error="x")
    assert people_or_empty(failed, now=10.0, stale_after_s=0.3) == ()


def test_threaded_start_and_stop(pcfg):
    src = FakeSource(time.monotonic)
    with Perception(src, FakeDetector(), pcfg) as p:
        time.sleep(0.2)
        assert p.latest().source_ok
    assert src.closes == 1
