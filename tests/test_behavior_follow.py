import pytest

from behavior.follow import Command, Lean, follow_axis
from behavior.idle import Scanner
from tests.behavior_helpers import person


def test_follow_axis_moves_part_way_from_the_base():
    assert follow_axis(10.0, 20.0, 0.3, 1.5, None) == pytest.approx(16.0)
    assert follow_axis(10.0, -20.0, 0.3, 1.5, 12.0) == pytest.approx(4.0)


def test_follow_axis_dead_zone_keeps_the_previous_command():
    assert follow_axis(10.0, 1.0, 0.3, 1.5, 12.0) == 12.0
    assert follow_axis(10.0, 1.0, 0.3, 1.5, None) == pytest.approx(10.3)   # nothing to keep yet


def test_command_is_a_value():
    assert Command(1.0, 2.0, 3.0) == Command(1.0, 2.0, 3.0)


def test_lean_enters_when_close_and_facing(preset):
    lean = Lean(preset)
    assert lean.update(person(distance=2.0, facing_score=0.9), 0.0) == 0.0
    assert lean.update(person(distance=1.4, facing_score=0.9), 0.1) == 5.0


def test_lean_holds_in_the_hysteresis_band_and_releases_beyond(preset):
    lean = Lean(preset)
    lean.update(person(distance=1.4, facing_score=0.9), 0.0)
    assert lean.update(person(distance=1.7, facing_score=0.9), 0.1) == 5.0
    assert lean.update(person(distance=1.9, facing_score=0.9), 0.2) == 0.0


def test_lean_releases_after_not_facing_for_a_while(preset):
    lean = Lean(preset)
    lean.update(person(distance=1.2, facing_score=0.9), 0.0)
    assert lean.update(person(distance=1.2, facing_score=0.1), 1.0) == 5.0
    assert lean.update(person(distance=1.2, facing_score=0.1), 1.9) == 5.0
    assert lean.update(person(distance=1.2, facing_score=0.1), 2.0) == 0.0


def test_unknown_distance_never_leans(preset):
    lean = Lean(preset)
    assert lean.update(person(distance=None, facing_score=1.0), 0.0) == 0.0


def test_zero_lean_preset(make_preset):
    lean = Lean(make_preset({"lean_deg": 0.0}))
    assert lean.update(person(distance=1.0, facing_score=1.0), 0.0) == 0.0


def test_scanner_sweeps_pauses_and_reverses(preset):
    s = Scanner(preset)                    # 15 deg/s, +/-40 deg, 1 s pause
    s.reset(0.0)
    angles = []
    t = 0.0
    for _ in range(200):                   # 10 s at 20 Hz
        t += 0.05
        angles.append(s.step(0.05, t))
    assert max(angles) == pytest.approx(40.0)
    assert min(angles) == pytest.approx(-40.0)
    first_top = angles.index(max(angles))
    assert angles[first_top:first_top + 20] == [40.0] * 20     # 1 s pause at the end


def test_scanner_starts_where_told(preset):
    s = Scanner(preset)
    s.reset(12.0)
    assert s.step(0.05, 0.05) == pytest.approx(12.75)
