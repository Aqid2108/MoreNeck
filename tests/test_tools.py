import pytest

from tools.calibrate import derive_joint
from tools.log_current import summarize


def test_derive_joint_insets_limits_and_rests_lean_where_it_settles_limp():
    j = derive_joint("lean", neutral_raw=934, positive_raw=1000, seen_min=801, seen_max=1066,
                     limp_raw=808)
    assert j == {"raw_min": 824, "raw_max": 1043, "neutral": 934, "rest": 808, "direction": 1}


def test_derive_joint_negative_direction_and_high_rest():
    j = derive_joint("nod", neutral_raw=2939, positive_raw=2850, seen_min=2721, seen_max=3156,
                     limp_raw=3154)
    assert j["direction"] == -1
    assert j["rest"] == 3154          # where nod settles limp, on its back stop


def test_derive_joint_turn_rests_at_neutral():
    j = derive_joint("turn", neutral_raw=2286, positive_raw=2400, seen_min=1850, seen_max=2941,
                     limp_raw=2290)
    assert j["rest"] == 2286


def test_derive_joint_rejects_no_movement():
    with pytest.raises(ValueError, match="did not move"):
        derive_joint("turn", 2286, 2290, 1850, 2941, 2286)


def test_derive_joint_rejects_neutral_outside_range():
    with pytest.raises(ValueError, match="neutral"):
        derive_joint("lean", 1070, 1000, 801, 1066, 808)


def test_derive_joint_rejects_wrapping_range():
    with pytest.raises(ValueError, match="wrap"):
        derive_joint("turn", 2286, 2400, 100, 4000, 2286)


def test_summarize_peak_and_holding_mean():
    rows = []
    for i in range(200):                          # 4 s at 50 Hz
        t = i / 50
        holding = t >= 1.0
        rows.append({"t": t, "joint": "nod", "current_ma": 900.0 if t < 1.0 else 300.0,
                     "holding": holding})
    s = summarize(rows, hold_window_s=2.0)["nod"]
    assert s["peak_ma"] == 900.0
    assert s["hold_mean_ma"] == pytest.approx(300.0)
    assert s["recommended_current_limit_ma"] == 1200      # ceil(1.3 * 900 / 50) * 50
    assert s["recommended_hold_limit_ma"] == 400          # ceil(1.3 * 300 / 50) * 50


def test_derive_joint_rejects_a_range_that_was_barely_explored():
    with pytest.raises(ValueError, match="only moved"):
        derive_joint("turn", neutral_raw=2061, positive_raw=1990, seen_min=2022, seen_max=2084,
                     limp_raw=2061)


def test_ticklog_records_one_row_per_answering_joint_and_saves(tmp_path):
    import csv

    from neck.bus import Reading
    from tools.ticklog import TickLog

    log = TickLog(extra=lambda: {"mode": "resting"})
    r = Reading(position=3000, speed=0, load=-64, voltage=7.4, temperature=28, status=0, moving=True,
                current_ma=604.5, error=0)
    log(1.5, {"nod": r, "turn": None}, {"nod": 3010.0, "turn": 2023.0}, {"nod": False, "turn": True})
    assert log.rows == [{"t": 1.5, "joint": "nod", "position": 3000, "setpoint": 3010.0,
                         "current_ma": 604.5, "load": -64, "temperature": 28, "voltage": 7.4,
                         "holding": False, "mode": "resting"}]
    path = log.save(tmp_path / "ticks.csv")
    saved = list(csv.DictReader(path.open()))
    assert saved[0]["mode"] == "resting" and saved[0]["current_ma"] == "604.5"


@pytest.mark.parametrize("script", ["tools/neck_shell.py", "tools/log_current.py"])
def test_tool_scripts_start_when_run_directly(script):
    # Answering "n" at the safety prompt exits before any bus access.
    import subprocess
    import sys

    result = subprocess.run([sys.executable, script], input="n\n", capture_output=True, text=True,
                            timeout=30)
    assert result.returncode == 0, result.stderr
