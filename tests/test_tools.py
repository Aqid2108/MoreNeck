import pytest

from tools.calibrate import derive_joint
from tools.log_current import summarize


def test_derive_joint_insets_limits_and_picks_rest_nearest_limp_end():
    j = derive_joint("lean", neutral_raw=934, positive_raw=1000, seen_min=801, seen_max=1066,
                     limp_raw=808)
    assert j == {"raw_min": 824, "raw_max": 1043, "neutral": 934, "rest": 824, "direction": 1}


def test_derive_joint_negative_direction_and_high_rest():
    j = derive_joint("nod", neutral_raw=2939, positive_raw=2850, seen_min=2721, seen_max=3156,
                     limp_raw=3154)
    assert j["direction"] == -1
    assert j["rest"] == 3133


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
