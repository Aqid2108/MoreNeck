import copy

import pytest

from neck.config import config_from_dict


def _joint(id_, raw_min, raw_max, neutral, rest, max_speed, max_accel):
    return {
        "id": id_, "raw_min": raw_min, "raw_max": raw_max, "neutral": neutral, "rest": rest,
        "direction": 1, "max_speed": max_speed, "max_accel": max_accel,
        "current_limit_ma": 800, "hold_current_limit_ma": 800,
        "temp_warn_c": 55, "temp_soft_c": 60, "temp_hard_c": 65,
    }


BASE = {
    "loop_hz": 50, "port_vid": 0x1A86, "port_pid": 0x55D3, "baudrate": 1_000_000,
    "voltage_warn_below": 6.8, "voltage_soft_below": 6.5,
    "voltage_hard_below": 6.0, "voltage_hard_above": 8.0,
    "tracking_max_error_deg": 8.0, "tracking_max_duration_s": 0.3,
    "spike_ticks": 3, "hold_window_s": 2.0, "hard_misses": 5,
    "soft_limit_margin_deg": 2.0, "soft_hold_timeout_s": 1.0,
    "rest_speed_deg_s": 15.0, "rest_torque_fraction": 0.5,
    "rest_done_within_deg": 3.0, "near_rest_deg": 5.0, "rest_timeout_s": 10.0,
    "joints": {
        "turn": _joint(4, 1873, 2918, 2286, 2286, 120.0, 400.0),
        "nod": _joint(3, 2744, 3133, 2939, 3133, 90.0, 300.0),
        "lean": _joint(1, 824, 1043, 934, 824, 45.0, 150.0),
    },
}


def build_config(overrides=None):
    d = copy.deepcopy(BASE)
    for key, value in (overrides or {}).items():
        if "." in key:
            joint, field = key.split(".", 1)
            d["joints"][joint][field] = value
        else:
            d[key] = value
    return config_from_dict(d)


@pytest.fixture
def cfg():
    return build_config()


@pytest.fixture
def make_cfg():
    return build_config
