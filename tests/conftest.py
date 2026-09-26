import copy

import pytest

from neck.config import config_from_dict


def _joint(id_, raw_min, raw_max, neutral, rest, max_speed, max_accel, rest_tolerance_deg=0.5):
    return {
        "id": id_, "raw_min": raw_min, "raw_max": raw_max, "neutral": neutral, "rest": rest,
        "rest_tolerance_deg": rest_tolerance_deg,
        "direction": 1, "max_speed": max_speed, "max_accel": max_accel,
        "current_limit_ma": 800, "hold_current_limit_ma": 800,
        "temp_warn_c": 55, "temp_soft_c": 60, "temp_hard_c": 65,
    }


BASE = {
    "loop_hz": 50, "port_vid": 0x1A86, "port_pid": 0x55D3, "baudrate": 1_000_000,
    "voltage_warn_below": 6.8, "voltage_soft_below": 6.5,
    "voltage_hard_below": 6.0, "voltage_hard_above": 8.0,
    "tracking_max_error_deg": 8.0, "tracking_max_duration_s": 0.3,
    "spike_ticks": 3, "fault_debounce_ticks": 3, "hold_window_s": 2.0, "hard_misses": 5,
    "soft_limit_margin_deg": 2.0, "soft_hold_timeout_s": 1.0,
    "rest_speed_deg_s": 15.0, "rest_torque_fraction": 0.5,
    "rest_timeout_s": 10.0, "limp_after_rest": True,
    "joints": {
        "turn": _joint(4, 1873, 2918, 2286, 2286, 120.0, 400.0),
        "nod": _joint(3, 2744, 3133, 2939, 3150, 90.0, 300.0, rest_tolerance_deg=0.5),
        "lean": _joint(1, 824, 1043, 934, 806, 45.0, 150.0, rest_tolerance_deg=1.0),
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


PBASE = {
    "camera_usb_id": "0C45:6366", "camera_name": "Webcam", "camera_index_fallback": None,
    "width": 1280, "height": 720, "rotate": 0, "hfov_deg": 60.0,
    "model": "data/models/yolo11n-pose.pt", "imgsz": 640,
    "person_score_min": 0.4, "keypoint_conf_min": 0.5,
    "track_iou_min": 0.2, "track_confirm_frames": 3, "max_coast_s": 1.0,
    "shoulder_width_m": 0.40, "body_height_m": 1.70, "head_width_m": 0.15,
    "distance_ema_alpha": 0.3, "facing_threshold": 0.6,
    "motion_window_s": 1.0, "approach_rate_mps": 0.3, "passing_rate_dps": 15.0,
    "source_timeout_s": 0.5, "stale_after_s": 0.3, "reopen_every_s": 2.0,
}


def build_pcfg(overrides=None):
    from perception.config import config_from_dict

    d = copy.deepcopy(PBASE)
    d.update(overrides or {})
    return config_from_dict(d)


@pytest.fixture
def pcfg():
    return build_pcfg()


@pytest.fixture
def make_pcfg():
    return build_pcfg


BBASE = {   # frozen copy of the spec's "calm" preset; tests override what they need
    "w_facing": 1.0, "w_near": 0.6, "w_approach": 0.5, "stickiness": 0.6, "switch_hold_s": 1.5,
    "follow_gain": 0.25, "glance_newcomer": True, "glance_movement": False, "glance_s": 1.0,
    "glance_cooldown_s": 3.0, "lost_hold_s": 2.0, "idle_scan": False, "scan_after_s": 5.0,
    "scan_amplitude_deg": 40.0, "scan_speed_dps": 15.0, "scan_nod_deg": 10.0, "scan_pause_s": 1.0,
    "lean_deg": 5.0, "avert_after_s": None, "avert_deg": 20.0, "avert_s": 2.0,
    "deadband_deg": 3.0, "lean_near_m": 1.5, "lean_release_m": 1.8, "lean_release_s": 1.0,
    "coast_factor": 0.5, "near_full_m": 1.0, "near_zero_m": 4.0, "newcomer_window_s": 0.5,
    "loop_hz": 20.0, "pursuit_hz": 1.0,
}


def build_preset(overrides=None):
    from behavior.config import preset_from_dict

    d = copy.deepcopy(BBASE)
    d.update(overrides or {})
    return preset_from_dict("test", d)


@pytest.fixture
def preset():
    return build_preset()


@pytest.fixture
def make_preset():
    return build_preset
