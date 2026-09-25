import copy

import pytest
import yaml

from neck.config import (
    DEFAULT_CONFIG, DEG_PER_COUNT, config_from_dict, load_config, update_joint_config,
)
from tests.conftest import BASE


def test_default_config_file_loads():
    c = load_config(DEFAULT_CONFIG)
    # Joint limits are rewritten by tools/calibrate.py; check identity and wiring, not values.
    assert {n: j.id for n, j in c.joints.items()} == {"turn": 4, "nod": 3, "lean": 1}
    assert c.port_vid == 0x1A86
    assert c.loop_hz == 50


def test_deg_raw_roundtrip_respects_direction(make_cfg):
    c = make_cfg({"turn.direction": -1})
    turn = c.joints["turn"]
    assert turn.to_raw(0.0) == 2286
    assert turn.to_raw(10.0) == pytest.approx(2286 - 10.0 / DEG_PER_COUNT)
    assert turn.to_deg(turn.to_raw(-17.5)) == pytest.approx(-17.5)


def test_clamp_raw(cfg):
    lean = cfg.joints["lean"]
    assert lean.clamp_raw(0) == 824
    assert lean.clamp_raw(5000) == 1043
    assert lean.clamp_raw(900.5) == 900.5


@pytest.mark.parametrize(
    "joint,field,value,message",
    [
        ("lean", "raw_min", 1100, "lean"),          # raw_min >= raw_max
        ("nod", "neutral", 3500, "nod"),            # neutral outside limits
        ("turn", "rest", 5000, "turn"),             # rest beyond 4095
        ("turn", "direction", 0, "turn"),           # direction must be +/-1
        ("nod", "id", 1, "id"),                     # duplicate id with lean
        ("lean", "temp_soft_c", 70, "lean"),        # warn < soft < hard violated
        ("turn", "raw_max", 5000, "turn"),          # beyond 4095
    ],
)
def test_validation_errors(joint, field, value, message):
    d = copy.deepcopy(BASE)
    d["joints"][joint][field] = value
    with pytest.raises(ValueError, match=message):
        config_from_dict(d)


def test_missing_joint_is_rejected():
    d = copy.deepcopy(BASE)
    del d["joints"]["nod"]
    with pytest.raises(ValueError, match="nod"):
        config_from_dict(d)


def test_update_joint_config_rewrites_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(copy.deepcopy(BASE)))
    update_joint_config(path, {"turn": {"neutral": 2300, "direction": -1}})
    c = load_config(path)
    assert c.joints["turn"].neutral == 2300
    assert c.joints["turn"].direction == -1
    assert c.joints["nod"].neutral == 2939


def test_default_config_has_bus_pid_and_rest_timeout():
    c = load_config(DEFAULT_CONFIG)
    assert c.port_pid == 0x55D3
    assert c.rest_timeout_s == 10.0


def test_loop_hz_must_be_positive(make_cfg):
    with pytest.raises(ValueError, match="loop_hz"):
        make_cfg({"loop_hz": 0})


def test_rest_may_sit_outside_soft_limits(cfg):
    # Lean is only stable limp on its back stop, which is outside the soft range look() may use.
    lean = cfg.joints["lean"]
    assert lean.rest < lean.raw_min
    assert lean.rest_tolerance_deg == 0.5
