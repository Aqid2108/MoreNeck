import copy

import pytest

from behavior.config import PRESETS_FILE, load_preset, load_presets, preset_from_dict
from tests.conftest import BBASE


def test_presets_file_has_the_three_presets_from_the_spec():
    presets = load_presets(PRESETS_FILE)
    assert set(presets) == {"calm", "curious", "shy"}
    calm, curious, shy = presets["calm"], presets["curious"], presets["shy"]
    assert (calm.follow_gain, calm.stickiness, calm.switch_hold_s, calm.lean_deg) == (0.25, 0.6, 1.5, 5.0)
    assert curious.idle_scan and curious.glance_movement and curious.lean_deg == 10.0
    assert shy.avert_after_s == 4.0 and shy.lean_deg == 0.0 and not shy.glance_newcomer
    assert calm.avert_after_s is None
    assert calm.deadband_deg == shy.deadband_deg == 3.0          # shared values merged in


def test_load_preset_by_name():
    assert load_preset("calm").name == "calm"
    with pytest.raises(ValueError, match="unknown preset 'loud'"):
        load_preset("loud")


@pytest.mark.parametrize("key,value,message", [
    ("follow_gain", 0.0, "follow_gain"),
    ("follow_gain", 1.5, "follow_gain"),
    ("near_full_m", 5.0, "near_full_m"),
    ("lean_near_m", 2.0, "lean_near_m"),
    ("w_facing", -1.0, "weights"),
    ("avert_after_s", 0.0, "avert_after_s"),
    ("loop_hz", 0.0, "positive"),
])
def test_validation_errors(key, value, message):
    d = copy.deepcopy(BBASE)
    d[key] = value
    with pytest.raises(ValueError, match=message):
        preset_from_dict("x", d)


def test_missing_and_unknown_keys():
    d = copy.deepcopy(BBASE)
    del d["follow_gain"]
    with pytest.raises(ValueError, match="missing: follow_gain"):
        preset_from_dict("x", d)
    d = copy.deepcopy(BBASE)
    d["folow_gain"] = 0.3
    with pytest.raises(ValueError, match="unknown.*folow_gain"):
        preset_from_dict("x", d)
