import copy

import pytest

from perception.config import DEFAULT_CONFIG, config_from_dict, load_config
from perception.types import KEYPOINT_NAMES, Detection, Person, Snapshot
from tests.conftest import PBASE


def test_default_config_file_loads():
    c = load_config(DEFAULT_CONFIG)
    assert c.camera_usb_id == "0C45:6366"
    assert c.camera_name == "Webcam"
    assert c.hfov_deg == 60.0


@pytest.mark.parametrize("key,value,message", [
    ("rotate", 45, "rotate"),
    ("hfov_deg", 200, "hfov"),
    ("width", 0, "width"),
    ("facing_threshold", 1.5, "0..1"),
    ("track_confirm_frames", 0, "track_confirm_frames"),
    ("max_coast_s", 0, "positive"),
    ("camera_usb_id", "0c456366", "camera_usb_id"),
])
def test_validation_errors(key, value, message):
    d = copy.deepcopy(PBASE)
    d[key] = value
    with pytest.raises(ValueError, match=message):
        config_from_dict(d)


def test_missing_and_unknown_keys_are_reported():
    d = copy.deepcopy(PBASE)
    del d["hfov_deg"]
    with pytest.raises(ValueError, match="missing: hfov_deg"):
        config_from_dict(d)
    d = copy.deepcopy(PBASE)
    d["hfvo_deg"] = 60
    with pytest.raises(ValueError, match="unknown.*hfvo_deg"):
        config_from_dict(d)


def test_contract_types_are_immutable():
    det = Detection(box=(0, 0, 10, 20), score=0.9, keypoints={"nose": (5, 2, 0.9)})
    person = Person(id=1, box=(0, 0, 10, 20), head_px=(5, 2), head_deg=(1.0, 2.0), world_yaw_deg=None,
                    distance_m=2.0, distance_source="body", facing_score=0.8, facing=True,
                    motion="still", confidence=0.9, visible=True, first_seen=0.0, last_seen=1.0)
    snap = Snapshot(timestamp=1.0, frame_size=(1280, 720), people=(person,), source_ok=True)
    assert snap.error is None
    assert "nose" in KEYPOINT_NAMES and det.keypoints["nose"][2] == 0.9
    with pytest.raises(AttributeError):
        snap.people = ()


def test_index_fallback_is_off_unless_configured():
    assert load_config(DEFAULT_CONFIG).camera_index_fallback is None
