"""Head-camera tests. Run only with the camera plugged in: .venv/bin/pytest -m camera -v"""
import time

import pytest

from perception.config import load_config
from perception.sources import CameraSource, find_camera

pytestmark = pytest.mark.camera


def test_head_camera_is_found_by_usb_id_or_name():
    cfg = load_config()
    assert find_camera(cfg.camera_usb_id, cfg.camera_name) is not None


def test_head_camera_delivers_fresh_frames():
    src = CameraSource.from_config(load_config())
    src.open()
    try:
        time.sleep(1.0)
        a = src.read()
        time.sleep(0.2)
        b = src.read()
        assert a is not None and b is not None and b.time > a.time
        print("frame size:", a.image.shape[1], "x", a.image.shape[0])
    finally:
        src.close()
