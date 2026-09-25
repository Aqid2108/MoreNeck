import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from perception.sources import CameraError, CameraSource, VideoFileSource, find_camera, rotate_image


def make_video(path, n=10, fps=10, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n):
        img = np.full((size[1], size[0], 3), i * 20, dtype=np.uint8)
        writer.write(img)
    writer.release()
    return path


def test_video_file_source_reads_frames_then_eof(tmp_path):
    src = VideoFileSource(make_video(tmp_path / "clip.avi"))
    src.open()
    frames = []
    while (f := src.read()) is not None:
        frames.append(f)
    assert len(frames) == 10
    assert [round(f.time, 2) for f in frames[:3]] == [0.0, 0.1, 0.2]
    assert frames[0].image.shape == (48, 64, 3)
    assert src.eof and src.read() is None
    src.close()


def test_realtime_video_uses_clock_timestamps(tmp_path):
    now = [100.0]
    src = VideoFileSource(make_video(tmp_path / "clip.avi"), realtime=True,
                          clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s))
    src.open()
    a, b = src.read(), src.read()
    assert a.time == pytest.approx(100.0) and b.time == pytest.approx(100.1)


def test_rotate_image():
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    assert rotate_image(img, 0).shape == (48, 64, 3)
    assert rotate_image(img, 90).shape == (64, 48, 3)
    assert rotate_image(img, 180).shape == (48, 64, 3)


def test_find_camera_prefers_usb_id_then_name():
    cams = [SimpleNamespace(index=1200, name="Webcam", vid=0x1234, pid=0x0001),
            SimpleNamespace(index=1201, name="MacBook Pro Camera", vid=None, pid=None),
            SimpleNamespace(index=1202, name="Webcam", vid=0x0C45, pid=0x6366)]
    assert find_camera("0C45:6366", "Webcam", enumerate_fn=lambda: cams) == 1202
    assert find_camera("FFFF:0000", "MacBook Pro Camera", enumerate_fn=lambda: cams) == 1201
    assert find_camera("FFFF:0000", "Nope", enumerate_fn=lambda: cams) is None


class FakeCapture:
    """Produces numbered frames as fast as read() is called (~1 ms each)."""

    def __init__(self, index, opened=True):
        self.index = index
        self.opened = opened
        self.count = 0
        self.props = {}
        self.released = False

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.props[prop] = value

    def read(self):
        time.sleep(0.001)
        self.count += 1
        return True, np.full((48, 64, 3), self.count % 256, dtype=np.uint8)

    def release(self):
        self.released = True


def test_camera_source_returns_only_the_latest_frame():
    caps = []
    src = CameraSource(capture_factory=lambda i: caps.append(FakeCapture(i)) or caps[-1],
                       find=lambda usb, name: 1202)
    src.open()
    try:
        time.sleep(0.1)
        first = src.read()
        assert first is not None
        assert int(first.image[0, 0, 0]) > 20          # skipped the backlog
        assert caps[0].index == 1202
        assert caps[0].props[cv2.CAP_PROP_FRAME_WIDTH] == 1280
        second = src.read()
        assert second is None or second.time > first.time
    finally:
        src.close()
    assert caps[0].released


def test_camera_source_falls_back_to_index_and_reports_failure():
    src = CameraSource(index_fallback=3, capture_factory=lambda i: FakeCapture(i, opened=False),
                       find=lambda usb, name: None)
    with pytest.raises(CameraError, match="index 3"):
        src.open()
