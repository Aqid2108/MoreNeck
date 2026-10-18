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


def test_find_camera_prefers_media_foundation_on_windows(monkeypatch):
    # As cv2-enumerate-cameras reports them on Windows: DirectShow (7xx) and MSMF (14xx), backend=0
    monkeypatch.setattr("perception.sources.sys.platform", "win32")
    cams = [SimpleNamespace(index=700, name="Laptop", vid=0x0408, pid=0x4036, backend=0),
            SimpleNamespace(index=701, name="Webcam", vid=0x0C45, pid=0x6366, backend=0),
            SimpleNamespace(index=1400, name="Laptop", vid=0x0408, pid=0x4036, backend=0),
            SimpleNamespace(index=1401, name="Webcam", vid=0x0C45, pid=0x6366, backend=0)]
    assert find_camera("0C45:6366", "Webcam", enumerate_fn=lambda: cams) == 1401


def test_directshow_camera_requests_mjpg_after_size(monkeypatch):
    monkeypatch.setattr("perception.sources.sys.platform", "win32")
    caps = []
    src = CameraSource(capture_factory=lambda i: caps.append(FakeCapture(i)) or caps[-1],
                       find=lambda usb, name: 701)
    src.open()
    src.close()
    props = list(caps[0].props)
    assert caps[0].props[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter_fourcc(*"MJPG")
    assert props.index(cv2.CAP_PROP_FOURCC) > props.index(cv2.CAP_PROP_FRAME_HEIGHT)


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


def test_missing_head_camera_is_an_error_without_an_explicit_fallback():
    # Index 0 on this Mac is the MacBook or iPhone camera, which does not move with the head.
    src = CameraSource(index_fallback=None, capture_factory=lambda i: FakeCapture(i),
                       find=lambda usb, name: None)
    with pytest.raises(CameraError, match="not found"):
        src.open()


class SlowCapture(FakeCapture):
    """read() blocks like AVFoundation after an unplug; records a release that happens mid-read."""

    def __init__(self, index):
        super().__init__(index)
        self.reading = False
        self.released_mid_read = False

    def read(self):
        self.reading = True
        time.sleep(0.3)
        self.reading = False
        self.count += 1
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        if self.reading:
            self.released_mid_read = True
        self.released = True


def test_close_never_releases_a_capture_mid_read_and_reopen_is_isolated():
    caps = []
    src = CameraSource(capture_factory=lambda i: caps.append(SlowCapture(i)) or caps[-1],
                       find=lambda usb, name: 1, join_timeout_s=0.05, first_frame_timeout_s=0)
    src.open()
    time.sleep(0.05)                     # reader is now blocked inside read()
    src.close()                          # join times out
    src.open()                           # reopen straight away
    time.sleep(0.8)
    src.close()
    time.sleep(0.5)
    assert not caps[0].released_mid_read
    assert caps[0].released
    assert caps[0].count == 1            # the old reader never touched the capture again
