"""Frame sources: the live head camera and video files, behind one interface."""
from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("perception.sources")

_ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    time: float


def rotate_image(img: np.ndarray, degrees: int) -> np.ndarray:
    return img if degrees == 0 else cv2.rotate(img, _ROTATIONS[degrees])


def _default_enumerate():
    from cv2_enumerate_cameras import enumerate_cameras

    return enumerate_cameras()


# Windows lists each camera twice, once per backend. Measured on the head camera at 1280x720:
# Media Foundation 30 fps (opens in ~2 s with hardware transforms off, see perception/__init__.py);
# DirectShow 18.7 fps with MJPG, 7 fps without. So Media Foundation first.
_WINDOWS_BACKEND_ORDER = (cv2.CAP_MSMF, cv2.CAP_DSHOW)


def _backend_of(index: int) -> int:
    # OpenCV encodes the backend in the index (701 = DirectShow device 1); cv2-enumerate-cameras
    # reports backend=0 on Windows, so derive it from the index.
    return index // 100 * 100


def _backend_rank(cam) -> int:
    backend = getattr(cam, "backend", 0) or _backend_of(getattr(cam, "index", 0))
    if sys.platform != "win32" or backend not in _WINDOWS_BACKEND_ORDER:
        return len(_WINDOWS_BACKEND_ORDER)
    return _WINDOWS_BACKEND_ORDER.index(backend)


def list_cameras(enumerate_fn=None) -> list:
    """All cameras OpenCV can open, preferred backend first."""
    return sorted((enumerate_fn or _default_enumerate)(), key=_backend_rank)


def find_camera(usb_id: str, name: str, enumerate_fn=None) -> int | None:
    """OpenCV index (backend-encoded) of the camera, by USB ID first, then by name."""
    cams = list_cameras(enumerate_fn)
    vid, pid = (int(x, 16) for x in usb_id.split(":"))
    for c in cams:
        if getattr(c, "vid", None) == vid and getattr(c, "pid", None) == pid:
            return c.index
    for c in cams:
        if c.name == name:
            return c.index
    return None


class VideoFileSource:
    def __init__(self, path, realtime: bool = False, clock=time.monotonic, sleep=time.sleep):
        self.path = str(path)
        self.realtime = realtime
        self.clock = clock
        self._sleep = sleep
        self._cap = None
        self._n = 0
        self._fps = 30.0
        self._start = 0.0
        self.eof = False

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise CameraError(f"could not open video {self.path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._n = 0
        self._start = self.clock()
        self.eof = False

    def read(self) -> Frame | None:
        if self.eof or self._cap is None:
            return None
        ok, img = self._cap.read()
        if not ok:
            self.eof = True
            return None
        offset = self._n / self._fps
        self._n += 1
        if not self.realtime:
            return Frame(img, offset)
        due = self._start + offset
        wait = due - self.clock()
        if wait > 0:
            self._sleep(wait)
        return Frame(img, due)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class CameraSource:
    """Live camera. A reader thread keeps only the newest frame, so consumers never see a backlog."""

    def __init__(self, usb_id: str = "0C45:6366", name: str = "Webcam", index_fallback: int | None = None,
                 width: int = 1280, height: int = 720, rotate: int = 0,
                 capture_factory=cv2.VideoCapture, find=find_camera, clock=time.monotonic,
                 join_timeout_s: float = 1.0, first_frame_timeout_s: float = 5.0):
        self.usb_id, self.name, self.index_fallback = usb_id, name, index_fallback
        self.width, self.height, self.rotate = width, height, rotate
        self._factory, self._find, self.clock = capture_factory, find, clock
        self.join_timeout_s = join_timeout_s
        self.first_frame_timeout_s = first_frame_timeout_s
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._delivered: Frame | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.opened_index: int | None = None

    @classmethod
    def from_config(cls, cfg) -> "CameraSource":
        return cls(cfg.camera_usb_id, cfg.camera_name, cfg.camera_index_fallback, cfg.width,
                   cfg.height, cfg.rotate)

    def open(self) -> None:
        index = self._find(self.usb_id, self.name)
        if index is None:
            if self.index_fallback is None:
                # Never guess: index 0 is usually a camera that doesn't move with the head
                # (the laptop's built-in one).
                raise CameraError(f"camera {self.usb_id} / {self.name!r} not found; is the head "
                                  "camera plugged in? (tools/check_devices.py lists cameras)")
            log.warning("camera %s / %r not found; using configured index %d", self.usb_id,
                        self.name, self.index_fallback)
            index = self.index_fallback
        cap = self._factory(index)
        if not cap.isOpened():
            cap.release()
            hint = ""
            if sys.platform == "win32":
                hint = ("; another app (Teams, Zoom, Camera) may be using it, or Windows Settings > "
                        "Privacy & security > Camera > 'Let desktop apps access your camera' is off")
            raise CameraError(f"could not open camera at index {index}{hint}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if sys.platform == "win32" and _backend_of(index) == cv2.CAP_DSHOW:
            # DirectShow defaults to YUY2, which USB 2 carries at only 7 fps at 720p. MJPG must be
            # requested after the size, or DirectShow ignores it.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.opened_index = index
        # Each reader owns its capture and stop event, so a reader that is still blocked in read()
        # after close() can neither be released under nor touch a later reader's capture.
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, args=(cap, self._stop),
                                        name="camera-reader", daemon=True)
        self._thread.start()
        # Windows' Media Foundation delivers the first frame ~1.7 s after the capture opens. Wait for
        # it, so callers' no-frame timeouts don't fire (and reopen the camera) while it warms up.
        if self.first_frame_timeout_s > 0:
            deadline = time.monotonic() + self.first_frame_timeout_s
            while time.monotonic() < deadline and self._thread.is_alive():
                with self._lock:
                    if self._latest is not None:
                        break
                time.sleep(0.01)
            else:
                log.warning("camera opened but no frame within %.1f s", self.first_frame_timeout_s)

    def _reader(self, cap, stop: threading.Event) -> None:
        try:
            while not stop.is_set():
                ok, img = cap.read()
                if stop.is_set():
                    break
                if not ok:
                    time.sleep(0.01)
                    continue
                frame = Frame(rotate_image(img, self.rotate), self.clock())
                with self._lock:
                    self._latest = frame
        finally:
            cap.release()

    def read(self) -> Frame | None:
        with self._lock:
            frame = self._latest
        if frame is None or frame is self._delivered:
            return None
        self._delivered = frame
        return frame

    def close(self) -> None:
        self._stop.set()                 # the reader releases its own capture when read() returns
        if self._thread is not None:
            self._thread.join(timeout=self.join_timeout_s)
            if self._thread.is_alive():
                log.warning("camera reader still blocked in read(); it will release on return")
            self._thread = None
        with self._lock:
            self._latest = self._delivered = None
