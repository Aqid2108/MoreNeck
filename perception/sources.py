"""Frame sources: the live head camera and video files, behind one interface."""
from __future__ import annotations

import logging
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


def find_camera(usb_id: str, name: str, enumerate_fn=None) -> int | None:
    """OpenCV index (backend-encoded) of the camera, by USB ID first, then by name."""
    cams = list((enumerate_fn or _default_enumerate)())
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
                 join_timeout_s: float = 1.0):
        self.usb_id, self.name, self.index_fallback = usb_id, name, index_fallback
        self.width, self.height, self.rotate = width, height, rotate
        self._factory, self._find, self.clock = capture_factory, find, clock
        self.join_timeout_s = join_timeout_s
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._delivered: Frame | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_config(cls, cfg) -> "CameraSource":
        return cls(cfg.camera_usb_id, cfg.camera_name, cfg.camera_index_fallback, cfg.width,
                   cfg.height, cfg.rotate)

    def open(self) -> None:
        index = self._find(self.usb_id, self.name)
        if index is None:
            if self.index_fallback is None:
                # Never guess: index 0 is usually a camera that doesn't move with the head.
                raise CameraError(f"camera {self.usb_id} / {self.name!r} not found")
            log.warning("camera %s / %r not found; using configured index %d", self.usb_id,
                        self.name, self.index_fallback)
            index = self.index_fallback
        cap = self._factory(index)
        if not cap.isOpened():
            cap.release()
            raise CameraError(f"could not open camera at index {index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Each reader owns its capture and stop event, so a reader that is still blocked in read()
        # after close() can neither be released under nor touch a later reader's capture.
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, args=(cap, self._stop),
                                        name="camera-reader", daemon=True)
        self._thread.start()

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
