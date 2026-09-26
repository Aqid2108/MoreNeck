"""Perception pipeline: source -> detector -> tracker -> features -> Snapshot.

Behavior reads latest() (or uses people_or_empty). No frames are ever written to disk.
"""
from __future__ import annotations

import logging
import threading
import time

from perception.features import FeatureExtractor
from perception.tracker import Tracker
from perception.types import Snapshot

log = logging.getLogger("perception.pipeline")


def people_or_empty(snapshot: Snapshot, now: float, stale_after_s: float) -> tuple:
    """The people in a snapshot, or () if it failed or is older than stale_after_s."""
    if not snapshot.source_ok or now - snapshot.timestamp > stale_after_s:
        return ()
    return snapshot.people


class Perception:
    def __init__(self, source, detector, cfg, pose_provider=None, on_snapshot=None,
                 clock=time.monotonic, sleep=time.sleep):
        self.source = source
        self.detector = detector
        self.cfg = cfg
        self.pose_provider = pose_provider
        self.on_snapshot = on_snapshot
        self.clock = clock
        self._sleep = sleep
        self.tracker = Tracker(cfg.track_iou_min, cfg.track_confirm_frames, cfg.max_coast_s)
        self.features = FeatureExtractor(cfg)
        self._lock = threading.Lock()
        self._latest = Snapshot(0.0, (0, 0), (), False, "not started")
        self._failing = False
        self._last_frame_at = 0.0
        self._last_reopen = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- lifecycle -------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    def start(self) -> None:
        self.source.open()
        self._last_frame_at = self._last_reopen = self.clock()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="perception", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.source.close()

    def latest(self) -> Snapshot:
        with self._lock:
            return self._latest

    # ---- loop ------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if self.step() is None:
                    self._sleep(0.005)
            except Exception:
                log.exception("perception step failed")
                self._sleep(0.05)

    def step(self) -> Snapshot | None:
        now = self.clock()
        frame = self.source.read()
        if frame is None:
            if now - self._last_frame_at > self.cfg.source_timeout_s:
                self._fail(now, "no frames from source")
                if now - self._last_reopen >= self.cfg.reopen_every_s:
                    self._reopen(now)
            return None
        self._last_frame_at = now
        height, width = frame.image.shape[:2]
        pose = None                      # sampled before inference, as close to capture as we can
        if self.pose_provider is not None:
            try:
                pose = self.pose_provider()
            except Exception as exc:
                log.warning("pose provider failed: %r", exc)
        try:
            detections = self.detector.detect(frame.image)
        except Exception as exc:
            self._fail(frame.time, f"detector error: {exc!r}")
            return self.latest()
        tracks = self.tracker.update(detections, frame.time)
        people = self.features.people(tracks, frame.time, (width, height), pose)
        if self._failing:
            log.info("perception recovered")
            self._failing = False
        snap = Snapshot(frame.time, (width, height), people, True, None,
                        tuple(pose) if pose is not None else None)
        self._publish(snap, frame.image)
        return snap

    def _fail(self, t: float, message: str) -> None:
        if not self._failing:
            log.error(message)
            self._failing = True
        self._publish(Snapshot(t, self.latest().frame_size, (), False, message), None)

    def _reopen(self, now: float) -> None:
        self._last_reopen = now
        try:
            self.source.close()
            self.source.open()
            log.info("source reopened")
        except Exception as exc:
            log.warning("source reopen failed: %r", exc)

    def _publish(self, snap: Snapshot, image) -> None:
        with self._lock:
            self._latest = snap
        if self.on_snapshot is not None:
            self.on_snapshot(snap, image)
