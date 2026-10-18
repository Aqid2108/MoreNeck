"""Live perception debug view. Nothing is saved.

Usage:
  python tools/perception_view.py              # head camera
  python tools/perception_view.py --video F    # a video file, played in real time
Press q or Esc to quit.
"""
import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # repo root, for `perception`/`tools` when run as a script

import perception  # noqa: E402,F401  (before cv2: sets Windows camera options)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

GREEN, GREY, RED, WHITE = (0, 200, 0), (150, 150, 150), (0, 0, 255), (255, 255, 255)


def draw_overlay(image: np.ndarray, snapshot, fps: float, latency_s: float) -> np.ndarray:
    for p in snapshot.people:
        colour = GREEN if p.visible else GREY
        x1, y1, x2, y2 = (int(v) for v in p.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        cv2.circle(image, (int(p.head_px[0]), int(p.head_px[1])), 5, colour, -1)
        dist = f"{p.distance_m:.1f}m" if p.distance_m is not None else "?m"
        label = f"#{p.id} {dist} {'facing' if p.facing else 'away'} {p.motion}"
        cv2.putText(image, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    status = f"{fps:4.1f} fps  {latency_s * 1000:4.0f} ms  {len(snapshot.people)} people"
    cv2.putText(image, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    if not snapshot.source_ok:
        cv2.putText(image, f"SOURCE FAILED: {snapshot.error}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, RED, 2)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", help="play a video file instead of the camera")
    args = parser.parse_args()

    from perception.config import load_config
    from perception.detector import YoloPoseDetector
    from perception.pipeline import Perception
    from perception.sources import CameraSource, VideoFileSource

    cfg = load_config()
    source = VideoFileSource(args.video, realtime=True) if args.video else CameraSource.from_config(cfg)
    latest = {"snap": None, "img": None, "times": []}

    def on_snapshot(snap, img):
        latest["snap"], latest["img"] = snap, img
        latest["times"] = (latest["times"] + [time.monotonic()])[-30:]

    with Perception(source, YoloPoseDetector.from_config(cfg, REPO), cfg, on_snapshot=on_snapshot):
        while True:
            snap, img, times = latest["snap"], latest["img"], latest["times"]
            if snap is not None:
                frame = img.copy() if img is not None else np.zeros((480, 640, 3), np.uint8)
                fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
                latency = time.monotonic() - snap.timestamp if not args.video else 0.0
                cv2.imshow("perception", draw_overlay(frame, snap, fps, latency))
            if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
