"""Evaluate perception on a video or image sequence, optionally against MOT ground truth.

Usage:
  python tools/eval_perception.py VIDEO [--gt gt.txt] [--fps 30]
VIDEO can be a file or an image-sequence pattern such as data/MOT17/.../img1/%06d.jpg
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # repo root, for `perception`/`tools` when run as a script


def parse_mot_gt(path):
    """MOT gt.txt -> {frame: [(id, (x1, y1, x2, y2))]}, pedestrians (class 1) with conf 1 only."""
    out = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        f = line.split(",")
        frame, tid, x, y, w, h, conf = int(f[0]), int(f[1]), *map(float, f[2:7])
        cls = int(float(f[7])) if len(f) > 7 else 1
        if conf == 1 and cls == 1:
            out[frame].append((tid, (x, y, x + w, y + h)))
    return dict(out)


def iou_distances(gt_boxes, hyp_boxes, min_iou: float = 0.5):
    """1 - IoU per (gt, hyp) pair, NaN where IoU < min_iou (motmetrics' convention).

    Replaces motmetrics.distances.iou_matrix, which uses np.asfarray (removed in NumPy 2).
    """
    import numpy as np

    from perception.tracker import iou

    d = np.full((len(gt_boxes), len(hyp_boxes)), np.nan)
    for i, g in enumerate(gt_boxes):
        for j, h in enumerate(hyp_boxes):
            v = iou(g, h)
            if v >= min_iou:
                d[i, j] = 1.0 - v
    return d


def mot_summary(gt_by_frame, hyp_by_frame) -> dict:
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=False)
    for frame in sorted(set(gt_by_frame) | set(hyp_by_frame)):
        gt = gt_by_frame.get(frame, [])
        hyp = hyp_by_frame.get(frame, [])
        dist = iou_distances([b for _, b in gt], [b for _, b in hyp])
        acc.update([i for i, _ in gt], [i for i, _ in hyp], dist, frameid=frame)
    s = mm.metrics.create().compute(acc, metrics=["num_switches", "mota", "idf1"], name="run")
    return {k: float(s[k].iloc[0]) for k in ("num_switches", "mota", "idf1")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--gt", help="MOT-format ground truth (gt.txt)")
    args = parser.parse_args()

    from perception.config import load_config
    from perception.detector import YoloPoseDetector
    from perception.pipeline import Perception
    from perception.sources import VideoFileSource

    cfg = load_config()
    source = VideoFileSource(args.video)
    p = Perception(source, YoloPoseDetector.from_config(cfg, REPO), cfg)
    source.open()
    hyp, step_times, tracks_seen, frame_no = {}, [], set(), 0
    start = time.monotonic()
    while True:
        t0 = time.monotonic()
        snap = p.step()
        if snap is None:
            if source.eof:
                break
            continue
        frame_no += 1
        step_times.append(time.monotonic() - t0)
        hyp[frame_no] = [(q.id, q.box) for q in snap.people if q.visible]
        tracks_seen.update(q.id for q in snap.people)
    source.close()
    elapsed = time.monotonic() - start
    step_times.sort()
    print(f"frames {frame_no}  fps {frame_no / elapsed:.1f}  "
          f"latency mean {1000 * sum(step_times) / len(step_times):.0f} ms  "
          f"p95 {1000 * step_times[int(0.95 * (len(step_times) - 1))]:.0f} ms  tracks {len(tracks_seen)}")
    if args.gt:
        s = mot_summary(parse_mot_gt(args.gt), hyp)
        print(f"ID switches {s['num_switches']:.0f}  MOTA {s['mota']:.3f}  IDF1 {s['idf1']:.3f}")


if __name__ == "__main__":
    main()
