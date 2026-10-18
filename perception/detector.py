"""Person detection. Only YoloPoseDetector touches Ultralytics/PyTorch.

Ultralytics is AGPL-3.0: fine for personal/research use; commercial use needs a licence or a
different backend behind the same detect() interface (MediaPipe is the documented alternative).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from perception.types import KEYPOINT_NAMES, Detection

# Keep Ultralytics offline in this process: no usage analytics, no automatic downloads (model
# weights are fetched only with the user's OK). Must be set before ultralytics is first imported.
os.environ.setdefault("YOLO_OFFLINE", "1")

COCO_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}   # first 7 COCO keypoints


def cpu_threads() -> int:
    """Torch CPU threads for inference. Torch's default (every core) is slower and jitterier on
    hybrid laptop CPUs, whose efficiency cores hold the rest back: on a Core Ultra 7 155H, 6 threads
    gave 64 ms median / 81 ms p95 vs 78-134 / 135 ms with all 16. Override: MORENECK_CPU_THREADS."""
    env = os.environ.get("MORENECK_CPU_THREADS")
    if env:
        return max(1, int(env))
    return max(1, min(6, (os.cpu_count() or 2) // 2))


def detections_from_arrays(boxes, scores, keypoints, score_min: float) -> list[Detection]:
    out = []
    for box, score, kps in zip(np.asarray(boxes), np.asarray(scores), np.asarray(keypoints)):
        if score < score_min:
            continue
        kp = {name: (float(kps[i][0]), float(kps[i][1]), float(kps[i][2]))
              for name, i in COCO_INDEX.items()}
        out.append(Detection(box=tuple(float(v) for v in box), score=float(score), keypoints=kp))
    return out


class YoloPoseDetector:
    def __init__(self, model_path, imgsz: int = 640, score_min: float = 0.4, device: str | None = None):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"pose model weights not found: {path} "
                                    "(download yolo11n-pose.pt into data/models/)")
        import torch
        from ultralytics import YOLO

        self.model = YOLO(str(path))
        self.imgsz = imgsz
        self.score_min = score_min
        # Apple GPU on Macs, otherwise CPU (deliberately never CUDA: the same setup on every laptop).
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        if self.device == "cpu":
            torch.set_num_threads(cpu_threads())

    @classmethod
    def from_config(cls, cfg, repo_root) -> "YoloPoseDetector":
        return cls(Path(repo_root) / cfg.model, cfg.imgsz, cfg.person_score_min)

    def detect(self, image) -> list[Detection]:
        r = self.model.predict(image, imgsz=self.imgsz, conf=self.score_min, device=self.device,
                               classes=[0], verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        return detections_from_arrays(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                      r.keypoints.data.cpu().numpy(), self.score_min)
