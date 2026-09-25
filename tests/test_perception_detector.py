from pathlib import Path

import numpy as np
import pytest

from perception.detector import COCO_INDEX, detections_from_arrays

REPO = Path(__file__).resolve().parent.parent
WEIGHTS = REPO / "data" / "models" / "yolo11n-pose.pt"


def test_coco_indices_for_used_keypoints():
    assert COCO_INDEX == {"nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
                          "left_shoulder": 5, "right_shoulder": 6}


def test_detections_from_arrays_maps_keypoints_and_filters_scores():
    boxes = np.array([[10, 20, 110, 320], [0, 0, 5, 5]], dtype=float)
    scores = np.array([0.9, 0.2])
    kps = np.zeros((2, 17, 3))
    kps[0, 0] = (60, 40, 0.95)       # nose
    kps[0, 6] = (90, 120, 0.8)       # right shoulder
    dets = detections_from_arrays(boxes, scores, kps, score_min=0.4)
    assert len(dets) == 1
    d = dets[0]
    assert d.box == (10.0, 20.0, 110.0, 320.0) and d.score == pytest.approx(0.9)
    assert d.keypoints["nose"] == pytest.approx((60, 40, 0.95))
    assert d.keypoints["right_shoulder"] == pytest.approx((90, 120, 0.8))
    assert set(d.keypoints) == set(COCO_INDEX)


def test_missing_weights_give_a_clear_error(tmp_path):
    pytest.importorskip("ultralytics")
    from perception.detector import YoloPoseDetector

    with pytest.raises(FileNotFoundError, match="yolo11n-pose.pt"):
        YoloPoseDetector(tmp_path / "yolo11n-pose.pt")


@pytest.mark.model
@pytest.mark.parametrize("image,min_people", [("bus.jpg", 3), ("zidane.jpg", 2)])
def test_finds_people_in_sample_images(image, min_people):
    import cv2
    from ultralytics.utils import ASSETS

    from perception.detector import YoloPoseDetector

    det = YoloPoseDetector(WEIGHTS)
    dets = det.detect(cv2.imread(str(ASSETS / image)))
    assert len(dets) >= min_people
    for d in dets:
        x1, y1, x2, y2 = d.box
        nx, ny, nc = d.keypoints["nose"]
        if nc >= 0.5:
            assert x1 - 5 <= nx <= x2 + 5 and y1 - 5 <= ny <= y2 + 5
