from types import SimpleNamespace

import numpy as np
import pytest

from perception.features import (FeatureExtractor, classify_motion, estimate_distance, facing_score,
                                  head_point)
from perception.geometry import focal_px
from perception.types import Detection

F = focal_px(1280, 60.0)      # ~1108.5 px


def frontal(cx=640, top=200, shoulder_px=110.85):
    kp = {
        "nose": (cx, top + 40, 0.9),
        "left_eye": (cx - 10, top + 30, 0.9), "right_eye": (cx + 10, top + 30, 0.9),
        "left_ear": (cx - 22, top + 35, 0.9), "right_ear": (cx + 22, top + 35, 0.9),
        "left_shoulder": (cx - shoulder_px / 2, top + 120, 0.9),
        "right_shoulder": (cx + shoulder_px / 2, top + 120, 0.9),
    }
    return Detection(box=(cx - 80, top, cx + 80, top + 470), score=0.9, keypoints=kp)


def profile(cx=640, top=200):
    kp = {
        "nose": (cx - 25, top + 40, 0.9),
        "left_eye": (cx - 18, top + 30, 0.9), "right_eye": (cx - 5, top + 30, 0.2),
        "left_ear": (cx + 5, top + 35, 0.9), "right_ear": (cx + 10, top + 35, 0.1),
    }
    return Detection(box=(cx - 60, top, cx + 60, top + 470), score=0.8, keypoints=kp)


def from_behind(box=(600, 0, 700, 719)):
    kp = {n: (650, 100, 0.1) for n in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")}
    return Detection(box=box, score=0.7, keypoints=kp)


def track(det, id_=1, visible=True, t=0.0, box=None):
    b = np.array(det.box, dtype=float)
    return SimpleNamespace(id=id_, detection=det, box=b if box is None else np.array(box, float),
                           last_box=b, visible=visible, first_seen=0.0, last_seen=t)


def test_head_point_is_mean_of_confident_head_keypoints():
    x, y = head_point(frontal(), 0.5)
    assert x == pytest.approx(640)
    assert 230 <= y <= 240


def test_person_from_behind_uses_box_fallbacks(pcfg):
    d = from_behind()
    assert head_point(d, 0.5) == pytest.approx((650, 0 + 0.1 * 719))
    assert facing_score(d, 0.5) == 0.0
    assert estimate_distance(d, (1280, 720), F, 0.0, pcfg) == (None, None)   # box touches both edges


def test_facing_scores_frontal_high_profile_low():
    assert facing_score(frontal(), 0.5) == pytest.approx(1.0)
    assert facing_score(profile(), 0.5) < 0.3


def test_distance_from_shoulders_when_facing(pcfg):
    d, src = estimate_distance(frontal(shoulder_px=F * 0.40 / 4.0), (1280, 720), F, 1.0, pcfg)
    assert src == "shoulders" and d == pytest.approx(4.0)


def test_side_on_skips_shoulders_and_uses_body(pcfg):
    det = frontal()                                       # box 470 px tall, inside the frame
    d, src = estimate_distance(det, (1280, 720), F, 0.2, pcfg)
    assert src == "body" and d == pytest.approx(1.70 * F / 470)


def test_head_spread_is_the_last_resort(pcfg):
    det = Detection(box=(500, 0, 780, 719), score=0.9, keypoints={
        "left_ear": (620, 100, 0.9), "right_ear": (660, 100, 0.9)})
    d, src = estimate_distance(det, (1280, 720), F, 0.0, pcfg)
    assert src == "head" and d == pytest.approx(0.15 * F / 40)


def hist(dist_rate=0.0, yaw_rate=0.0, seconds=1.0, n=31):
    ts = np.linspace(0, seconds, n)
    return [(t, 3.0 + dist_rate * t, 5.0 + yaw_rate * t) for t in ts]


@pytest.mark.parametrize("h,expected", [
    (hist(dist_rate=-0.6), "approaching"),
    (hist(dist_rate=+0.6), "leaving"),
    (hist(yaw_rate=30.0), "passing"),
    (hist(), "still"),
    (hist(seconds=0.3, n=10), "unknown"),
])
def test_motion_classes(pcfg, h, expected):
    assert classify_motion(h, pcfg) == expected


def test_angles_use_actual_frame_size_not_config(pcfg):
    fx = FeatureExtractor(pcfg)                           # config says 1280x720
    det = frontal(cx=320, top=100)
    (p,) = fx.people([track(det)], 0.0, (640, 480))
    assert p.head_deg[0] == pytest.approx(0.0, abs=0.1)   # centred in a 640-wide frame


def test_centred_person_while_neck_turns_is_passing(pcfg):
    fx = FeatureExtractor(pcfg)
    for i in range(31):
        t = i / 30
        (p,) = fx.people([track(frontal(), t=t)], t, (1280, 720), pose=(30.0 * t, 0.0))
    assert p.world_yaw_deg == pytest.approx(30.0, abs=0.5)
    assert p.motion == "passing"


def test_centred_person_without_neck_pose_is_still(pcfg):
    fx = FeatureExtractor(pcfg)
    for i in range(31):
        t = i / 30
        (p,) = fx.people([track(frontal(), t=t)], t, (1280, 720))
    assert p.world_yaw_deg is None and p.motion == "still"


def test_distance_is_smoothed(pcfg):
    fx = FeatureExtractor(pcfg)
    near = frontal(shoulder_px=F * 0.40 / 2.0)            # 2 m
    far = frontal(shoulder_px=F * 0.40 / 4.0)             # 4 m
    fx.people([track(near)], 0.0, (1280, 720))
    (p,) = fx.people([track(far, t=0.1)], 0.1, (1280, 720))
    assert p.distance_m == pytest.approx(0.3 * 4.0 + 0.7 * 2.0)
    assert p.distance_source == "shoulders"


def test_coasting_person_head_follows_predicted_box(pcfg):
    fx = FeatureExtractor(pcfg)
    det = frontal()
    shifted = np.array(det.box) + np.array([50, 0, 50, 0])
    (p,) = fx.people([track(det, visible=False, box=shifted)], 0.0, (1280, 720))
    assert p.head_px[0] == pytest.approx(head_point(det, 0.5)[0] + 50)
    assert p.visible is False


def test_state_is_dropped_for_tracks_that_disappear(pcfg):
    fx = FeatureExtractor(pcfg)
    fx.people([track(frontal(), id_=7)], 0.0, (1280, 720))
    fx.people([], 0.1, (1280, 720))
    assert fx._state == {}


def test_rotated_camera_uses_sensor_width_for_focal_length(make_pcfg):
    cfg = make_pcfg({"rotate": 90})      # 1280x720 sensor, image delivered as 720 wide x 1280 tall
    fx = FeatureExtractor(cfg)
    det = Detection(box=(0, 500, 60, 900), score=0.9, keypoints={"nose": (0, 600, 0.9)})
    (p,) = fx.people([track(det)], 0.0, (720, 1280))
    assert p.head_deg[0] == pytest.approx(np.degrees(np.arctan(360 / F)), abs=0.05)   # ~18.0, not 30


def test_pose_dropout_does_not_fake_passing(pcfg):
    # Static person, neck held at 40 deg, pose provider stops answering part-way.
    fx = FeatureExtractor(pcfg)
    for i in range(31):
        t = i / 30
        pose = (40.0, 0.0) if t < 0.67 else None
        (p,) = fx.people([track(frontal(), t=t)], t, (1280, 720), pose=pose)
    assert p.motion in ("still", "unknown")
