import math
import random

import pytest

from behavior.kinematics import aim_for, camera_to_world, world_to_camera, world_vector


def test_facing_forward_the_angles_just_add():
    assert aim_for(0.0, 0.0, 0.0, 20.0, 0.0) == pytest.approx((20.0, 0.0))
    assert aim_for(0.0, 0.0, 0.0, 0.0, 15.0) == pytest.approx((0.0, 15.0))
    assert aim_for(10.0, 5.0, 0.0, 0.0, 0.0) == pytest.approx((10.0, 5.0))    # already centred


def test_camera_and_world_round_trip():
    rng = random.Random(3)
    for _ in range(200):
        pose = (rng.uniform(-60, 60), rng.uniform(0, 40), rng.uniform(0, 10))
        yaw, pitch = rng.uniform(-30, 30), rng.uniform(-17, 17)
        v = camera_to_world(*pose, yaw, pitch)
        assert world_to_camera(*pose, v) == pytest.approx((yaw, pitch), abs=1e-6)


def test_aiming_centres_the_target_for_any_pose():
    # Turn sits above nod: turned far, nodding mostly rolls the view, so naive adding creeps.
    rng = random.Random(4)
    for _ in range(200):
        pose = (rng.uniform(-60, 60), rng.uniform(0, 40), rng.uniform(0, 10))
        yaw, pitch = rng.uniform(-25, 25), rng.uniform(-15, 15)
        v = camera_to_world(*pose, yaw, pitch)
        turn, nod = aim_for(*pose, yaw, pitch)
        assert world_to_camera(turn, nod, pose[2], v) == pytest.approx((0.0, 0.0), abs=1e-6)


def test_turned_far_the_needed_nod_is_more_than_the_elevation():
    v = world_vector(58.7, 10.0)                     # 58.7 deg to the left, 10 deg up
    turn, nod = aim_for(58.7, 0.0, 0.0, *world_to_camera(58.7, 0.0, 0.0, v))
    az, el = math.radians(58.7), math.radians(10.0)
    # turn happens in the tilted plane, so it is a little less than the azimuth ...
    assert turn == pytest.approx(math.degrees(math.asin(math.cos(el) * math.sin(az))))
    # ... and the tilt needed is well above the 10 deg elevation
    assert nod == pytest.approx(math.degrees(math.atan2(math.sin(el), math.cos(el) * math.cos(az))))
    assert nod > 18.0
