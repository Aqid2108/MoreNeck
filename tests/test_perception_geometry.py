import math

import pytest

from perception.geometry import angles_to_pixel, focal_px, hfov_from_shift, pixel_to_angles


def test_focal_length_from_fov():
    assert focal_px(1280, 60.0) == pytest.approx(640 / math.tan(math.radians(30)))


def test_centre_is_straight_ahead():
    assert pixel_to_angles(640, 360, 1280, 720, 60.0) == pytest.approx((0.0, 0.0))


def test_image_left_is_positive_yaw_and_top_is_positive_pitch():
    yaw, pitch = pixel_to_angles(0, 0, 1280, 720, 60.0)
    assert yaw == pytest.approx(30.0)
    assert pitch > 0


def test_round_trip():
    x, y = angles_to_pixel(-12.5, 7.0, 1280, 720, 60.0)
    assert pixel_to_angles(x, y, 1280, 720, 60.0) == pytest.approx((-12.5, 7.0))


def test_hfov_recovered_from_a_known_turn():
    f = focal_px(1280, 62.0)
    shift = f * math.tan(math.radians(20.0))    # a centred feature after turning 20 degrees
    assert hfov_from_shift(1280, 20.0, shift) == pytest.approx(62.0)
