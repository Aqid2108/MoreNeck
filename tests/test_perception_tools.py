import subprocess
import sys

import numpy as np
import pytest

from perception.types import Person, Snapshot


def person(id_=1, box=(100, 100, 200, 400)):
    return Person(id=id_, box=box, head_px=(150, 130), head_deg=(5.0, 2.0), world_yaw_deg=None,
                  distance_m=2.5, distance_source="body", facing_score=0.9, facing=True,
                  motion="still", confidence=0.9, visible=True, first_seen=0.0, last_seen=1.0)


def test_draw_overlay_draws_people_and_keeps_shape():
    from tools.perception_view import draw_overlay

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    snap = Snapshot(timestamp=1.0, frame_size=(640, 480), people=(person(),), source_ok=True)
    out = draw_overlay(img.copy(), snap, fps=29.7, latency_s=0.041)
    assert out.shape == img.shape and out.any()


def test_draw_overlay_shows_failures():
    from tools.perception_view import draw_overlay

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    snap = Snapshot(timestamp=1.0, frame_size=(640, 480), people=(), source_ok=False, error="no frames")
    assert draw_overlay(img.copy(), snap, fps=0.0, latency_s=0.0).any()


def test_parse_mot_gt_keeps_pedestrians_only(tmp_path):
    from tools.eval_perception import parse_mot_gt

    gt = tmp_path / "gt.txt"
    gt.write_text("1,1,10,20,30,40,1,1,1.0\n"
                  "1,2,50,60,30,40,0,1,1.0\n"       # ignored (conf 0)
                  "2,1,12,20,30,40,1,1,0.9\n"
                  "2,3,90,90,10,10,1,7,1.0\n")      # ignored (class 7, not pedestrian)
    parsed = parse_mot_gt(gt)
    assert parsed == {1: [(1, (10.0, 20.0, 40.0, 60.0))], 2: [(1, (12.0, 20.0, 42.0, 60.0))]}


def test_mot_summary_counts_id_switches():
    from tools.eval_perception import mot_summary

    gt = {f: [(1, (0.0, 0.0, 10.0, 10.0))] for f in range(1, 11)}
    perfect = {f: [(7, (0.0, 0.0, 10.0, 10.0))] for f in range(1, 11)}
    swapped = {f: [(7 if f <= 5 else 8, (0.0, 0.0, 10.0, 10.0))] for f in range(1, 11)}
    assert mot_summary(gt, perfect)["num_switches"] == 0
    assert mot_summary(gt, perfect)["idf1"] == pytest.approx(1.0)
    assert mot_summary(gt, swapped)["num_switches"] == 1


@pytest.mark.parametrize("script", ["tools/perception_view.py", "tools/eval_perception.py"])
def test_tool_scripts_show_help(script):
    result = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
