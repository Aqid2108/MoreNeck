import pytest

from behavior.attention import Attention, closeness, score
from tests.behavior_helpers import person


def test_closeness_ramp(preset):
    assert closeness(1.0, preset) == 1.0
    assert closeness(4.0, preset) == 0.0
    assert closeness(2.5, preset) == pytest.approx(0.5)
    assert closeness(0.4, preset) == 1.0


def test_unknown_distance_scores_zero_closeness(preset):
    assert closeness(None, preset) == 0.0
    assert score(person(distance=None, facing_score=0.0), preset) == 0.0


def test_score_components(preset):
    base = score(person(distance=2.5, facing_score=0.8), preset)
    assert base == pytest.approx(0.8 + 0.6 * 0.5)
    assert score(person(distance=2.5, facing_score=0.8, motion="approaching"), preset) == pytest.approx(base + 0.5)
    assert score(person(distance=2.5, facing_score=0.8), preset, is_target=True) == pytest.approx(base + 0.6)
    assert score(person(distance=2.5, facing_score=0.8, visible=False), preset) == pytest.approx(base * 0.5)


def test_picks_the_highest_scorer_and_ignores_low_confidence(preset):
    att = Attention(preset)
    people = [person(1, facing_score=0.2), person(2, facing_score=0.9),
              person(3, facing_score=1.0, confidence=0.3)]
    assert att.update(people, 0.0) == 2


def test_stickiness_prevents_small_advantages_from_switching(preset):
    att = Attention(preset)
    att.update([person(1, facing_score=0.8)], 0.0)
    for i in range(100):
        t = i * 0.05
        assert att.update([person(1, facing_score=0.8), person(2, facing_score=1.0)], t) == 1


def test_switch_needs_a_sustained_lead(preset):
    att = Attention(preset)
    att.update([person(1, facing_score=0.2)], 0.0)
    rivals = [person(1, facing_score=0.2), person(2, facing_score=1.0, motion="approaching")]
    assert att.update(rivals, 1.0) == 1
    assert att.update(rivals, 2.4) == 1                 # 1.4 s < switch_hold_s (1.5)
    assert att.update(rivals, 2.6) == 2


def test_an_interrupted_lead_restarts_the_timer(preset):
    att = Attention(preset)
    att.update([person(1, facing_score=0.2)], 0.0)
    strong = [person(1, facing_score=0.2), person(2, facing_score=1.0, motion="approaching")]
    weak = [person(1, facing_score=0.2), person(2, facing_score=0.1)]
    att.update(strong, 1.0)
    att.update(weak, 2.0)
    att.update(strong, 2.1)
    assert att.update(strong, 3.0) == 1                 # only 0.9 s since the lead restarted
    assert att.update(strong, 3.7) == 2


def test_target_disappearing_clears_it(preset):
    att = Attention(preset)
    att.update([person(1)], 0.0)
    assert att.update([], 0.1) is None


def test_newcomer_glance_and_cooldown(preset):
    att = Attention(preset)
    att.update([person(1)], 0.0)
    newcomer = person(2, first_seen=10.0)
    assert att.glance_candidate([person(1), newcomer], 10.2) == 2
    other = person(3, first_seen=11.0)
    assert att.glance_candidate([person(1), other], 11.1) is None      # cooldown (3 s)
    later = person(4, first_seen=13.5)
    assert att.glance_candidate([person(1), later], 13.6) == 4


def test_old_arrivals_and_the_target_are_not_newcomers(preset):
    att = Attention(preset)
    att.update([person(1, first_seen=10.0)], 10.0)
    assert att.glance_candidate([person(1, first_seen=10.0)], 10.1) is None
    assert att.glance_candidate([person(1), person(2, first_seen=5.0)], 10.1) is None


def test_movement_glance_only_when_enabled(make_preset):
    passing = [person(1), person(2, motion="passing")]
    calm = Attention(make_preset({"glance_movement": False}))
    calm.update([person(1)], 0.0)
    assert calm.glance_candidate(passing, 5.0) is None
    curious = Attention(make_preset({"glance_movement": True}))
    curious.update([person(1)], 0.0)
    assert curious.glance_candidate(passing, 5.0) == 2


def test_newcomer_glance_disabled(make_preset):
    att = Attention(make_preset({"glance_newcomer": False}))
    att.update([person(1)], 0.0)
    assert att.glance_candidate([person(1), person(2, first_seen=10.0)], 10.1) is None


def test_a_one_frame_confidence_dip_does_not_hand_attention_away(preset):
    # Real detector scores for a turned or partly hidden person hover around 0.4-0.6.
    att = Attention(preset)
    att.update([person(1, facing_score=0.9), person(2, facing_score=0.5)], 0.0)
    assert att.target_id == 1
    att.update([person(1, facing_score=0.9, confidence=0.45), person(2, facing_score=0.5)], 0.05)
    for i in range(2, 40):
        assert att.update([person(1, facing_score=0.9), person(2, facing_score=0.5)], i * 0.05) == 1


def test_low_confidence_detections_cannot_become_the_target(preset):
    att = Attention(preset)                      # min_confidence 0.6
    assert att.update([person(1, confidence=0.55, facing_score=1.0)], 0.0) is None
    assert att.update([person(1, confidence=0.65, facing_score=1.0)], 0.1) == 1
