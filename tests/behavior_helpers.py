"""Shared builders for behavior tests."""
from perception.types import Person


def person(id_=1, yaw=0.0, pitch=0.0, distance=2.0, facing_score=0.8, motion="still",
           first_seen=-10.0, confidence=0.9, visible=True, t=0.0, head_from_keypoints=True):
    return Person(id=id_, box=(0.0, 0.0, 1.0, 1.0), head_px=(0.0, 0.0), head_deg=(yaw, pitch),
                  world_yaw_deg=None, distance_m=distance,
                  distance_source="body" if distance is not None else None,
                  facing_score=facing_score, facing=facing_score >= 0.6, motion=motion,
                  confidence=confidence, visible=visible, first_seen=first_seen, last_seen=t,
                  head_from_keypoints=head_from_keypoints)
