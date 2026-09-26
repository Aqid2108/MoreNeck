from perception.tracker import Tracker, iou
from perception.types import Detection

FPS = 30
DT = 1 / FPS


def det(x, y=100, w=100, h=300):
    return Detection(box=(x, y, x + w, y + h), score=0.9)


def ids(tracks):
    return sorted(t.id for t in tracks)


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == 50 / 150


def test_person_is_confirmed_after_three_frames():
    tr = Tracker()
    assert tr.update([det(100)], 0 * DT) == []
    assert tr.update([det(101)], 1 * DT) == []
    tracks = tr.update([det(102)], 2 * DT)
    assert len(tracks) == 1 and tracks[0].id == 1 and tracks[0].visible


def test_one_or_two_frame_flicker_is_never_reported():
    tr = Tracker()
    for i in range(2):
        assert tr.update([det(500)], i * DT) == []
    for i in range(2, 10):
        assert tr.update([], i * DT) == []


def test_two_people_crossing_keep_their_ids():
    tr = Tracker()
    first = None
    for i in range(80):
        a = det(100 + 10 * i, y=100)          # moving right
        b = det(900 - 10 * i, y=140)          # moving left, passes through a around i = 40
        tracks = tr.update([a, b], i * DT)
        if len(tracks) == 2 and first is None:
            first = {t.id: t.box[0] for t in tracks}
    assert first is not None
    right_mover = min(first, key=first.get)   # the one that started on the left
    final = {t.id: t.box[0] for t in tracks}
    assert final[right_mover] > 800           # still the same id, now on the right


def test_short_occlusion_keeps_id_and_coasts():
    tr = Tracker()
    for i in range(30):
        tracks = tr.update([det(100 + 5 * i)], i * DT)
    (before,) = tracks
    for i in range(30, 45):                   # 0.5 s hidden
        (coasting,) = tr.update([], i * DT)
        assert coasting.id == before.id and not coasting.visible
    (after,) = tr.update([det(100 + 5 * 45)], 45 * DT)
    assert after.id == before.id and after.visible


def test_long_occlusion_gets_a_new_id():
    tr = Tracker()
    for i in range(30):
        tracks = tr.update([det(100)], i * DT)
    old_id = tracks[0].id
    for i in range(30, 90):                   # 2 s hidden
        tracks = tr.update([], i * DT)
    assert tracks == []
    for i in range(90, 93):
        tracks = tr.update([det(100)], i * DT)
    assert len(tracks) == 1 and tracks[0].id != old_id


def test_ids_are_never_reused():
    tr = Tracker(max_coast_s=0.1)
    seen = []
    t = 0.0
    for _ in range(3):
        for _ in range(3):
            tracks = tr.update([det(100)], t)
            t += DT
        seen.append(tracks[0].id)
        for _ in range(10):
            tr.update([], t)
            t += DT
    assert seen == sorted(set(seen))


def test_head_rotation_shift_keeps_ids_through_fast_turns():
    def track_with(shift):
        tr = Tracker()
        for i in range(3):
            tracks = tr.update([det(300, w=60)], i * DT)
        first = tracks[0].id
        for i in range(3, 13):                       # the head turns: the person slides 70 px/frame
            x = 300 + 70 * (i - 2)
            tracks = tr.update([det(x, w=60)], i * DT, shift_px=(70.0, 0.0) if shift else (0.0, 0.0))
        return first, tracks

    first, tracks = track_with(shift=True)
    assert [t.id for t in tracks if t.visible] == [first]
    first, tracks = track_with(shift=False)          # without compensation the track is lost
    assert first not in [t.id for t in tracks if t.visible]
