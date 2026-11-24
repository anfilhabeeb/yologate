from app.detector import Detection
from app.tracker import Tracker


def box(x, y, w=40, h=20, label="car"):
    return Detection(x, y, x + w, y + h, 0.9, label)


def test_keeps_identity_while_moving():
    tr = Tracker()
    ids = []
    for x in range(0, 200, 10):
        tracks = tr.update([box(x, 100)])
        assert len(tracks) == 1
        ids.append(tracks[0].id)
    assert len(set(ids)) == 1


def test_two_objects_get_two_ids():
    tr = Tracker()
    tracks = tr.update([box(0, 0), box(300, 300)])
    assert len({t.id for t in tracks}) == 2
    tracks = tr.update([box(8, 0), box(308, 300)])
    assert len({t.id for t in tracks}) == 2
    assert len(tr.tracks) == 2


def test_fast_object_without_overlap_matched_by_distance():
    tr = Tracker()
    a = tr.update([box(0, 0)])[0].id
    b = tr.update([box(50, 0)])[0].id  # no IoU overlap, but within distance limit
    assert a == b


def test_track_expires_after_missed_frames():
    tr = Tracker(max_missed=2)
    tr.update([box(0, 0)])
    for _ in range(3):
        tr.update([])
    assert tr.tracks == {}


def test_object_leaving_does_not_jump_to_new_object_far_away():
    tr = Tracker()
    first = None
    for x in range(300, 440, 20):  # moving right, then leaves the frame
        first = tr.update([box(x, 100, w=60)])[0].id
    tr.update([])
    tr.update([])
    new = tr.update([box(0, 100, w=60)])[0]  # a different vehicle enters on the left
    assert new.id != first
