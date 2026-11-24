from app.geometry import crossing_direction, segments_intersect

H = ((0.0, 0.5), (1.0, 0.5))  # horizontal line drawn left -> right


def test_downward_crossing_is_in():
    assert crossing_direction(*H, (0.5, 0.4), (0.5, 0.6)) == "in"


def test_upward_crossing_is_out():
    assert crossing_direction(*H, (0.5, 0.6), (0.5, 0.4)) == "out"


def test_no_crossing_same_side():
    assert crossing_direction(*H, (0.5, 0.1), (0.6, 0.3)) is None


def test_crossing_outside_segment_ignored():
    short = ((0.0, 0.5), (0.3, 0.5))
    assert crossing_direction(*short, (0.8, 0.4), (0.8, 0.6)) is None


def test_vertical_line_right_to_left_is_in():
    v = ((0.5, 0.0), (0.5, 1.0))
    assert crossing_direction(*v, (0.6, 0.5), (0.4, 0.5)) == "in"
    assert crossing_direction(*v, (0.4, 0.5), (0.6, 0.5)) == "out"


def test_touching_then_leaving_counts_once():
    assert crossing_direction(*H, (0.5, 0.4), (0.5, 0.5)) == "in"
    assert crossing_direction(*H, (0.5, 0.5), (0.5, 0.6)) is None


def test_segments_intersect():
    assert segments_intersect((0, 0), (1, 1), (0, 1), (1, 0))
    assert not segments_intersect((0, 0), (1, 0), (0, 1), (1, 1))
