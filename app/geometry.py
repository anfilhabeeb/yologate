"""Pure geometry helpers for line-crossing detection.

Lines are stored in *normalised* coordinates (0..1 relative to frame width /
height) so they are independent of stream resolution.
"""
from __future__ import annotations

from typing import Optional, Tuple

Point = Tuple[float, float]


def side(a: Point, b: Point, p: Point) -> float:
    """Signed area: >0 if p is left of a->b, <0 if right, 0 if on the line."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """True if segment p1-p2 properly intersects (or touches) segment q1-q2."""
    d1 = side(q1, q2, p1)
    d2 = side(q1, q2, p2)
    d3 = side(p1, p2, q1)
    d4 = side(p1, p2, q2)
    if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
        return True

    def on_seg(a: Point, b: Point, c: Point) -> bool:
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])

    if d1 == 0 and on_seg(q1, q2, p1):
        return True
    if d2 == 0 and on_seg(q1, q2, p2):
        return True
    if d3 == 0 and on_seg(p1, p2, q1):
        return True
    if d4 == 0 and on_seg(p1, p2, q2):
        return True
    return False


def crossing_direction(a: Point, b: Point, prev: Point, curr: Point) -> Optional[str]:
    """Return "in"/"out" if the movement prev->curr crosses segment a-b, else None.

    Image coordinates have y growing downward. "in" means moving from the
    negative to the positive side of a->b: for a line drawn left-to-right that
    is a downward movement, for a line drawn top-to-bottom it is right-to-left.
    "out" is the opposite direction.
    """
    s_prev = side(a, b, prev)
    s_curr = side(a, b, curr)
    if s_prev == s_curr or (s_prev == 0 and s_curr == 0):
        return None
    # Must change side (touching the line from one side counts once it leaves).
    if (s_prev > 0 and s_curr > 0) or (s_prev < 0 and s_curr < 0):
        return None
    if s_prev == 0:
        return None  # started on the line; the previous step already counted it
    if not segments_intersect(prev, curr, a, b):
        return None
    return "in" if s_prev < 0 else "out"
