"""A small IoU + centroid multi-object tracker.

Each camera gets its own tracker instance, so one YOLO model can be shared by
all cameras while identities stay per-stream.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .detector import Detection


def iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / max(area_a + area_b - inter, 1e-9)


@dataclass
class Track:
    id: int
    det: Detection
    prev_center: Tuple[float, float]
    center: Tuple[float, float]
    missed: int = 0
    hits: int = 1
    velocity: Tuple[float, float] = (0.0, 0.0)
    counted_lines: set = field(default_factory=set)


class Tracker:
    def __init__(self, iou_threshold: float = 0.2, max_missed: int = 10, max_dist_ratio: float = 1.5):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        # Fallback centroid matching: allow movement up to this multiple of box size.
        self.max_dist_ratio = max_dist_ratio
        self.tracks: Dict[int, Track] = {}
        self._ids = itertools.count(1)

    def _score(self, t: Track, d: Detection) -> float:
        s = iou(t.det, d)
        if s >= self.iou_threshold:
            return 1.0 + s
        # Fast-moving objects at low FPS may not overlap: match on distance to
        # the position predicted from the track's velocity, within a gate that
        # grows only slightly while the track is unseen (a large gate would let
        # a vehicle leaving one edge "jump" to a new one entering elsewhere).
        size = max(t.det.x2 - t.det.x1, t.det.y2 - t.det.y1, 1.0)
        steps = t.missed + 1
        px = t.center[0] + t.velocity[0] * steps
        py = t.center[1] + t.velocity[1] * steps
        dx = px - d.center[0]
        dy = py - d.center[1]
        dist = (dx * dx + dy * dy) ** 0.5
        limit = size * self.max_dist_ratio * min(1 + 0.25 * t.missed, 2.0)
        if dist <= limit:
            return 1.0 - dist / limit  # in (0, 1]
        return 0.0

    def update(self, detections: List[Detection]) -> List[Track]:
        pairs = []
        for tid, t in self.tracks.items():
            for di, d in enumerate(detections):
                if d.label != t.det.label and iou(t.det, d) < 0.5:
                    continue  # avoid swapping between classes unless overlapping strongly
                s = self._score(t, d)
                if s > 0:
                    pairs.append((s, tid, di))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for _, tid, di in pairs:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid)
            used_d.add(di)
            t = self.tracks[tid]
            d = detections[di]
            steps = t.missed + 1
            vx = (d.center[0] - t.center[0]) / steps
            vy = (d.center[1] - t.center[1]) / steps
            if t.hits == 1:
                t.velocity = (vx, vy)
            else:
                t.velocity = (0.5 * t.velocity[0] + 0.5 * vx, 0.5 * t.velocity[1] + 0.5 * vy)
            t.prev_center = t.center
            t.center = d.center
            t.det = d
            t.missed = 0
            t.hits += 1

        for tid in list(self.tracks):
            if tid not in used_t:
                t = self.tracks[tid]
                t.missed += 1
                t.prev_center = t.center
                if t.missed > self.max_missed:
                    del self.tracks[tid]

        for di, d in enumerate(detections):
            if di not in used_d:
                tid = next(self._ids)
                self.tracks[tid] = Track(tid, d, d.center, d.center)

        return [t for t in self.tracks.values() if t.missed == 0]
