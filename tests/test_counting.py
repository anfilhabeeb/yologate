import numpy as np

from app.agent import CameraWorker
from app.config import Settings
from app.detector import Detection


class ScriptedDetector:
    """Returns pre-scripted detections, one list per call."""

    def __init__(self, script):
        self.script = list(script)

    def detect(self, frame):
        return self.script.pop(0) if self.script else []


def make_worker(script, lines):
    cam = {"id": 1, "name": "t", "url": "none", "lines": lines}
    events = []
    w = CameraWorker(cam, ScriptedDetector(script), Settings(), on_event=events.append)
    return w, events


FRAME = np.zeros((100, 200, 3), np.uint8)
VLINE = {"id": "L1", "name": "mid", "x1": 0.5, "y1": 0.0, "x2": 0.5, "y2": 1.0}


def car(x, label="car"):
    return Detection(x, 40, x + 20, 60, 0.9, label)


def test_vehicle_crossing_counted_once():
    # car moves left -> right across x=100 (centre), then lingers around the line
    xs = [40, 60, 80, 100, 110, 95, 120, 140]
    w, events = make_worker([[car(x)] for x in xs], [VLINE])
    for _ in xs:
        w.process_frame(FRAME)
    assert len(events) == 1
    assert events[0].direction == "out"  # left->right over a top->bottom line
    assert w.get_counts()["L1"]["out"] == {"car": 1}


def test_two_directions_and_classes():
    script = []
    for i in range(8):
        script.append([car(10 + i * 20, "car"), car(170 - i * 20, "truck")])
    # objects pass each other; keep them on different rows to avoid confusion
    for frame in script:
        frame[1].y1, frame[1].y2 = 5, 25
    w, events = make_worker(script, [VLINE])
    for _ in script:
        w.process_frame(FRAME)
    c = w.get_counts()["L1"]
    assert c["out"] == {"car": 1}
    assert c["in"] == {"truck": 1}
    assert w.status()["total"] == 2


def test_multiple_lines_and_reset():
    lines = [VLINE, {"id": "L2", "name": "q", "x1": 0.25, "y1": 0, "x2": 0.25, "y2": 1}]
    xs = list(range(0, 180, 15))
    w, events = make_worker([[car(x)] for x in xs], lines)
    for _ in xs:
        w.process_frame(FRAME)
    assert {e.line_id for e in events} == {"L1", "L2"}
    w.reset_counts()
    assert w.status()["total"] == 0


def test_annotated_jpeg_produced():
    w, _ = make_worker([[car(50)]], [VLINE])
    w.process_frame(FRAME)
    assert w.latest_jpeg().startswith(b"\xff\xd8")
    assert w.latest_jpeg(raw=True).startswith(b"\xff\xd8")
