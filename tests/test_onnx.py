"""YOLO via the bundled ONNX model + OpenCV DNN (no PyTorch needed)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.agent import CameraWorker
from app.config import DEFAULT_ONNX_MODEL, Settings
from app.detector import OnnxDetector, build_detector
from fake_camera import Scene

SPRITE = Path(__file__).resolve().parents[1] / "tools" / "assets" / "bus.jpg"

pytestmark = pytest.mark.skipif(not DEFAULT_ONNX_MODEL.exists(), reason="ONNX model not present")


@pytest.fixture(scope="module")
def det():
    return OnnxDetector(Settings())


def test_detects_bus_with_correct_box(det):
    frame = Scene(sprite=cv2.imread(str(SPRITE)), vehicles=1, sprite_w=300).frame(60)
    dets = det.detect(frame)
    buses = [d for d in dets if d.label in ("bus", "truck")]
    assert len(buses) == 1, dets
    # the sprite is drawn at x = 60*12 - 300 = 420 .. 720, centred vertically at y = 324
    b = buses[0]
    assert abs(b.x1 - 420) < 25 and abs(b.x2 - 720) < 25
    assert abs((b.y1 + b.y2) / 2 - 324) < 25


def test_empty_frame_has_no_detections(det):
    assert det.detect(np.full((480, 640, 3), 90, np.uint8)) == []


def test_counts_bus_crossing(det):
    scene = Scene(sprite=cv2.imread(str(SPRITE)), vehicles=1, speed=25)
    cam = {"id": 1, "name": "onnx", "url": "x",
           "lines": [{"id": "L", "name": "L", "x1": 0.5, "y1": 0.1, "x2": 0.5, "y2": 0.95}]}
    events = []
    w = CameraWorker(cam, det, Settings(), on_event=events.append)
    for i in range(60):
        w.process_frame(scene.frame(i))
    assert len(events) == 1
    assert events[0].direction == "out" and events[0].label in ("bus", "truck")


def test_build_detector_onnx_mode():
    assert isinstance(build_detector(Settings(detector="onnx")), OnnxDetector)
