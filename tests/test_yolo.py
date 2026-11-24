"""End-to-end test with the real YOLO model on a synthetic video of a bus.

Skipped automatically if ultralytics is not installed or the model weights
cannot be loaded (e.g. offline CI without a cached model).
"""
import os
from pathlib import Path

import cv2
import pytest

ultralytics = pytest.importorskip("ultralytics")

from app.agent import CameraWorker, FrameSource  # noqa: E402
from app.config import Settings  # noqa: E402
from fake_camera import Scene  # noqa: E402


@pytest.fixture(scope="module")
def yolo_detector():
    from app.detector import YoloDetector

    try:
        return YoloDetector(Settings(model_path=os.environ.get("VC_MODEL", "yolo11n.pt"), imgsz=640))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"YOLO model unavailable: {exc}")


@pytest.fixture(scope="module")
def bus_sprite():
    img = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    return img[229:728, 3:796]


def test_yolo_detects_bus(yolo_detector, bus_sprite):
    dets = yolo_detector.detect(Scene(sprite=bus_sprite, vehicles=1).frame(30))
    assert any(d.label in ("bus", "truck") for d in dets)


def test_yolo_counts_bus_crossing_from_video_file(yolo_detector, bus_sprite, tmp_path):
    scene = Scene(sprite=bus_sprite, vehicles=1, speed=25)
    path = str(tmp_path / "traffic.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (scene.w, scene.h))
    for i in range(60):
        vw.write(scene.frame(i))
    vw.release()

    cam = {"id": 1, "name": "yolo", "url": path,
           "lines": [{"id": "L", "name": "L", "x1": 0.5, "y1": 0.1, "x2": 0.5, "y2": 0.95}]}
    events = []
    w = CameraWorker(cam, yolo_detector, Settings(), on_event=events.append)
    src = FrameSource(path)
    src.open()
    try:
        while True:
            try:
                frame = src.read()
            except Exception:  # end of file
                break
            w.process_frame(frame)
    finally:
        src.close()
    assert len(events) == 1, events
    assert events[0].direction == "out"
    assert events[0].label in ("bus", "truck")
