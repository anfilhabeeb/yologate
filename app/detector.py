"""Object detectors. `YoloDetector` wraps Ultralytics YOLO; `DummyDetector`
is a lightweight stand-in used by tests and for environments without a model."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol

import numpy as np

from .config import VEHICLE_CLASSES, Settings

log = logging.getLogger(__name__)


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> List[Detection]: ...


class YoloDetector:
    """Thread-safe YOLO vehicle detector (Ultralytics + PyTorch) shared by all camera workers."""

    def __init__(self, settings: Settings):
        from ultralytics import YOLO  # imported lazily: heavy dependency

        self.name = f"yolo ({settings.model_path})"
        self.settings = settings
        self.model = YOLO(settings.model_path)
        self.classes = list(VEHICLE_CLASSES.keys())
        self._lock = threading.Lock()
        log.info("Loaded YOLO model %s on %s", settings.model_path, settings.device)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        with self._lock:
            results = self.model.predict(
                frame,
                conf=self.settings.confidence,
                imgsz=self.settings.imgsz,
                device=self.settings.device,
                classes=self.classes,
                verbose=False,
            )
        out: List[Detection] = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            label = VEHICLE_CLASSES.get(int(k))
            if label:
                out.append(Detection(float(x1), float(y1), float(x2), float(y2), float(c), label))
        return out


class OnnxDetector:
    """YOLO (v8/11, ONNX export) run with OpenCV DNN - no PyTorch required.

    Expects the standard Ultralytics detection output of shape
    (1, 4 + num_classes, N): cx, cy, w, h followed by per-class scores.
    """

    def __init__(self, settings: Settings, path: str | None = None, input_size: int = 640, nms: float = 0.45):
        import cv2

        path = path or settings.onnx_model_path
        self.net = cv2.dnn.readNetFromONNX(path)
        self.name = f"yolo-onnx ({Path(path).name}, OpenCV DNN)"
        self.settings = settings
        self.size = input_size  # the bundled model is exported with a fixed 640x640 input
        self.nms = nms
        self.vehicle_ids = np.array(list(VEHICLE_CLASSES.keys()))
        self._lock = threading.Lock()
        log.info("Loaded ONNX model %s with OpenCV DNN", path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        import cv2

        h, w = frame.shape[:2]
        s = self.size
        r = min(s / w, s / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        dx, dy = (s - nw) // 2, (s - nh) // 2
        canvas = np.full((s, s, 3), 114, np.uint8)  # letterbox, as in Ultralytics
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (s, s), swapRB=True, crop=False)
        with self._lock:
            self.net.setInput(blob)
            out = self.net.forward()

        pred = out[0].T  # (N, 4 + classes)
        scores = pred[:, 4:]
        cls = scores.argmax(axis=1)
        conf = scores[np.arange(len(cls)), cls]
        keep = (conf >= self.settings.confidence) & np.isin(cls, self.vehicle_ids)
        if not keep.any():
            return []
        pred, cls, conf = pred[keep], cls[keep], conf[keep]
        x1 = (pred[:, 0] - pred[:, 2] / 2 - dx) / r
        y1 = (pred[:, 1] - pred[:, 3] / 2 - dy) / r
        bw, bh = pred[:, 2] / r, pred[:, 3] / r
        boxes = [[float(a), float(b), float(c), float(d)] for a, b, c, d in zip(x1, y1, bw, bh)]
        idx = cv2.dnn.NMSBoxesBatched(boxes, conf.tolist(), cls.tolist(), self.settings.confidence, self.nms)
        out_dets: List[Detection] = []
        for i in np.array(idx).flatten():
            bx, by, bw_, bh_ = boxes[i]
            out_dets.append(Detection(
                max(0.0, bx), max(0.0, by), min(float(w), bx + bw_), min(float(h), by + bh_),
                float(conf[i]), VEHICLE_CLASSES[int(cls[i])],
            ))
        return out_dets


class DummyDetector:
    """Detects bright blobs as "car". Good enough for synthetic test videos."""

    name = "dummy (bright blobs, not YOLO)"

    def __init__(self, threshold: int = 200, min_area: int = 50):
        self.threshold = threshold
        self.min_area = min_area

    def detect(self, frame: np.ndarray) -> List[Detection]:
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h >= self.min_area:
                out.append(Detection(x, y, x + w, y + h, 0.99, "car"))
        return out


def build_detector(settings: Settings) -> Detector:
    if settings.detector == "dummy":
        log.warning("Using DummyDetector (VC_DETECTOR=dummy) - not YOLO")
        return DummyDetector()
    if settings.detector == "onnx" or settings.model_path.endswith(".onnx"):
        path = settings.model_path if settings.model_path.endswith(".onnx") else None
        return OnnxDetector(settings, path)
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        log.warning("ultralytics/PyTorch not installed - using the ONNX model with OpenCV DNN")
        return OnnxDetector(settings)
    return YoloDetector(settings)
