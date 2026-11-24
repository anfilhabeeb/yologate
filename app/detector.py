"""Object detectors. `YoloDetector` wraps Ultralytics YOLO; `DummyDetector`
is a lightweight stand-in used by tests and for environments without a model."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
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
    """Thread-safe YOLO vehicle detector shared by all camera workers."""

    def __init__(self, settings: Settings):
        from ultralytics import YOLO  # imported lazily: heavy dependency

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


class DummyDetector:
    """Detects bright blobs as "car". Good enough for synthetic test videos."""

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
    return YoloDetector(settings)
