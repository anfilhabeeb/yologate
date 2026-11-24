"""Runtime settings, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("VC_DATA_DIR", "data")))
    # "yolo" for real detection, "dummy" for tests / environments without a model.
    detector: str = field(default_factory=lambda: _env("VC_DETECTOR", "yolo"))
    model_path: str = field(default_factory=lambda: _env("VC_MODEL", "yolo11n.pt"))
    device: str = field(default_factory=lambda: _env("VC_DEVICE", "cpu"))
    confidence: float = field(default_factory=lambda: float(_env("VC_CONFIDENCE", "0.35")))
    imgsz: int = field(default_factory=lambda: int(_env("VC_IMGSZ", "640")))
    # Max frames per second processed by each camera worker (0 = unlimited).
    max_fps: float = field(default_factory=lambda: float(_env("VC_MAX_FPS", "10")))
    # Seconds between reconnect attempts after a stream error.
    reconnect_delay: float = field(default_factory=lambda: float(_env("VC_RECONNECT_DELAY", "3")))
    jpeg_quality: int = field(default_factory=lambda: int(_env("VC_JPEG_QUALITY", "75")))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "vehicle_counter.db"


# COCO class ids considered "vehicles".
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
