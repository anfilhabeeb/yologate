"""Camera agent: supervises one worker per camera.

Each `CameraWorker` runs two threads:
  * a reader that keeps only the most recent frame from the stream (so slow
    inference never builds up latency), and
  * a processor that runs detection, tracking and line-crossing counting and
    renders an annotated JPEG for the web portal.

`CameraAgent` owns all workers, shares one detector between them, restarts
workers whose threads died, and keeps them in sync with the database.
"""
from __future__ import annotations

import logging
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from .config import Settings
from .db import Database
from .detector import Detector
from .geometry import crossing_direction
from .tracker import Tracker

log = logging.getLogger(__name__)

LINE_COLORS = [(0, 255, 255), (255, 0, 255), (0, 165, 255), (255, 255, 0), (0, 255, 0)]


# --------------------------------------------------------------------------
# Frame sources
# --------------------------------------------------------------------------
class SourceError(RuntimeError):
    pass


class FrameSource:
    """Wraps an OpenCV VideoCapture (MJPEG/RTSP/HLS/file) or a JPEG snapshot URL."""

    def __init__(self, url: str, stream_type: str = "auto", timeout: float = 10.0):
        self.url = url
        self.stream_type = stream_type
        self.timeout = timeout
        self.cap: Optional[cv2.VideoCapture] = None
        self.mode = ""
        self.is_file = False
        self.file_fps = 0.0

    def open(self) -> None:
        if self.stream_type in ("auto", "video"):
            cap = cv2.VideoCapture(self.url)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    self.cap = cap
                    self.mode = "video"
                    self._first = frame
                    self.is_file = "://" not in self.url
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    self.file_fps = fps if self.is_file and 0 < fps < 240 else 0.0
                    return
            cap.release()
            if self.stream_type == "video":
                raise SourceError(f"Could not open video stream {self.url}")
        if self.stream_type in ("auto", "snapshot") and self.url.startswith(("http://", "https://")):
            frame = self._fetch_snapshot()
            self.mode = "snapshot"
            self._first = frame
            return
        raise SourceError(f"Could not open stream {self.url}")

    def _fetch_snapshot(self) -> np.ndarray:
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout) as resp:
                data = resp.read()
        except Exception as exc:  # noqa: BLE001
            raise SourceError(f"Snapshot request failed: {exc}") from exc
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise SourceError("Snapshot URL did not return a decodable image")
        return frame

    def read(self) -> np.ndarray:
        first = getattr(self, "_first", None)
        if first is not None:
            self._first = None
            return first
        if self.mode == "video" and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise SourceError("Stream ended or read failed")
            return frame
        if self.mode == "snapshot":
            return self._fetch_snapshot()
        raise SourceError("Source not open")

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------
@dataclass
class CrossingEvent:
    camera_id: int
    line_id: str
    direction: str
    label: str
    track_id: int
    ts: float


def _empty_counts() -> Dict[str, Dict[str, int]]:
    return {"in": {}, "out": {}}


class CameraWorker:
    def __init__(
        self,
        camera: Dict[str, Any],
        detector: Detector,
        settings: Settings,
        on_event: Optional[Callable[[CrossingEvent], None]] = None,
        initial_counts: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None,
        source_factory: Callable[[str, str], FrameSource] = FrameSource,
    ):
        self.camera_id = int(camera["id"])
        self.name = camera["name"]
        self.url = camera["url"]
        self.stream_type = camera.get("stream_type", "auto")
        self.detector = detector
        self.settings = settings
        self.on_event = on_event
        self.source_factory = source_factory

        self._lock = threading.Lock()
        self._frame_cond = threading.Condition()
        self._stop = threading.Event()
        self._latest_raw: Optional[np.ndarray] = None
        self._raw_seq = 0
        self._latest_jpeg: Optional[bytes] = None
        self._latest_raw_jpeg: Optional[bytes] = None
        self.jpeg_seq = 0

        self.lines: List[Dict[str, Any]] = []
        self.counts: Dict[str, Dict[str, Dict[str, int]]] = {}
        self.set_lines(camera.get("lines", []))
        for lid, c in (initial_counts or {}).items():
            if lid in self.counts:
                for direction in ("in", "out"):
                    self.counts[lid][direction].update(c.get(direction, {}))

        self.tracker = Tracker()
        self.state = "stopped"
        self.last_error = ""
        self.fps = 0.0
        self.frame_size: Optional[tuple[int, int]] = None
        self.last_frame_ts = 0.0
        self.frames_processed = 0
        self.active_tracks = 0
        self._threads: List[threading.Thread] = []

    # ---- configuration -------------------------------------------------
    def set_lines(self, lines: List[Dict[str, Any]]) -> None:
        with self._lock:
            self.lines = [dict(l) for l in lines]
            new_counts = {}
            for l in self.lines:
                new_counts[l["id"]] = self.counts.get(l["id"], _empty_counts())
            self.counts = new_counts

    def reset_counts(self) -> None:
        with self._lock:
            self.counts = {l["id"]: _empty_counts() for l in self.lines}
            for t in self.tracker.tracks.values():
                t.counted_lines.clear()

    # ---- lifecycle -----------------------------------------------------
    def start(self) -> None:
        if self.is_alive():
            return
        self._stop.clear()
        self.state = "connecting"
        self._threads = [
            threading.Thread(target=self._reader_loop, name=f"cam{self.camera_id}-reader", daemon=True),
            threading.Thread(target=self._process_loop, name=f"cam{self.camera_id}-proc", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self, join: bool = True) -> None:
        self._stop.set()
        with self._frame_cond:
            self._frame_cond.notify_all()
        if join:
            for t in self._threads:
                t.join(timeout=5)
        self.state = "stopped"

    def is_alive(self) -> bool:
        return bool(self._threads) and all(t.is_alive() for t in self._threads)

    # ---- reader --------------------------------------------------------
    def _reader_loop(self) -> None:
        while not self._stop.is_set():
            source = self.source_factory(self.url, self.stream_type)
            try:
                self.state = "connecting"
                source.open()
                self.state = "running"
                self.last_error = ""
                log.info("Camera %s connected (%s)", self.camera_id, source.mode)
                min_interval = 0.0
                if source.is_file and source.file_fps:
                    min_interval = 1.0 / source.file_fps  # play files in real time
                elif source.mode == "snapshot" and self.settings.max_fps > 0:
                    min_interval = 1.0 / self.settings.max_fps
                while not self._stop.is_set():
                    t0 = time.monotonic()
                    frame = source.read()
                    with self._frame_cond:
                        self._latest_raw = frame
                        self._raw_seq += 1
                        self._frame_cond.notify_all()
                    if min_interval:
                        dt = min_interval - (time.monotonic() - t0)
                        if dt > 0:
                            self._stop.wait(dt)
            except Exception as exc:  # noqa: BLE001
                if self._stop.is_set():
                    break
                self.state = "error"
                self.last_error = str(exc)
                log.warning("Camera %s: %s; retrying in %.1fs", self.camera_id, exc, self.settings.reconnect_delay)
                self._stop.wait(self.settings.reconnect_delay)
            finally:
                source.close()

    # ---- processor -----------------------------------------------------
    def _process_loop(self) -> None:
        last_seq = 0
        fps_t0, fps_n = time.monotonic(), 0
        min_interval = 1.0 / self.settings.max_fps if self.settings.max_fps > 0 else 0.0
        while not self._stop.is_set():
            with self._frame_cond:
                while self._raw_seq == last_seq and not self._stop.is_set():
                    self._frame_cond.wait(timeout=1.0)
                if self._stop.is_set():
                    break
                frame = self._latest_raw
                last_seq = self._raw_seq
            if frame is None:
                continue
            t0 = time.monotonic()
            try:
                self.process_frame(frame)
            except Exception as exc:  # noqa: BLE001
                log.exception("Camera %s processing error", self.camera_id)
                self.last_error = f"processing: {exc}"
            fps_n += 1
            now = time.monotonic()
            if now - fps_t0 >= 1.0:
                self.fps = fps_n / (now - fps_t0)
                fps_t0, fps_n = now, 0
            if min_interval:
                dt = min_interval - (now - t0)
                if dt > 0:
                    self._stop.wait(dt)

    def process_frame(self, frame: np.ndarray) -> List[CrossingEvent]:
        """Detect, track and count on one frame. Public so tests can drive it."""
        h, w = frame.shape[:2]
        self.frame_size = (w, h)
        detections = self.detector.detect(frame)
        events: List[CrossingEvent] = []
        now = time.time()
        with self._lock:
            tracks = self.tracker.update(detections)
            self.active_tracks = len(tracks)
            for t in tracks:
                if t.prev_center == t.center:
                    continue
                prev = (t.prev_center[0] / w, t.prev_center[1] / h)
                curr = (t.center[0] / w, t.center[1] / h)
                for line in self.lines:
                    if line["id"] in t.counted_lines:
                        continue
                    direction = crossing_direction(
                        (line["x1"], line["y1"]), (line["x2"], line["y2"]), prev, curr
                    )
                    if direction is None:
                        continue
                    t.counted_lines.add(line["id"])
                    bucket = self.counts[line["id"]][direction]
                    bucket[t.det.label] = bucket.get(t.det.label, 0) + 1
                    events.append(CrossingEvent(self.camera_id, line["id"], direction, t.det.label, t.id, now))
            lines = [dict(l) for l in self.lines]
            counts = {k: {d: dict(v) for d, v in c.items()} for k, c in self.counts.items()}
        for ev in events:
            if self.on_event:
                try:
                    self.on_event(ev)
                except Exception:  # noqa: BLE001
                    log.exception("on_event failed")

        annotated = self._annotate(frame.copy(), tracks, lines, counts)
        q = [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.jpeg_quality]
        ok1, jpg = cv2.imencode(".jpg", annotated, q)
        ok2, raw = cv2.imencode(".jpg", frame, q)
        with self._lock:
            if ok1:
                self._latest_jpeg = jpg.tobytes()
            if ok2:
                self._latest_raw_jpeg = raw.tobytes()
            self.jpeg_seq += 1
        self.last_frame_ts = now
        self.frames_processed += 1
        return events

    @staticmethod
    def _annotate(frame, tracks, lines, counts) -> np.ndarray:
        h, w = frame.shape[:2]
        for t in tracks:
            d = t.det
            cv2.rectangle(frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 200, 0), 2)
            cv2.putText(frame, f"#{t.id} {d.label} {d.confidence:.2f}", (int(d.x1), max(12, int(d.y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, (int(t.center[0]), int(t.center[1])), 3, (0, 0, 255), -1)
        y = 22
        for i, l in enumerate(lines):
            color = LINE_COLORS[i % len(LINE_COLORS)]
            p1 = (int(l["x1"] * w), int(l["y1"] * h))
            p2 = (int(l["x2"] * w), int(l["y2"] * h))
            cv2.line(frame, p1, p2, color, 3)
            c = counts.get(l["id"], _empty_counts())
            n_in, n_out = sum(c["in"].values()), sum(c["out"].values())
            text = f"{l.get('name') or l['id']}: in {n_in} | out {n_out} | total {n_in + n_out}"
            cv2.putText(frame, l.get("name") or l["id"], p1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (6, y - th - 6), (14 + tw, y + 6), (0, 0, 0), -1)
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            y += th + 14
        return frame

    # ---- accessors -----------------------------------------------------
    def latest_jpeg(self, raw: bool = False) -> Optional[bytes]:
        with self._lock:
            return self._latest_raw_jpeg if raw else self._latest_jpeg

    def get_counts(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        with self._lock:
            return {k: {d: dict(v) for d, v in c.items()} for k, c in self.counts.items()}

    def status(self) -> Dict[str, Any]:
        counts = self.get_counts()
        total = sum(sum(v.values()) for c in counts.values() for v in c.values())
        return {
            "state": self.state,
            "alive": self.is_alive(),
            "fps": round(self.fps, 2),
            "last_error": self.last_error,
            "frame_size": self.frame_size,
            "last_frame_ts": self.last_frame_ts,
            "frames_processed": self.frames_processed,
            "active_tracks": self.active_tracks,
            "counts": counts,
            "total": total,
        }


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------
class CameraAgent:
    """Manages a pool of camera workers sharing one detector."""

    def __init__(self, db: Database, detector: Detector, settings: Settings, supervise_interval: float = 5.0):
        self.db = db
        self.detector = detector
        self.settings = settings
        self.workers: Dict[int, CameraWorker] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._supervise_interval = supervise_interval
        self._supervisor: Optional[threading.Thread] = None

    def _on_event(self, ev: CrossingEvent) -> None:
        self.db.add_event(ev.camera_id, ev.line_id, ev.direction, ev.label, ev.track_id, ev.ts)

    def start(self) -> None:
        for cam in self.db.list_cameras():
            if cam["enabled"]:
                self.start_camera(cam)
        self._stop.clear()
        self._supervisor = threading.Thread(target=self._supervise, name="agent-supervisor", daemon=True)
        self._supervisor.start()
        log.info("Camera agent started with %d camera(s)", len(self.workers))

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            for w in self.workers.values():
                w.stop(join=False)
            for w in self.workers.values():
                w.stop(join=True)
            self.workers.clear()

    def _supervise(self) -> None:
        while not self._stop.wait(self._supervise_interval):
            with self._lock:
                for cam_id, w in list(self.workers.items()):
                    if not w.is_alive():
                        log.warning("Worker for camera %s died; restarting", cam_id)
                        cam = self.db.get_camera(cam_id)
                        if cam and cam["enabled"]:
                            self.start_camera(cam)

    def start_camera(self, cam: Dict[str, Any]) -> CameraWorker:
        with self._lock:
            old = self.workers.pop(cam["id"], None)
            if old:
                old.stop()
            w = CameraWorker(cam, self.detector, self.settings, on_event=self._on_event,
                             initial_counts=self.db.count_summary(cam["id"]))
            self.workers[cam["id"]] = w
            w.start()
            return w

    def stop_camera(self, cam_id: int) -> None:
        with self._lock:
            w = self.workers.pop(cam_id, None)
        if w:
            w.stop()

    def update_lines(self, cam_id: int, lines: List[Dict[str, Any]]) -> None:
        with self._lock:
            w = self.workers.get(cam_id)
        if w:
            w.set_lines(lines)

    def reset_counts(self, cam_id: int) -> None:
        self.db.clear_events(cam_id)
        with self._lock:
            w = self.workers.get(cam_id)
        if w:
            w.reset_counts()

    def get(self, cam_id: int) -> Optional[CameraWorker]:
        with self._lock:
            return self.workers.get(cam_id)

    def status(self, cam_id: int) -> Dict[str, Any]:
        w = self.get(cam_id)
        if w:
            return w.status()
        counts = self.db.count_summary(cam_id)
        total = sum(sum(v.values()) for c in counts.values() for v in c.values())
        return {"state": "stopped", "alive": False, "fps": 0, "last_error": "", "frame_size": None,
                "last_frame_ts": 0, "frames_processed": 0, "active_tracks": 0, "counts": counts, "total": total}
