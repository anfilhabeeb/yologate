"""SQLite persistence for cameras, counting lines and crossing events."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    url TEXT NOT NULL,
    stream_type TEXT NOT NULL DEFAULT 'auto',
    enabled INTEGER NOT NULL DEFAULT 1,
    lines TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id INTEGER NOT NULL,
    line_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    label TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id, ts);
"""


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- cameras -------------------------------------------------------
    @staticmethod
    def _cam(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["enabled"] = bool(d["enabled"])
        d["lines"] = json.loads(d["lines"])
        return d

    def list_cameras(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM cameras ORDER BY id").fetchall()
        return [self._cam(r) for r in rows]

    def get_camera(self, cam_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM cameras WHERE id=?", (cam_id,)).fetchone()
        return self._cam(row) if row else None

    def add_camera(self, name: str, url: str, stream_type: str = "auto", enabled: bool = True) -> Dict[str, Any]:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO cameras(name,url,stream_type,enabled,lines,created_at) VALUES(?,?,?,?,?,?)",
                (name, url, stream_type, int(enabled), "[]", time.time()),
            )
            self._conn.commit()
            cam_id = cur.lastrowid
        return self.get_camera(cam_id)  # type: ignore[return-value]

    def update_camera(self, cam_id: int, **fields: Any) -> Optional[Dict[str, Any]]:
        allowed = {"name", "url", "stream_type", "enabled", "lines"}
        sets, vals = [], []
        for k, v in fields.items():
            if k not in allowed or v is None:
                continue
            if k == "lines":
                v = json.dumps(v)
            if k == "enabled":
                v = int(v)
            sets.append(f"{k}=?")
            vals.append(v)
        if sets:
            with self._lock:
                self._conn.execute(f"UPDATE cameras SET {', '.join(sets)} WHERE id=?", (*vals, cam_id))
                self._conn.commit()
        return self.get_camera(cam_id)

    def delete_camera(self, cam_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM cameras WHERE id=?", (cam_id,))
            self._conn.execute("DELETE FROM events WHERE camera_id=?", (cam_id,))
            self._conn.commit()
            return cur.rowcount > 0

    # ---- events --------------------------------------------------------
    def add_event(self, camera_id: int, line_id: str, direction: str, label: str, track_id: int, ts: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO events(camera_id,line_id,direction,label,track_id,ts) VALUES(?,?,?,?,?,?)",
                (camera_id, line_id, direction, label, track_id, ts),
            )
            self._conn.commit()

    def recent_events(self, camera_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE camera_id=? ORDER BY id DESC LIMIT ?", (camera_id, limit)
            ).fetchall()
        return [dict(r) for r in rows]

    def count_summary(self, camera_id: int) -> Dict[str, Dict[str, Dict[str, int]]]:
        """{line_id: {direction: {label: n}}} for all stored events of a camera."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT line_id, direction, label, COUNT(*) n FROM events WHERE camera_id=? "
                "GROUP BY line_id, direction, label",
                (camera_id,),
            ).fetchall()
        out: Dict[str, Dict[str, Dict[str, int]]] = {}
        for r in rows:
            out.setdefault(r["line_id"], {}).setdefault(r["direction"], {})[r["label"]] = r["n"]
        return out

    def clear_events(self, camera_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM events WHERE camera_id=?", (camera_id,))
            self._conn.commit()
