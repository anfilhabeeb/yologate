"""FastAPI web portal + REST API."""
from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .agent import CameraAgent
from .config import Settings
from .db import Database
from .detector import build_detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STATIC_DIR = Path(__file__).parent / "static"


class CameraIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    url: str = Field(min_length=1, max_length=2000)
    stream_type: Literal["auto", "video", "snapshot"] = "auto"
    enabled: bool = True


class CameraPatch(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    url: Optional[str] = Field(default=None, min_length=1, max_length=2000)
    stream_type: Optional[Literal["auto", "video", "snapshot"]] = None
    enabled: Optional[bool] = None


class LineIn(BaseModel):
    id: Optional[str] = None
    name: str = ""
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)


def create_app(settings: Optional[Settings] = None, detector=None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = Database(settings.db_path)
        det = detector or build_detector(settings)
        agent = CameraAgent(db, det, settings)
        agent.start()
        app.state.db, app.state.agent, app.state.settings = db, agent, settings
        try:
            yield
        finally:
            agent.stop()
            db.close()

    app = FastAPI(title="Vehicle Counter", version=__version__, lifespan=lifespan)

    def db_(request: Request) -> Database:
        return request.app.state.db

    def agent_(request: Request) -> CameraAgent:
        return request.app.state.agent

    def camera_or_404(request: Request, cam_id: int) -> dict:
        cam = db_(request).get_camera(cam_id)
        if not cam:
            raise HTTPException(404, "Camera not found")
        return cam

    def with_status(request: Request, cam: dict) -> dict:
        return {**cam, "status": agent_(request).status(cam["id"])}

    # ---- pages ---------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/camera/{cam_id}", include_in_schema=False)
    def camera_page(cam_id: int):
        return FileResponse(STATIC_DIR / "camera.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ---- api -----------------------------------------------------------
    @app.get("/api/health")
    def health(request: Request):
        agent = agent_(request)
        return {"status": "ok", "version": __version__, "detector": settings.detector,
                "model": settings.model_path if settings.detector == "yolo" else None,
                "cameras": len(db_(request).list_cameras()), "workers": len(agent.workers)}

    @app.get("/api/cameras")
    def list_cameras(request: Request):
        return [with_status(request, c) for c in db_(request).list_cameras()]

    @app.post("/api/cameras", status_code=201)
    def add_camera(request: Request, body: CameraIn):
        cam = db_(request).add_camera(body.name.strip(), body.url.strip(), body.stream_type, body.enabled)
        if cam["enabled"]:
            agent_(request).start_camera(cam)
        return with_status(request, cam)

    @app.get("/api/cameras/{cam_id}")
    def get_camera(request: Request, cam_id: int):
        return with_status(request, camera_or_404(request, cam_id))

    @app.patch("/api/cameras/{cam_id}")
    def update_camera(request: Request, cam_id: int, body: CameraPatch):
        camera_or_404(request, cam_id)
        cam = db_(request).update_camera(cam_id, **body.model_dump(exclude_none=True))
        agent = agent_(request)
        if cam["enabled"]:
            agent.start_camera(cam)  # restart with new url/settings
        else:
            agent.stop_camera(cam_id)
        return with_status(request, cam)

    @app.delete("/api/cameras/{cam_id}", status_code=204)
    def delete_camera(request: Request, cam_id: int):
        camera_or_404(request, cam_id)
        agent_(request).stop_camera(cam_id)
        db_(request).delete_camera(cam_id)
        return Response(status_code=204)

    @app.put("/api/cameras/{cam_id}/lines")
    def set_lines(request: Request, cam_id: int, lines: List[LineIn]):
        camera_or_404(request, cam_id)
        clean = []
        for i, l in enumerate(lines):
            d = l.model_dump()
            if (d["x1"], d["y1"]) == (d["x2"], d["y2"]):
                raise HTTPException(422, f"Line {i + 1} has zero length")
            d["id"] = d["id"] or uuid.uuid4().hex[:8]
            d["name"] = d["name"].strip() or f"Line {i + 1}"
            clean.append(d)
        cam = db_(request).update_camera(cam_id, lines=clean)
        agent_(request).update_lines(cam_id, clean)
        return with_status(request, cam)

    @app.get("/api/cameras/{cam_id}/stats")
    def stats(request: Request, cam_id: int):
        camera_or_404(request, cam_id)
        return agent_(request).status(cam_id)

    @app.get("/api/cameras/{cam_id}/events")
    def events(request: Request, cam_id: int, limit: int = 50):
        camera_or_404(request, cam_id)
        return db_(request).recent_events(cam_id, max(1, min(limit, 500)))

    @app.post("/api/cameras/{cam_id}/reset")
    def reset(request: Request, cam_id: int):
        camera_or_404(request, cam_id)
        agent_(request).reset_counts(cam_id)
        return agent_(request).status(cam_id)

    @app.get("/api/cameras/{cam_id}/snapshot")
    def snapshot(request: Request, cam_id: int, raw: bool = False):
        camera_or_404(request, cam_id)
        w = agent_(request).get(cam_id)
        jpg = w.latest_jpeg(raw=raw) if w else None
        if not jpg:
            return JSONResponse({"detail": "No frame available yet"}, status_code=503)
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/cameras/{cam_id}/stream")
    async def stream(request: Request, cam_id: int, raw: bool = False):
        camera_or_404(request, cam_id)
        agent = agent_(request)
        boundary = "frame"

        async def gen():
            last = -1
            while not await request.is_disconnected():
                w = agent.get(cam_id)
                if w is not None and w.jpeg_seq != last:
                    jpg = w.latest_jpeg(raw=raw)
                    if jpg:
                        last = w.jpeg_seq
                        yield (f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                               f"Content-Length: {len(jpg)}\r\n\r\n").encode() + jpg + b"\r\n"
                await asyncio.sleep(0.04)

        return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}",
                                 headers={"Cache-Control": "no-store"})

    return app


app = create_app()
