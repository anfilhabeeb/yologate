# Vehicle Counter

A web portal that connects to live camera streams, detects vehicles with **YOLO**
(Ultralytics YOLO11) and counts every vehicle whose path crosses a line you draw
on the camera view. Runs on Linux, CPU-only by default.

![Camera view](docs/camera.png)

## Features

- **Live HTTP camera streams**: MJPEG over HTTP, JPEG snapshot URLs (polled),
  RTSP, HLS, or a local video file. Auto-reconnects when a stream drops.
- **Add, edit, start/stop and delete cameras** from the portal.
- **Draw counting lines** on each camera (click and drag). Several lines per camera are supported,
  and an arrow shows the **IN** direction. Use *Flip IN* to reverse it.
- **Per-line counts** split by direction (in/out) and vehicle type
  (car, motorcycle, bus, truck, bicycle), plus a log of recent crossings.
- **Camera agent**: one worker per camera (a stream reader plus a detector/tracker),
  a shared YOLO model and a supervisor that restarts dead workers.
- Counts are stored in SQLite and survive restarts. The annotated live view
  is served as MJPEG.

![Dashboard](docs/dashboard.png)

## How it works

```
camera ──► reader thread (keeps latest frame only)
                │
                ▼
         processor thread ─► YOLO (shared, vehicle classes only)
                │          ─► IoU/centroid tracker (per camera, velocity-predicted)
                │          ─► line-crossing check (segment intersection, counted once per track)
                ▼
     annotated JPEG ─► /api/cameras/{id}/stream (MJPEG)  ─► web portal
     crossing events ─► SQLite ─► /api/cameras/{id}/stats, /events
```

A vehicle is counted when its tracked **centre point** moves from one side of the
line segment to the other. Each track is counted at most once per line, so
vehicles that stop or jitter on the line aren't counted twice.

## Quick start (Linux)

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # optional: CPU-only torch
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000, click **+ Add camera**, enter the stream URL, then
open the camera and click **Draw lines**. The YOLO weights (`yolo11n.pt`) are
downloaded automatically on first start.

### No camera at hand? Use the synthetic camera

```bash
python tools/fake_camera.py --port 8081 \
  --sprite venv/lib/python3*/site-packages/ultralytics/assets/bus.jpg --crop 3,229,796,728
```

Then add the camera `http://localhost:8081/video` (MJPEG) or
`http://localhost:8081/snapshot.jpg` (stream type *JPEG snapshot*).

### Docker

```bash
docker compose up -d                 # portal on :8000, data in the vc-data volume
docker compose --profile demo up -d  # also starts a fake camera at http://fake-camera:8081/video
```

Or run the image that CI publishes:

```bash
docker run -d -p 8000:8000 -v vc-data:/data ghcr.io/anfilhabeeb/vehicle_counter:latest
```

### systemd service

```bash
sudo ./deploy/install.sh     # installs to /opt/vehicle_counter, starts vehicle-counter.service
```

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `VC_MODEL` | `yolo11n.pt` | YOLO weights (e.g. `yolo11s.pt` for better accuracy) |
| `VC_DEVICE` | `cpu` | `cpu`, `0` (first CUDA GPU), … |
| `VC_CONFIDENCE` | `0.35` | Detection confidence threshold |
| `VC_IMGSZ` | `640` | Inference image size |
| `VC_MAX_FPS` | `10` | Max processed frames/s per camera |
| `VC_RECONNECT_DELAY` | `3` | Seconds between reconnect attempts |
| `VC_DATA_DIR` | `data` | Where the SQLite database is stored |
| `VC_DETECTOR` | `yolo` | `dummy` = bright-blob detector for tests without a model |

## REST API

Interactive docs at `/docs`.

| Method | Path | |
|---|---|---|
| GET | `/api/health` | Health and detector info |
| GET / POST | `/api/cameras` | List cameras / add `{name, url, stream_type}` |
| GET / PATCH / DELETE | `/api/cameras/{id}` | Get, update (`name`, `url`, `stream_type`, `enabled`), delete |
| PUT | `/api/cameras/{id}/lines` | Replace lines: `[{name, x1, y1, x2, y2}]` in 0..1 coordinates |
| GET | `/api/cameras/{id}/stats` | Live state, fps and counts per line/direction/type |
| GET | `/api/cameras/{id}/events` | Recent crossing events |
| POST | `/api/cameras/{id}/reset` | Reset counts |
| GET | `/api/cameras/{id}/stream` | Annotated MJPEG stream (`?raw=1` for unannotated) |
| GET | `/api/cameras/{id}/snapshot` | Latest annotated JPEG (`?raw=1` for unannotated) |

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

The suite covers the geometry, the tracker and the counting logic. It also starts a
real HTTP MJPEG/snapshot camera and checks counts through the API, and runs YOLO end to end
on a generated traffic video. GitHub Actions runs all tests on every push, builds the Docker
image, smoke-tests it and publishes it to GHCR.

## Notes

- The portal has no authentication. Run it on a trusted network or behind a reverse proxy with auth.
- Stream URLs with credentials (`http://user:pass@cam/...`) are stored in plain text in the SQLite DB.
- For many cameras or higher FPS, use a GPU (`VC_DEVICE=0` with CUDA torch).
