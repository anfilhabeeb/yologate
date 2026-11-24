# 🚦 YOLOgate

**AI vehicle counting for any camera.**

A web portal that connects to live camera streams, detects vehicles with **YOLO**
(Ultralytics YOLO11) and counts every vehicle whose path crosses a line you draw
on the camera view. Runs on Linux, CPU-only by default.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/anfilhabeeb/yologate?ref=claude/loving-gates-aghs13)

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

## Try it in GitHub Codespaces (no install)

1. Click the **Open in GitHub Codespaces** button above. You can also use
   **Code → Codespaces → Create codespace** on GitHub.
2. Wait for setup to finish (about 1 minute). The codespace uses the **lite** install, which runs
   YOLO from the bundled ONNX model with OpenCV, so the large PyTorch download isn't needed.
3. The portal opens in a new browser tab (port **8000**). A **Demo camera** with a counting line
   is already added, and its count goes up as the synthetic buses cross the line.
   If the tab doesn't open, go to the **Ports** panel and open port 8000.
4. Add your own cameras with **+ Add camera**.

Notes:
- A codespace runs in GitHub's cloud, so it can only reach cameras with a **public** URL.
  Cameras on your local network (e.g. `http://192.168.x.x/...`) aren't reachable from it.
- To restart the app, run `bash tools/demo.sh`. Logs are in `/tmp/yologate.log`.
- Codespaces stop after about 30 minutes of inactivity. Stop or delete yours at
  https://github.com/codespaces to save your free hours.

## Quick start (Linux)

**Lite install (recommended to start, no PyTorch, about 300 MB):** YOLO11n runs from
`models/yolo11n.onnx` through OpenCV DNN.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-lite.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Full install (Ultralytics + PyTorch, about 6 GB with CUDA; use it for GPUs or other `.pt` models):**

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # optional: CPU-only torch
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000, click **+ Add camera**, enter the stream URL, then
open the camera and click **Draw lines**. With the full install, the YOLO weights
(`yolo11n.pt`) are downloaded automatically on first start. If PyTorch isn't
installed, the portal automatically uses the bundled ONNX model.

### No camera at hand? Use the synthetic camera

```bash
python tools/fake_camera.py --port 8081 --sprite tools/assets/bus.jpg
```

Or run `bash tools/demo.sh`, which starts the portal and the fake camera and adds the camera with a counting line.

Then add the camera `http://localhost:8081/video` (MJPEG) or
`http://localhost:8081/snapshot.jpg` (stream type *JPEG snapshot*).

### Windows (no Docker)

1. Install Python 3.12 from https://www.python.org/downloads/ and tick **"Add python.exe to PATH"**.
2. Download this repository (**Code → Download ZIP**) and unzip it.
3. Double-click **`start_windows.bat`**. The first run installs about 300 MB of dependencies.
4. Open http://localhost:8000.

### Docker

```bash
docker compose up -d                 # portal on :8000, data in the yologate-data volume
docker compose --profile demo up -d  # also starts a fake camera at http://fake-camera:8081/video
```

The image is the **lite** build (ONNX + OpenCV, no PyTorch, about 0.5 GB). For the
full Ultralytics/PyTorch build, run `docker build --build-arg FULL=1 -t yologate:full .`.

Or run the image that CI publishes:

```bash
docker run -d --name yologate -p 8000:8000 -v yologate-data:/data ghcr.io/anfilhabeeb/yologate:latest
```

### systemd service

```bash
sudo ./deploy/install.sh     # installs to /opt/yologate, starts yologate.service
```

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `VC_MODEL` | `yolo11n.pt` | YOLO weights for the full install (e.g. `yolo11s.pt`), or a `.onnx` file |
| `VC_ONNX_MODEL` | `models/yolo11n.onnx` | ONNX model used when PyTorch isn't installed |
| `VC_DEVICE` | `cpu` | `cpu`, `0` (first CUDA GPU), … |
| `VC_CONFIDENCE` | `0.35` | Detection confidence threshold |
| `VC_IMGSZ` | `640` | Inference image size |
| `VC_MAX_FPS` | `10` | Max processed frames/s per camera |
| `VC_RECONNECT_DELAY` | `3` | Seconds between reconnect attempts |
| `VC_DATA_DIR` | `data` | Where the SQLite database is stored |
| `VC_DETECTOR` | `yolo` | `yolo` (PyTorch if installed, else ONNX), `onnx` (force OpenCV DNN), `dummy` (tests) |

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

- CI publishes the Docker image as `ghcr.io/anfilhabeeb/yologate`.
  If you rename the repository to `yologate`, CI publishes to `ghcr.io/anfilhabeeb/yologate`; update
  `docker-compose.yml` to match.
- Environment variables keep the `VC_` prefix.

- YOLO11 model weights and the sample bus image come from Ultralytics and are licensed under
  AGPL-3.0. Check Ultralytics' licensing before commercial use.
- The portal has no authentication. Run it on a trusted network or behind a reverse proxy with auth.
- Stream URLs with credentials (`http://user:pass@cam/...`) are stored in plain text in the SQLite DB.
- For many cameras or higher FPS, use a GPU (`VC_DEVICE=0` with CUDA torch).
