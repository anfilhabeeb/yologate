#!/usr/bin/env bash
# Start the portal plus a synthetic demo camera, and add that camera with a
# counting line if no cameras exist yet. Used by Codespaces; works anywhere.
#   bash tools/demo.sh            then open http://localhost:8000
set -uo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8000}"
CAM_PORT="${CAM_PORT:-8081}"
# Use the first Python that has the dependencies (pip installs go to python3,
# the apt fallback in .devcontainer/setup.sh installs for /usr/bin/python3).
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for cand in python3 /usr/bin/python3 python; do
    if command -v "$cand" >/dev/null && "$cand" -c "import cv2, fastapi, uvicorn" 2>/dev/null; then
      PY="$cand"; break
    fi
  done
fi
if [ -z "$PY" ]; then
  echo "ERROR: no Python with the dependencies found - run: bash .devcontainer/setup.sh"
  exit 1
fi
echo "Using $PY"

pkill -f "tools/fake_camera.py" 2>/dev/null || true
pkill -f "uvicorn app.main:app" 2>/dev/null || true

$PY tools/fake_camera.py --port "$CAM_PORT" --sprite tools/assets/bus.jpg --speed 14 --vehicles 2 &
$PY -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" &

for _ in $(seq 1 120); do
  curl -fs "http://localhost:$PORT/api/health" >/dev/null && break
  sleep 1
done

if [ "$(curl -fs "http://localhost:$PORT/api/cameras")" = "[]" ]; then
  id=$(curl -fs -X POST "http://localhost:$PORT/api/cameras" -H 'Content-Type: application/json' \
    -d "{\"name\":\"Demo camera\",\"url\":\"http://127.0.0.1:$CAM_PORT/video\"}" | $PY -c 'import sys, json; print(json.load(sys.stdin)["id"])')
  curl -fs -X PUT "http://localhost:$PORT/api/cameras/$id/lines" -H 'Content-Type: application/json' \
    -d '[{"name":"Center line","x1":0.5,"y1":0.25,"x2":0.5,"y2":0.95}]' >/dev/null
  echo "Added demo camera #$id with a counting line"
fi
echo "YOLOgate is running on port $PORT"
wait
