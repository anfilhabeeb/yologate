#!/usr/bin/env bash
# One-time setup when the Codespace is created.
# Installs the lite requirements only: YOLO runs from models/yolo11n.onnx with
# OpenCV, so the large PyTorch download is not needed.
# For the full Ultralytics/PyTorch stack instead, run: FULL=1 bash .devcontainer/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PIP="python3 -m pip install --timeout 120 --retries 10"
if [ "${FULL:-0}" = "1" ]; then
  $PIP torch torchvision --index-url https://download.pytorch.org/whl/cpu
  $PIP -r requirements-dev.txt
else
  $PIP -r requirements-lite.txt pytest httpx
fi
python3 -c "import cv2, fastapi, uvicorn, numpy; print('Vehicle Counter dependencies OK')"
