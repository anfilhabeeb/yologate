#!/usr/bin/env bash
# One-time setup when the Codespace is created.
# Installs the lite requirements only: YOLO runs from models/yolo11n.onnx with
# OpenCV, so the large PyTorch download is not needed.
#
# If pypi.org can't be reached (seen on some codespaces: SSL EOF errors), it
# falls back to Debian's own packages via apt, which use a different server.
#
#   bash .devcontainer/setup.sh            # pip, falling back to apt
#   USE_APT=1 bash .devcontainer/setup.sh  # go straight to apt
#   FULL=1 bash .devcontainer/setup.sh     # full Ultralytics + PyTorch stack (pip only)
set -uo pipefail
cd "$(dirname "$0")/.."
CHECK="import cv2, fastapi, uvicorn, numpy, pydantic"

if [ "${FULL:-0}" = "1" ]; then
  set -e
  PIP="python3 -m pip install --timeout 120 --retries 10"
  $PIP torch torchvision --index-url https://download.pytorch.org/whl/cpu
  $PIP -r requirements-dev.txt
  python3 -c "$CHECK" && echo "YOLOgate dependencies OK (python3, full)"
  exit 0
fi

if [ "${USE_APT:-0}" != "1" ]; then
  echo ">>> Installing with pip (pypi.org)..."
  if python3 -m pip install --timeout 30 --retries 2 -r requirements-lite.txt \
     && python3 -c "$CHECK"; then
    echo "YOLOgate dependencies OK (python3, pip)"
    exit 0
  fi
  echo ">>> pip could not install from pypi.org - falling back to Debian packages (apt)"
fi

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  python3-opencv python3-numpy python3-fastapi python3-uvicorn python3-pydantic \
  python3-pytest python3-httpx
if /usr/bin/python3 -c "$CHECK"; then
  echo "YOLOgate dependencies OK (/usr/bin/python3, apt)"
else
  echo "ERROR: dependencies could not be installed with pip or apt"
  exit 1
fi
