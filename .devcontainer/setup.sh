#!/usr/bin/env bash
# One-time setup when the Codespace is created.
set -euo pipefail
sudo apt-get update && sudo apt-get install -y --no-install-recommends libgl1 libglib2.0-0
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
# Download the YOLO weights now so the first start is fast.
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
