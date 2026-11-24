#!/usr/bin/env bash
# Install YOLOgate as a systemd service on a Linux host (Debian/Ubuntu/RHEL).
# Usage: sudo ./deploy/install.sh [install_dir]   (default /opt/yologate)
set -euo pipefail
DIR="${1:-/opt/yologate}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"

if ! id yologate >/dev/null 2>&1; then useradd -r -m -d /var/lib/yologate -s /usr/sbin/nologin yologate; fi
mkdir -p "$DIR" /var/lib/yologate
cp -r "$SRC/app" "$SRC/tools" "$SRC/requirements.txt" "$DIR/"
python3 -m venv "$DIR/venv"
"$DIR/venv/bin/pip" install --upgrade pip
"$DIR/venv/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cpu || true
"$DIR/venv/bin/pip" install -r "$DIR/requirements.txt"
(cd /var/lib/yologate && "$DIR/venv/bin/python" -c "from ultralytics import YOLO; YOLO('yolo11n.pt')")
chown -R yologate:yologate /var/lib/yologate

sed "s#@DIR@#$DIR#g" "$SRC/deploy/yologate.service" > /etc/systemd/system/yologate.service
systemctl daemon-reload
systemctl enable --now yologate
echo "YOLOgate running on http://$(hostname -I | awk '{print $1}'):8000"
