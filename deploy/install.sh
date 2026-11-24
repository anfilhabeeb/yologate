#!/usr/bin/env bash
# Install Vehicle Counter as a systemd service on a Linux host (Debian/Ubuntu/RHEL).
# Usage: sudo ./deploy/install.sh [install_dir]   (default /opt/vehicle_counter)
set -euo pipefail
DIR="${1:-/opt/vehicle_counter}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"

if ! id vc >/dev/null 2>&1; then useradd -r -m -d /var/lib/vehicle_counter -s /usr/sbin/nologin vc; fi
mkdir -p "$DIR" /var/lib/vehicle_counter
cp -r "$SRC/app" "$SRC/tools" "$SRC/requirements.txt" "$DIR/"
python3 -m venv "$DIR/venv"
"$DIR/venv/bin/pip" install --upgrade pip
"$DIR/venv/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cpu || true
"$DIR/venv/bin/pip" install -r "$DIR/requirements.txt"
(cd /var/lib/vehicle_counter && "$DIR/venv/bin/python" -c "from ultralytics import YOLO; YOLO('yolo11n.pt')")
chown -R vc:vc /var/lib/vehicle_counter

sed "s#@DIR@#$DIR#g" "$SRC/deploy/vehicle-counter.service" > /etc/systemd/system/vehicle-counter.service
systemctl daemon-reload
systemctl enable --now vehicle-counter
echo "Vehicle Counter running on http://$(hostname -I | awk '{print $1}'):8000"
