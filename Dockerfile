# Lite image (default): YOLO11n runs from models/yolo11n.onnx with OpenCV DNN,
# no PyTorch, ~0.5 GB.  Full Ultralytics/PyTorch image (~several GB):
#   docker build --build-arg FULL=1 -t yologate:full .
FROM python:3.12-slim

ARG FULL=0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    VC_DATA_DIR=/data YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app
COPY requirements*.txt ./
RUN if [ "$FULL" = "1" ]; then \
        apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/* \
        && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
        && pip install -r requirements.txt \
        && mkdir -p /models && cd /models && python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"; \
    else \
        pip install -r requirements-lite.txt; \
    fi
ENV VC_MODEL=/models/yolo11n.pt

COPY app ./app
COPY models ./models
COPY tools ./tools

RUN useradd -r -u 10001 yologate && mkdir -p /data && chown yologate /data
USER yologate
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=4)" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
