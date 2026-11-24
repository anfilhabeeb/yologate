FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    VC_DATA_DIR=/data VC_MODEL=/models/yolo11n.pt YOLO_CONFIG_DIR=/tmp/Ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU-only PyTorch keeps the image small; use a CUDA base image + default torch for GPUs.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Bake the YOLO weights into the image so the container works offline.
RUN mkdir -p /models && cd /models && python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

COPY app ./app
COPY tools ./tools

RUN useradd -r -u 10001 vc && mkdir -p /data && chown vc /data
USER vc
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fs http://localhost:8000/api/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
