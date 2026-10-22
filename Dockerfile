# syntax=docker/dockerfile:1.7
# ---- frontend build ---------------------------------------------------------
FROM node:22.11.0-bookworm-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- runtime: API + worker (same image, different command) -----------------
FROM python:3.12.7-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    TOUCHLINE_DATA_DIR=/data TOUCHLINE_MODEL_PATH=/app/models/object_detection_yolox_2022nov.onnx \
    TOUCHLINE_FRONTEND_DIST=/app/frontend/dist
# libgomp1 / libglib2.0-0: runtime libs used by the OpenCV wheel on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend/ backend/
COPY models/ models/
COPY --from=ui /ui/dist frontend/dist
# Fail the build if the pinned weights are missing/corrupt or OpenCV is not 5.x.
RUN python -c "import hashlib,cv2,sys; d=hashlib.sha256(open('/app/models/object_detection_yolox_2022nov.onnx','rb').read()).hexdigest(); \
assert d=='c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063', d; assert cv2.__version__.startswith('5.'), cv2.__version__; print('OpenCV', cv2.__version__)"
RUN useradd -r -u 10001 touchline && mkdir -p /data && chown touchline /data
USER touchline
WORKDIR /app/backend
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4).status==200 else 1)"
CMD ["python", "-m", "uvicorn", "touchline.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
