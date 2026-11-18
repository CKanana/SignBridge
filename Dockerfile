# SignBridge animation API container.
#
# Build:  docker build -t signbridge .
# Run:    docker run --rm -p 8000:8000 -e SIGNBRIDGE_API_KEY=change-me signbridge
#
# Rendering needs ffmpeg (H.264 output). OpenCV is the headless wheel, so no
# X11/GL runtime is installed.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg/ffprobe: MP4 probe + H.264 transcode.
# libglib2.0-0/libgomp1: runtime libraries used by the OpenCV and NumPy wheels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg \
      libglib2.0-0 \
      libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so Pose/ changes do not invalidate the pip layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The service never writes inside the repository.
RUN mkdir -p /tmp/signbridge-outputs

ENV SIGNBRIDGE_ROOT=/app \
    OUTPUT_DIR=/tmp/signbridge-outputs \
    SIGNBRIDGE_HEADLESS=1 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT','8000') + '/health', timeout=5)" || exit 1

# Single worker: rendering is CPU-bound and serialised in-process.
CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 120"]
