# OCR + ID-detection service (CPU). Build: docker compose build   (x86_64 only: no Linux arm64 Paddle wheels)
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /bin/

# OpenCV (pulled in by PaddleX) needs libGL/glib; Paddle needs OpenMP.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PADDLE_PDX_CACHE_HOME=/opt/paddlex

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY app ./app
COPY config ./config
RUN uv sync --frozen --no-dev

# Bake the OCR models into the image at build time; the running container never downloads anything.
RUN uv run --no-sync python -m app.cli download-models

RUN useradd --create-home --uid 10001 app \
    && chown -R app /opt/paddlex \
    && mkdir -p /tmp/ocr-id-service && chown app /tmp/ocr-id-service
USER app

# ENVIRONMENT=production: the service refuses to start with an unsafe configuration (see README, "Docker deployment").
ENV ENVIRONMENT=production \
    LOG_FORMAT=json \
    UPLOAD_DIR=/tmp/ocr-id-service \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=8)" || exit 1

CMD ["uv", "run", "--no-sync", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
