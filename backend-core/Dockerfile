# syntax=docker/dockerfile:1
# CloudPSO Voice Agent — backend-core (Phase 1 skeleton).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --- dependency layer: cached until requirements.txt changes ----------------
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- non-root runtime user --------------------------------------------------
RUN groupadd --system --gid 1001 appuser \
    && useradd --system --uid 1001 --gid appuser --create-home appuser

# --- application layer (changes most often, so copied last) -----------------
COPY --chown=appuser:appuser pyproject.toml ./
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser proto ./proto

USER appuser

EXPOSE 8000 50051

# Container-internal bind port is fixed at 8000; the *published* host port is
# configurable via BACKEND_CORE_PORT in .env (see docker-compose.yml).
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

CMD ["python", "-m", "app.main"]
