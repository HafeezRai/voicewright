# syntax=docker/dockerfile:1
# CloudPSO Voice Agent — ai-engine (Phase 1 skeleton).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Phase 1 must NOT pull the multi-GB ML stack (torch, transformers, onnxruntime,
# aiortc, av, silero-vad). Default off keeps the scaffold image light and
# buildable. Phase 3 flips this to true -- or, preferably, moves the ML stack to
# a separate GPU/ONNX-Runtime image stage.
ARG INSTALL_ML_STACK=false

WORKDIR /app

# --- dependency layer: cached until the manifests change --------------------
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt \
    && if [ "$INSTALL_ML_STACK" = "true" ]; then \
         pip install --no-cache-dir -r requirements.txt; \
       fi

# --- non-root runtime user --------------------------------------------------
RUN groupadd --system --gid 1001 appuser \
    && useradd --system --uid 1001 --gid appuser --create-home appuser \
    && mkdir -p /models && chown appuser:appuser /models

# --- application layer (changes most often, so copied last) -----------------
COPY --chown=appuser:appuser pyproject.toml ./
COPY --chown=appuser:appuser app ./app

USER appuser

EXPOSE 8001

# Container-internal bind port is fixed at 8001; the *published* host port is
# configurable via AI_ENGINE_PORT in .env (see docker-compose.yml).
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=2).status == 200 else 1)"

CMD ["python", "-m", "app.main"]
