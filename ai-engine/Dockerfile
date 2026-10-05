# syntax=docker/dockerfile:1
# CloudPSO Voice Agent — ai-engine (Phase 3: STT -> LLM -> TTS).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libsndfile is required by `soundfile` (WAV IO). sherpa-onnx and
# ctranslate2 ship their own runtimes inside the wheels, so they need nothing
# else beyond the base image's libstdc++/libgomp.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- dependency layer: cached until the manifests change --------------------
COPY requirements.txt requirements-dev.txt ./

# torch is installed from the CPU-only index first. PyPI's default linux wheel
# drags in ~2 GB of CUDA libraries this service never uses (there is no GPU in
# the STT/TTS path, and the MiniLM embedder runs on torch CPU). Installing it
# here means the requirements.txt pass below sees it already satisfied.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2,<3.0" \
    && pip install --no-cache-dir -r requirements.txt

# --- non-root runtime user --------------------------------------------------
RUN groupadd --system --gid 1001 appuser \
    && useradd --system --uid 1001 --gid appuser --create-home appuser \
    && mkdir -p /models && chown appuser:appuser /models

# --- application layer (changes most often, so copied last) -----------------
COPY --chown=appuser:appuser pyproject.toml ./
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser proto ./proto
COPY --chown=appuser:appuser scripts ./scripts

USER appuser

# HTTP (8001) and gRPC (50052). Both are fixed in-container; the host mapping is
# configured by AI_ENGINE_PORT / AI_ENGINE_GRPC_PORT in .env.
EXPOSE 8001 50052

# /health never touches a model, so it answers immediately even on a cold
# container with an empty /models volume.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=2).status == 200 else 1)"

CMD ["python", "-m", "app.main"]
