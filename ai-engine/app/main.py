"""FastAPI application entrypoint for ai-engine.

Phase 1 exposed ``GET /health`` only. Phase 3 adds the real STT -> NLU -> TTS
pipeline, its Prometheus metrics, and a gRPC server (the ``VoiceGateway``
contract) started alongside HTTP.

The ``GET /health`` contract is frozen from Phase 1 and is unchanged.

No model is loaded at import or startup: every engine is constructed lazily and
resolves its weights on first use, so the process starts (and answers ``/health``)
in milliseconds and never downloads anything until a request needs it.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import grpc
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from app.audio_io import to_wav_bytes
from app.config import get_settings
from app.grpc.server import start_grpc_server
from app.pipeline import VoicePipeline, build_pipeline

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)

#: The single process-wide pipeline. Constructing it loads no weights.
pipeline: VoicePipeline = build_pipeline(settings)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start the gRPC server alongside HTTP; shut everything down cleanly."""
    grpc_server: grpc.aio.Server | None = None
    if settings.grpc_enabled:
        try:
            grpc_server, port = await start_grpc_server(settings, pipeline)
            logger.info("gRPC server bound on port %s", port)
        except Exception:  # pragma: no cover - port clash etc. must not kill HTTP
            logger.exception("gRPC server failed to start; HTTP continues")
            grpc_server = None
    try:
        yield
    finally:
        pipeline.stt.close()
        pipeline.nlu.close()
        pipeline.tts.close()
        if grpc_server is not None:
            await grpc_server.stop(grace=1.0)


app = FastAPI(
    title="CloudPSO Voice Agent — ai-engine",
    version="0.3.0",
    description="STT -> NLU -> TTS pipeline service (Phase 3).",
    lifespan=lifespan,
)

# CORS for the Phase 4 browser dashboard. Additive and off the frozen contract:
# origins come from CLOUDEPSO_CORS_ALLOW_ORIGINS and default to the local
# dashboard only (never "*").
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness/readiness probe. Contract fixed at Phase 1."""
    return {"status": "ok", "service": settings.service_name}


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus exposition endpoint (scraped by infra/prometheus)."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Pipeline endpoints
#
# These are additive: the frozen surface is /health. They exist so the
# Phase 4 dashboard (and `curl`) can drive each stage independently without
# standing up a WebRTC peer.
# ---------------------------------------------------------------------------


class NLURequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    speed: float | None = Field(default=None, gt=0.0, le=4.0)


def _intent_payload(intent) -> dict:
    return {
        "intent": intent.intent,
        "confidence": round(float(intent.confidence), 4),
        "backend": intent.backend,
        "slots": {s.name: s.value for s in intent.slots},
        "scores": {k: round(float(v), 4) for k, v in (intent.scores or {}).items()},
    }


@app.post("/stt")
async def stt(request: Request) -> dict:
    """Transcribe a WAV body. Returns the text and per-stage timing."""
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty request body; send WAV bytes")
    try:
        audio, rate = await _read_wav_bytes(body)
        result = await _run_off_loop(pipeline.transcribe, audio, rate)
    except HTTPException:
        raise
    except Exception as exc:
        _raise_unavailable("STT", exc)
    return {
        "text": result.text,
        "language": result.language,
        "duration_s": round(result.duration_s, 3),
        "backend": result.backend,
    }


@app.post("/nlu")
async def nlu(payload: NLURequest) -> dict:
    """Classify a transcript into an intent (no audio involved)."""
    try:
        intent = await _run_off_loop(pipeline.classify, payload.text)
    except Exception as exc:
        _raise_unavailable("NLU", exc)
    return _intent_payload(intent)


@app.post("/tts")
async def tts(payload: TTSRequest) -> Response:
    """Synthesise text and return a WAV body."""
    try:
        result = await _run_off_loop(
            lambda: pipeline.synthesize(payload.text, speed=payload.speed)
        )
    except Exception as exc:
        _raise_unavailable("TTS", exc)
    wav = to_wav_bytes(result.audio, result.sample_rate)
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(result.sample_rate),
            "X-Duration-Seconds": f"{result.duration_s:.3f}",
        },
    )


@app.post("/pipeline")
async def run_pipeline(request: Request, include_audio: bool = False) -> dict:
    """Full STT -> NLU -> TTS turn from a WAV body."""
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty request body; send WAV bytes")
    try:
        audio, rate = await _read_wav_bytes(body)
        result = await _run_off_loop(pipeline.process_turn, audio, rate)
    except HTTPException:
        raise
    except Exception as exc:
        _raise_unavailable("pipeline", exc)

    payload = {
        "transcript": result.transcript.text,
        "intent": _intent_payload(result.intent),
        "reply_text": result.reply_text,
        "audio": {
            "sample_rate": result.speech.sample_rate,
            "duration_s": round(result.speech.duration_s, 3),
            "cancelled": result.speech.cancelled,
        },
        "timings_ms": {k: round(v * 1000, 2) for k, v in result.timings.items()},
    }
    if include_audio:
        payload["audio"]["wav_base64"] = base64.b64encode(
            to_wav_bytes(result.speech.audio, result.speech.sample_rate)
        ).decode("ascii")
    return payload


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _run_off_loop(fn, *args):
    """Run a blocking, CPU-bound pipeline call on a worker thread."""
    return await asyncio.to_thread(fn, *args)


async def _read_wav_bytes(body: bytes):
    """Decode WAV bytes to (float32 mono, sample_rate) without touching disk."""
    try:
        data, rate = sf.read(io.BytesIO(body), dtype="float32", always_2d=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode WAV: {exc}") from exc
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    return data, int(rate)


def _raise_unavailable(stage: str, exc: Exception) -> None:
    logger.warning("%s unavailable: %s", stage, exc)
    raise HTTPException(status_code=503, detail=f"{stage} backend unavailable: {exc}")


def main() -> None:
    """Run the service with uvicorn (container entrypoint / `python -m app.main`)."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
