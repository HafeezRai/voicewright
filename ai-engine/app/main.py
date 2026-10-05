"""FastAPI application entrypoint for ai-engine.

Phase 1 exposed ``GET /health`` only. Phase 3 added the real STT -> NLU -> TTS
pipeline, its Prometheus metrics, and a gRPC server (the ``VoiceGateway``
contract) started alongside HTTP. The intent-classification stage has since been
replaced by a generative LLM, so the middle stage is now ``POST /llm`` and the
pipeline is ``STT -> LLM -> TTS``.

The ``GET /health`` contract is frozen from Phase 1 and is unchanged.

No model is loaded at import or startup: every engine is constructed lazily and
resolves its weights on first use, so the process starts (and answers ``/health``)
in milliseconds and never downloads anything until a request needs it. The LLM
is the one exception, and deliberately so: ``AI_ENGINE_LLM_WARMUP`` warms it on a
*background thread*, so a cold multi-gigabyte model load is paid before the first
caller arrives without ever delaying ``/health``.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import threading
from collections import OrderedDict
from collections.abc import AsyncIterator, Sequence
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
from app.llm.history import ConversationHistory
from app.pipeline import VoicePipeline, build_pipeline

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)

#: The single process-wide pipeline. Constructing it loads no weights.
pipeline: VoicePipeline = build_pipeline(settings)

#: How many distinct sessions keep history in memory. A demo has a handful; the
#: cap exists so an unauthenticated caller cannot grow the process without bound
#: by sending a fresh ``session_id`` on every request.
_MAX_SESSIONS = 256


class SessionHistoryStore:
    """The smallest thing that can carry conversation history across requests.

    The pipeline is stateless -- it is shared by every session in the process --
    so history has to live where the session does. This is a locked LRU of
    :class:`ConversationHistory`, one per ``session_id``. A request without a
    ``session_id`` gets no history at all, which makes the default behaviour
    identical to a pipeline with no memory.
    """

    def __init__(self, max_turns: int, max_sessions: int = _MAX_SESSIONS):
        self._max_turns = max_turns
        self._max_sessions = max_sessions
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, ConversationHistory] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self._max_turns > 0

    def snapshot(self, session_id: str | None) -> list:
        """Prior messages for this session (empty for no/unknown session)."""
        if not session_id or not self.enabled:
            return []
        with self._lock:
            history = self._sessions.get(session_id)
            if history is None:
                return []
            self._sessions.move_to_end(session_id)
            return history.snapshot()

    def record(self, session_id: str | None, user_text: str, assistant_text: str) -> None:
        """Append one exchange, creating the session if needed."""
        if not session_id or not self.enabled:
            return
        if not user_text.strip() or not assistant_text.strip():
            return
        with self._lock:
            history = self._sessions.get(session_id)
            if history is None:
                history = ConversationHistory(self._max_turns)
                self._sessions[session_id] = history
            else:
                self._sessions.move_to_end(session_id)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
            history.add_turn(user_text, assistant_text)


_sessions = SessionHistoryStore(settings.llm_history_turns)


def _warm_llm(target: VoicePipeline) -> None:
    """Load the model off the request path. Failures are logged, never raised."""
    try:
        target.llm.load()
    except Exception:
        logger.warning(
            "LLM warm-up failed; the first turn will pay the model-load cost",
            exc_info=True,
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start the gRPC server and (optionally) warm the LLM; shut down cleanly."""
    grpc_server: grpc.aio.Server | None = None
    if settings.grpc_enabled:
        try:
            grpc_server, port = await start_grpc_server(settings, pipeline)
            logger.info("gRPC server bound on port %s", port)
        except Exception:  # pragma: no cover - port clash etc. must not kill HTTP
            logger.exception("gRPC server failed to start; HTTP continues")
            grpc_server = None

    # A background thread, not an await: /health must answer immediately even
    # while 4.66 GB of weights are being paged in.
    if settings.llm_warmup:
        threading.Thread(
            target=_warm_llm, args=(pipeline,), name="llm-warmup", daemon=True
        ).start()

    try:
        yield
    finally:
        pipeline.stt.close()
        pipeline.llm.close()
        pipeline.tts.close()
        if grpc_server is not None:
            await grpc_server.stop(grace=1.0)


app = FastAPI(
    title="CloudPSO Voice Agent — ai-engine",
    version="0.3.0",
    description="STT -> LLM -> TTS pipeline service.",
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


class LLMRequest(BaseModel):
    """A generation request. ``/nlu`` this is not: the reply is free-form prose."""

    text: str = Field(min_length=1, max_length=4000)
    #: Overrides AI_ENGINE_LLM_SYSTEM_PROMPT for this call.
    system_prompt: str | None = Field(default=None, max_length=4000)
    #: Opt in to conversation memory; omitted means a single stateless turn.
    session_id: str | None = Field(default=None, max_length=200)


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    speed: float | None = Field(default=None, gt=0.0, le=4.0)


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


@app.post("/llm")
async def llm(payload: LLMRequest) -> dict:
    """Generate a reply from a transcript (no audio involved)."""
    history = _sessions.snapshot(payload.session_id)
    try:
        result = await _run_off_loop(
            lambda: pipeline.generate(
                payload.text,
                system_prompt=payload.system_prompt,
                history=history,
            )
        )
    except Exception as exc:
        _raise_unavailable("LLM", exc)
    _sessions.record(payload.session_id, payload.text, result.text)
    return result.as_dict()


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
async def run_pipeline(
    request: Request,
    include_audio: bool = False,
    system_prompt: str | None = None,
    session_id: str | None = None,
) -> dict:
    """Full STT -> LLM -> TTS turn from a WAV body.

    ``system_prompt`` overrides the configured persona for this call, and
    ``session_id`` opts in to conversation memory.
    """
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty request body; send WAV bytes")
    history: Sequence = _sessions.snapshot(session_id)
    try:
        audio, rate = await _read_wav_bytes(body)
        result = await _run_off_loop(
            lambda: pipeline.process_turn(
                audio, rate, system_prompt=system_prompt, history=history
            )
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_unavailable("pipeline", exc)

    _sessions.record(session_id, result.transcript.text, result.reply_text)
    payload = {
        "transcript": result.transcript.text,
        "reply_text": result.reply_text,
        "model": result.llm.model,
        "provider": result.llm.provider,
        "usage": result.llm.usage.as_dict() if result.llm.usage else None,
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
