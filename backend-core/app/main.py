"""FastAPI application entrypoint for backend-core.

Phase 2 adds the WebRTC gateway wiring: a WebSocket signalling endpoint, the
per-session audio pipeline (VAD -> barge-in), Prometheus metrics, and an
optional gRPC server started alongside the HTTP app.

The ``GET /health`` contract is frozen from Phase 1 and is unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import grpc
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.audio.vad import build_vad
from app.config import Settings, get_settings
from app.grpc.server import start_grpc_server
from app.ws.manager import SessionManager
from app.ws.signaling import router as signaling_router

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)


def _vad_factory(config: Settings):
    """Build a fresh (stateful) VAD per session from configuration."""
    return lambda: build_vad(
        config.vad_backend,
        model_path=config.vad_model_path or None,
        threshold=config.vad_threshold,
        sample_rate=config.audio_sample_rate,
        frame_samples=config.vad_frame_samples,
    )


#: Process-wide registry of live voice sessions.
session_manager = SessionManager(
    _vad_factory(settings),
    frame_samples=settings.vad_frame_samples,
    frame_duration_ms=settings.frame_duration_ms,
    bargein_enabled=settings.bargein_enabled,
    min_speech_ms=settings.bargein_min_speech_ms,
    min_silence_ms=settings.bargein_min_silence_ms,
    ice_servers=settings.ice_servers,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start the gRPC server alongside HTTP; shut everything down cleanly."""
    grpc_server: grpc.aio.Server | None = None
    if settings.grpc_enabled:
        try:
            grpc_server, port = await start_grpc_server(
                settings,
                active_sessions_provider=lambda: session_manager.active_count,
            )
            logger.info("gRPC server bound on port %s", port)
        except Exception:  # pragma: no cover - port clash etc. must not kill HTTP
            logger.exception("gRPC server failed to start; HTTP continues")
            grpc_server = None
    try:
        yield
    finally:
        await session_manager.close_all()
        if grpc_server is not None:
            await grpc_server.stop(grace=1.0)


app = FastAPI(
    title="CloudPSO Voice Agent — backend-core",
    version="0.2.0",
    description="WebRTC gateway, WebSocket management and barge-in (Phase 2).",
    lifespan=lifespan,
)

app.state.session_manager = session_manager
app.include_router(signaling_router)

# CORS for the Phase 4 browser dashboard. Additive and off the frozen contract:
# origins come from CLOUDEPSO_CORS_ALLOW_ORIGINS and default to the local
# dashboard only (never "*"). WebSockets are not subject to CORS, so /ws/signaling
# is unaffected.
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
