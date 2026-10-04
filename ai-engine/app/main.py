"""FastAPI application entrypoint for ai-engine.

Phase 1: a health endpoint only. The STT -> NLU -> TTS pipeline is Phase 3 work.

Deliberately free of ML imports: `torch`, `transformers` and `onnxruntime` are
declared as dependencies but never imported here, so the service (and its tests)
start without the multi-GB model stack present.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app.config import get_settings

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(
    title="CloudPSO Voice Agent — ai-engine",
    version="0.1.0",
    description="STT -> NLU -> TTS pipeline service (Phase 3).",
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness/readiness probe. Contract fixed at Phase 1."""
    return {"status": "ok", "service": settings.service_name}


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
