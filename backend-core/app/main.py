"""FastAPI application entrypoint for backend-core.

Phase 1: a health endpoint only. The WebRTC gateway, WebSocket manager and
barge-in controller are Phase 2 work and deliberately absent.
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
    title="CloudPSO Voice Agent — backend-core",
    version="0.1.0",
    description="WebRTC gateway, WebSocket management and barge-in (Phase 2).",
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
