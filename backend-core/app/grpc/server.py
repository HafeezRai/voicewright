"""gRPC gateway service.

Phase 2 implements the *transport* half of the contract:

* ``StreamAudio`` — accepts the bidirectional audio stream and acknowledges each
  ingress frame with a :class:`FrameAck`. It deliberately does **not** transcribe
  or synthesise: interpreting audio is Phase 3 (ai-engine) work, and faking a
  transcript here would be a lie in the wire format.
* ``Health`` — status/load probe backed by the live session registry and
  Prometheus counters.

``grpc.aio`` is used throughout so the servicer coexists with the FastAPI event
loop in a single process.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable

import grpc
from prometheus_client import Counter

from app.config import Settings
from app.gen import voice_gateway_pb2 as pb
from app.gen import voice_gateway_pb2_grpc as pb_grpc
from app.metrics import VAD_FRAMES_PROCESSED

logger = logging.getLogger(__name__)


def _counter_total(counter: Counter) -> int:
    """Sum every labelled sample series of a Prometheus counter."""
    total = 0
    for metric in counter.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total"):
                total += int(sample.value)
    return total


class VoiceGatewayServicer(pb_grpc.VoiceGatewayServicer):
    """Minimal servable implementation of the VoiceGateway contract."""

    def __init__(
        self,
        *,
        service_name: str = "backend-core",
        active_sessions_provider: Callable[[], int] | None = None,
    ):
        self._service_name = service_name
        self._active_sessions_provider = active_sessions_provider or (lambda: 0)

    async def StreamAudio(
        self,
        request_iterator: AsyncIterator[pb.AudioFrame],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[pb.VoiceEvent]:
        """Acknowledge each ingress audio frame; no STT/TTS here (Phase 3)."""
        count = 0
        async for frame in request_iterator:
            count += 1
            yield pb.VoiceEvent(
                frame_ack=pb.FrameAck(sequence=frame.sequence, session_id=frame.session_id)
            )
        logger.debug("StreamAudio completed after %d frames", count)

    async def Health(
        self, request: pb.HealthRequest, context: grpc.aio.ServicerContext
    ) -> pb.HealthResponse:
        return pb.HealthResponse(
            status="ok",
            service=self._service_name,
            active_sessions=int(self._active_sessions_provider()),
            vad_frames_processed=_counter_total(VAD_FRAMES_PROCESSED),
        )


async def create_grpc_server(
    settings: Settings,
    *,
    active_sessions_provider: Callable[[], int] | None = None,
    host: str = "0.0.0.0",
    port: int | None = None,
) -> tuple[grpc.aio.Server, int]:
    """Build (but do not start) an aio gRPC server. Returns (server, bound_port).

    ``port=0`` binds an ephemeral port — used by tests to avoid clashing.
    """
    server = grpc.aio.server()
    servicer = VoiceGatewayServicer(
        service_name=settings.service_name,
        active_sessions_provider=active_sessions_provider,
    )
    pb_grpc.add_VoiceGatewayServicer_to_server(servicer, server)
    bound_port = server.add_insecure_port(f"{host}:{port if port is not None else settings.grpc_port}")
    if bound_port == 0:  # pragma: no cover - bind failure
        raise RuntimeError("failed to bind gRPC port")
    return server, bound_port


async def start_grpc_server(
    settings: Settings,
    *,
    active_sessions_provider: Callable[[], int] | None = None,
) -> tuple[grpc.aio.Server, int]:
    """Create and start the gRPC server. Returns (server, bound_port)."""
    server, bound_port = await create_grpc_server(
        settings, active_sessions_provider=active_sessions_provider
    )
    await server.start()
    logger.info("gRPC VoiceGateway listening on :%d", bound_port)
    return server, bound_port
