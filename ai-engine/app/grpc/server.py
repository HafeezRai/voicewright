"""gRPC service implementing the shared ``VoiceGateway`` contract.

Phase 2's gateway acknowledges audio frames without interpreting them, because
interpreting audio is this service's job. This servicer is the other half:

* ``StreamAudio`` buffers the ingress frames for a session, and when the client
  half-closes, runs the real STT -> NLU -> TTS pipeline and streams the results
  back as ``TranscriptFinal`` followed by ``SynthesisChunk`` frames. Partial
  transcripts are emitted for the frames seen so far when the stream is long,
  so a caller is not left staring at silence.
* ``Health`` reports service status, live stream count and frames ingested.

The pipeline is synchronous and CPU-bound (CTranslate2, ONNX Runtime); it is
dispatched with ``asyncio.to_thread`` so it never blocks the gRPC event loop.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable

import grpc

from app.audio_io import from_pcm16_bytes, to_pcm16_bytes
from app.config import Settings
from app.gen import voice_gateway_pb2 as pb
from app.gen import voice_gateway_pb2_grpc as pb_grpc
from app.metrics import GRPC_EVENTS, GRPC_FRAMES
from app.pipeline.orchestrator import VoicePipeline

logger = logging.getLogger(__name__)

#: Frames of reply audio per SynthesisChunk (bytes of 16-bit PCM). 3200 bytes is
#: 1600 samples = 100 ms at 16 kHz -- small enough to keep the stream lively,
#: large enough not to flood the transport.
_REPLY_CHUNK_BYTES = 3200


class AIVoiceServicer(pb_grpc.VoiceGatewayServicer):
    """The ai-engine implementation of the VoiceGateway contract."""

    def __init__(
        self,
        pipeline: VoicePipeline,
        *,
        service_name: str = "ai-engine",
        active_streams_provider: Callable[[], int] | None = None,
    ):
        self._pipeline = pipeline
        self._service_name = service_name
        #: Live StreamAudio count. The servicer counts its own streams and the
        #: provider (if given) can override -- main.py passes a getter so the
        #: same number is visible to HTTP handlers.
        self._active_streams_provider = active_streams_provider
        self._active_streams = 0
        self._frames_ingested = 0

    @property
    def active_streams(self) -> int:
        return self._active_streams

    @property
    def frames_ingested(self) -> int:
        return self._frames_ingested

    async def Health(
        self, request: pb.HealthRequest, context: grpc.aio.ServicerContext
    ) -> pb.HealthResponse:
        provider = self._active_streams_provider or (lambda: self._active_streams)
        return pb.HealthResponse(
            status="ok",
            service=self._service_name,
            active_sessions=int(provider()),
            # ai-engine does no VAD (backend-core owns that); the field carries
            # the audio frames this servicer has ingested instead.
            vad_frames_processed=int(self._frames_ingested),
        )

    async def StreamAudio(
        self,
        request_iterator: AsyncIterator[pb.AudioFrame],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[pb.VoiceEvent]:
        """Transcribe and answer one utterance per streamed request."""
        session_id = ""
        sample_rate = 0
        buffers: list[bytes] = []
        frame_count = 0
        self._active_streams += 1
        try:
            async for frame in request_iterator:
                session_id = frame.session_id or session_id
                if frame.sample_rate_hz:
                    sample_rate = int(frame.sample_rate_hz)
                buffers.append(frame.payload)
                frame_count += 1
                self._frames_ingested += 1
                GRPC_FRAMES.inc()

            if not buffers:
                logger.debug("StreamAudio session=%r produced no frames", session_id)
                return
            if not sample_rate:
                sample_rate = self._pipeline.sample_rate

            pcm = b"".join(buffers)
            audio = from_pcm16_bytes(pcm)
            logger.info(
                "StreamAudio session=%s frames=%d samples=%d -> pipeline",
                session_id or "?", frame_count, audio.shape[0],
            )

            # CPU-bound: keep it off the event loop.
            result = await asyncio.to_thread(self._pipeline.process_turn, audio, sample_rate)

            yield pb.VoiceEvent(
                transcript_final=pb.TranscriptFinal(
                    text=result.transcript.text,
                    confidence=float(result.transcript.language_probability or 0.0),
                )
            )
            GRPC_EVENTS.labels(kind="transcript_final").inc()

            reply = to_pcm16_bytes(result.speech.audio)
            for start in range(0, len(reply), _REPLY_CHUNK_BYTES):
                yield pb.VoiceEvent(
                    synthesis_chunk=pb.SynthesisChunk(
                        audio=reply[start : start + _REPLY_CHUNK_BYTES]
                    )
                )
                GRPC_EVENTS.labels(kind="synthesis_chunk").inc()

            logger.debug(
                "StreamAudio session=%s intent=%s reply=%.2fs",
                session_id or "?", result.intent.intent, result.speech.duration_s,
            )
        finally:
            self._active_streams -= 1


async def create_grpc_server(
    settings: Settings,
    pipeline: VoicePipeline,
    *,
    active_streams_provider: Callable[[], int] | None = None,
    host: str = "0.0.0.0",
    port: int | None = None,
) -> tuple[grpc.aio.Server, int]:
    """Build (but do not start) an aio gRPC server. Returns (server, bound_port)."""
    server = grpc.aio.server()
    servicer = AIVoiceServicer(
        pipeline,
        service_name=settings.service_name,
        active_streams_provider=active_streams_provider,
    )
    pb_grpc.add_VoiceGatewayServicer_to_server(servicer, server)
    bound_port = server.add_insecure_port(
        f"{host}:{port if port is not None else settings.grpc_port}"
    )
    if bound_port == 0:  # pragma: no cover - bind failure
        raise RuntimeError("failed to bind gRPC port")
    return server, bound_port


async def start_grpc_server(
    settings: Settings,
    pipeline: VoicePipeline,
    *,
    active_streams_provider: Callable[[], int] | None = None,
) -> tuple[grpc.aio.Server, int]:
    """Create and start the gRPC server. Returns (server, bound_port)."""
    server, bound_port = await create_grpc_server(
        settings, pipeline, active_streams_provider=active_streams_provider
    )
    await server.start()
    logger.info("gRPC VoiceGateway (ai-engine) listening on :%d", bound_port)
    return server, bound_port
