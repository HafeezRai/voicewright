"""The gRPC VoiceGateway servicer, exercised over a real ephemeral port.

The same ``voice_gateway.proto`` backend-core serves is served here, but this
servicer does the real work: it buffers the ingress frames, transcribes them,
classifies the transcript and streams synthesised reply audio back.
"""

from __future__ import annotations

import asyncio

import grpc
import numpy as np

from app.audio_io import load_wav, to_pcm16_bytes
from app.config import get_settings
from app.gen import voice_gateway_pb2 as pb
from app.gen import voice_gateway_pb2_grpc as pb_grpc
from app.grpc.server import create_grpc_server


def _run(pipeline_engine, speech_wav) -> None:
    audio, rate = load_wav(str(speech_wav))
    pcm = to_pcm16_bytes(audio)
    # Split into 32 ms (512-sample) frames, the way the gateway sends them.
    frame_bytes = 512 * 2

    async def _main() -> None:
        server, port = await create_grpc_server(
            get_settings(), pipeline_engine, host="127.0.0.1", port=0
        )
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                stub = pb_grpc.VoiceGatewayStub(channel)

                health = await stub.Health(pb.HealthRequest())
                assert health.status == "ok"
                assert health.service == "ai-engine"

                async def frames():
                    for i, start in enumerate(range(0, len(pcm), frame_bytes)):
                        yield pb.AudioFrame(
                            payload=pcm[start : start + frame_bytes],
                            sample_rate_hz=rate,
                            sequence=i,
                            session_id="test-session",
                        )

                events = [event async for event in stub.StreamAudio(frames())]

            finals = [e for e in events if e.WhichOneof("event") == "transcript_final"]
            chunks = [e for e in events if e.WhichOneof("event") == "synthesis_chunk"]

            assert len(finals) == 1, f"expected one final transcript, got {len(finals)}"
            assert finals[0].transcript_final.text.strip()

            assert chunks, "no synthesis audio streamed back"
            reply = b"".join(c.synthesis_chunk.audio for c in chunks)
            assert len(reply) > 16000, "reply audio is implausibly short"
            samples = np.frombuffer(reply, dtype="<i2")
            assert np.any(samples != 0), "reply audio is silent"
        finally:
            await server.stop(grace=0)

    asyncio.run(_main())


def test_grpc_stream_transcribes_and_synthesises(pipeline_engine, speech_wav) -> None:
    _run(pipeline_engine, speech_wav)


def test_grpc_health_counts_frames_after_a_stream(pipeline_engine, speech_wav) -> None:
    """Health reflects the frames the servicer has ingested."""

    async def _main() -> None:
        pipeline = pipeline_engine
        from app.grpc.server import AIVoiceServicer

        servicer = AIVoiceServicer(pipeline, service_name="ai-engine")
        server = grpc.aio.server()
        pb_grpc.add_VoiceGatewayServicer_to_server(servicer, server)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        try:
            audio, rate = load_wav(str(speech_wav))
            pcm = to_pcm16_bytes(audio)[: 512 * 2 * 4]  # just 4 frames

            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                stub = pb_grpc.VoiceGatewayStub(channel)

                async def frames():
                    for i in range(4):
                        yield pb.AudioFrame(
                            payload=pcm[i * 1024 : (i + 1) * 1024],
                            sample_rate_hz=rate,
                            sequence=i,
                            session_id="s",
                        )

                async for _ in stub.StreamAudio(frames()):
                    pass

                health = await stub.Health(pb.HealthRequest())
                assert health.vad_frames_processed == 4
                assert health.active_sessions == 0
        finally:
            await server.stop(grace=0)

    asyncio.run(_main())
