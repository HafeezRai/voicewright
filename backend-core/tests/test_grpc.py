"""gRPC VoiceGateway service — exercised over a real (ephemeral) port."""

from __future__ import annotations

import asyncio

import grpc

from app.config import get_settings
from app.gen import voice_gateway_pb2 as pb
from app.gen import voice_gateway_pb2_grpc as pb_grpc
from app.grpc.server import create_grpc_server


async def _run_grpc() -> None:
    server, port = await create_grpc_server(
        get_settings(),
        active_sessions_provider=lambda: 3,
        host="127.0.0.1",
        port=0,  # ephemeral: never clashes with a real service
    )
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = pb_grpc.VoiceGatewayStub(channel)

            health = await stub.Health(pb.HealthRequest())
            assert health.status == "ok"
            assert health.service == "backend-core"
            assert health.active_sessions == 3

            async def frames():
                for i in range(3):
                    yield pb.AudioFrame(
                        payload=b"\x00\x01\x02\x03",
                        sample_rate_hz=16000,
                        sequence=i,
                        session_id="s1",
                    )

            acks = [event.frame_ack async for event in stub.StreamAudio(frames())]
            assert [a.sequence for a in acks] == [0, 1, 2]
            assert all(a.session_id == "s1" for a in acks)
    finally:
        await server.stop(grace=0)


def test_grpc_health_and_stream_audio() -> None:
    asyncio.run(_run_grpc())
