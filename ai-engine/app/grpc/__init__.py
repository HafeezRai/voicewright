"""gRPC serving for the VoiceGateway contract."""

from __future__ import annotations

from app.grpc.server import AIVoiceServicer, create_grpc_server, start_grpc_server

__all__ = ["AIVoiceServicer", "create_grpc_server", "start_grpc_server"]
