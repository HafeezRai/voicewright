"""WebSocket signalling endpoint and session lifecycle."""

from __future__ import annotations

import asyncio
import time

from aiortc import RTCPeerConnection
from fastapi.testclient import TestClient

from app.main import app, session_manager

client = TestClient(app)


def _make_offer_sdp() -> str:
    """Build a real SDP offer string (recvonly audio) for the gateway to answer."""

    async def _generate() -> str:
        pc = RTCPeerConnection()
        pc.addTransceiver("audio", direction="recvonly")
        await pc.setLocalDescription(await pc.createOffer())
        assert pc.localDescription is not None
        sdp = pc.localDescription.sdp
        await pc.close()
        return sdp

    return asyncio.run(_generate())


def _wait_sync(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_ping_pong() -> None:
    with client.websocket_connect("/ws/signaling") as ws:
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}


def test_unknown_message_type_returns_error() -> None:
    with client.websocket_connect("/ws/signaling") as ws:
        ws.send_json({"type": "nonsense"})
        message = ws.receive_json()
        assert message["type"] == "error"
        assert "nonsense" in message["message"]


def test_ice_candidate_before_offer_is_rejected() -> None:
    with client.websocket_connect("/ws/signaling") as ws:
        ws.send_json({"type": "ice-candidate", "candidate": {"candidate": "x"}})
        message = ws.receive_json()
        assert message["type"] == "error"


def test_offer_returns_answer_and_registers_session() -> None:
    offer_sdp = _make_offer_sdp()
    before = session_manager.active_count

    with client.websocket_connect("/ws/signaling") as ws:
        ws.send_json({"type": "offer", "sdp": offer_sdp})
        message = ws.receive_json()
        assert message["type"] == "answer"
        assert message["session_id"]
        assert "m=audio" in message["sdp"]
        session_id = message["session_id"]
        assert session_manager.get(session_id) is not None
        assert session_manager.active_count == before + 1

    # Closing the socket must tear the session down (no leak).
    assert _wait_sync(lambda: session_manager.get(session_id) is None)
    assert _wait_sync(lambda: session_manager.active_count == before)


def test_playback_control_before_offer_is_rejected() -> None:
    with client.websocket_connect("/ws/signaling") as ws:
        ws.send_json({"type": "playback", "state": "start"})
        assert ws.receive_json()["type"] == "error"
