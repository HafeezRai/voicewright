"""WebSocket signalling endpoint.

Transport for everything that is not media: SDP offer/answer, trickle-ICE, and
out-of-band control such as barge-in notifications. Media itself flows over the
WebRTC peer connection established by this exchange.

Wire protocol (JSON, one object per frame)::

    client -> server
      {"type": "offer",   "sdp": "<sdp>", "session_id": "<optional>"}
      {"type": "ice-candidate", "candidate": {"candidate": ..., "sdpMid": ..., "sdpMLineIndex": ...}}
      {"type": "playback", "state": "start" | "stop"}   # drives the STUB sink
      {"type": "close"}
      {"type": "ping"}

    server -> client
      {"type": "ready", "session_id": "<id>"}
      {"type": "answer", "sdp": "<sdp>", "session_id": "<id>"}
      {"type": "ice-candidate", "candidate": {...}}
      {"type": "state", "state": "connected" | ...}
      {"type": "barge-in", "duration_ms": <int>}
      {"type": "pong"}
      {"type": "error", "message": "<reason>"}

All sends are funnelled through a single writer task via a queue, because
aiortc callbacks fire from arbitrary points on the event loop and Starlette
does not support concurrent sends on one socket.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.metrics import WS_CONNECTIONS
from app.pipeline.bargein import BargeInEventKind
from app.ws.manager import SessionManager, VoiceSession

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/signaling")
async def signaling(websocket: WebSocket) -> None:
    """Accept a signalling connection and drive one voice session over it."""
    await websocket.accept()
    manager: SessionManager = websocket.app.state.session_manager

    send_queue: asyncio.Queue[dict] = asyncio.Queue()

    async def writer() -> None:
        while True:
            message = await send_queue.get()
            await websocket.send_json(message)

    writer_task = asyncio.create_task(writer())

    def emit(message: dict) -> None:
        """Enqueue a message for the writer task (safe from sync callbacks)."""
        send_queue.put_nowait(message)

    session: VoiceSession | None = None
    WS_CONNECTIONS.inc()
    try:
        while True:
            message = await websocket.receive_json()
            msg_type = message.get("type")

            if msg_type == "offer":
                if session is not None:
                    emit({"type": "error", "message": "session already established"})
                    continue
                session = await manager.create(
                    session_id=message.get("session_id"),
                    on_ice_candidate=lambda c: emit({"type": "ice-candidate", "candidate": c}),
                    on_state_change=lambda s: emit({"type": "state", "state": s.value}),
                    on_barge_in=_barge_in_emitter(emit),
                )
                answer = await session.gateway.handle_offer(message["sdp"])
                emit(
                    {
                        "type": "answer",
                        "sdp": answer.sdp,
                        "session_id": session.id,
                    }
                )

            elif msg_type == "ice-candidate":
                if session is None:
                    emit({"type": "error", "message": "no session; send offer first"})
                    continue
                await session.gateway.add_ice_candidate(message.get("candidate"))

            elif msg_type == "playback":
                # Controls the Phase 2 STUB sink only — Phase 3 owns real TTS.
                if session is None:
                    emit({"type": "error", "message": "no session; send offer first"})
                    continue
                if message.get("state") == "start":
                    session.sink.start()
                else:
                    session.sink.stop()

            elif msg_type == "ping":
                emit({"type": "pong"})

            elif msg_type == "close":
                break

            else:
                emit({"type": "error", "message": f"unknown message type: {msg_type!r}"})

    except WebSocketDisconnect:
        logger.debug("signalling websocket disconnected")
    except Exception:  # pragma: no cover - defensive: never leak a session
        logger.exception("signalling handler failed")
    finally:
        writer_task.cancel()
        try:
            await writer_task
        except (asyncio.CancelledError, RuntimeError):
            pass
        WS_CONNECTIONS.dec()
        if session is not None:
            await manager.destroy(session.id)


def _barge_in_emitter(emit):
    def _emit_barge_in(event) -> None:
        if event.kind is BargeInEventKind.BARGE_IN:
            emit({"type": "barge-in", "duration_ms": event.duration_ms})

    return _emit_barge_in
