"""Per-session state and the process-wide session registry.

A :class:`VoiceSession` owns exactly one :class:`~app.webrtc.gateway.WebRTCSession`
plus the per-session audio pipeline: VAD -> barge-in -> playback sink. The
:class:`SessionManager` creates, tracks and — critically — *tears down* those
sessions so that a dropped WebSocket never leaks a peer connection or an
audio-consumer task.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable

from app.audio.vad import VAD
from app.metrics import (
    ACTIVE_WEBRTC_SESSIONS,
    BARGEIN_EVENTS,
    TOTAL_WEBRTC_SESSIONS,
    VAD_FRAMES_PROCESSED,
)
from app.pipeline.bargein import (
    BargeInController,
    BargeInEvent,
    BargeInEventKind,
    StubPlaybackSink,
)
from app.webrtc.gateway import WebRTCSession, WebRTCSessionState

logger = logging.getLogger(__name__)

VADFactory = Callable[[], VAD]


class VoiceSession:
    """One call: a peer connection, a VAD, a barge-in controller and a sink."""

    def __init__(
        self,
        session_id: str,
        *,
        vad: VAD,
        frame_samples: int,
        frame_duration_ms: int,
        bargein_enabled: bool = True,
        min_speech_ms: int = 200,
        min_silence_ms: int = 400,
        ice_servers: list[dict] | None = None,
        on_ice_candidate: Callable[[dict], None] | None = None,
        on_state_change: Callable[[WebRTCSessionState], None] | None = None,
        on_barge_in: Callable[[BargeInEvent], None] | None = None,
    ):
        self.id = session_id
        self.vad = vad
        self.sink = StubPlaybackSink()
        self.bargein = BargeInController(
            self.sink,
            frame_duration_ms=frame_duration_ms,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            enabled=bargein_enabled,
        )
        self._on_barge_in = on_barge_in
        self._closed = False
        self.gateway = WebRTCSession(
            session_id,
            on_audio_frame=self._on_audio_frame,
            on_ice_candidate=on_ice_candidate,
            on_state_change=on_state_change,
            sample_rate=16000,
            frame_samples=frame_samples,
            ice_servers=ice_servers,
        )

    # -- pipeline ------------------------------------------------------------
    def _on_audio_frame(self, frame) -> None:
        """VAD the frame, then advance barge-in. Runs on the event loop thread."""
        speech = self.vad.is_speech(frame)
        VAD_FRAMES_PROCESSED.labels(backend=self.vad.backend_name, speech=str(speech)).inc()
        for event in self.bargein.process(speech):
            # Count the interrupt exactly where the controller fires it: once
            # per BARGE_IN event (never per speech frame, never for SPEECH_END).
            if event.kind is BargeInEventKind.BARGE_IN:
                BARGEIN_EVENTS.inc()
            if self._on_barge_in is not None:
                self._on_barge_in(event)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.gateway.close()


class SessionManager:
    """Registry of live :class:`VoiceSession` objects."""

    def __init__(
        self,
        vad_factory: VADFactory,
        *,
        frame_samples: int = 512,
        frame_duration_ms: int = 32,
        bargein_enabled: bool = True,
        min_speech_ms: int = 200,
        min_silence_ms: int = 400,
        ice_servers: list[dict] | None = None,
    ):
        self._vad_factory = vad_factory
        self._frame_samples = frame_samples
        self._frame_duration_ms = frame_duration_ms
        self._bargein_enabled = bargein_enabled
        self._min_speech_ms = min_speech_ms
        self._min_silence_ms = min_silence_ms
        self._ice_servers = ice_servers or []
        self._sessions: dict[str, VoiceSession] = {}
        self._lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    def get(self, session_id: str) -> VoiceSession | None:
        return self._sessions.get(session_id)

    async def create(
        self,
        *,
        session_id: str | None = None,
        on_ice_candidate: Callable[[dict], None] | None = None,
        on_state_change: Callable[[WebRTCSessionState], None] | None = None,
        on_barge_in: Callable[[BargeInEvent], None] | None = None,
    ) -> VoiceSession:
        sid = session_id or uuid.uuid4().hex
        async with self._lock:
            if sid in self._sessions:
                raise ValueError(f"session {sid!r} already exists")
            session = VoiceSession(
                sid,
                vad=self._vad_factory(),
                frame_samples=self._frame_samples,
                frame_duration_ms=self._frame_duration_ms,
                bargein_enabled=self._bargein_enabled,
                min_speech_ms=self._min_speech_ms,
                min_silence_ms=self._min_silence_ms,
                ice_servers=self._ice_servers,
                on_ice_candidate=on_ice_candidate,
                on_state_change=on_state_change,
                on_barge_in=on_barge_in,
            )
            self._sessions[sid] = session
        ACTIVE_WEBRTC_SESSIONS.set(len(self._sessions))
        TOTAL_WEBRTC_SESSIONS.inc()
        logger.info("created session %s (active=%d)", sid, len(self._sessions))
        return session

    async def destroy(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return
        try:
            await session.close()
        finally:
            ACTIVE_WEBRTC_SESSIONS.set(len(self._sessions))
            logger.info("destroyed session %s (active=%d)", session_id, len(self._sessions))

    async def close_all(self) -> None:
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        await asyncio.gather(*(s.close() for s in sessions), return_exceptions=True)
        ACTIVE_WEBRTC_SESSIONS.set(0)
