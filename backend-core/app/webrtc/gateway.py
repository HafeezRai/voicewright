"""WebRTC gateway — one aiortc ``RTCPeerConnection`` per voice session.

Responsibilities:

* SDP offer/answer exchange (:meth:`WebRTCSession.handle_offer`).
* Trickle-ICE: remote candidates in, local candidates out via a callback.
* Consume the remote audio track (the user's microphone), normalise it to
  16 kHz mono float32, re-chunk it to a fixed frame size and hand each frame to
  ``on_audio_frame``. VAD/barge-in live *above* this layer — the gateway does
  not know what a frame means, only how to move audio.
* Lifecycle and leak-free teardown (:meth:`close`).

The gateway is deliberately importable and drivable without a running server:
the integration test builds two peer connections in one process against this
class.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from enum import Enum

import av
import numpy as np
from aiortc import (
    RTCConfiguration,
    RTCIceCandidate,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)

logger = logging.getLogger(__name__)

#: Called with one frame of mono float32 audio at the target sample rate.
AudioFrameCallback = Callable[[np.ndarray], None]
#: Called with a serialisable local ICE candidate dict.
IceCandidateCallback = Callable[[dict], None]
#: Called whenever the peer connection state changes.
StateChangeCallback = Callable[["WebRTCSessionState"], None]


class WebRTCSessionState(str, Enum):
    NEW = "new"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"
    DISCONNECTED = "disconnected"
    CLOSED = "closed"


def _candidate_to_dict(candidate: RTCIceCandidate) -> dict:
    """Serialise an aiortc candidate, tolerating the sdpMid/sdp_mid rename."""
    sdp_mid = getattr(candidate, "sdpMid", None)
    if sdp_mid is None:
        sdp_mid = getattr(candidate, "sdp_mid", None)
    mline = getattr(candidate, "sdpMLineIndex", None)
    if mline is None:
        mline = getattr(candidate, "sdp_mline_index", None)
    return {"candidate": candidate.candidate, "sdpMid": sdp_mid, "sdpMLineIndex": mline}


class WebRTCSession:
    """A single peer connection plus its audio-consuming task."""

    def __init__(
        self,
        session_id: str,
        *,
        on_audio_frame: AudioFrameCallback | None = None,
        on_ice_candidate: IceCandidateCallback | None = None,
        on_state_change: StateChangeCallback | None = None,
        sample_rate: int = 16000,
        frame_samples: int = 512,
        ice_servers: list[dict] | None = None,
    ):
        self.id = session_id
        self._on_audio_frame = on_audio_frame
        self._on_ice_candidate = on_ice_candidate
        self._on_state_change = on_state_change
        self._sample_rate = sample_rate
        self._frame_samples = frame_samples

        config = RTCConfiguration(
            iceServers=[RTCIceServer(**server) for server in (ice_servers or [])]
        )
        self.pc = RTCPeerConnection(configuration=config)
        self._state = WebRTCSessionState.NEW
        self._audio_tasks: set[asyncio.Task] = set()
        self._resamplers: dict[str, av.AudioResampler] = {}
        self._buffers: dict[str, np.ndarray] = {}
        self._remote_audio_tracks = 0

        self.pc.on("icecandidate")(self._handle_local_candidate)
        self.pc.on("connectionstatechange")(self._handle_connection_state)
        self.pc.on("track")(self._handle_track)

    # -- properties ----------------------------------------------------------
    @property
    def state(self) -> WebRTCSessionState:
        return self._state

    @property
    def remote_audio_tracks(self) -> int:
        """How many remote audio tracks this session has seen (test/diagnostics)."""
        return self._remote_audio_tracks

    @property
    def connection_state(self) -> str:
        """Raw aiortc connection state string."""
        return self.pc.connectionState

    # -- signalling ----------------------------------------------------------
    async def handle_offer(self, sdp: str, type_: str = "offer") -> RTCSessionDescription:
        """Apply a remote offer and return the local answer."""
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type_))
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        # aiortc exposes the negotiated local description; fall back to the raw
        # answer if it is not yet populated.
        return self.pc.localDescription or answer

    async def add_ice_candidate(self, candidate: dict | None) -> None:
        """Add a remote ICE candidate. ``None``/empty signals end-of-candidates."""
        if not candidate or not candidate.get("candidate"):
            return
        await self.pc.addIceCandidate(
            RTCIceCandidate(
                candidate=candidate["candidate"],
                sdpMid=candidate.get("sdpMid"),
                sdpMLineIndex=candidate.get("sdpMLineIndex"),
            )
        )

    async def close(self) -> None:
        """Close the connection and cancel every consumer task (no leaks)."""
        for task in list(self._audio_tasks):
            task.cancel()
        if self._audio_tasks:
            await asyncio.gather(*self._audio_tasks, return_exceptions=True)
        self._audio_tasks.clear()
        self._resamplers.clear()
        self._buffers.clear()
        await self.pc.close()
        self._set_state(WebRTCSessionState.CLOSED)

    # -- aiortc callbacks ----------------------------------------------------
    def _handle_local_candidate(self, candidate: RTCIceCandidate | None) -> None:
        if candidate is None or self._on_ice_candidate is None:
            return
        self._on_ice_candidate(_candidate_to_dict(candidate))

    def _handle_connection_state(self) -> None:
        mapping = {
            "new": WebRTCSessionState.NEW,
            "connecting": WebRTCSessionState.CONNECTING,
            "connected": WebRTCSessionState.CONNECTED,
            "failed": WebRTCSessionState.FAILED,
            "disconnected": WebRTCSessionState.DISCONNECTED,
            "closed": WebRTCSessionState.CLOSED,
        }
        self._set_state(mapping.get(self.pc.connectionState, WebRTCSessionState.NEW))

    def _set_state(self, state: WebRTCSessionState) -> None:
        if state is self._state:
            return
        self._state = state
        if self._on_state_change is not None:
            self._on_state_change(state)

    def _handle_track(self, track) -> None:
        if getattr(track, "kind", None) != "audio":
            logger.debug("ignoring non-audio track %r", getattr(track, "kind", None))
            return
        self._remote_audio_tracks += 1
        task = asyncio.ensure_future(self._consume_audio(track))
        self._audio_tasks.add(task)
        task.add_done_callback(self._audio_tasks.discard)

    # -- audio consumption ---------------------------------------------------
    async def _consume_audio(self, track) -> None:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=self._sample_rate)
        self._resamplers[track.id] = resampler
        self._buffers[track.id] = np.zeros(0, dtype=np.float32)
        try:
            while True:
                frame = await track.recv()
                pcm = self._to_mono_float(frame, resampler)
                if pcm.size == 0:
                    continue
                self._dispatch(track.id, pcm)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - network/track teardown races
            logger.debug("audio track consumer for %s ended", track.id, exc_info=True)

    @staticmethod
    def _to_mono_float(frame: av.AudioFrame, resampler: av.AudioResampler) -> np.ndarray:
        out = resampler.resample(frame)
        if out is None:
            return np.zeros(0, dtype=np.float32)
        if isinstance(out, av.AudioFrame):
            out = [out]
        chunks = []
        for f in out:
            arr = f.to_ndarray()
            # mono s16 -> shape (1, n); be defensive about layout.
            if arr.ndim == 2 and arr.shape[0] > 1:
                arr = arr.astype(np.float32).mean(axis=0)
            else:
                arr = arr.reshape(-1).astype(np.float32)
            chunks.append(arr / 32768.0)
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def _dispatch(self, track_id: str, pcm: np.ndarray) -> None:
        """Re-chunk the stream into exactly ``frame_samples`` and emit frames."""
        buffer = np.concatenate([self._buffers.get(track_id, np.zeros(0, np.float32)), pcm])
        while buffer.size >= self._frame_samples:
            frame = buffer[: self._frame_samples]
            buffer = buffer[self._frame_samples :]
            if self._on_audio_frame is not None:
                self._on_audio_frame(frame)
        self._buffers[track_id] = buffer
