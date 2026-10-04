"""Real in-process WebRTC integration test.

Two aiortc peer connections in one process complete a genuine SDP offer/answer
handshake through :class:`WebRTCSession` (the gateway). The test then proves
data actually flowed in both directions:

* **media** — the offerer sends a 440 Hz tone; the gateway's ``on_audio_frame``
  callback receives real 512-sample, 16 kHz frames with non-zero energy.
* **data** — a WebRTC data channel carries ``ping-from-offerer`` up, and the
  gateway replies ``pong``.

Nothing is mocked; this is the evidence that the gateway works without Docker.
"""

from __future__ import annotations

import asyncio
from fractions import Fraction

import av
import numpy as np
import pytest
from aiortc import RTCIceCandidate, RTCPeerConnection
from aiortc.mediastreams import MediaStreamTrack

from app.webrtc.gateway import WebRTCSession, WebRTCSessionState

SAMPLE_RATE = 48000
SAMPLES = 480  # 10 ms @ 48 kHz


class ToneTrack(MediaStreamTrack):
    """A synthetic audio source: a steady sine tone, generated on demand."""

    kind = "audio"

    def __init__(self, freq: float = 440.0):
        super().__init__()
        self._freq = freq
        self._index = 0

    async def recv(self) -> av.AudioFrame:
        frame = av.AudioFrame(format="s16", layout="mono", samples=SAMPLES)
        frame.sample_rate = SAMPLE_RATE
        t = (self._index + np.arange(SAMPLES)) / SAMPLE_RATE
        data = (np.sin(2 * np.pi * self._freq * t) * 0.5 * 32767).astype(np.int16)
        frame.planes[0].update(data.tobytes())
        frame.pts = self._index
        frame.time_base = Fraction(1, SAMPLE_RATE)
        self._index += SAMPLES
        await asyncio.sleep(0)
        return frame


async def _wait_until(predicate, timeout: float = 15.0, interval: float = 0.02) -> None:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition not met within timeout")


def _candidate_dict(candidate) -> dict:
    sdp_mid = getattr(candidate, "sdpMid", None)
    if sdp_mid is None:
        sdp_mid = getattr(candidate, "sdp_mid", None)
    return {
        "candidate": candidate.candidate,
        "sdpMid": sdp_mid,
        "sdpMLineIndex": getattr(candidate, "sdpMLineIndex", None),
    }


async def _run_handshake() -> None:
    received_frames: list[np.ndarray] = []
    answerer_candidates: list[dict] = []
    inbound_datachannel_messages: list[str] = []
    offerer_pongs: list[str] = []

    gateway = WebRTCSession(
        "itest-session",
        on_audio_frame=received_frames.append,
        on_ice_candidate=answerer_candidates.append,
        sample_rate=16000,
        frame_samples=512,
    )

    offerer = RTCPeerConnection()
    offerer_candidate_queue: asyncio.Queue = asyncio.Queue()
    offerer.on("icecandidate")(
        lambda c: offerer_candidate_queue.put_nowait(c) if c is not None else None
    )

    @gateway.pc.on("datachannel")
    def _on_datachannel(channel) -> None:
        @channel.on("message")
        def _on_message(message) -> None:
            inbound_datachannel_messages.append(message)
            channel.send("pong")

    offerer.addTrack(ToneTrack())
    data_channel = offerer.createDataChannel("chat")

    @data_channel.on("message")
    def _on_pong(message) -> None:
        offerer_pongs.append(message)

    # --- genuine offer/answer through the gateway ---------------------------
    await offerer.setLocalDescription(await offerer.createOffer())
    answer = await gateway.handle_offer(
        offerer.localDescription.sdp, offerer.localDescription.type
    )
    assert answer.type == "answer"
    assert "m=audio" in answer.sdp
    await offerer.setRemoteDescription(answer)

    # --- trickle ICE both directions ----------------------------------------
    async def pump_offerer_candidates() -> None:
        while True:
            candidate = await offerer_candidate_queue.get()
            await gateway.add_ice_candidate(_candidate_dict(candidate))

    async def pump_answerer_candidates() -> None:
        while True:
            while answerer_candidates:
                candidate = answerer_candidates.pop(0)
                await offerer.addIceCandidate(
                    RTCIceCandidate(
                        candidate=candidate["candidate"],
                        sdpMid=candidate["sdpMid"],
                        sdpMLineIndex=candidate["sdpMLineIndex"],
                    )
                )
            await asyncio.sleep(0.01)

    pump_in = asyncio.create_task(pump_offerer_candidates())
    pump_out = asyncio.create_task(pump_answerer_candidates())

    try:
        await _wait_until(lambda: gateway.connection_state == "connected")
        assert gateway.state == WebRTCSessionState.CONNECTED
        assert gateway.remote_audio_tracks == 1

        # --- data channel payload, both directions --------------------------
        await _wait_until(lambda: data_channel.readyState == "open")
        data_channel.send("ping-from-offerer")
        await _wait_until(lambda: offerer_pongs)
        assert inbound_datachannel_messages == ["ping-from-offerer"]
        assert offerer_pongs == ["pong"]

        # --- media payload --------------------------------------------------
        await _wait_until(lambda: len(received_frames) >= 5)
        for frame in received_frames[:5]:
            assert frame.shape == (512,)
        energy = float(np.sqrt(np.mean(np.square(received_frames[0]))))
        assert energy > 0.01, f"expected an audible tone, got RMS {energy}"
    finally:
        pump_in.cancel()
        pump_out.cancel()
        await gateway.close()
        await offerer.close()

    assert gateway.state == WebRTCSessionState.CLOSED
    assert gateway.pc.connectionState == "closed"


def test_offer_answer_handshake_media_and_datachannel() -> None:
    asyncio.run(_run_handshake())


def test_close_is_idempotent() -> None:
    async def _run() -> None:
        gateway = WebRTCSession("close-test")
        await gateway.close()
        await gateway.close()  # must not raise
        assert gateway.state == WebRTCSessionState.CLOSED

    asyncio.run(_run())
