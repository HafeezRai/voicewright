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
from aiortc import RTCPeerConnection
from aiortc.mediastreams import MediaStreamTrack
from aiortc.sdp import candidate_from_sdp

from app.webrtc.gateway import WebRTCSession, WebRTCSessionState, _candidate_to_dict

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


def _sdp_media_mid(sdp: str) -> str | None:
    """The mid of the (first) m-line, as an aiortc candidate needs for sdpMid."""
    for line in sdp.splitlines():
        if line.startswith("a=mid:"):
            return line[len("a=mid:") :].strip()
    return None


def _sdp_candidates(sdp: str) -> list[str]:
    """Every inline ICE candidate in an SDP, as bare attribute strings.

    aiortc 1.10 gathers candidates inline into the local description and never
    fires an ``icecandidate`` event, so the candidates a browser would trickle
    are read back out of the SDP here.
    """
    prefix = "a=candidate:"
    return [line[len(prefix) :] for line in sdp.splitlines() if line.startswith(prefix)]


def _candidate_from_dict(init: dict) -> "object":
    """Build an aiortc candidate from a browser-style ICE init dict."""
    raw = init["candidate"]
    sdp = raw[len("candidate:") :] if raw.startswith("candidate:") else raw
    candidate = candidate_from_sdp(sdp)
    candidate.sdpMid = init.get("sdpMid")
    candidate.sdpMLineIndex = init.get("sdpMLineIndex")
    return candidate


async def _run_handshake() -> None:
    received_frames: list[np.ndarray] = []
    inbound_datachannel_messages: list[str] = []
    offerer_pongs: list[str] = []
    # Trickle-ICE bookkeeping: the test must prove candidates actually crossed
    # in both directions, not just that nothing raised.
    offerer_candidates_to_gateway = 0
    answerer_candidates_to_offerer = 0

    gateway = WebRTCSession(
        "itest-session",
        on_audio_frame=received_frames.append,
        sample_rate=16000,
        frame_samples=512,
    )

    offerer = RTCPeerConnection()

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
    #
    # aiortc 1.10 never fires an ``icecandidate`` event: it gathers candidates
    # inline into the local description. So the candidates a browser would
    # trickle are read out of each side's SDP and pumped through the real trickle
    # entry points — the gateway's ``add_ice_candidate`` (inbound) and the
    # gateway's ``_candidate_to_dict`` serializer feeding ``addIceCandidate``
    # (outbound). Both pumps are genuine tasks whose exceptions must surface.
    offerer_candidate_lines = _sdp_candidates(offerer.localDescription.sdp)
    gateway_candidate_lines = _sdp_candidates(answer.sdp)
    offerer_mid = _sdp_media_mid(offerer.localDescription.sdp)
    answer_mid = _sdp_media_mid(answer.sdp)
    assert offerer_candidate_lines, "offerer gathered no candidates to trickle"
    assert gateway_candidate_lines, "gateway gathered no candidates to trickle"

    async def pump_offerer_candidates() -> None:
        nonlocal offerer_candidates_to_gateway
        for line in offerer_candidate_lines:
            # Exactly the browser payload the signalling handler forwards.
            await gateway.add_ice_candidate(
                {
                    "candidate": "candidate:" + line,
                    "sdpMid": offerer_mid,
                    "sdpMLineIndex": 0,
                }
            )
            offerer_candidates_to_gateway += 1

    async def pump_answerer_candidates() -> None:
        nonlocal answerer_candidates_to_offerer
        for line in gateway_candidate_lines:
            # Drive the gateway's own outbound serializer, then hand the result
            # to the offerer the way a browser's addIceCandidate would.
            candidate = candidate_from_sdp(line)
            candidate.sdpMid = answer_mid
            candidate.sdpMLineIndex = 0
            await offerer.addIceCandidate(_candidate_from_dict(_candidate_to_dict(candidate)))
            answerer_candidates_to_offerer += 1

    pump_in = asyncio.create_task(pump_offerer_candidates())
    pump_out = asyncio.create_task(pump_answerer_candidates())
    pumps = (pump_in, pump_out)
    pump_results: list = []

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

        # --- trickle ICE actually flowed, both directions -------------------
        # Await the pumps explicitly. Unlike the old test, which cancelled them
        # without awaiting and so silently swallowed a raising task, this raises
        # the pump's exception right here if `add_ice_candidate` (or the outbound
        # serializer) failed.
        await asyncio.gather(pump_in, pump_out)
        assert offerer_candidates_to_gateway == len(offerer_candidate_lines), (
            "not every offerer candidate was accepted by the gateway"
        )
        assert answerer_candidates_to_offerer == len(gateway_candidate_lines), (
            "not every gateway candidate reached the offerer"
        )
    finally:
        for task in pumps:
            task.cancel()
        # Belt-and-suspenders: collect any outcome that was not awaited above
        # (e.g. if the try body raised first). Cancellation is expected.
        pump_results = await asyncio.gather(*pumps, return_exceptions=True)
        await gateway.close()
        await offerer.close()

    for result in pump_results:
        if isinstance(result, BaseException) and not isinstance(
            result, asyncio.CancelledError
        ):
            # Re-raise so the failure names the real cause (e.g. TypeError from
            # a bad RTCIceCandidate) instead of a misleading timeout.
            raise result

    assert gateway.state == WebRTCSessionState.CLOSED
    assert gateway.pc.connectionState == "closed"


def test_offer_answer_handshake_media_and_datachannel() -> None:
    asyncio.run(_run_handshake())


def test_add_ice_candidate_accepts_browser_init() -> None:
    """A realistic browser ICE candidate is accepted without raising.

    Regression guard for the aiortc >= 1.10 ``RTCIceCandidate`` API change: the
    old ``RTCIceCandidate(candidate=...)`` construction raised ``TypeError`` and
    tore the session down. Empty/EOF payloads must be no-ops, not errors.
    """

    async def _run() -> None:
        gateway = WebRTCSession("candidate-test")
        try:
            await gateway.add_ice_candidate(
                {
                    "candidate": (
                        "candidate:842163049 1 udp 1677729535 127.0.0.1 50000 "
                        "typ host generation 0 ufrag abcd"
                    ),
                    "sdpMid": "0",
                    "sdpMLineIndex": 0,
                }
            )
            # End-of-candidates and empty/None payloads are signalled, not errors.
            await gateway.add_ice_candidate(None)
            await gateway.add_ice_candidate({"candidate": "", "sdpMid": "0"})
        finally:
            await gateway.close()

    asyncio.run(_run())


def test_candidate_to_dict_browser_roundtrip() -> None:
    """The outbound serializer emits a valid browser-style ICE init dict.

    aiortc >= 1.10's candidate has no ``candidate`` attribute, so the old
    ``_candidate_to_dict`` raised ``AttributeError``; this checks the SDP string
    is rebuilt with the ``candidate:`` prefix and survives a parse round-trip.
    """
    candidate = candidate_from_sdp(
        "842163049 1 udp 1677729535 127.0.0.1 50000 typ host"
    )
    candidate.sdpMid = "0"
    candidate.sdpMLineIndex = 0

    init = _candidate_to_dict(candidate)
    assert init["candidate"].startswith("candidate:")
    assert init["sdpMid"] == "0"
    assert init["sdpMLineIndex"] == 0

    # The dict parses back into an equivalent candidate.
    reparsed = _candidate_from_dict(init)
    assert reparsed.component == candidate.component
    assert reparsed.ip == candidate.ip
    assert reparsed.port == candidate.port
    assert reparsed.type == candidate.type


def test_close_is_idempotent() -> None:
    async def _run() -> None:
        gateway = WebRTCSession("close-test")
        await gateway.close()
        await gateway.close()  # must not raise
        assert gateway.state == WebRTCSessionState.CLOSED

    asyncio.run(_run())
