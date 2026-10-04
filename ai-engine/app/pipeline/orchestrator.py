"""STT -> NLU -> TTS orchestration.

Two paths over the same three stages:

* :meth:`VoicePipeline.process_turn` -- the **batch** path. Audio in, a
  :class:`TurnResult` (transcript, intent, synthesised reply) out. This is what
  the HTTP API, the gRPC servicer and the end-to-end test use.

* :meth:`VoicePipeline.stream_turn` -- the **streaming** path. Same stages, but
  the reply audio is yielded chunk by chunk and can be cut short mid-utterance.

* :meth:`VoicePipeline.process_stream` -- the **chunked ingress** path: audio
  arrives as a sequence of chunks (as it does over gRPC / WebRTC), is buffered,
  and is then run through the batch path. Buffering is deliberate: Whisper's
  accuracy on a 300 ms fragment is poor, so the honest design is to accumulate a
  full utterance (the gateway's VAD decides where that ends) and transcribe once.

Where barge-in cancels in-flight TTS
------------------------------------
In :meth:`stream_turn`. The loop pulls audio chunks from
:meth:`TTSEngine.stream` and, before each one, tests
``PlaybackSink.is_speaking`` / the sink's cancel token. When the barge-in
controller (backend-core) calls ``PlaybackSink.stop()`` -- which it does on
``min_speech_ms`` of continuous user speech -- the token is set and:

  * the loop stops yielding immediately (playback ends at the current chunk), and
  * the cancel token is passed into ``TTSEngine.stream``, so a backend that can
    interrupt synthesis (sherpa-onnx does, via its generation callback) aborts
    the forward pass rather than finishing an utterance nobody will hear.

Everything before synthesis (STT, NLU) is not cancellable: those are short,
bounded calls, and killing them mid-flight would leave the session in a worse
state than letting them finish.
"""

from __future__ import annotations

import enum
import logging
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from app.audio_io import duration_s
from app.metrics import (
    NLU_LATENCY,
    NLU_REQUESTS,
    PIPELINE_LATENCY,
    PIPELINE_REQUESTS,
    STT_AUDIO_SECONDS,
    STT_LATENCY,
    STT_REQUESTS,
    TTS_AUDIO_SECONDS,
    TTS_CANCELLED,
    TTS_LATENCY,
    TTS_REQUESTS,
)
from app.nlu.base import IntentResult, NLUEngine
from app.pipeline.responder import respond
from app.stt.base import STTEngine, TranscriptionResult
from app.tts.base import CancelToken, SynthesisResult, TTSEngine
from app.tts.sink import TTSPlaybackSink

logger = logging.getLogger(__name__)


class EventKind(enum.StrEnum):
    """Kinds of :class:`PipelineEvent` emitted by :meth:`stream_turn`."""

    TRANSCRIPT_FINAL = "transcript_final"
    INTENT = "intent"
    AUDIO_CHUNK = "audio_chunk"
    DONE = "done"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class PipelineEvent:
    """One step of a streaming turn."""

    kind: EventKind
    text: str | None = None
    intent: IntentResult | None = None
    audio: np.ndarray | None = None
    sample_rate: int = 0


@dataclass
class TurnResult:
    """Everything one batch turn produced."""

    transcript: TranscriptionResult
    intent: IntentResult
    reply_text: str
    speech: SynthesisResult
    stt_latency_s: float = 0.0
    nlu_latency_s: float = 0.0
    tts_latency_s: float = 0.0
    total_latency_s: float = 0.0
    cancelled: bool = False
    #: Per-stage timings, for the dashboard and for latency accounting.
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def audio_duration_s(self) -> float:
        return self.speech.duration_s


class VoicePipeline:
    """Wires the three stages together. Holds no per-turn state."""

    def __init__(
        self,
        stt: STTEngine,
        nlu: NLUEngine,
        tts: TTSEngine,
        *,
        sample_rate: int = 16000,
        chunk_samples: int = 3200,
    ):
        self.stt = stt
        self.nlu = nlu
        self.tts = tts
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples

    # -- single stages (public, metric-recording) ---------------------------
    #
    # The HTTP API exposes /stt, /nlu and /tts individually. They go through
    # these rather than the engines directly so that every path into a stage
    # records the same metrics -- otherwise calling /nlu would be invisible in
    # Prometheus while the identical NLU call inside a turn was counted.
    def transcribe(self, audio: np.ndarray | str, sample_rate: int | None = None):
        """Transcribe one utterance, recording STT metrics."""
        t0 = time.perf_counter()
        try:
            transcript = self.stt.transcribe(audio, sample_rate)
        except Exception:
            STT_REQUESTS.labels(status="error").inc()
            raise
        STT_REQUESTS.labels(status="ok").inc()
        STT_LATENCY.observe(time.perf_counter() - t0)
        if transcript.duration_s:
            STT_AUDIO_SECONDS.observe(transcript.duration_s)
        return transcript

    def classify(self, text: str) -> IntentResult:
        """Classify one utterance, recording NLU metrics."""
        return self._classify(text)

    def synthesize(self, text: str, *, speed: float | None = None) -> SynthesisResult:
        """Synthesise one utterance, recording TTS metrics."""
        return self._synthesize(text, speed=speed)

    # -- batch path ---------------------------------------------------------
    def process_turn(
        self,
        audio: np.ndarray | str,
        sample_rate: int | None = None,
    ) -> TurnResult:
        """Run one full STT -> NLU -> TTS turn and return the result."""
        started = time.perf_counter()

        t0 = time.perf_counter()
        try:
            transcript = self.transcribe(audio, sample_rate)
        except Exception:
            PIPELINE_REQUESTS.labels(status="error", nlu_backend="?").inc()
            raise
        stt_s = time.perf_counter() - t0

        return self._turn_from_transcript(transcript, stt_s, started)

    def process_stream(
        self,
        chunks: Iterable[np.ndarray],
        sample_rate: int | None = None,
    ) -> TurnResult:
        """Buffer an audio chunk stream into one utterance, then run it.

        See the module docstring: transcription happens once, on the whole
        buffered utterance, not per chunk.
        """
        rate = sample_rate or self.sample_rate
        parts = [np.asarray(c, dtype=np.float32) for c in chunks if c is not None and len(c)]
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        logger.debug("ingested %.2fs of audio from %d chunks",
                     duration_s(audio, rate), len(parts))
        return self.process_turn(audio, rate)

    def process_text(self, text: str) -> tuple[IntentResult, str, SynthesisResult]:
        """Run NLU -> TTS only (no audio in). Used by the /nlu and /tts tests."""
        intent = self._classify(text)
        reply = respond(intent)
        speech = self._synthesize(reply)
        return intent, reply, speech

    # -- streaming path -----------------------------------------------------
    def stream_turn(
        self,
        audio: np.ndarray | str,
        sample_rate: int | None = None,
        *,
        sink: TTSPlaybackSink | None = None,
    ) -> Iterator[PipelineEvent]:
        """Yield the turn as it happens, cutting the reply off on barge-in.

        The cancel point is the ``for chunk in ...`` loop below; see the module
        docstring for the full explanation.
        """
        started = time.perf_counter()
        t0 = time.perf_counter()
        transcript = self.transcribe(audio, sample_rate)
        stt_s = time.perf_counter() - t0
        yield PipelineEvent(EventKind.TRANSCRIPT_FINAL, text=transcript.text)

        # Go through _classify so the streaming path records the same NLU
        # metrics as the batch path.
        t1 = time.perf_counter()
        intent = self._classify(transcript.text)
        nlu_s = time.perf_counter() - t1
        logger.debug("stream turn: stt=%.3fs nlu=%.3fs backend=%s",
                     stt_s, nlu_s, intent.backend)
        yield PipelineEvent(EventKind.INTENT, intent=intent)

        reply = respond(intent)
        sink = sink or TTSPlaybackSink()
        sink.start()
        token: CancelToken = sink.cancel_token

        t2 = time.perf_counter()
        first_chunk_s: float | None = None
        produced = 0
        cancelled = False
        try:
            for chunk in self.tts.stream(
                reply, chunk_samples=self.chunk_samples, cancel=token
            ):
                # ---- BARGE-IN CANCEL POINT ------------------------------
                if not sink.is_speaking or token.cancelled:
                    cancelled = True
                    break
                # ---------------------------------------------------------
                if first_chunk_s is None:
                    first_chunk_s = time.perf_counter() - t2
                produced += len(chunk)
                yield PipelineEvent(
                    EventKind.AUDIO_CHUNK, audio=chunk, sample_rate=self.tts.sample_rate
                )
            # The generator can also end early *because* of the cancel (the
            # backend stops producing), in which case the loop above exits
            # normally and the check inside it never runs again. Catch that
            # here, or a cancelled turn would be reported as a completed one.
            if not cancelled and (token.cancelled or not sink.is_speaking):
                cancelled = True
        finally:
            tts_s = time.perf_counter() - t2
            TTS_LATENCY.observe(tts_s)
            if produced and self.tts.sample_rate:
                TTS_AUDIO_SECONDS.observe(produced / float(self.tts.sample_rate))
            if sink.is_speaking:
                sink.finish()

        if cancelled:
            TTS_CANCELLED.inc()
            TTS_REQUESTS.labels(status="cancelled").inc()
            yield PipelineEvent(EventKind.CANCELLED, text=reply)
            return

        TTS_REQUESTS.labels(status="ok").inc()
        if first_chunk_s is not None:
            logger.debug("tts first chunk in %.3fs", first_chunk_s)
        PIPELINE_REQUESTS.labels(status="ok", nlu_backend=intent.backend).inc()
        PIPELINE_LATENCY.observe(time.perf_counter() - started)
        yield PipelineEvent(EventKind.DONE, text=reply, intent=intent)

    # -- internals ----------------------------------------------------------
    def _classify(self, text: str) -> IntentResult:
        t1 = time.perf_counter()
        try:
            intent = self.nlu.classify(text)
        except Exception:
            NLU_REQUESTS.labels(backend=self.nlu.name, status="error").inc()
            raise
        NLU_LATENCY.labels(backend=intent.backend or self.nlu.name).observe(
            time.perf_counter() - t1
        )
        NLU_REQUESTS.labels(backend=intent.backend or self.nlu.name, status="ok").inc()
        return intent

    def _synthesize(self, text: str, *, speed: float | None = None) -> SynthesisResult:
        t2 = time.perf_counter()
        try:
            speech = self.tts.synthesize(text, speed=speed)
        except Exception:
            TTS_REQUESTS.labels(status="error").inc()
            raise
        TTS_LATENCY.observe(time.perf_counter() - t2)
        TTS_REQUESTS.labels(status="ok").inc()
        if speech.duration_s:
            TTS_AUDIO_SECONDS.observe(speech.duration_s)
        return speech

    def _turn_from_transcript(
        self, transcript: TranscriptionResult, stt_s: float, started: float
    ) -> TurnResult:
        t1 = time.perf_counter()
        intent = self._classify(transcript.text)
        nlu_s = time.perf_counter() - t1

        reply = respond(intent)
        t2 = time.perf_counter()
        speech = self._synthesize(reply)
        tts_s = time.perf_counter() - t2
        total = time.perf_counter() - started

        PIPELINE_REQUESTS.labels(status="ok", nlu_backend=intent.backend).inc()
        PIPELINE_LATENCY.observe(total)
        return TurnResult(
            transcript=transcript,
            intent=intent,
            reply_text=reply,
            speech=speech,
            stt_latency_s=stt_s,
            nlu_latency_s=nlu_s,
            tts_latency_s=tts_s,
            total_latency_s=total,
            timings={"stt": stt_s, "nlu": nlu_s, "tts": tts_s, "total": total},
        )
