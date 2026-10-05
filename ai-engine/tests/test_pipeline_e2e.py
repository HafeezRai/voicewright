"""End-to-end pipeline: WAV in -> STT -> LLM -> TTS -> WAV out.

This is the headline evidence for the pipeline. Two independent paths in:

1. the Phase 2 speech fixture (a real recording), and
2. a **closed loop** -- text is synthesised by the TTS, transcribed by the STT,
   answered, and synthesised again. If either model is broken the loop fails.

Whisper and VITS are the real models. Generation is the deterministic fake, so
these assert *structure* -- that audio came out, that timings add up, that a
cancel stops the turn -- never specific wording, which no honest test of a
generative model could.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.audio_io import load_wav, rms
from app.cancellation import CancelToken
from app.llm.base import ChatMessage, LLMUnavailableError
from app.pipeline import EventKind, VoicePipeline
from app.tts import TTSPlaybackSink
from tests.conftest import FAKE_REPLY, FakeLLM


def test_batch_turn_from_a_real_recording(pipeline_engine, speech_wav) -> None:
    audio, rate = load_wav(str(speech_wav))
    result = pipeline_engine.process_turn(audio, rate)

    # STT produced real words.
    assert result.transcript.text.strip()
    assert result.transcript.language == "en"

    # The LLM produced a reply and recorded which model answered.
    assert result.reply_text.strip()
    assert result.llm.model
    assert result.llm.provider == "fake"
    assert result.llm.usage is not None and result.llm.usage.total_tokens > 0
    assert result.system_prompt and result.system_prompt == pipeline_engine.system_prompt

    # TTS produced audible speech for the reply.
    assert result.speech.duration_s > 0.5
    assert rms(result.speech.audio) > 0.01
    assert result.speech.to_wav_bytes()[:4] == b"RIFF"

    # Timing is broken out per stage and adds up.
    assert result.timings["stt"] > 0
    assert result.timings["llm"] > 0
    assert result.timings["tts"] > 0
    assert result.total_latency_s >= result.timings["stt"]
    assert set(result.timings) == {"stt", "llm", "tts", "total"}


def test_closed_loop_transcript_drives_generation(pipeline_engine, tts_engine) -> None:
    """Synthesise a request, hear it back, and generate from what was heard."""
    spoken = tts_engine.synthesize("I would like to book an appointment")
    result = pipeline_engine.process_turn(spoken.audio, spoken.sample_rate)

    assert result.transcript.text.strip()
    # The transcript is what was sent to the model, not something else.
    assert pipeline_engine.llm.last_messages[-1].content == result.transcript.text
    assert result.reply_text.strip()


def test_reply_audio_is_a_playable_wav(pipeline_engine, tts_engine) -> None:
    spoken = tts_engine.synthesize("I want to book an appointment")
    result = pipeline_engine.process_turn(spoken.audio, spoken.sample_rate)
    wav = result.speech.to_wav_bytes()

    import io
    import wave

    with wave.open(io.BytesIO(wav), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 16000
        assert handle.getnframes() > 8000  # > 0.5 s


def test_chunked_ingress_matches_batch(pipeline_engine, speech_wav) -> None:
    """The chunked-ingress path buffers and gives the same answer as batch."""
    from app.audio_io import iter_chunks

    audio, rate = load_wav(str(speech_wav))
    batch = pipeline_engine.process_turn(audio, rate)
    chunked = pipeline_engine.process_stream(iter_chunks(audio, 512), rate)

    assert chunked.transcript.text == batch.transcript.text
    assert chunked.reply_text == batch.reply_text


def test_streaming_turn_emits_the_expected_events(pipeline_engine, tts_engine) -> None:
    spoken = tts_engine.synthesize("I need to cancel my account")
    events = list(pipeline_engine.stream_turn(spoken.audio, spoken.sample_rate))

    kinds = [e.kind for e in events]
    assert kinds[0] is EventKind.TRANSCRIPT_FINAL
    assert EventKind.REPLY_DELTA in kinds
    assert kinds[-1] is EventKind.DONE
    assert kinds.count(EventKind.AUDIO_CHUNK) >= 2

    transcript_event = events[0]
    assert transcript_event.text.strip()

    delta = next(e for e in events if e.kind is EventKind.REPLY_DELTA)
    assert delta.text == FAKE_REPLY
    done = next(e for e in events if e.kind is EventKind.DONE)
    assert done.text == FAKE_REPLY

    audio_events = [e for e in events if e.kind is EventKind.AUDIO_CHUNK]
    joined = np.concatenate([e.audio for e in audio_events])
    assert rms(joined) > 0.01
    assert all(e.sample_rate == 16000 for e in audio_events)


def _reply_audio_len(pipeline_engine, spoken, *, cancel_after: int | None) -> tuple[int, list]:
    """Stream a turn, optionally stopping the sink after N audio chunks."""
    sink = TTSPlaybackSink()
    produced = 0
    seen = []
    for event in pipeline_engine.stream_turn(spoken.audio, spoken.sample_rate, sink=sink):
        seen.append(event.kind)
        if event.kind is EventKind.AUDIO_CHUNK:
            produced += len(event.audio)
            if cancel_after is not None and seen.count(EventKind.AUDIO_CHUNK) >= cancel_after:
                sink.stop()  # what the barge-in controller does
    return produced, seen


def test_streaming_turn_is_cancelled_by_the_playback_sink(pipeline_engine, tts_engine) -> None:
    """Barge-in: stopping the sink mid-reply cuts the stream and reports CANCELLED."""
    spoken = tts_engine.synthesize("I need to cancel my account")

    # Reference: the same turn, streamed without a cancel.
    full_len, _ = _reply_audio_len(pipeline_engine, spoken, cancel_after=None)
    produced, seen = _reply_audio_len(pipeline_engine, spoken, cancel_after=1)

    assert EventKind.CANCELLED in seen
    assert EventKind.DONE not in seen, "a cancelled turn must not also report DONE"
    assert produced < full_len * 0.5, (
        f"cancel barely shortened the reply ({produced} of {full_len} samples)"
    )


def test_pipeline_holds_no_per_turn_state(pipeline_engine, tts_engine) -> None:
    """Two turns in a row must not interfere (the orchestrator is stateless)."""
    spoken = tts_engine.synthesize("I would like to book an appointment")
    first = pipeline_engine.process_turn(spoken.audio, spoken.sample_rate)
    second = pipeline_engine.process_turn(spoken.audio, spoken.sample_rate)

    assert first.reply_text == second.reply_text == FAKE_REPLY
    assert first.history_turns == 0 and second.history_turns == 0
    # Nothing accumulated on the pipeline itself.
    assert len(pipeline_engine.llm.calls) == 2


def test_history_is_replayed_into_the_prompt(pipeline_engine, tts_engine) -> None:
    """History is a per-call parameter; when passed, it reaches the model."""
    spoken = tts_engine.synthesize("what about Tuesday")
    history = [
        ChatMessage(role="user", content="book me an appointment"),
        ChatMessage(role="assistant", content="Certainly, what day suits you?"),
    ]
    result = pipeline_engine.process_turn(
        spoken.audio, spoken.sample_rate, history=history
    )

    sent = pipeline_engine.llm.last_messages
    assert [m.content for m in sent] == [
        pipeline_engine.system_prompt,
        "book me an appointment",
        "Certainly, what day suits you?",
        result.transcript.text,
    ]
    assert result.history_turns == 2


def test_a_blank_system_prompt_omits_the_system_message(stt_engine, tts_engine) -> None:
    llm = FakeLLM()
    pipeline = VoicePipeline(
        stt_engine, llm, tts_engine, sample_rate=16000, chunk_samples=3200, system_prompt=""
    )
    pipeline.generate("hello")
    assert [m.role for m in llm.last_messages] == ["user"]


# ---------------------------------------------------------------------------
# generation failures
# ---------------------------------------------------------------------------


def test_provider_failure_surfaces_as_an_error(stt_engine, tts_engine, speech_wav) -> None:
    """Default configuration fails loudly rather than speaking nothing."""
    llm = FakeLLM(fail=LLMUnavailableError("ollama is down"))
    pipeline = VoicePipeline(stt_engine, llm, tts_engine, sample_rate=16000)
    audio, rate = load_wav(str(speech_wav))

    with pytest.raises(LLMUnavailableError):
        pipeline.process_turn(audio, rate)


def test_a_configured_fallback_reply_keeps_the_demo_audible(
    stt_engine, tts_engine, speech_wav
) -> None:
    llm = FakeLLM(fail=LLMUnavailableError("ollama is down"))
    pipeline = VoicePipeline(
        stt_engine,
        llm,
        tts_engine,
        sample_rate=16000,
        fallback_reply="Sorry, I am having trouble thinking right now.",
    )
    audio, rate = load_wav(str(speech_wav))
    result = pipeline.process_turn(audio, rate)

    assert result.reply_text == "Sorry, I am having trouble thinking right now."
    assert result.llm.fallback is True
    assert result.speech.duration_s > 0.5


# ---------------------------------------------------------------------------
# barge-in during generation
# ---------------------------------------------------------------------------


def test_streaming_turn_is_interruptible_while_the_model_is_thinking(
    stt_engine, tts_engine, speech_wav
) -> None:
    """The agent must be interruptible *before* it produces any audio.

    ``sink.start()`` runs before the generation call precisely so a caller who
    talks over the silence is heard. This is the case that catches a regression
    in that ordering: if ``start()`` moved back after generation, it would
    re-arm the cancel token that the barge-in already set, and the turn would
    finish and report ``DONE`` instead of ``CANCELLED``.
    """
    sink = TTSPlaybackSink()

    class InterruptedLLM(FakeLLM):
        """Stops the sink *during* generation, as the barge-in controller does."""

        def generate(self, messages):
            sink.stop()
            return super().generate(messages)

    pipeline = VoicePipeline(stt_engine, InterruptedLLM(), tts_engine, sample_rate=16000)
    audio, rate = load_wav(str(speech_wav))

    events = list(pipeline.stream_turn(audio, rate, sink=sink))
    kinds = [e.kind for e in events]

    assert EventKind.TRANSCRIPT_FINAL in kinds
    assert EventKind.REPLY_DELTA in kinds, "generation should still have run"
    assert EventKind.CANCELLED in kinds
    assert EventKind.DONE not in kinds, "an abandoned turn must not report DONE"
    assert EventKind.AUDIO_CHUNK not in kinds, (
        "the turn was abandoned while thinking, so nothing should have been spoken"
    )


def test_fake_llm_stream_stops_on_cancel() -> None:
    """The incremental path itself honours the token, not just the turn loop."""
    llm = FakeLLM(delta_delay_s=0.01)
    token = CancelToken()
    pieces = []
    for piece in llm.stream([ChatMessage(role="user", content="hi")], cancel=token):
        pieces.append(piece)
        if len(pieces) == 2:
            token.cancel()

    assert len(pieces) == 2
    assert len(pieces) < len(FAKE_REPLY.split())
