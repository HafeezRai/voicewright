"""End-to-end pipeline: WAV in -> STT -> NLU -> TTS -> WAV out.

This is the headline evidence for Phase 3. Two independent paths in:

1. the Phase 2 speech fixture (a real recording), and
2. a **closed loop** -- text is synthesised by the TTS, transcribed by the STT,
   classified, and answered. If either model is broken the loop fails, and the
   expected intent is known exactly because we chose the input text.

Nothing is mocked.
"""

from __future__ import annotations

import numpy as np

from app.audio_io import load_wav, rms
from app.nlu import INTENT_NAMES
from app.pipeline import EventKind
from app.tts import TTSPlaybackSink

ROUND_TRIP_CASES = [
    ("I want to book an appointment", "book_appointment"),
    ("I need to cancel my account", "cancel_account"),
    ("what is your return policy", "return_policy"),
]


def test_batch_turn_from_a_real_recording(pipeline_engine, speech_wav) -> None:
    audio, rate = load_wav(str(speech_wav))
    result = pipeline_engine.process_turn(audio, rate)

    # STT produced real words.
    assert result.transcript.text.strip()
    assert result.transcript.language == "en"

    # NLU produced a real intent from the catalogue.
    assert result.intent.intent in INTENT_NAMES
    assert result.intent.backend, "the intent must record which backend answered"

    # TTS produced audible speech for the reply.
    assert result.reply_text.strip()
    assert result.speech.duration_s > 0.5
    assert rms(result.speech.audio) > 0.01
    assert result.speech.to_wav_bytes()[:4] == b"RIFF"

    # Timing is broken out per stage and adds up.
    assert result.timings["stt"] > 0
    assert result.timings["nlu"] > 0
    assert result.timings["tts"] > 0
    assert result.total_latency_s >= result.timings["stt"]


def test_closed_loop_intent_is_correct(pipeline_engine, tts_engine) -> None:
    """Synthesise a request, hear it back, and check the intent it produces."""
    for phrase, expected in ROUND_TRIP_CASES:
        spoken = tts_engine.synthesize(phrase)          # text -> audio
        result = pipeline_engine.process_turn(spoken.audio, spoken.sample_rate)
        assert result.intent.intent == expected, (
            f"{phrase!r} transcribed as {result.transcript.text!r} "
            f"classified as {result.intent.intent!r}"
        )
        assert result.intent.confidence > 0.3


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
    assert chunked.intent.intent == batch.intent.intent


def test_streaming_turn_emits_the_expected_events(pipeline_engine, tts_engine) -> None:
    spoken = tts_engine.synthesize("I need to cancel my account")
    events = list(pipeline_engine.stream_turn(spoken.audio, spoken.sample_rate))

    kinds = [e.kind for e in events]
    assert kinds[0] is EventKind.TRANSCRIPT_FINAL
    assert EventKind.INTENT in kinds
    assert kinds[-1] is EventKind.DONE
    assert kinds.count(EventKind.AUDIO_CHUNK) >= 2

    transcript_event = events[0]
    assert transcript_event.text.strip()

    intent_event = next(e for e in events if e.kind is EventKind.INTENT)
    assert intent_event.intent is not None
    assert intent_event.intent.intent == "cancel_account"

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


def test_process_text_runs_nlu_and_tts_only(pipeline_engine) -> None:
    intent, reply, speech = pipeline_engine.process_text("I want to book an appointment")
    assert intent.intent == "book_appointment"
    assert "book" in reply.lower()
    assert speech.duration_s > 0.5


def test_pipeline_holds_no_per_turn_state(pipeline_engine, tts_engine) -> None:
    """Two turns in a row must not interfere (the orchestrator is stateless)."""
    for phrase, expected in ROUND_TRIP_CASES:
        spoken = tts_engine.synthesize(phrase)
        a = pipeline_engine.process_turn(spoken.audio, spoken.sample_rate)
        b = pipeline_engine.process_turn(spoken.audio, spoken.sample_rate)
        assert a.intent.intent == b.intent.intent == expected
