"""Real TTS: synthesise speech with sherpa-onnx and inspect the waveform.

The assertions are about *audibility*, not just length: a correct WAV header on a
buffer of zeros would pass a naive length check, so RMS and peak are asserted too.
"""

from __future__ import annotations

import io
import wave

import numpy as np

from app.audio_io import rms
from app.tts import SynthesisResult, TTSEngine, TTSPlaybackSink
from app.tts.base import CancelToken

TEXT = "I can help you cancel your account."


def test_synthesises_real_audio(tts_engine) -> None:
    result = tts_engine.synthesize(TEXT)

    assert isinstance(result, SynthesisResult)
    assert isinstance(tts_engine, TTSEngine)
    assert result.sample_rate == 16000
    assert result.audio.dtype == np.float32
    assert result.audio.ndim == 1

    # Duration: ~2 s of speech for an 8-word sentence. Generous bounds, but
    # tight enough that "returned a single silent frame" fails.
    assert 0.5 < result.duration_s < 8.0, f"implausible duration {result.duration_s}s"

    # Audibility: real speech, not a buffer of zeros.
    assert rms(result.audio) > 0.01, "synthesised audio is (near) silent"
    assert float(np.max(np.abs(result.audio))) > 0.1
    assert not np.allclose(result.audio, result.audio[0]), "audio is a constant"


def test_wav_bytes_are_a_valid_wav(tts_engine) -> None:
    result = tts_engine.synthesize(TEXT)
    wav = result.to_wav_bytes()

    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    with wave.open(io.BytesIO(wav), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 16000
        assert handle.getsampwidth() == 2
        frames = handle.readframes(handle.getnframes())
    assert len(frames) > 32000  # > 1 s of 16-bit mono audio
    pcm = np.frombuffer(frames, dtype="<i2")
    assert np.any(pcm != 0), "WAV payload is all-zero samples"


def test_stream_yields_chunks_covering_the_utterance(tts_engine) -> None:
    chunks = list(tts_engine.stream(TEXT, chunk_samples=3200))
    assert len(chunks) >= 2, "expected the utterance to span several chunks"
    assert all(c.dtype == np.float32 for c in chunks)
    # Every chunk but the last is exactly one chunk long.
    assert all(len(c) == 3200 for c in chunks[:-1])

    joined = np.concatenate(chunks)
    assert rms(joined) > 0.01
    # NOTE: sherpa-onnx/Piper is not sample-exact reproducible across calls, so
    # the streamed length is checked against duration bounds rather than against
    # a second, separately-generated utterance.
    assert 0.5 < len(joined) / 16000 < 8.0


def test_stream_honours_a_cancel(tts_engine) -> None:
    """A cancelled stream stops early.

    The reference is another *stream* of the same text, not ``synthesize``,
    because Piper's output length varies slightly between calls; comparing two
    streams keeps the comparison on one code path.
    """
    full_len = sum(len(c) for c in tts_engine.stream(TEXT, chunk_samples=3200))

    token = CancelToken()
    produced = 0
    for i, chunk in enumerate(tts_engine.stream(TEXT, chunk_samples=3200, cancel=token)):
        produced += len(chunk)
        if i >= 1:
            token.cancel()

    assert token.cancelled
    assert produced < full_len * 0.5, (
        f"cancel barely shortened the stream ({produced} of {full_len} samples)"
    )


def test_empty_text_is_a_no_op(tts_engine) -> None:
    result = tts_engine.synthesize("   ")
    assert len(result.audio) == 0
    assert list(tts_engine.stream("", chunk_samples=3200)) == []


# ---------------------------------------------------------------------------
# PlaybackSink
# ---------------------------------------------------------------------------


def test_playback_sink_conforms_to_the_contract() -> None:
    sink = TTSPlaybackSink()
    assert sink.is_speaking is False

    sink.start()
    assert sink.is_speaking is True
    assert not sink.cancelled

    sink.stop()
    assert sink.is_speaking is False
    assert sink.cancelled is True

    # Idempotent: a second stop is harmless and records another call.
    sink.stop()
    assert sink.is_speaking is False
    assert sink.stop_calls == 2


def test_playback_sink_reset_rearms_the_token() -> None:
    sink = TTSPlaybackSink()
    sink.start()
    sink.stop()
    assert sink.cancelled

    sink.reset()
    assert not sink.cancelled
    assert not sink.is_speaking


def test_playback_sink_cancel_token_stops_a_real_stream(tts_engine) -> None:
    """The sink's token is the one the TTS stream polls."""
    full_len = sum(len(c) for c in tts_engine.stream(TEXT, chunk_samples=3200))

    sink = TTSPlaybackSink()
    sink.start()

    produced = 0
    for i, chunk in enumerate(
        tts_engine.stream(TEXT, chunk_samples=3200, cancel=sink.cancel_token)
    ):
        produced += len(chunk)
        if i >= 1:
            sink.stop()  # what the barge-in controller does

    assert sink.cancelled
    assert produced < full_len * 0.5
