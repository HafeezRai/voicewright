"""Real STT: transcribe a genuine speech recording with faster-whisper.

No mocks. The fixture is the same 2 s speech excerpt Phase 2 uses for its VAD
tests, so a regression in either stage is visible against one shared input.
"""

from __future__ import annotations

from app.audio_io import load_wav, rms
from app.stt import STTEngine


def test_speech_wav_is_real_audio(speech_wav) -> None:
    """Guard the guard: the fixture must be speech, not silence."""
    audio, rate = load_wav(str(speech_wav))
    assert rate == 16000
    assert audio.shape[0] == 32000  # 2.0 s
    assert rms(audio) > 0.01, "fixture looks like silence; the STT assertions below would be moot"


def test_transcribes_real_speech(stt_engine, speech_wav) -> None:
    audio, rate = load_wav(str(speech_wav))
    result = stt_engine.transcribe(audio, rate)

    assert isinstance(stt_engine, STTEngine)
    assert result.text.strip(), "STT returned empty text for a real speech clip"
    # The clip is an English news-reading excerpt; whisper may return either
    # casing, so only assert on the words themselves.
    assert any(ch.isalpha() for ch in result.text), f"no words in transcript: {result.text!r}"
    assert len(result.text.split()) >= 2, f"implausibly short transcript: {result.text!r}"
    assert result.language == "en"
    assert result.duration_s == 2.0
    assert result.backend == "faster-whisper"
    assert result.segments, "expected at least one timestamped segment"
    assert 0.0 <= result.segments[0].start_s <= result.segments[-1].end_s <= 2.5


def test_transcribe_accepts_a_path(stt_engine, speech_wav) -> None:
    """Transcribing by filename must give the same answer as by array."""
    from_array = stt_engine.transcribe(*load_wav(str(speech_wav)))
    from_path = stt_engine.transcribe(str(speech_wav))
    assert from_path.text.strip() == from_array.text.strip()


def test_empty_audio_yields_empty_text(stt_engine) -> None:
    import numpy as np

    result = stt_engine.transcribe(np.zeros(1600, dtype=np.float32), 16000)
    assert result.text.strip() == ""
