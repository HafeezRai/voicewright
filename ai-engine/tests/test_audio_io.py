"""Pure audio helpers."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from app.audio_io import (
    duration_s,
    ensure_rate,
    from_pcm16_bytes,
    iter_chunks,
    load_wav,
    peak,
    resample_linear,
    rms,
    to_pcm16_bytes,
    to_wav_bytes,
)


def test_load_wav_reads_mono_float32(speech_wav) -> None:
    audio, rate = load_wav(str(speech_wav))
    assert rate == 16000
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert audio.shape[0] == 32000
    assert -1.0 <= audio.min() and audio.max() <= 1.0


def test_load_wav_downmixes_stereo(tmp_path) -> None:
    stereo = np.stack([np.full(800, 0.5, np.float32), np.full(800, -0.5, np.float32)], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(path, stereo, 16000, subtype="PCM_16")

    audio, rate = load_wav(str(path))
    assert rate == 16000
    assert audio.ndim == 1
    assert abs(float(audio.mean())) < 1e-3, "opposite channels should cancel on downmix"


def test_pcm16_round_trip() -> None:
    original = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    restored = from_pcm16_bytes(to_pcm16_bytes(original))
    assert np.allclose(original, restored, atol=1e-4)


def test_pcm16_clips_instead_of_wrapping() -> None:
    loud = np.array([2.0, -2.0], dtype=np.float32)
    pcm = np.frombuffer(to_pcm16_bytes(loud), dtype="<i2")
    # Wrapping would flip the sign; clipping saturates.
    assert pcm[0] > 32000 and pcm[1] < -32000


def test_to_wav_bytes_round_trips_through_soundfile() -> None:
    audio = (np.sin(np.linspace(0, 40 * np.pi, 16000)) * 0.4).astype(np.float32)
    wav = to_wav_bytes(audio, 16000)
    assert wav[:4] == b"RIFF"
    restored, rate = sf.read(__import__("io").BytesIO(wav), dtype="float32")
    assert rate == 16000
    assert len(restored) == len(audio)
    assert np.allclose(audio, restored, atol=1e-3)


def test_resample_linear_halves_the_sample_count() -> None:
    audio = np.sin(np.linspace(0, 20 * np.pi, 16000)).astype(np.float32)
    down = resample_linear(audio, 16000, 8000)
    assert abs(len(down) - 8000) <= 1
    assert down.dtype == np.float32


def test_resample_is_a_no_op_at_the_same_rate() -> None:
    audio = np.ones(100, dtype=np.float32)
    assert ensure_rate(audio, 16000, 16000) is audio or np.array_equal(
        ensure_rate(audio, 16000, 16000), audio
    )


def test_rms_and_peak() -> None:
    assert rms(np.zeros(10, np.float32)) == 0.0
    assert rms(np.ones(10, np.float32)) == 1.0
    assert peak(np.array([-0.5, 0.25], np.float32)) == 0.5
    assert rms(np.zeros(0, np.float32)) == 0.0
    assert peak(np.zeros(0, np.float32)) == 0.0


def test_duration_s() -> None:
    assert duration_s(np.zeros(16000, np.float32), 16000) == 1.0
    assert duration_s(np.zeros(0, np.float32), 16000) == 0.0
    assert duration_s(np.zeros(10, np.float32), 0) == 0.0


def test_iter_chunks_covers_everything() -> None:
    audio = np.arange(1000, dtype=np.float32)
    chunks = list(iter_chunks(audio, 256))
    assert [len(c) for c in chunks] == [256, 256, 256, 232]
    assert np.array_equal(np.concatenate(chunks), audio)


def test_iter_chunks_empty_and_short() -> None:
    assert list(iter_chunks(np.zeros(0, np.float32), 256)) == []
    short = list(iter_chunks(np.ones(10, np.float32), 256))
    assert len(short) == 1 and len(short[0]) == 10
