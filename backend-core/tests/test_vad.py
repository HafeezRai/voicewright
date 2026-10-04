"""VAD backends.

The Silero tests are *real* inference: they load the ONNX model committed at
``app/audio/models/silero_vad.onnx`` and classify a genuine speech recording
(``tests/data/speech_16k.wav``, a 2 s excerpt of the MIT-licensed silero-vad
test fixture) against silence. Nothing here is mocked.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from app.audio.vad import (
    DEFAULT_MODEL_PATH,
    SILERO_FRAME_SAMPLES,
    EnergyVAD,
    SileroVAD,
    VADUnavailableError,
    build_vad,
    create_vad,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
SPEECH_WAV = DATA_DIR / "speech_16k.wav"


def _read_wav_mono16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def _frames(signal: np.ndarray) -> list[np.ndarray]:
    n = len(signal) // SILERO_FRAME_SAMPLES
    return [signal[i * SILERO_FRAME_SAMPLES : (i + 1) * SILERO_FRAME_SAMPLES] for i in range(n)]


# ---------------------------------------------------------------------------
# EnergyVAD — deterministic fallback
# ---------------------------------------------------------------------------


def test_energy_vad_separates_silence_from_tone() -> None:
    vad = EnergyVAD(threshold_rms=0.01)
    tone = 0.5 * np.sin(2 * np.pi * 220 * np.arange(SILERO_FRAME_SAMPLES) / 16000.0)
    assert vad.is_speech(np.zeros(SILERO_FRAME_SAMPLES, dtype=np.float32)) is False
    assert vad.is_speech(tone.astype(np.float32)) is True


def test_energy_vad_empty_frame_is_not_speech() -> None:
    assert EnergyVAD().is_speech(np.zeros(0, dtype=np.float32)) is False


def test_energy_vad_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError):
        EnergyVAD(threshold_rms=-1.0)


# ---------------------------------------------------------------------------
# SileroVAD — real ONNX inference
# ---------------------------------------------------------------------------


def test_model_asset_is_present() -> None:
    assert DEFAULT_MODEL_PATH.is_file(), "silero_vad.onnx must be committed under app/audio/models"


def test_silero_loads_and_runs_on_silence() -> None:
    vad = SileroVAD(threshold=0.5)
    assert vad.backend_name == "silero"
    assert vad.frame_samples == SILERO_FRAME_SAMPLES

    probs = [vad.probability(np.zeros(SILERO_FRAME_SAMPLES, dtype=np.float32)) for _ in range(10)]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert max(probs) < 0.5, f"silence should not be speech, got probs={probs}"
    assert vad.is_speech(np.zeros(SILERO_FRAME_SAMPLES, dtype=np.float32)) is False


def test_silero_classifies_real_speech() -> None:
    """The decisive test: real speech audio must be detected as speech."""
    assert SPEECH_WAV.is_file(), "speech fixture missing"
    vad = SileroVAD(threshold=0.5)
    frames = _frames(_read_wav_mono16k(SPEECH_WAV))
    assert len(frames) >= 30, "fixture should be ~2 s of audio"

    probabilities = [vad.probability(f) for f in frames]
    speech_frames = [p for p in probabilities if p >= 0.5]

    # A speech fixture must be dominated by speech frames, not silence.
    assert len(speech_frames) / len(probabilities) > 0.5, (
        f"expected mostly speech, got {len(speech_frames)}/{len(probabilities)} "
        f"frames (max prob {max(probabilities):.3f})"
    )


def test_silero_state_streams_and_resets() -> None:
    """Streaming state carries across frames; reset() returns to a clean slate."""
    vad = SileroVAD(threshold=0.5)
    frames = _frames(_read_wav_mono16k(SPEECH_WAV))[:5]
    streamed = [vad.probability(f) for f in frames]

    vad.reset()
    replayed = [vad.probability(f) for f in frames]
    # Same frames after reset must reproduce the same probabilities.
    assert streamed == pytest.approx(replayed, abs=1e-6)


def test_silero_rejects_wrong_frame_size() -> None:
    vad = SileroVAD(threshold=0.5)
    with pytest.raises(ValueError):
        vad.probability(np.zeros(256, dtype=np.float32))


def test_silero_rejects_missing_model() -> None:
    with pytest.raises(VADUnavailableError):
        SileroVAD(model_path="/nonexistent/silero_vad.onnx")


# ---------------------------------------------------------------------------
# factory / fallback behaviour
# ---------------------------------------------------------------------------


def test_create_vad_returns_requested_backend() -> None:
    assert create_vad("energy").backend_name == "energy"
    assert create_vad("silero").backend_name == "silero"


def test_create_vad_rejects_unknown_backend() -> None:
    with pytest.raises(VADUnavailableError):
        create_vad("nonsense")


def test_build_vad_falls_back_to_energy_when_model_missing() -> None:
    vad = build_vad("silero", model_path="/nonexistent/silero_vad.onnx")
    assert vad.backend_name == "energy"
