"""Phase 2 additions to the config layer (defaults, derivations, validation)."""

from __future__ import annotations

import pytest

from app.config import Settings

PHASE2_VARS = (
    "BACKEND_CORE_AUDIO_SAMPLE_RATE",
    "BACKEND_CORE_VAD_FRAME_SAMPLES",
    "BACKEND_CORE_VAD_BACKEND",
    "BACKEND_CORE_VAD_MODEL_PATH",
    "BACKEND_CORE_VAD_THRESHOLD",
    "BACKEND_CORE_BARGEIN_ENABLED",
    "BACKEND_CORE_BARGEIN_MIN_SPEECH_MS",
    "BACKEND_CORE_BARGEIN_MIN_SILENCE_MS",
    "BACKEND_CORE_WEBRTC_STUN_SERVER",
    "BACKEND_CORE_GRPC_ENABLED",
)


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in PHASE2_VARS:
        monkeypatch.delenv(var, raising=False)


def test_phase2_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    settings = Settings()
    assert settings.audio_sample_rate == 16000
    assert settings.vad_frame_samples == 512
    assert settings.vad_backend == "silero"
    assert settings.vad_threshold == 0.5
    assert settings.bargein_enabled is True
    assert settings.bargein_min_speech_ms == 200
    assert settings.bargein_min_silence_ms == 400
    assert settings.grpc_enabled is True


def test_frame_duration_derived_from_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    assert Settings().frame_duration_ms == 32  # 512 samples @ 16 kHz

    monkeypatch.setenv("BACKEND_CORE_VAD_FRAME_SAMPLES", "256")
    assert Settings().frame_duration_ms == 16


def test_ice_servers_from_stun(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    assert Settings().ice_servers == []

    monkeypatch.setenv("BACKEND_CORE_WEBRTC_STUN_SERVER", "stun:stun.l.google.com:19302")
    assert Settings().ice_servers == [{"urls": ["stun:stun.l.google.com:19302"]}]


def test_vad_backend_is_constrained(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("BACKEND_CORE_VAD_BACKEND", "magic")
    with pytest.raises(Exception):
        Settings()


def test_vad_threshold_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("BACKEND_CORE_VAD_THRESHOLD", "1.5")
    with pytest.raises(Exception):
        Settings()
