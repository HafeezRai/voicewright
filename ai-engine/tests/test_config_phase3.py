"""Phase 3 additions to the config layer.

Phase 1's ``test_health.py`` already covers the frozen settings; this file covers
the backend selection, the writable-model-root fallback and the derived paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings

_CLEARED = (
    "AI_ENGINE_STT_BACKEND",
    "AI_ENGINE_STT_MODEL",
    "AI_ENGINE_STT_COMPUTE_TYPE",
    "AI_ENGINE_NLU_BACKEND",
    "AI_ENGINE_NLU_MODEL",
    "AI_ENGINE_NLU_HF_MODEL",
    "AI_ENGINE_TTS_BACKEND",
    "AI_ENGINE_TTS_MODEL",
    "AI_ENGINE_TTS_SPEED",
    "AI_ENGINE_DEVICE",
    "AI_ENGINE_MODEL_CACHE",
    "AI_ENGINE_LAZY_MODELS",
    "AI_ENGINE_GRPC_ENABLED",
    "AI_ENGINE_GRPC_PORT",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for var in _CLEARED:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_phase3_defaults(clean_env) -> None:
    s = Settings()
    assert s.stt_backend == "faster-whisper"
    assert s.stt_model == "base.en"
    assert s.nlu_backend == "auto"
    assert s.nlu_model == "convaiinnovations/laya"
    assert s.tts_backend == "sherpa"
    assert s.lazy_models is True
    assert s.grpc_port == 50052
    assert s.tts_speed == 1.0


def test_backend_choices_are_normalised(clean_env) -> None:
    clean_env.setenv("AI_ENGINE_NLU_BACKEND", "HF")
    clean_env.setenv("AI_ENGINE_STT_BACKEND", "Faster-Whisper")
    s = Settings()
    assert s.nlu_backend == "hf"
    assert s.stt_backend == "faster-whisper"


def test_unknown_backend_is_rejected(clean_env) -> None:
    clean_env.setenv("AI_ENGINE_NLU_BACKEND", "gpt")
    with pytest.raises(Exception):
        Settings()


def test_mps_is_an_accepted_device(clean_env) -> None:
    """Apple Silicon GPU is a first-class choice, not a typo."""
    clean_env.setenv("AI_ENGINE_DEVICE", "MPS")
    assert Settings().device == "mps"


def test_tts_speed_is_bounded(clean_env) -> None:
    clean_env.setenv("AI_ENGINE_TTS_SPEED", "0")
    with pytest.raises(Exception):
        Settings()
    clean_env.setenv("AI_ENGINE_TTS_SPEED", "5")
    with pytest.raises(Exception):
        Settings()


def test_models_root_falls_back_when_the_cache_is_unwritable(clean_env, tmp_path) -> None:
    """The container path /models does not exist on a dev host; fall back."""
    missing = tmp_path / "definitely-not-here"
    clean_env.setenv("AI_ENGINE_MODEL_CACHE", str(missing))
    s = Settings()
    assert s.models_root.is_dir()
    assert s.models_root != missing
    assert s.models_root.name == "models"


def test_models_root_honours_a_writable_cache(clean_env, tmp_path) -> None:
    writable = tmp_path / "cache"
    writable.mkdir()
    clean_env.setenv("AI_ENGINE_MODEL_CACHE", str(writable))
    assert Settings().models_root == writable


def test_derived_paths_hang_off_models_root(clean_env, tmp_path) -> None:
    writable = tmp_path / "cache"
    writable.mkdir()
    clean_env.setenv("AI_ENGINE_MODEL_CACHE", str(writable))
    clean_env.setenv("AI_ENGINE_TTS_MODEL", "some-voice")
    s = Settings()
    assert s.hf_home == writable / "hf"
    assert s.hf_hub_cache == writable / "hf" / "hub"
    assert s.tts_model_dir == writable / "tts" / "some-voice"
    assert isinstance(s.tts_model_dir, Path)
