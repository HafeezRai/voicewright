"""Config layer: backend selection, derived paths, and the LLM stage's settings.

Phase 1's ``test_health.py`` already covers the frozen settings. This file covers
the backend choices, the writable-model-root fallback, the derived paths, and
the retirement of the ``AI_ENGINE_NLU_*`` surface.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from app.config import Settings

_CLEARED = (
    "AI_ENGINE_STT_BACKEND",
    "AI_ENGINE_STT_MODEL",
    "AI_ENGINE_STT_COMPUTE_TYPE",
    "AI_ENGINE_LLM_BACKEND",
    "AI_ENGINE_LLM_PROVIDER",
    "AI_ENGINE_LLM_MODEL",
    "AI_ENGINE_LLM_BASE_URL",
    "AI_ENGINE_LLM_SYSTEM_PROMPT",
    "AI_ENGINE_EMBEDDING_MODEL",
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
    assert s.tts_backend == "sherpa"
    assert s.lazy_models is True
    assert s.grpc_port == 50052
    assert s.tts_speed == 1.0


def test_backend_choices_are_normalised(clean_env) -> None:
    clean_env.setenv("AI_ENGINE_STT_BACKEND", "Faster-Whisper")
    clean_env.setenv("AI_ENGINE_LLM_BACKEND", "OpenAI_Compat")
    s = Settings()
    assert s.stt_backend == "faster-whisper"
    assert s.llm_backend == "openai_compat"


def test_unknown_backend_is_rejected(clean_env) -> None:
    clean_env.setenv("AI_ENGINE_LLM_BACKEND", "gpt")
    with pytest.raises(Exception):
        Settings()


def test_mps_is_an_accepted_device(clean_env) -> None:
    """Apple Silicon GPU is a first-class choice, not a typo."""
    clean_env.setenv("AI_ENGINE_DEVICE", "MPS")
    assert Settings().device == "mps"


def _mps_available() -> bool:
    """Mirror of the production probe, so the test agrees with the code."""
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:  # pragma: no cover - torch absent on a slim CI image
        return False


def test_unset_device_auto_detects(clean_env) -> None:
    """No AI_ENGINE_DEVICE -> best available runtime, without raising.

    Written to pass on every host: MPS where torch reports it (this Apple
    Silicon machine), CPU on Linux CI / containers / Intel Macs.
    """
    device = Settings().device
    assert device in {"mps", "cpu"}
    assert device == ("mps" if _mps_available() else "cpu")


def test_auto_device_is_equivalent_to_unset(clean_env) -> None:
    clean_env.setenv("AI_ENGINE_DEVICE", "auto")
    assert Settings().device in {"mps", "cpu"}


def test_explicit_device_beats_auto_detection(clean_env) -> None:
    """An explicit cpu/mps must win over detection, whatever the host offers."""
    clean_env.setenv("AI_ENGINE_DEVICE", "cpu")
    assert Settings().device == "cpu"
    clean_env.setenv("AI_ENGINE_DEVICE", "mps")
    assert Settings().device == "mps"


def test_tts_speed_is_bounded(clean_env) -> None:
    clean_env.setenv("AI_ENGINE_TTS_SPEED", "0")
    with pytest.raises(Exception):
        Settings()
    clean_env.setenv("AI_ENGINE_TTS_SPEED", "5")
    with pytest.raises(Exception):
        Settings()


# ---------------------------------------------------------------------------
# the LLM stage
# ---------------------------------------------------------------------------


def test_llm_defaults_point_at_the_local_model(clean_env) -> None:
    s = Settings()
    assert s.llm_backend == "openai_compat"
    assert s.llm_provider == "ollama"
    # The tag is `llama3:8b`; bare `llama3` does not exist here and 404s.
    assert s.llm_model == "llama3:8b"
    assert s.llm_max_tokens == 160
    assert s.llm_temperature == 0.4
    assert s.llm_stream is True
    assert s.llm_history_turns == 6
    assert s.llm_allow_reasoning_models is False
    # Empty by default: fail loudly rather than speak a canned line.
    assert s.llm_fallback_reply == ""
    assert s.llm_system_prompt.strip()


def test_the_system_prompt_is_configurable(clean_env) -> None:
    clean_env.setenv("AI_ENGINE_LLM_SYSTEM_PROMPT", "You are a laconic lighthouse keeper.")
    assert Settings().llm_system_prompt == "You are a laconic lighthouse keeper."


def test_history_turns_is_bounded(clean_env) -> None:
    clean_env.setenv("AI_ENGINE_LLM_HISTORY_TURNS", "-1")
    with pytest.raises(Exception):
        Settings()
    clean_env.setenv("AI_ENGINE_LLM_HISTORY_TURNS", "0")
    assert Settings().llm_history_turns == 0


# ---------------------------------------------------------------------------
# the retired NLU surface
# ---------------------------------------------------------------------------


def test_the_nlu_fields_are_gone(clean_env) -> None:
    s = Settings()
    assert not hasattr(s, "nlu_backend")
    assert not hasattr(s, "nlu_model")
    assert not hasattr(s, "nlu_hf_model")


def test_the_encoder_was_renamed_not_dropped(clean_env) -> None:
    """The MiniLM weights are still needed -- for retrieval, not classification."""
    assert Settings().embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    clean_env.setenv("AI_ENGINE_EMBEDDING_MODEL", "some/encoder")
    assert Settings().embedding_model == "some/encoder"


def test_a_stale_nlu_variable_is_ignored_but_warned_about(
    clean_env, caplog: pytest.LogCaptureFixture
) -> None:
    """``extra="ignore"`` would drop it in silence -- the worst outcome."""
    clean_env.setenv("AI_ENGINE_NLU_BACKEND", "auto")
    with caplog.at_level(logging.WARNING, logger="app.config"):
        Settings()

    assert any(
        "AI_ENGINE_NLU_BACKEND" in record.getMessage() for record in caplog.records
    ), "a retired NLU variable was silently ignored"
    # It must still load: an unknown variable may not stop the service booting.
    assert Settings().service_name == "ai-engine"


def test_no_warning_when_there_is_nothing_legacy(
    clean_env, caplog: pytest.LogCaptureFixture
) -> None:
    clean_env.delenv("AI_ENGINE_NLU_BACKEND", raising=False)
    with caplog.at_level(logging.WARNING, logger="app.config"):
        Settings()
    assert not [r for r in caplog.records if "NLU" in r.getMessage()]


# ---------------------------------------------------------------------------
# derived paths
# ---------------------------------------------------------------------------


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
