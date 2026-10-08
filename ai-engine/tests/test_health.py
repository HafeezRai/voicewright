"""Phase 1 contract tests for ai-engine.

Covers the two things Phase 1 promises:
  1. the FastAPI app exposes GET /health with the frozen response shape
  2. the config layer loads and validates settings from the environment

Runs against `requirements-dev.txt` — no torch/transformers/onnxruntime needed.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app

client = TestClient(app)

_CLEARED_VARS = (
    "VOICEWRIGHT_ENV",
    "VOICEWRIGHT_LOG_LEVEL",
    "VOICEWRIGHT_CONFIG_PREFIX",
    "AI_ENGINE_HOST",
    "AI_ENGINE_PORT",
    "AI_ENGINE_MODEL_CACHE",
    "AI_ENGINE_STT_MODEL",
    "AI_ENGINE_NLU_MODEL",
    "AI_ENGINE_TTS_MODEL",
    "AI_ENGINE_DEVICE",
    "AI_ENGINE_VAD_THRESHOLD",
    "AI_ENGINE_SAMPLE_RATE_HZ",
)


def test_health_returns_200_and_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "ai-engine"}


def test_health_reports_configured_service_name() -> None:
    assert client.get("/health").json()["service"] == "ai-engine"


# --------------------------------------------------------------------------
# config layer
# --------------------------------------------------------------------------


def test_settings_load_defaults_with_empty_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _CLEARED_VARS:
        monkeypatch.delenv(var, raising=False)

    settings = Settings()

    assert settings.service_name == "ai-engine"
    assert settings.env == "development"
    assert settings.log_level == "INFO"
    assert settings.config_prefix == "voicewright"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8001
    assert settings.model_cache == "/models"
    # The unset default auto-detects: MPS on Apple Silicon, CPU otherwise.
    assert settings.device in {"mps", "cpu"}
    assert settings.vad_threshold == 0.5
    assert settings.sample_rate_hz == 16000


def test_settings_read_and_normalise_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICEWRIGHT_ENV", "production")
    monkeypatch.setenv("VOICEWRIGHT_LOG_LEVEL", "warning")  # lower-case on purpose
    monkeypatch.setenv("AI_ENGINE_PORT", "9100")
    monkeypatch.setenv("AI_ENGINE_DEVICE", "CUDA")  # upper-case on purpose
    monkeypatch.setenv("AI_ENGINE_MODEL_CACHE", "/opt/models")

    settings = Settings()

    assert settings.env == "production"
    assert settings.log_level == "WARNING"
    assert settings.port == 9100
    assert settings.device == "cuda"
    assert settings.model_cache == "/opt/models"
    assert settings.is_production is True


def test_settings_rejects_out_of_range_vad_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_ENGINE_VAD_THRESHOLD", "1.5")
    with pytest.raises(Exception):
        Settings()


def test_settings_rejects_unknown_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_ENGINE_DEVICE", "tpu")
    with pytest.raises(Exception):
        Settings()


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
