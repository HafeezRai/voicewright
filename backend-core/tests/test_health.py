"""Phase 1 contract tests for backend-core.

Covers the two things Phase 1 promises:
  1. the FastAPI app exposes GET /health with the frozen response shape
  2. the config layer loads and validates settings from the environment
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app

client = TestClient(app)


def test_health_returns_200_and_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "backend-core"}


def test_health_reports_configured_service_name() -> None:
    assert client.get("/health").json()["service"] == "backend-core"


# --------------------------------------------------------------------------
# config layer
# --------------------------------------------------------------------------


def test_settings_load_defaults_with_empty_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "CLOUDEPSO_ENV",
        "CLOUDEPSO_LOG_LEVEL",
        "CLOUDEPSO_CONFIG_PREFIX",
        "BACKEND_CORE_HOST",
        "BACKEND_CORE_PORT",
        "BACKEND_CORE_GRPC_PORT",
        "AI_ENGINE_URL",
    ):
        monkeypatch.delenv(var, raising=False)

    settings = Settings()

    assert settings.service_name == "backend-core"
    assert settings.env == "development"
    assert settings.log_level == "INFO"
    assert settings.config_prefix == "cloudpso"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.grpc_port == 50051
    assert settings.ai_engine_url == "http://ai-engine:8001"
    assert settings.is_production is False


def test_settings_read_and_normalise_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDEPSO_ENV", "production")
    monkeypatch.setenv("CLOUDEPSO_LOG_LEVEL", "debug")  # lower-case on purpose
    monkeypatch.setenv("BACKEND_CORE_PORT", "9000")
    monkeypatch.setenv("CLOUDEPSO_CONFIG_PREFIX", "pso")

    settings = Settings()

    assert settings.env == "production"
    assert settings.log_level == "DEBUG"
    assert settings.port == 9000
    assert settings.config_prefix == "pso"
    assert settings.is_production is True


def test_settings_rejects_invalid_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDEPSO_LOG_LEVEL", "CHATTY")
    with pytest.raises(Exception):
        Settings()


def test_settings_rejects_out_of_range_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BACKEND_CORE_PORT", "70000")
    with pytest.raises(Exception):
        Settings()


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
