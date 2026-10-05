"""LLM backend selection: which host a Settings object points at.

No connection is opened -- :func:`build_llm` only wires up a client.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.llm.base import LLMUnavailableError
from app.llm.factory import PROVIDER_BASE_URLS, build_llm, resolve_base_url
from app.llm.openai_compat import OpenAICompatLLM

_CLEARED = (
    "AI_ENGINE_LLM_BACKEND",
    "AI_ENGINE_LLM_PROVIDER",
    "AI_ENGINE_LLM_BASE_URL",
    "AI_ENGINE_LLM_MODEL",
    "AI_ENGINE_LLM_API_KEY",
    "AI_ENGINE_LLM_STREAM",
    "AI_ENGINE_LLM_MAX_TOKENS",
    "AI_ENGINE_LLM_TEMPERATURE",
    "AI_ENGINE_LLM_EXTRA_BODY",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for var in _CLEARED:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_defaults_target_the_local_ollama() -> None:
    settings = Settings()
    assert settings.llm_backend == "openai_compat"
    assert settings.llm_provider == "ollama"
    assert settings.llm_base_url == ""
    # The tag is `llama3:8b`; the bare `llama3` does not exist on this host.
    assert settings.llm_model == "llama3:8b"
    assert settings.llm_api_key is None
    assert settings.llm_max_retries == 0
    assert settings.llm_extra_body == {"keep_alive": "30m"}


def test_base_url_is_derived_from_the_provider(clean_env) -> None:
    for provider, expected in PROVIDER_BASE_URLS.items():
        clean_env.setenv("AI_ENGINE_LLM_PROVIDER", provider)
        assert resolve_base_url(Settings()) == expected


def test_an_explicit_base_url_wins(clean_env) -> None:
    clean_env.setenv("AI_ENGINE_LLM_PROVIDER", "ollama")
    clean_env.setenv("AI_ENGINE_LLM_BASE_URL", "http://box.internal:8000/v1/")
    assert resolve_base_url(Settings()) == "http://box.internal:8000/v1/"


def test_an_unknown_provider_without_a_base_url_is_an_error(clean_env) -> None:
    """A provider typo must not silently send the demo to Ollama."""
    clean_env.setenv("AI_ENGINE_LLM_PROVIDER", "typo-d")
    with pytest.raises(LLMUnavailableError) as excinfo:
        resolve_base_url(Settings())
    assert "typo-d" in str(excinfo.value)


def test_an_unknown_provider_with_a_base_url_is_accepted(clean_env) -> None:
    """Any OpenAI-compatible host works; the provider name is only a default."""
    clean_env.setenv("AI_ENGINE_LLM_PROVIDER", "some-in-house-gateway")
    clean_env.setenv("AI_ENGINE_LLM_BASE_URL", "http://gateway.local/v1")
    clean_env.setenv("AI_ENGINE_LLM_MODEL", "in-house-7b")
    engine = build_llm(Settings())
    assert engine.endpoint == "http://gateway.local/v1/chat/completions"


def test_provider_names_are_normalised(clean_env) -> None:
    clean_env.setenv("AI_ENGINE_LLM_PROVIDER", "  DeepSeek ")
    assert Settings().llm_provider == "deepseek"
    assert resolve_base_url(Settings()) == PROVIDER_BASE_URLS["deepseek"]


def test_build_llm_returns_a_configured_engine(clean_env) -> None:
    clean_env.setenv("AI_ENGINE_LLM_PROVIDER", "deepseek")
    clean_env.setenv("AI_ENGINE_LLM_MODEL", "deepseek-chat")
    clean_env.setenv("AI_ENGINE_LLM_API_KEY", "sk-test")
    clean_env.setenv("AI_ENGINE_LLM_MAX_TOKENS", "77")
    clean_env.setenv("AI_ENGINE_LLM_TEMPERATURE", "0.9")

    engine = build_llm(Settings())
    assert isinstance(engine, OpenAICompatLLM)
    assert engine.provider == "deepseek"
    assert engine.model == "deepseek-chat"
    assert engine.endpoint == "https://api.deepseek.com/v1/chat/completions"
    assert engine._max_tokens == 77
    assert engine._temperature == 0.9
    assert engine._api_key == "sk-test"


def test_the_api_key_is_not_a_plain_string_on_settings(clean_env) -> None:
    """A secret in the config object must not be printable by accident."""
    clean_env.setenv("AI_ENGINE_LLM_API_KEY", "sk-do-not-log")
    settings = Settings()
    assert settings.llm_api_key is not None
    assert "sk-do-not-log" not in repr(settings.llm_api_key)
    assert settings.llm_api_key.get_secret_value() == "sk-do-not-log"


def test_max_tokens_is_bounded(clean_env) -> None:
    clean_env.setenv("AI_ENGINE_LLM_MAX_TOKENS", "0")
    with pytest.raises(Exception):
        Settings()


def test_provider_swap_needs_no_code_change(clean_env) -> None:
    """Same client, same call path -- only the endpoint and model differ."""
    ollama = build_llm(Settings())

    clean_env.setenv("AI_ENGINE_LLM_PROVIDER", "deepseek")
    clean_env.setenv("AI_ENGINE_LLM_MODEL", "deepseek-chat")
    clean_env.setenv("AI_ENGINE_LLM_API_KEY", "sk-test")
    deepseek = build_llm(Settings())

    assert type(ollama) is type(deepseek)
    assert ollama.endpoint != deepseek.endpoint
    assert deepseek._headers()["Authorization"] == "Bearer sk-test"
    assert "Authorization" not in ollama._headers()
