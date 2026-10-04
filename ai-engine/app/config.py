"""Configuration layer for ai-engine.

Settings are loaded from the process environment and validated on construction,
with defaults chosen so the service boots from an entirely empty environment.

Model/VAD fields are *declared* here in Phase 1 so downstream phases have one
place to read them from; none of them trigger a model download on import.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
Environment = Literal["development", "staging", "production"]
Device = Literal["cpu", "cuda"]


class Settings(BaseSettings):
    """Validated runtime settings for the ai-engine service."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # --- shared / cross-service ---------------------------------------------
    service_name: str = "ai-engine"
    config_prefix: str = Field(default="cloudpso", alias="CLOUDEPSO_CONFIG_PREFIX")
    env: Environment = Field(default="development", alias="CLOUDEPSO_ENV")
    log_level: LogLevel = Field(default="INFO", alias="CLOUDEPSO_LOG_LEVEL")

    # --- HTTP server ---------------------------------------------------------
    host: str = Field(default="0.0.0.0", alias="AI_ENGINE_HOST")
    port: int = Field(default=8001, ge=1, le=65535, alias="AI_ENGINE_PORT")

    # --- models --------------------------------------------------------------
    model_cache: str = Field(default="/models", alias="AI_ENGINE_MODEL_CACHE")
    stt_model: str = Field(default="openai/whisper-small", alias="AI_ENGINE_STT_MODEL")
    nlu_model: str = Field(
        default="microsoft/deberta-v3-base", alias="AI_ENGINE_NLU_MODEL"
    )
    tts_model: str = Field(default="facebook/fastspeech2-en-ljspeech", alias="AI_ENGINE_TTS_MODEL")
    device: Device = Field(default="cpu", alias="AI_ENGINE_DEVICE")

    # --- voice activity detection -------------------------------------------
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0, alias="AI_ENGINE_VAD_THRESHOLD")
    sample_rate_hz: int = Field(default=16000, gt=0, alias="AI_ENGINE_SAMPLE_RATE_HZ")

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("device", mode="before")
    @classmethod
    def _normalise_device(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value

    @property
    def is_production(self) -> bool:
        return self.env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""
    return Settings()
