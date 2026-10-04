"""Configuration layer for backend-core.

Settings are loaded from the process environment (populated by docker-compose,
a .env file, or the shell) and validated on construction. Defaults are chosen so
the service boots with an entirely empty environment.

`Settings()` reads the environment at *instantiation* time, which keeps it
trivially testable: mutate the environment, construct a fresh instance, assert.
`get_settings()` adds process-wide caching for request-path use.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
Environment = Literal["development", "staging", "production"]


class Settings(BaseSettings):
    """Validated runtime settings for the backend-core service."""

    model_config = SettingsConfigDict(
        # No env prefix: each field declares its full environment variable name
        # via `alias`, matching the names documented in .env.example.
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # --- shared / cross-service ---------------------------------------------
    service_name: str = "backend-core"
    config_prefix: str = Field(default="cloudpso", alias="CLOUDEPSO_CONFIG_PREFIX")
    env: Environment = Field(default="development", alias="CLOUDEPSO_ENV")
    log_level: LogLevel = Field(default="INFO", alias="CLOUDEPSO_LOG_LEVEL")

    # --- HTTP / WebSocket server --------------------------------------------
    host: str = Field(default="0.0.0.0", alias="BACKEND_CORE_HOST")
    port: int = Field(default=8000, ge=1, le=65535, alias="BACKEND_CORE_PORT")

    # --- gRPC ----------------------------------------------------------------
    grpc_port: int = Field(default=50051, ge=1, le=65535, alias="BACKEND_CORE_GRPC_PORT")

    # --- downstream services -------------------------------------------------
    ai_engine_url: str = Field(default="http://ai-engine:8001", alias="AI_ENGINE_URL")

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: object) -> object:
        # Accept "info" / "Info" and normalise to the canonical upper-case form.
        return value.upper() if isinstance(value, str) else value

    @property
    def is_production(self) -> bool:
        return self.env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""
    return Settings()
