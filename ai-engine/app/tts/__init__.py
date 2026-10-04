"""Text-to-speech stage."""

from __future__ import annotations

from app.config import Settings
from app.tts.base import (
    CancelToken,
    SynthesisResult,
    TTSEngine,
    TTSUnavailableError,
)
from app.tts.sherpa_engine import SherpaTTS
from app.tts.sink import TTSPlaybackSink

__all__ = [
    "CancelToken",
    "SherpaTTS",
    "SynthesisResult",
    "TTSEngine",
    "TTSPlaybackSink",
    "TTSUnavailableError",
    "build_tts",
]


def build_tts(settings: Settings) -> TTSEngine:
    """Construct the configured TTS engine (weights load on first use)."""
    if settings.tts_backend == "sherpa":
        return SherpaTTS(
            settings.tts_model_dir,
            speaker_id=settings.tts_speaker_id,
            speed=settings.tts_speed,
            sample_rate=settings.sample_rate_hz,
        )
    raise TTSUnavailableError(f"unknown TTS backend: {settings.tts_backend!r}")
