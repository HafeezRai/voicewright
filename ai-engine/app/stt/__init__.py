"""Speech-to-text stage."""

from __future__ import annotations

from app.config import Settings
from app.stt.base import (
    STTEngine,
    STTUnavailableError,
    TranscriptionResult,
    TranscriptSegment,
)
from app.stt.faster_whisper_engine import FasterWhisperSTT

__all__ = [
    "FasterWhisperSTT",
    "STTEngine",
    "STTUnavailableError",
    "TranscriptSegment",
    "TranscriptionResult",
    "build_stt",
]


def build_stt(settings: Settings) -> STTEngine:
    """Construct the configured STT engine (weights load on first use)."""
    if settings.stt_backend == "faster-whisper":
        return FasterWhisperSTT(
            settings.stt_model,
            device=settings.device,
            compute_type=settings.stt_compute_type,
            beam_size=settings.stt_beam_size,
            sample_rate=settings.sample_rate_hz,
            download_root=str(settings.hf_hub_cache),
            hf_cache_dir=str(settings.hf_home),
        )
    raise STTUnavailableError(f"unknown STT backend: {settings.stt_backend!r}")
