"""Small audio helpers shared by the STT, TTS and pipeline modules.

Everything here works on mono ``float32`` numpy arrays in ``[-1, 1]`` -- the
common currency between the ML runtimes (faster-whisper, sherpa-onnx), the
WebRTC ingress (16-bit PCM) and the WAV files used in tests.

Deliberately dependency-light: ``numpy`` + ``soundfile`` only. No scipy/librosa,
which keeps the container small and the resampling path auditable.
"""

from __future__ import annotations

import io
from collections.abc import Iterator

import numpy as np
import soundfile as sf

#: Tolerated absolute peak when accepting caller-supplied float audio.
_PEAK_EPS = 1e-6


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a WAV file as mono ``float32`` in ``[-1, 1]``.

    Multi-channel input is downmixed by averaging; this matches what the WebRTC
    gateway does before audio reaches the VAD.
    """
    data, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), int(sample_rate)


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resampler.

    Adequate for speech at the rates this service deals with (8/16/22.05/44.1/48
    kHz). It is *not* an anti-aliasing resampler; for the down-sampling this
    service actually performs (e.g. 48k -> 16k) the STT model's own mel
    front-end is far more tolerant of the resulting high-frequency artefacts
    than the transcript would suggest. If aliasing ever shows up in transcripts,
    replace this with a polyphase filter -- the call site does not change.
    """
    if src_rate == dst_rate or samples.size == 0:
        return np.ascontiguousarray(samples, dtype=np.float32)
    duration = samples.shape[0] / float(src_rate)
    target_len = max(1, round(duration * dst_rate))
    src_idx = np.linspace(0.0, samples.shape[0] - 1, num=samples.shape[0], dtype=np.float64)
    dst_idx = np.linspace(0.0, samples.shape[0] - 1, num=target_len, dtype=np.float64)
    out = np.interp(dst_idx, src_idx, samples.astype(np.float64))
    return np.ascontiguousarray(out, dtype=np.float32)


def ensure_rate(samples: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    """Resample ``samples`` to ``target_rate`` if it is not already there."""
    return resample_linear(samples, sample_rate, target_rate)


def rms(samples: np.ndarray) -> float:
    """Root-mean-square amplitude; 0.0 for an empty signal."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def peak(samples: np.ndarray) -> float:
    """Largest absolute sample value."""
    if samples.size == 0:
        return 0.0
    return float(np.max(np.abs(samples)))


def duration_s(samples: np.ndarray, sample_rate: int) -> float:
    """Duration in seconds of ``samples`` at ``sample_rate``."""
    if sample_rate <= 0:
        return 0.0
    return samples.shape[0] / float(sample_rate)


def to_pcm16_bytes(samples: np.ndarray) -> bytes:
    """Convert float32 ``[-1, 1]`` to little-endian signed 16-bit PCM bytes.

    Clipped rather than wrapped, so an over-driven sample becomes full scale
    instead of a loud click of the opposite sign.
    """
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def from_pcm16_bytes(payload: bytes) -> np.ndarray:
    """Inverse of :func:`to_pcm16_bytes`."""
    if not payload:
        return np.zeros(0, dtype=np.float32)
    return (np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0).copy()


def to_wav_bytes(samples: np.ndarray, sample_rate: int, *, subtype: str = "PCM_16") -> bytes:
    """Serialise mono ``float32`` audio to an in-memory WAV file."""
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(samples, dtype=np.float32), int(sample_rate),
             format="WAV", subtype=subtype)
    return buffer.getvalue()


def iter_chunks(samples: np.ndarray, chunk_samples: int) -> Iterator[np.ndarray]:
    """Yield consecutive, non-overlapping slices of ``samples``.

    The final chunk may be shorter than ``chunk_samples``. Yields exactly one
    (possibly short) chunk when the whole signal is shorter than a chunk.
    """
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    total = samples.shape[0]
    if total == 0:
        return
    for start in range(0, total, chunk_samples):
        yield samples[start : start + chunk_samples]
