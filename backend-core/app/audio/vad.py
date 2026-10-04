"""Voice-activity detection.

Two interchangeable backends sit behind the :class:`VAD` interface:

* :class:`SileroVAD` — the real Silero VAD neural model run through ONNX
  Runtime. This is the production backend (CLAUDE.md lists ONNX Runtime and
  Silero VAD as required tech; ONNX avoids a multi-GB PyTorch dependency).
* :class:`EnergyVAD` — a deterministic RMS-threshold detector with no model
  file and no native dependency. Used for CI, for tests that need byte-exact
  determinism, and as a graceful fallback if the ONNX model cannot be loaded.

Frames are mono ``float32`` numpy arrays in ``[-1.0, 1.0]``.

Silero contract (v5 ONNX export, verified against the model shipped in this
repo): at 16 kHz the network consumes a *576*-sample window — the current
512-sample frame prefixed by the previous frame's trailing 64 samples, which
the model uses as left-context between hops. Feeding bare 512-sample chunks
silently collapses the output to ~0, so the context is mandatory, not an
optimisation.
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Silero v5 constants (16 kHz branch).
SILERO_FRAME_SAMPLES = 512
SILERO_CONTEXT_SAMPLES = 64
SILERO_STATE_SHAPE = (2, 1, 128)

# Packaged model location; the ONNX asset is committed so tests run offline.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "silero_vad.onnx"


class VADUnavailableError(RuntimeError):
    """Raised when a requested VAD backend cannot be constructed."""


class VAD(abc.ABC):
    """Interface every VAD backend implements.

    ``is_speech`` is synchronous and allocation-light: it is called once per
    audio frame (every 32 ms at 16 kHz) on the WebRTC consumer task.
    """

    @property
    @abc.abstractmethod
    def backend_name(self) -> str:
        """Stable identifier used as a Prometheus label, e.g. ``"silero"``."""

    @property
    @abc.abstractmethod
    def frame_samples(self) -> int:
        """Number of samples the backend expects per ``is_speech`` call."""

    @abc.abstractmethod
    def is_speech(self, frame: np.ndarray) -> bool:
        """Classify one frame of mono float32 audio in ``[-1, 1]``."""

    def reset(self) -> None:
        """Drop any streaming state. Default is a no-op (stateless backends)."""


class EnergyVAD(VAD):
    """Deterministic RMS-threshold VAD. No model, no native dependencies."""

    def __init__(self, threshold_rms: float = 0.01, frame_samples: int = SILERO_FRAME_SAMPLES):
        if threshold_rms < 0.0:
            raise ValueError("threshold_rms must be non-negative")
        self._threshold_rms = threshold_rms
        self._frame_samples = frame_samples

    @property
    def backend_name(self) -> str:
        return "energy"

    @property
    def frame_samples(self) -> int:
        return self._frame_samples

    @property
    def threshold_rms(self) -> float:
        return self._threshold_rms

    def is_speech(self, frame: np.ndarray) -> bool:
        samples = np.asarray(frame, dtype=np.float64).reshape(-1)
        if samples.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(samples))))
        return rms >= self._threshold_rms


class SileroVAD(VAD):
    """Real Silero VAD, executed with ONNX Runtime on the CPU."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        threshold: float = 0.5,
        sample_rate: int = 16000,
    ):
        if sample_rate != 16000:
            raise VADUnavailableError(
                "the packaged Silero ONNX model supports only 16000 Hz input"
            )
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not path.is_file():
            raise VADUnavailableError(f"Silero ONNX model not found at {path}")

        # Imported lazily so importing this module never requires onnxruntime.
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on install
            raise VADUnavailableError("onnxruntime is not installed") from exc

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        try:
            self._session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # pragma: no cover - corrupt model file
            raise VADUnavailableError(f"could not load Silero ONNX model: {exc}") from exc

        self._model_path = path
        self._threshold = float(threshold)
        self._sr = np.asarray(sample_rate, dtype=np.int64)
        self._last_probability = 0.0
        self.reset()

    @property
    def backend_name(self) -> str:
        return "silero"

    @property
    def frame_samples(self) -> int:
        return SILERO_FRAME_SAMPLES

    @property
    def last_probability(self) -> float:
        """Speech probability from the most recent frame (for diagnostics)."""
        return self._last_probability

    @property
    def model_path(self) -> Path:
        return self._model_path

    def reset(self) -> None:
        self._state = np.zeros(SILERO_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, SILERO_CONTEXT_SAMPLES), dtype=np.float32)
        self._last_probability = 0.0

    def probability(self, frame: np.ndarray) -> float:
        """Run one frame and return the raw speech probability without thresholding."""
        samples = np.asarray(frame, dtype=np.float32).reshape(1, -1)
        if samples.shape[1] != SILERO_FRAME_SAMPLES:
            raise ValueError(
                f"Silero requires exactly {SILERO_FRAME_SAMPLES} samples per frame, "
                f"got {samples.shape[1]}"
            )
        window = np.concatenate([self._context, samples], axis=1)
        outputs = self._session.run(
            None, {"input": window, "state": self._state, "sr": self._sr}
        )
        out, self._state = outputs[0], outputs[1]
        self._context = window[..., -SILERO_CONTEXT_SAMPLES:]
        self._last_probability = float(np.squeeze(out))
        return self._last_probability

    def is_speech(self, frame: np.ndarray) -> bool:
        return self.probability(frame) >= self._threshold


def pcm16_bytes_to_float32(data: bytes) -> np.ndarray:
    """Convert little-endian signed 16-bit PCM bytes to float32 in ``[-1, 1]``."""
    if len(data) % 2:
        raise ValueError("PCM16 payload must have an even number of bytes")
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def create_vad(
    backend: str = "silero",
    *,
    model_path: str | Path | None = None,
    threshold: float = 0.5,
    sample_rate: int = 16000,
    frame_samples: int = SILERO_FRAME_SAMPLES,
) -> VAD:
    """Construct a VAD backend, raising :class:`VADUnavailableError` on failure."""
    if backend == "silero":
        return SileroVAD(model_path=model_path, threshold=threshold, sample_rate=sample_rate)
    if backend == "energy":
        # Map the speech probability threshold onto an RMS floor so the same
        # config value tunes both backends.
        return EnergyVAD(threshold_rms=max(threshold, 0.01) * 0.02, frame_samples=frame_samples)
    raise VADUnavailableError(f"unknown VAD backend: {backend!r}")


def build_vad(
    backend: str = "silero",
    *,
    model_path: str | Path | None = None,
    threshold: float = 0.5,
    sample_rate: int = 16000,
    frame_samples: int = SILERO_FRAME_SAMPLES,
) -> VAD:
    """Construct the requested VAD, degrading to the energy backend if needed.

    Called on the request path (session creation), so a missing/broken ONNX
    model must not take the whole service down — it logs loudly and continues
    with the deterministic fallback.
    """
    try:
        return create_vad(
            backend,
            model_path=model_path,
            threshold=threshold,
            sample_rate=sample_rate,
            frame_samples=frame_samples,
        )
    except VADUnavailableError as exc:
        if backend == "energy":
            raise
        logger.warning("VAD backend %r unavailable (%s); falling back to energy VAD", backend, exc)
        return EnergyVAD(
            threshold_rms=max(threshold, 0.01) * 0.02, frame_samples=frame_samples
        )
