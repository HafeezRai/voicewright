"""sherpa-onnx TTS backend (Piper/VITS voices).

Why sherpa-onnx rather than ``piper-tts``:

* **Licence.** ``piper-tts`` 1.8 ships under GPL-3.0-or-later, which is not
  acceptable for an enterprise product. ``sherpa-onnx`` is Apache-2.0, and the
  Piper voices it runs are MIT-licensed ONNX files.
* **Runtime.** It carries its own prebuilt ONNX Runtime; there is no PyTorch in
  the inference path at all, so the container stays small and the model is a
  plain ``.onnx`` file -- the exact artifact Phase 5 hands to Triton.
* **Cancellation.** ``OfflineTts.generate`` accepts a callback that can stop
  generation mid-utterance, which is what makes barge-in able to cut the agent
  off *during* synthesis rather than only between chunks.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from app.tts.base import CancelToken, SynthesisResult, TTSEngine, TTSUnavailableError

logger = logging.getLogger(__name__)

#: Sentinel pushed by the producer thread when generation ends.
_DONE = object()


class SherpaTTS(TTSEngine):
    """VITS/Piper synthesis through sherpa-onnx."""

    name = "sherpa"

    def __init__(
        self,
        model_dir: str | Path,
        *,
        num_threads: int = 2,
        speaker_id: int = 0,
        speed: float = 1.0,
        sample_rate: int = 16000,
    ):
        self._model_dir = Path(model_dir)
        self._num_threads = num_threads
        self._speaker_id = speaker_id
        self._speed = speed
        self._sample_rate = sample_rate
        self._tts = None
        self._lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------
    @property
    def sample_rate(self) -> int:
        return self._tts.sample_rate if self._tts is not None else self._sample_rate

    @property
    def is_loaded(self) -> bool:
        return self._tts is not None

    def _onnx_path(self) -> Path:
        """The voice's ``.onnx`` file, whatever the voice calls it."""
        candidates = sorted(self._model_dir.glob("*.onnx"))
        if not candidates:
            raise TTSUnavailableError(
                f"no .onnx voice found in {self._model_dir}; "
                "run scripts/fetch_models.sh to download the Piper voice"
            )
        return candidates[0]

    def _load(self):
        if self._tts is not None:
            return self._tts
        with self._lock:
            if self._tts is not None:
                return self._tts
            try:
                import sherpa_onnx
            except Exception as exc:  # pragma: no cover - import guard
                raise TTSUnavailableError(
                    "sherpa-onnx is not installed; `pip install sherpa-onnx`"
                ) from exc
            if not self._model_dir.is_dir():
                raise TTSUnavailableError(
                    f"TTS model directory {self._model_dir} does not exist; "
                    "run scripts/fetch_models.sh"
                )
            model_path = self._onnx_path()
            tokens = self._model_dir / "tokens.txt"
            data_dir = self._model_dir / "espeak-ng-data"
            logger.info("loading sherpa-onnx TTS voice=%s", model_path.name)
            config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=str(model_path),
                        tokens=str(tokens),
                        data_dir=str(data_dir) if data_dir.is_dir() else "",
                    ),
                    num_threads=self._num_threads,
                ),
            )
            try:
                self._tts = sherpa_onnx.OfflineTts(config)
            except Exception as exc:
                raise TTSUnavailableError(f"failed to load sherpa-onnx voice: {exc}") from exc
            self._sample_rate = int(self._tts.sample_rate)
            return self._tts

    def load(self) -> None:
        self._load()

    def close(self) -> None:
        self._tts = None

    # -- inference ----------------------------------------------------------
    def synthesize(
        self,
        text: str,
        *,
        speed: float | None = None,
        speaker_id: int | None = None,
    ) -> SynthesisResult:
        tts = self._load()
        if not text or not text.strip():
            return SynthesisResult(
                audio=np.zeros(0, dtype=np.float32),
                sample_rate=self.sample_rate,
                backend=self.name,
            )
        audio = tts.generate(
            text,
            sid=self._speaker_id if speaker_id is None else int(speaker_id),
            speed=self._speed if speed is None else float(speed),
        )
        samples = np.asarray(audio.samples, dtype=np.float32)
        logger.debug("tts %r -> %.2fs @%d Hz", text[:60], len(samples) / self.sample_rate,
                     self.sample_rate)
        return SynthesisResult(audio=samples, sample_rate=self.sample_rate, backend=self.name)

    def stream(
        self,
        text: str,
        *,
        chunk_samples: int,
        speed: float | None = None,
        speaker_id: int | None = None,
        cancel: CancelToken | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield audio chunks as sherpa-onnx produces them.

        Generation runs on a worker thread and hands chunks to this generator
        through a queue. When ``cancel`` is set the generator stops consuming and
        the worker's callback returns non-zero, which aborts the forward pass --
        so a barge-in stops the model, not merely the playback loop.
        """
        tts = self._load()
        if not text or not text.strip():
            return

        chunks: queue.Queue = queue.Queue(maxsize=64)
        stop = threading.Event()

        def _callback(samples: np.ndarray, progress: float) -> int:
            # Return non-zero to make sherpa-onnx stop generating.
            if stop.is_set() or (cancel is not None and cancel.cancelled):
                return 1
            try:
                chunks.put(np.asarray(samples, dtype=np.float32), timeout=1.0)
            except queue.Full:  # pragma: no cover - consumer stalled
                pass
            return 0

        def _produce() -> None:
            try:
                tts.generate(
                    text,
                    sid=self._speaker_id if speaker_id is None else int(speaker_id),
                    speed=self._speed if speed is None else float(speed),
                    callback=_callback,
                )
            except Exception:  # pragma: no cover - surfaced on the consumer side
                logger.exception("TTS generation failed")
            finally:
                chunks.put(_DONE)

        worker = threading.Thread(target=_produce, name="tts-synth", daemon=True)
        worker.start()
        try:
            while True:
                item = chunks.get()
                if item is _DONE:
                    break
                # sherpa-onnx may hand back one large buffer (or several); the
                # cancel check has to be per *yielded* chunk, not per callback,
                # or a single-buffer utterance could never be interrupted.
                for start in range(0, len(item), chunk_samples):
                    if cancel is not None and cancel.cancelled:
                        stop.set()
                        return
                    yield item[start : start + chunk_samples]
        finally:
            stop.set()
            worker.join(timeout=5.0)
