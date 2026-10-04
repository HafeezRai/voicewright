"""Triton python-backend model: text -> Piper phoneme ids.

Reference implementation of the TTS front-end step (espeak-ng phonemisation +
the Piper ``phoneme_id_map``). It is the piece of a TTS serving path that cannot
live inside the VITS ONNX graph, and it is what an ensemble/BLS would call
before ``piper_tts``.

NOT EXECUTED on the verification host: it needs an espeak-ng python binding
(``espeakng-loader`` / ``piper-phonemize``), which is deliberately not a
dependency of this repository (piper-tts is GPL-3.0 and was rejected in Phase 3;
sherpa-onnx's phonemiser is C++ and not importable). This file exists so the
serving topology is concrete and reviewable; treat it as unverified code until
it has been run against a real Triton python backend.

The layout/contract it implements:
    input  TEXT           : [n_texts]                 (bytes)
    output input          : [n_texts, max_phonemes]   (int64)
    output input_lengths  : [n_texts]                 (int64)
    output scales         : [3]                       (float32)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:  # pragma: no cover - import guard; see module docstring
    import espeakng_loader
    import piper_phonemize
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "tts_frontend needs an espeak-ng phonemiser (espeakng-loader + "
        "piper-phonemize). This is expected to be absent in the repo's verify "
        "environment; install it in the Triton image that serves this model."
    ) from exc


class TritonPythonModel:
    """Triton python-backend entry point."""

    def initialize(self, args):
        params = json.loads(args["model_config"])["parameters"]
        self._voice_dir = Path(params["VOICE_DIR"]["string_value"])
        self._id_map = json.loads(
            (next(self._voice_dir.glob("*.onnx")).with_suffix(".onnx.json")).read_text()
        )["phoneme_id_map"]
        espeakng_loader.make_library_available()
        # Default VITS noise/length scales; length_scale is the speaking-rate knob.
        self._scales = np.asarray([0.667, 1.0, 0.8], dtype=np.float32)

    def _ids(self, text: str) -> list[int]:
        # espeak IPA phonemes -> Piper id sequence, wrapped in BOS/EOS the way
        # the Piper front-end does (`^` ... `$`).
        phonemes = piper_phonemize.phonemize_espeak(text, "en-us")[0]
        ids = [self._id_map["^"][0]]
        for ph in phonemes:
            ids.extend(self._id_map.get(ph, []))
        ids.append(self._id_map["$"][0])
        return ids

    def execute(self, requests):
        responses = []
        for request in requests:
            texts = [t.decode("utf-8") for t in request.inputs()[0].as_numpy()]
            sequences = [self._ids(t) for t in texts]
            width = max(len(s) for s in sequences)
            padded = np.zeros((len(sequences), width), dtype=np.int64)
            lengths = np.zeros((len(sequences),), dtype=np.int64)
            for i, seq in enumerate(sequences):
                padded[i, : len(seq)] = seq
                lengths[i] = len(seq)

            from triton_python_backend_utils import InferenceResponse, Tensor

            responses.append(
                InferenceResponse(
                    request,
                    [
                        Tensor("input", padded),
                        Tensor("input_lengths", lengths),
                        Tensor("scales", self._scales),
                    ],
                    error=None,
                )
            )
        return responses
