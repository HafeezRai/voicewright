"""Numerical parity harness: ONNX Runtime vs the source model.

This is the evidence that the exported graphs are *equivalent*, not merely
loadable. For each exported model it runs fixed inputs through both the source
(PyTorch / the shipped graph) and the exported ONNX graph, and reports the
maximum absolute difference.

What "parity" means per model, honestly stated:

* **minilm_nlu** -- PyTorch ``AutoModel`` vs ONNX Runtime, on both the raw
  ``last_hidden_state`` and the mean-pooled, L2-normalised embedding the
  knowledge-base encoder actually produces. The latter is the number that
  matters: it is the vector retrieval compares against.
* **piper_tts** -- there is no PyTorch source (the voice is authored as ONNX), so
  a cross-framework diff is not defined. Instead this checks the graph runs under
  raw ONNX Runtime, is finite, and is *deterministic* at ``noise_scale=0`` --
  the property a reproducible serving path needs. No parity figure is claimed.

Tolerances: fp32 ONNX Runtime uses different kernel fusion and accumulation order
than PyTorch, so exact equality is not expected. ``abs_tol`` is set at 1e-3 --
three orders of magnitude below the ~1.0 magnitude of the activations, which is
comfortably inside any difference that would change a retrieval result.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np

#: Activation-scale-relative tolerance for fp32 graph re-execution.
ABS_TOL = 1e-3

_SENTENCES = [
    "book a meeting with sales tomorrow",
    "what is the weather like today",
    "I would like to speak to a human agent",
]


def _ort_session(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean-pool then L2-normalise -- mirrors app/knowledge/embeddings.EmbeddingModel."""
    m = mask[..., None].astype(np.float32)
    pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
    norm = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.clip(norm, 1e-9, None)


def parity_minilm(
    onnx_path: Path,
    models_root: Path,
    *,
    seq_lens: tuple[int, ...] = (16, 7, 3),
) -> dict[str, Any]:
    """PyTorch vs ONNX Runtime parity for the MiniLM embedding encoder."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    from export_onnx import MINILM_REPO, _snapshot, pin_hf_offline

    pin_hf_offline(models_root)
    snap = _snapshot(models_root, MINILM_REPO)
    tokenizer = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
    model = AutoModel.from_pretrained(str(snap), local_files_only=True).eval()
    session = _ort_session(onnx_path)

    hidden_diff = 0.0
    embed_diff = 0.0
    per_len: list[dict[str, Any]] = []
    for seq_len in seq_lens:
        batch = tokenizer(
            _SENTENCES,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        )
        with torch.no_grad():
            ref_hidden = model(**batch).last_hidden_state.numpy()
        ort_hidden = session.run(
            None,
            {
                "input_ids": batch["input_ids"].numpy(),
                "attention_mask": batch["attention_mask"].numpy(),
                "token_type_ids": batch["token_type_ids"].numpy(),
            },
        )[0]

        h = float(np.max(np.abs(ort_hidden - ref_hidden)))
        mask = batch["attention_mask"].numpy()
        e = float(np.max(np.abs(_pool(ort_hidden, mask) - _pool(ref_hidden, mask))))
        hidden_diff = max(hidden_diff, h)
        embed_diff = max(embed_diff, e)
        per_len.append({"seq_len": seq_len, "shape": list(ort_hidden.shape),
                        "hidden_max_abs_diff": h, "embedding_max_abs_diff": e})

    return {
        "model": "minilm_nlu",
        "framework": "torch -> onnxruntime",
        "hidden_max_abs_diff": hidden_diff,
        "embedding_max_abs_diff": embed_diff,
        "abs_tol": ABS_TOL,
        "passed": hidden_diff <= ABS_TOL and embed_diff <= ABS_TOL,
        "per_seq_len": per_len,
    }


def check_piper(onnx_path: Path) -> dict[str, Any]:
    """Determinism + finiteness check for the shipped Piper VITS graph.

    No cross-framework parity is claimed -- the voice is authored as ONNX and has
    no PyTorch source. What is checked is that raw ONNX Runtime (the runtime a
    Triton ONNX backend uses) executes the real graph and is deterministic at
    ``noise_scale=0``. Phoneme ids are built from the graph's own
    ``phoneme_id_map`` sidecar so no external phonemiser is needed.
    """
    import json

    import onnxruntime as ort

    meta = json.loads(Path(str(onnx_path) + ".json").read_text())
    id_map = meta["phoneme_id_map"]
    # A short, valid phoneme id sequence: BOS + a few real ids + EOS.
    symbols = ["^", "h", "ə", "l", "oʊ", "$"]
    ids = [id_map[s][0] for s in symbols if s in id_map]
    phonemes = np.asarray([ids], dtype=np.int64)
    lengths = np.asarray([len(ids)], dtype=np.int64)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def synth(noise: float, length: float, noise_w: float) -> np.ndarray:
        scales = np.asarray([noise, length, noise_w], dtype=np.float32)
        return session.run(
            None,
            {"input": phonemes, "input_lengths": lengths, "scales": scales},
        )[0]

    a = synth(0.0, 1.0, 0.0)
    b = synth(0.0, 1.0, 0.0)
    det = float(np.max(np.abs(a - b)))
    return {
        "model": "piper_tts",
        "framework": "onnxruntime (authored ONNX; no torch source)",
        "samples": int(a.size),
        "duration_s": round(float(a.size) / 16000.0, 4),
        "finite": bool(np.isfinite(a).all()),
        "determinism_max_abs_diff": det,
        "abs_tol": 0.0,
        "passed": bool(np.isfinite(a).all() and det == 0.0),
        "note": "No cross-framework parity claimed: no PyTorch source exists for this voice.",
    }


def find_artifact(artifacts_dir: Path, name: str) -> Path:
    """Locate an exported artifact, tolerating the exact filename chosen by the exporter."""
    if name == "piper_tts":
        from export_onnx import DEFAULT_MODELS_ROOT, PIPER_VOICE_DIRNAME

        voices = glob.glob(str(DEFAULT_MODELS_ROOT / "tts" / PIPER_VOICE_DIRNAME / "*.onnx"))
        if not voices:
            raise FileNotFoundError("no Piper voice found")
        return Path(voices[0])
    candidate = artifacts_dir / f"{name}.onnx"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"{candidate} not found; run `python mlops-pipeline/export/export_onnx.py --model all` first"
        )
    return candidate
