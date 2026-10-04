"""ONNX export for the CloudPSO ai-engine models (Phase 5).

This is the bridge between the checkpoints Phase 3 selected and the artifacts a
Triton inference server can actually load. It is deliberately *explicit* about
which stages get an ONNX graph and which do not, because two of the four models
are already in their serving format:

    stage                source                      onnx?   triton backend
    -------------------  --------------------------  ------  -----------------
    NLU  (fallback)      all-MiniLM-L6-v2 (torch)    EXPORT  onnxruntime
    NLU  (primary)       Laya ModernBERT (torch)     EXPORT  onnxruntime
    TTS                  Piper VITS (already .onnx)  none    onnxruntime
    STT                  faster-whisper (CT2 dir)    none    ctranslate2

STT is the case worth being careful about. ``faster-whisper`` ships a
**CTranslate2** model directory, not a PyTorch checkpoint: ``model.bin`` +
``config.json`` + ``tokenizer.json`` + ``vocabulary.txt``. CTranslate2 is its own
optimised runtime (quantisation, fused kernels, int8) and is what Triton's
``ctranslate2`` backend consumes directly. Round-tripping those weights through
ONNX would throw away the quantisation and add a conversion step with no upside,
so **no export is attempted** -- the correct Triton target is the CTranslate2
backend and :func:`describe_whisper_ct2` records the artifact for its config.

TTS is the same story in a different flavour: the Piper voice *is* an ``.onnx``
file authored by the Piper training pipeline. There is no PyTorch source to
compare against, so no export and no PyTorch-parity claim; :func:`stage_piper`
verifies the graph loads under raw ONNX Runtime and records its I/O signature.

Everything here is offline-first: model directories are resolved from the
ai-engine model cache and HuggingFace is pinned to ``local_files_only`` so an
export run cannot silently depend on the network.

Run it (invoke by path -- *not* ``python -m`` from ``mlops-pipeline/``, because
that directory contains ``triton/`` and would shadow the real ``triton`` package
that ``import torch`` needs; see ``tests/conftest.py``)::

    python mlops-pipeline/export/export_onnx.py --model all
    python mlops-pipeline/export/export_onnx.py --model minilm laya
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# export/ -> mlops-pipeline/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
AI_ENGINE_ROOT = _REPO_ROOT / "ai-engine"
DEFAULT_MODELS_ROOT = AI_ENGINE_ROOT / "models"
DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

#: HuggingFace repo ids, matching ai-engine/app/config.py defaults.
MINILM_REPO = "sentence-transformers/all-MiniLM-L6-v2"
LAYA_REPO = "convaiinnovations/laya"
WHISPER_CT2_REPO = "Systran/faster-whisper-base.en"
PIPER_VOICE_DIRNAME = "vits-piper-en_US-amy-low"

#: Pinned opset. 17 is the highest opset the CUDA/CPU ONNX Runtime builds Triton
#: ships with a stable operator set for, and what both exporters below target.
ONNX_OPSET = 17

#: Export-time sequence length for the tracer. The graph is exported with
#: dynamic batch/sequence axes (see ``dynamic_axes`` below); this only fixes the
#: concrete shape the tracer *sees*, and is verified to be irrelevant by the
#: dynamic-shape parity test in ``tests/test_export_parity.py``.
_TRACE_SEQ_LEN = 16


# --------------------------------------------------------------------------
# Offline HuggingFace cache resolution
# --------------------------------------------------------------------------
def pin_hf_offline(models_root: Path) -> None:
    """Point HuggingFace at the ai-engine cache and forbid network access.

    ``setdefault`` keeps an operator's explicit ``HF_HOME``; the offline flags
    are set unconditionally so an export either uses the local cache or fails
    loudly rather than quietly downloading a second copy of a checkpoint.
    """
    os.environ.setdefault("HF_HOME", str(models_root / "hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _snapshot(models_root: Path, repo_id: str) -> Path:
    """Resolve the single cached snapshot directory for ``repo_id``.

    ``models--<org>--<name>/snapshots/<rev>`` is the huggingface_hub layout the
    ai-engine cache uses (see app/config.py: hf_hub_cache).
    """
    repo_dir = models_root / "hf" / "hub" / ("models--" + repo_id.replace("/", "--"))
    snapshots = sorted((repo_dir / "snapshots").glob("*"))
    if not snapshots:
        raise FileNotFoundError(
            f"no cached snapshot for {repo_id!r} under {repo_dir}. "
            "Run ai-engine/scripts/fetch_models.sh first."
        )
    return snapshots[-1]


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------
@dataclass
class Artifact:
    """One exportable/stageable model artifact and the facts a Triton config needs."""

    model: str
    stage: str
    source: str
    status: str  # "exported" | "already-onnx" | "native-ct2" | "failed"
    path: str = ""
    size_bytes: int = 0
    sha256: str = ""
    opset: int = 0
    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    triton_backend: str = ""
    notes: str = ""
    seconds: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _rel(path: Path) -> str:
    """Serialise a path relative to the repo root, so the manifest is portable.

    Absolute paths would bake this machine's checkout location into a committed
    file; ``verify_manifest.py`` resolves these back against the repo root.

    Deliberately uses ``absolute()`` rather than ``resolve()``: the HuggingFace
    cache stores snapshots as symlinks into ``blobs/``, and resolving them would
    record a content-addressed blob path that says nothing about which model it
    belongs to.
    """
    try:
        return os.path.relpath(Path(path).absolute(), _REPO_ROOT)
    except ValueError:
        return str(path)


def _sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _ort_signature(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read the input/output signature of an ONNX graph via ONNX Runtime."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = [
        {"name": i.name, "type": i.type, "shape": [str(d) for d in i.shape]}
        for i in session.get_inputs()
    ]
    outputs = [
        {"name": o.name, "type": o.type, "shape": [str(d) for d in o.shape]}
        for o in session.get_outputs()
    ]
    return inputs, outputs


# --------------------------------------------------------------------------
# Exporters
# --------------------------------------------------------------------------
def export_minilm(
    models_root: Path = DEFAULT_MODELS_ROOT,
    out_dir: Path = DEFAULT_ARTIFACTS_DIR,
) -> Artifact:
    """Export the HuggingFace NLU fallback (all-MiniLM-L6-v2) to ONNX.

    A plain ``BertModel`` encoder, so the graph is ``input_ids`` /
    ``attention_mask`` / ``token_type_ids`` -> ``last_hidden_state`` with
    dynamic batch and sequence axes. The ai-engine NLU embeds by mean-pooling
    ``last_hidden_state`` and L2-normalising, which is arithmetic done outside
    the graph in both the PyTorch and Triton paths -- so the graph boundary is
    the hidden state, not the embedding.

    ``AutoModel`` for a BERT returns *two* tensors (``last_hidden_state`` and the
    unused ``pooler_output``). The model is wrapped so the graph exposes only the
    hidden state the service consumes -- otherwise the ONNX artifact carries a
    dead output that the Triton config would have to declare and the server
    would have to allocate on every request.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    class _EncoderOnly(torch.nn.Module):
        """Return only ``last_hidden_state``; drop the unused pooler output."""

        def __init__(self, backbone: torch.nn.Module) -> None:
            super().__init__()
            self.backbone = backbone

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).last_hidden_state

    pin_hf_offline(models_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = _snapshot(models_root, MINILM_REPO)
    out_path = out_dir / "minilm_nlu.onnx"

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
    backbone = AutoModel.from_pretrained(str(snap), local_files_only=True).eval()
    model = _EncoderOnly(backbone).eval()

    batch = tokenizer(
        ["book a meeting with sales tomorrow", "what is the weather"],
        padding="max_length",
        truncation=True,
        max_length=_TRACE_SEQ_LEN,
        return_tensors="pt",
    )
    args = (batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])

    torch.onnx.export(
        model,
        args,
        str(out_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        dynamo=False,  # TorchScript tracer: wider operator coverage for BERT
    )
    inputs, outputs = _ort_signature(out_path)
    return Artifact(
        model="minilm_nlu",
        stage="nlu_fallback",
        source=MINILM_REPO,
        status="exported",
        path=_rel(out_path),
        size_bytes=out_path.stat().st_size,
        sha256=_sha256(out_path),
        opset=ONNX_OPSET,
        inputs=inputs,
        outputs=outputs,
        triton_backend="onnxruntime",
        notes="Mean-pool + L2-norm stay outside the graph (match ai-engine HFIntentNLU).",
        seconds=round(time.time() - t0, 2),
    )


def _load_laya_encoder(models_root: Path):
    """Load the Laya ModernBERT encoder as a plain ``ModernBertModel``.

    Laya's checkpoint stores one ``model.safetensors`` for the whole RL agent
    (policy head + encoder are in the same file) with the encoder's weights
    namespaced ``encoder.*``, and the encoder's *config* in ``encoder/config.json``.
    The encoder is what a Triton ONNX backend serves, so we rebuild it from the
    config and load only the ``encoder.*`` tensors.
    """
    import torch  # noqa: F401 - ensures torch is importable before safetensors load
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoTokenizer, ModernBertModel

    snap = _snapshot(models_root, LAYA_REPO)
    config = AutoConfig.from_pretrained(str(snap / "encoder"), local_files_only=True)
    model = ModernBertModel(config).eval()
    state = load_file(str(snap / "model.safetensors"))
    encoder_state = {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")}
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Laya encoder state_dict mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected"
        )
    tokenizer = AutoTokenizer.from_pretrained(str(snap / "tokenizer"), local_files_only=True)
    return model, tokenizer


def export_laya_encoder(
    models_root: Path = DEFAULT_MODELS_ROOT,
    out_dir: Path = DEFAULT_ARTIFACTS_DIR,
) -> Artifact:
    """Export Laya's ModernBERT-large encoder to ONNX.

    Phase 3's README notes that Laya bundles an ONNX *runtime* but not its
    ``export_onnx.py``, so this is an independent ``torch.onnx.export`` of the
    encoder body. Only the encoder is exported: the RL policy head (``act_head``
    / ``scorer``, ~36 tensors) is small and cheap to keep in the client, and the
    expensive part -- the 28-layer, 1024-hidden transformer whose embeddings the
    policy head consumes -- is what benefits from GPU batching in Triton.

    The 1.6 GiB fp32 graph is written to the gitignored ``artifacts/`` dir; only
    its hash and signature are committed (see ``manifest.json``).
    """
    import torch

    pin_hf_offline(models_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = _snapshot(models_root, LAYA_REPO)
    out_path = out_dir / "laya_encoder.onnx"

    t0 = time.time()
    model, tokenizer = _load_laya_encoder(models_root)
    batch = tokenizer(
        ["book a meeting with sales tomorrow", "hi"],
        padding="max_length",
        truncation=True,
        max_length=_TRACE_SEQ_LEN,
        return_tensors="pt",
    )

    torch.onnx.export(
        model,
        (batch["input_ids"], batch["attention_mask"]),
        str(out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    inputs, outputs = _ort_signature(out_path)
    return Artifact(
        model="laya_encoder",
        stage="nlu_primary",
        source=LAYA_REPO,
        status="exported",
        path=_rel(out_path),
        size_bytes=out_path.stat().st_size,
        sha256=_sha256(out_path),
        opset=ONNX_OPSET,
        inputs=inputs,
        outputs=outputs,
        triton_backend="onnxruntime",
        notes=(
            "Encoder body only (ModernBertModel); Laya's act_head/scorer stay in "
            "the client. Tokenizer/temperature artefacts must be shipped alongside."
        ),
        seconds=round(time.time() - t0, 2),
    )


def stage_piper(
    models_root: Path = DEFAULT_MODELS_ROOT,
    out_dir: Path = DEFAULT_ARTIFACTS_DIR,
) -> Artifact:
    """Stage the Piper VITS voice -- already ONNX, so this verifies, not exports.

    The sherpa-onnx Piper voice ships as ``en_US-amy-low.onnx`` with a
    ``tokens.txt`` and ``espeak-ng-data`` phonemiser. ONNX Runtime (the runtime
    Triton's ONNX backend uses) loads the graph directly; this function records
    its real I/O signature so the Triton ``config.pbtxt`` matches the artifact
    rather than a guess.
    """
    pin_hf_offline(models_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    voice_dir = models_root / "tts" / PIPER_VOICE_DIRNAME
    onnx_files = sorted(voice_dir.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(
            f"no Piper .onnx voice under {voice_dir}; run ai-engine/scripts/fetch_models.sh"
        )
    onnx_path = onnx_files[0]
    inputs, outputs = _ort_signature(onnx_path)
    return Artifact(
        model="piper_tts",
        stage="tts",
        source=_rel(onnx_path),
        status="already-onnx",
        path=_rel(onnx_path),
        size_bytes=onnx_path.stat().st_size,
        sha256=_sha256(onnx_path),
        opset=0,  # authored by the Piper pipeline; opset not declared in-band
        inputs=inputs,
        outputs=outputs,
        triton_backend="onnxruntime",
        notes=(
            "Authored as ONNX by the Piper pipeline -- nothing to export. I/O "
            "signature read from the real graph via ONNX Runtime."
        ),
    )


def describe_whisper_ct2(
    models_root: Path = DEFAULT_MODELS_ROOT,
    out_dir: Path | None = None,  # accepted for a uniform exporter signature; unused
) -> Artifact:
    """Record the faster-whisper CTranslate2 directory for the CTranslate2 backend.

    Deliberately does **not** export to ONNX: see the module docstring. The
    Triton ``ctranslate2`` backend loads ``model.bin`` (with ``config.json`` and
    the tokenizer alongside) exactly as faster-whisper does.
    """
    del out_dir  # this exporter produces no artifact of its own
    snap = _snapshot(models_root, WHISPER_CT2_REPO)
    model_bin = snap / "model.bin"
    if not model_bin.is_file():
        raise FileNotFoundError(f"expected CTranslate2 model.bin in {snap}")
    return Artifact(
        model="whisper_ct2",
        stage="stt",
        source=WHISPER_CT2_REPO,
        status="native-ct2",
        path=_rel(model_bin),
        size_bytes=model_bin.stat().st_size,
        sha256=_sha256(model_bin),
        inputs=[],
        outputs=[],
        triton_backend="ctranslate2",
        notes=(
            "CTranslate2 model directory (model.bin + config.json + tokenizer + "
            "vocabulary). Served by Triton's ctranslate2 backend; converting to "
            "ONNX would discard the int8 quantisation. No export by design."
        ),
    )


EXPORTERS = {
    "minilm": export_minilm,
    "laya": export_laya_encoder,
    "piper": stage_piper,
    "whisper": describe_whisper_ct2,
}


def run(models: list[str], models_root: Path, out_dir: Path) -> list[Artifact]:
    """Run the named exporters, collecting failures rather than aborting."""
    results: list[Artifact] = []
    for name in models:
        fn = EXPORTERS[name]
        try:
            art = fn(models_root=models_root, out_dir=out_dir)
        except Exception as exc:  # noqa: BLE001 - a failed stage must not hide the rest
            art = Artifact(
                model=name,
                stage="?",
                source="?",
                status="failed",
                notes=f"{type(exc).__name__}: {exc}",
            )
        results.append(art)
        flag = "ok " if art.status != "failed" else "FAIL"
        print(f"[{flag}] {art.model:14s} {art.status:13s} {art.notes[:70]}", flush=True)
    return results


def write_manifest(results: list[Artifact], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.json"
    payload = {
        "generated_by": "mlops-pipeline/export/export_onnx.py",
        "opset": ONNX_OPSET,
        "artifacts": [r.to_json() for r in results],
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export ai-engine models to ONNX / stage native ones.")
    parser.add_argument(
        "--model",
        nargs="+",
        default=["all"],
        choices=["all", *EXPORTERS],
        help="Which models to process (default: all).",
    )
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args(argv)

    models = list(EXPORTERS) if "all" in args.model else args.model
    results = run(models, args.models_root, args.out_dir)
    manifest = write_manifest(results, args.out_dir)
    print(f"\nmanifest -> {manifest}")
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
