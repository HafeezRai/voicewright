"""Stage real model artifacts into the Triton model repository's version dirs.

A Triton repository needs each model's weights under a numeric version
directory (``<model>/1/``). Those weights are large (the Laya encoder graph
alone is ~1.6 GiB) and are gitignored, so they are *staged*, not committed:
``config.pbtxt`` files live in git; the binaries do not.

Run the exporter first, then this::

    python mlops-pipeline/export/export_onnx.py --model all
    python mlops-pipeline/export/populate_triton_repo.py            # all models
    python mlops-pipeline/export/populate_triton_repo.py minilm_nlu # one model

The repository is not runnable here (this host has no NVIDIA GPU); staging is
what makes it runnable *elsewhere*. ``--check`` verifies the expected files are
present without copying anything.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from export_onnx import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_MODELS_ROOT,
    PIPER_VOICE_DIRNAME,
    _snapshot,
    WHISPER_CT2_REPO,
)

TRITON_REPO = Path(__file__).resolve().parents[1] / "triton" / "model_repository"


def _ensure_clean_version_dir(model: str) -> Path:
    version_dir = TRITON_REPO / model / "1"
    if version_dir.exists():
        shutil.rmtree(version_dir)
    version_dir.mkdir(parents=True)
    return version_dir


def stage_minilm(check: bool = False) -> list[Path]:
    return _stage_onnx("minilm_nlu", DEFAULT_ARTIFACTS_DIR / "minilm_nlu.onnx", check)


def stage_laya(check: bool = False) -> list[Path]:
    return _stage_onnx("laya_encoder", DEFAULT_ARTIFACTS_DIR / "laya_encoder.onnx", check)


def _stage_onnx(model: str, src: Path, check: bool) -> list[Path]:
    if check:
        target = TRITON_REPO / model / "1" / "model.onnx"
        return [target]
    if not src.is_file():
        raise FileNotFoundError(f"{src} missing; run export_onnx.py first")
    version_dir = _ensure_clean_version_dir(model)
    target = version_dir / "model.onnx"
    shutil.copy2(src, target)
    return [target]


def stage_piper(check: bool = False) -> list[Path]:
    voice = DEFAULT_MODELS_ROOT / "tts" / PIPER_VOICE_DIRNAME
    if check:
        return [TRITON_REPO / "piper_tts" / "1" / "en_US-amy-low.onnx"]
    onnx_files = sorted(voice.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"no Piper voice under {voice}")
    version_dir = _ensure_clean_version_dir("piper_tts")
    # The VITS graph plus everything the front-end needs to drive it.
    written = [shutil.copy2(onnx_files[0], version_dir / onnx_files[0].name)]
    written.append(shutil.copy2(voice / "tokens.txt", version_dir / "tokens.txt"))
    for sidecar in (Path(str(onnx_files[0]) + ".json"), voice / "MODEL_CARD"):
        if sidecar.is_file():
            written.append(shutil.copy2(sidecar, version_dir / sidecar.name))
    data_dir = voice / "espeak-ng-data"
    if data_dir.is_dir():
        shutil.copytree(data_dir, version_dir / "espeak-ng-data")
        written.append(version_dir / "espeak-ng-data")
    return written


def stage_whisper(check: bool = False) -> list[Path]:
    snap = _snapshot(DEFAULT_MODELS_ROOT, WHISPER_CT2_REPO)
    target = TRITON_REPO / "whisper_ct2" / "1" / "model"
    if check:
        return [target / "model.bin"]
    version_dir = _ensure_clean_version_dir("whisper_ct2")
    # CTranslate2 backend convention: the converted model directory is nested
    # under the version dir as `model/`.
    shutil.copytree(snap, version_dir / "model",
                    ignore=shutil.ignore_patterns("*.safetensors", ".cache"))
    return [version_dir / "model" / "model.bin"]


STAGERS = {
    "minilm_nlu": stage_minilm,
    "laya_encoder": stage_laya,
    "piper_tts": stage_piper,
    "whisper_ct2": stage_whisper,
}
# tts_frontend is a python-backend model whose code is committed directly.

#: Files each model's version dir must contain for Triton to load it.
REQUIRED = {
    "minilm_nlu": ["1/model.onnx"],
    "laya_encoder": ["1/model.onnx"],
    "piper_tts": ["1/en_US-amy-low.onnx", "1/tokens.txt"],
    "whisper_ct2": ["1/model/model.bin"],
    "tts_frontend": ["1/model.py"],
}


def check_all() -> int:
    failures = 0
    for model, rels in REQUIRED.items():
        missing = [r for r in rels if not (TRITON_REPO / model / r).exists()]
        if missing:
            failures += 1
            print(f"  MISSING {model}: {missing}")
        else:
            print(f"  ok      {model}: {rels}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # `nargs="*"` + `choices` mis-validates the empty default in argparse, so the
    # choice check is done by hand.
    parser.add_argument("models", nargs="*", metavar="MODEL")
    parser.add_argument("--check", action="store_true", help="verify without copying")
    args = parser.parse_args(argv)
    unknown = [m for m in args.models if m not in STAGERS]
    if unknown:
        parser.error(f"unknown model(s) {unknown}; choose from {list(STAGERS)}")

    if args.check:
        failures = check_all()
        print(f"\n{'all required artifacts present' if not failures else f'{failures} model(s) incomplete'}")
        return 1 if failures else 0

    models = args.models or list(STAGERS)
    for model in models:
        try:
            written = STAGERS[model]()
            total = sum(p.stat().st_size for p in written if p.is_file())
            print(f"  staged {model:14s} -> {len(written)} path(s), {total / 1e6:.1f} MB")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {model:14s} {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    print(f"\nrepository ready under {TRITON_REPO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
