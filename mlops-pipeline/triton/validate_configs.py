"""Validate every ``config.pbtxt`` against Triton's real ``ModelConfig`` schema.

Triton itself fails *silently* on a malformed config: the model simply stays
UNAVAILABLE and the server logs nothing fatal, so a typo can look like a model
that never loads for unrelated reasons. This script catches that class of error
cheaply by parsing each config with the actual Triton protobuf
(``tritonclient.grpc.model_config_pb2``, shipped by the ``tritonclient``
package) and then checking the invariants Triton enforces at load time.

It is a *static* check: it never starts a Triton server and cannot prove a model
would load or serve correctly. What it proves is that the config is a valid
``ModelConfig`` and internally consistent.

Run directly (a script puts its *own* directory on ``sys.path``, not its parent,
so this does not shadow the real ``triton`` package -- see the note in
``../tests/conftest.py``)::

    python mlops-pipeline/triton/validate_configs.py

Exits non-zero if any config fails.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from google.protobuf import text_format

try:
    from tritonclient.grpc import model_config_pb2
except Exception as exc:  # pragma: no cover - dependency guard
    print(f"FATAL: tritonclient is required to validate configs ({exc})", file=sys.stderr)
    raise SystemExit(2)

REPO_ROOT = Path(__file__).resolve().parent / "model_repository"

#: Backends this repository intentionally uses. A typo'd backend name ("onnx",
#: "onnx_runtime") would parse fine but never load.
KNOWN_BACKENDS = {"onnxruntime", "ctranslate2", "python", "tensorrt", "pytorch"}


@dataclass
class Result:
    model: str
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_one(config_path: Path) -> Result:
    """Parse and check a single config.pbtxt. Never raises for config errors."""
    errors: list[str] = []
    model_name = config_path.parent.name

    try:
        raw = config_path.read_text()
    except OSError as exc:
        return Result(model_name, [f"cannot read: {exc}"])

    config = model_config_pb2.ModelConfig()
    try:
        text_format.Parse(raw, config)
    except text_format.ParseError as exc:
        return Result(model_name, [f"protobuf parse error: {exc}"])

    # --- invariants Triton enforces at load -------------------------------
    if config.name != model_name:
        errors.append(
            f"name {config.name!r} != directory name {model_name!r} "
            "(Triton requires an exact match)"
        )
    if not config.backend:
        errors.append("backend is empty")
    elif config.backend not in KNOWN_BACKENDS:
        errors.append(f"backend {config.backend!r} is not one of {sorted(KNOWN_BACKENDS)}")
    if not config.input:
        errors.append("no inputs declared")
    if not config.output:
        errors.append("no outputs declared")

    has_dynamic = config.dynamic_batching.max_queue_delay_microseconds > 0 or any(
        config.dynamic_batching.preferred_batch_size
    )
    if has_dynamic and config.max_batch_size == 0:
        errors.append("dynamic_batching is set but max_batch_size is 0 (Triton rejects this)")

    for kind, tensors in (("input", config.input), ("output", config.output)):
        for tensor in tensors:
            if tensor.data_type == 0:
                errors.append(f"{kind} {tensor.name!r} has no data_type")
            if not tensor.name:
                errors.append(f"{kind} #? has an empty name")
            # A negative dim is valid (dynamic); a zero dim is not.
            if any(d == 0 for d in tensor.dims):
                errors.append(f"{kind} {tensor.name!r} has a zero-sized dim {list(tensor.dims)}")

    # --- version directory sanity (weights live under a numeric version) ---
    version_dirs = [p for p in config_path.parent.iterdir() if p.is_dir()]
    if version_dirs:
        for vd in version_dirs:
            if not vd.name.isdigit():
                errors.append(f"version dir {vd.name!r} is not numeric")

    return Result(model_name, errors)


def validate_all(repo: Path = REPO_ROOT) -> list[Result]:
    configs = sorted(repo.glob("*/config.pbtxt"))
    if not configs:
        raise SystemExit(f"no config.pbtxt files found under {repo}")
    return [validate_one(p) for p in configs]


def main() -> int:
    results = validate_all()
    width = max(len(r.model) for r in results)
    failed = 0
    for r in results:
        if r.ok:
            print(f"  OK   {r.model:<{width}}  (parsed as triton.ModelConfig)")
        else:
            failed += 1
            print(f"  FAIL {r.model:<{width}}")
            for e in r.errors:
                print(f"         - {e}")
    print(f"\n{len(results) - failed}/{len(results)} configs valid")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
