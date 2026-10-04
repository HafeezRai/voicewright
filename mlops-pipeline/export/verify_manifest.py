"""Verify staged artifacts match the sha256 values recorded in the manifest.

The promotion gate from ``configs/promotion.yaml``: production must serve
exactly the bytes that passed numerical parity, so before promotion the staged
binaries are re-hashed and compared against ``artifacts/manifest.json``.

Run::

    python mlops-pipeline/export/verify_manifest.py

Exits non-zero on any mismatch, missing file, or missing manifest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from export_onnx import DEFAULT_ARTIFACTS_DIR, _REPO_ROOT, _sha256

MANIFEST = DEFAULT_ARTIFACTS_DIR / "manifest.json"


def main() -> int:
    if not MANIFEST.is_file():
        print(f"FATAL: {MANIFEST} missing; run export_onnx.py first", file=sys.stderr)
        return 2

    payload = json.loads(MANIFEST.read_text())
    failures = 0
    checked = 0
    for art in payload["artifacts"]:
        if art["status"] in {"failed"} or not art.get("sha256"):
            continue
        # Manifest paths are repo-relative (see export_onnx._rel); resolve them
        # so the check works from any checkout.
        path = Path(art["path"])
        if not path.is_absolute():
            path = _REPO_ROOT / path
        if not path.is_file():
            print(f"  MISSING  {art['model']:14s} {path}")
            failures += 1
            continue
        actual = _sha256(path)
        checked += 1
        if actual != art["sha256"]:
            print(f"  MISMATCH {art['model']:14s} manifest={art['sha256'][:12]} actual={actual[:12]}")
            failures += 1
        else:
            print(f"  ok       {art['model']:14s} {art['sha256'][:12]} ({art['size_bytes'] / 1e6:.1f} MB)")

    print(f"\n{checked} artifact(s) verified, {failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
