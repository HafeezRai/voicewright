"""Make the export modules importable without shadowing third-party packages.

IMPORTANT — why this adds ``export/`` and not ``mlops-pipeline/``:

``mlops-pipeline/triton/`` is a *directory named ``triton``*. If
``mlops-pipeline`` were placed on ``sys.path``, ``import triton`` would resolve to
that directory (as an implicit namespace package) instead of the real PyTorch
Triton package. ``torch._dynamo`` imports ``triton.language`` unconditionally
during ``import torch``, so the shadowing raises
``AttributeError: module 'triton' has no attribute 'language'`` deep inside
``torch._dynamo`` -- which surfaces, confusingly, as
``ImportError: cannot import name 'GenerationMixin' from 'transformers.generation'``.

Adding only ``mlops-pipeline/export`` keeps the export/parity modules importable
while leaving ``triton`` unshadowed. The Triton config validator is loaded by
file path for the same reason (see ``test_triton_configs.py``); running it
directly (``python mlops-pipeline/triton/validate_configs.py``) is also safe,
because a script puts *its own* directory -- not its parent -- on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXPORT_DIR = Path(__file__).resolve().parents[1] / "export"
if str(_EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPORT_DIR))
