"""Make the export/parity modules importable from the tests.

``export_onnx`` and ``parity`` live in ``mlops-pipeline/export/`` and are
imported by name, so that directory is added to ``sys.path``.

This conftest deliberately adds ``export/`` rather than ``mlops-pipeline/``
itself. The tests are run both from the repository root and from inside
``mlops-pipeline/`` (e.g. ``cd mlops-pipeline && python -m pytest``), and the
latter puts ``mlops-pipeline/`` on ``sys.path``. Adding ``export/`` keeps the
imports working either way without widening what is importable.

The Triton model repository is *not* importable code and intentionally no
longer lives in a directory named ``triton`` -- it is ``triton_repo/`` -- so that
nothing in this package can shadow the real PyTorch ``triton`` package (a
namespace package with no ``language`` attribute breaks the *guarded* import
inside ``torch._dynamo``, surfacing as a misleading
``ImportError: cannot import name 'GenerationMixin' from 'transformers.generation'``).
See ``test_triton_configs.py`` for how the validator is loaded by file path.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXPORT_DIR = Path(__file__).resolve().parents[1] / "export"
if str(_EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPORT_DIR))
