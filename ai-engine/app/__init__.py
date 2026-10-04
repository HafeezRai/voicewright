"""CloudPSO Voice Agent — ai-engine application package.

Phase 1 scope: importable app skeleton + configuration layer only.
The STT -> NLU -> TTS pipeline is Phase 3 work.

Nothing in this package imports torch/transformers/onnxruntime/aiortc at module
level, so `/health` is testable without the multi-GB ML stack installed.
"""

__version__ = "0.1.0"
