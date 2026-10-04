# mlops-pipeline

Phase 5 territory — **empty skeleton in Phase 1**. Only the directory shape and
this description exist; no code, configs, or model artifacts are present yet.

## Intended role

The bridge between trained models and the `ai-engine` runtime:

- **`export/`** — scripts that convert PyTorch/HuggingFace checkpoints into
  optimised **ONNX** artifacts (dynamic axes, opset pinning, quantisation), plus
  parity checks that the exported graph matches the eager model.
- **`triton/`** — **Triton Inference Server** model repository layout
  (`<model_name>/<version>/model.onnx` + `config.pbtxt`), ensemble/dynamic
  batching configuration, and the model-control settings the engine calls into.
- **`configs/`** — pipeline configuration: which checkpoints map to which
  exported models, per-environment Triton endpoints, and CI/CD promotion gates
  (staging -> production).

Owned by Phase 5. Earlier phases must not add model configs here.
