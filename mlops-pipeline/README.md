# mlops-pipeline

Phase 5. The bridge from the checkpoints `ai-engine` selected to the artifacts a
**Triton Inference Server** can load: ONNX export, numerical-parity evidence, and
the Triton model repository.

This is no longer a skeleton — it contains runnable tooling that has been run and
whose results are reported below.

```
ai-engine/models/            export/                          triton/
  (checkpoints)     ──►        export_onnx.py       ──►        model_repository/
                              parity.py                        <model>/config.pbtxt
                              populate_triton_repo.py          <model>/1/<weights>
                              verify_manifest.py
```

## What is validated vs. what has been executed

Read this table before trusting anything elsewhere in the repository. This host
is **arm64 macOS with no NVIDIA GPU**, so Triton and Kubernetes **cannot run
here**, and no claim below says otherwise.

| Artifact | Status | Evidence |
|---|---|---|
| `export/export_onnx.py` — export MiniLM & Laya, stage Piper/Whisper | **EXECUTED** | All four models processed; `manifest.json` written with real sha256 + I/O signatures |
| Numerical parity (MiniLM, Laya) | **EXECUTED** | PyTorch vs ONNX Runtime on real inputs; max-abs-diff 1.79e-05 / 2.16e-04 (see below) |
| Piper ONNX determinism check | **EXECUTED** | Raw ONNX Runtime forward pass; determinism diff 0.0 |
| STT CTranslate2 load + transcription | **EXECUTED** | `faster-whisper` transcribes `speech_16k.wav` → "percent global wealth." |
| `triton/model_repository/*/config.pbtxt` | **STATICALLY VALIDATED** | Parsed against the real `triton.ModelConfig` protobuf + consistency checks (5/5 valid) |
| `triton/docker-compose.triton.yml` | **STATICALLY VALIDATED** | `docker compose config` exits 0 |
| **Triton Inference Server itself** | **NEVER RUN** | No NVIDIA GPU on this host; `nvcr.io/nvidia/tritonserver` cannot start here |
| `tts_frontend/1/model.py` (python backend) | **NOT EXECUTED** | Reference implementation; needs an espeak-ng phonemiser that is not installed here |

The Triton configs are **statically validated only**. A `config.pbtxt` that
parses is not a model that serves. Nothing in this directory has been run behind
a Triton server.

## Layout

| Path | Purpose |
|---|---|
| `export/export_onnx.py` | Export / stage every model; writes `artifacts/manifest.json` |
| `export/parity.py` | PyTorch↔ONNX Runtime numerical-parity harness |
| `export/populate_triton_repo.py` | Copy artifacts into `<model>/1/` version dirs |
| `export/verify_manifest.py` | Re-hash artifacts and compare against the manifest (CI gate) |
| `triton/model_repository/` | One `<model>/config.pbtxt` per model |
| `triton/validate_configs.py` | Parse every config against Triton's protobuf schema |
| `triton/docker-compose.triton.yml` | GPU deployment of Triton (documentation; not run here) |
| `configs/models.yaml` | Stage → artifact → Triton-model mapping |
| `configs/promotion.yaml` | CI promotion gates (staging → production) |
| `tests/` | Export-parity and Triton-config tests |

## Why STT is not exported to ONNX

`faster-whisper`'s artifact is a **CTranslate2** model directory (`model.bin` +
`config.json` + `tokenizer.json` + `vocabulary.txt`), not a PyTorch checkpoint.
CTranslate2 is its own optimised runtime — int8 quantisation, fused kernels — and
is exactly what Triton's `ctranslate2` backend consumes. Converting those weights
to ONNX would **discard the quantisation** and add a lossy round-trip for no
benefit. The correct Triton target is the CTranslate2 backend, and
`whisper_ct2/config.pbtxt` configures it. See `configs/models.yaml`.

Likewise **TTS is already ONNX**: the Piper voice is an `.onnx` file authored by
the Piper training pipeline. There is no PyTorch source, so there is no export and
no cross-framework parity claim — only a determinism/finiteness check.

## Why the configs are validated against Triton's own protobuf

Triton fails **silently** on a malformed config: the model simply stays
`UNAVAILABLE` and no fatal error is logged. `validate_configs.py` therefore parses
each file with the genuine `tritonclient.grpc.model_config_pb2.ModelConfig` and
checks the invariants Triton enforces (name == directory name, backend is real,
`dynamic_batching` requires `max_batch_size > 0`, version dirs are numeric).

## Export results (this machine)

```
[ok ] minilm_nlu     exported      torch.onnx.export, opset 17, 90.4 MB
[ok ] laya_encoder   exported      torch.onnx.export, opset 17, 1.58 GB
[ok ] piper_tts      already-onnx  authored as ONNX by the Piper pipeline
[ok ] whisper_ct2    native-ct2    CTranslate2 directory; no export by design
```

Numerical parity — **real numbers from this run**:

| Model | Comparison | Max abs diff | Tolerance |
|---|---|---|---|
| `minilm_nlu` | PyTorch vs ORT, `last_hidden_state` | **1.79e-05** | 1e-3 |
| `minilm_nlu` | PyTorch vs ORT, mean-pooled embedding | **3.28e-07** | 1e-3 |
| `laya_encoder` | PyTorch vs ORT, `last_hidden_state` | **2.16e-04** | 1e-3 |
| `piper_tts` | ORT determinism @ `noise_scale=0` | **0.0** | exact |

`minilm_nlu` and `laya_encoder` were re-checked at three different sequence
lengths (16 / 7 / 3 and 16 / 11 / 3) to prove the exported dynamic axes are real
rather than frozen at the trace length. The 1e-3 tolerance is three orders below
the activation scale (~1.0) and far inside the margin between intent centroids.

### A real bug the export caught

`AutoModel` for a BERT returns **two** tensors — `last_hidden_state` and an
unused `pooler_output`. The first export produced both, with the second
auto-named `"738"`. Left alone that would have forced the Triton config to
declare a dead output that the server allocates on every request.
`export_minilm` now wraps the backbone to return only `last_hidden_state`, so the
graph boundary matches what the service actually consumes.

## ⚠️ A real footgun: `triton/` shadows the `triton` package

`mlops-pipeline/triton/` is a directory named `triton`. If `mlops-pipeline/` is
put on `sys.path` — which `python -m pytest` does when the working directory is
`mlops-pipeline/` — then `import triton` resolves to this directory instead of the
real PyTorch Triton package. `torch._dynamo` imports `triton.language` during
`import torch`, so the shadowing surfaces as:

```
ImportError: cannot import name 'GenerationMixin' from 'transformers.generation'
```

which is a maddening error for a missing GPU kernel. **Always run these tools by
path from the repository root** (`python mlops-pipeline/export/export_onnx.py`,
`python -m pytest mlops-pipeline/tests/`), which is what the `Makefile` and the
CI workflow do, and what `tests/conftest.py` arranges.

## Running it

```bash
# 0. one-time: export tooling into the ai-engine venv
ai-engine/.venv/bin/pip install -r mlops-pipeline/requirements-export.txt

# 1. export / stage every model  (Laya is ~1.6 GB and takes ~35 s)
ai-engine/.venv/bin/python mlops-pipeline/export/export_onnx.py --model all
ai-engine/.venv/bin/python mlops-pipeline/export/export_onnx.py --model minilm   # fast path

# 2. numerical parity (the headline evidence)
ai-engine/.venv/bin/python -m pytest -o addopts="" -s mlops-pipeline/tests/test_export_parity.py

# 3. validate the Triton configs against the real schema
ai-engine/.venv/bin/python mlops-pipeline/triton/validate_configs.py

# 4. verify artifacts against the manifest
ai-engine/.venv/bin/python mlops-pipeline/export/verify_manifest.py

# 5. stage weights into the Triton repository version dirs
ai-engine/.venv/bin/python mlops-pipeline/export/populate_triton_repo.py
ai-engine/.venv/bin/python mlops-pipeline/export/populate_triton_repo.py --check
```

Or `make -C mlops-pipeline help`.

## Running Triton (on a GPU host — not this one)

```bash
python mlops-pipeline/export/export_onnx.py --model all
python mlops-pipeline/export/populate_triton_repo.py
docker compose -f mlops-pipeline/triton/docker-compose.triton.yml up
```

`config.pbtxt` for `whisper_ct2` follows the Whisper-supporting CTranslate2
backend's example
([PD-Mera/ctranslate2-triton-backend](https://github.com/PD-Mera/ctranslate2-triton-backend)),
whose tensor names (`FEATURES`, `PROMPTS_IDS`, `OUTPUT_IDS`) differ from the base
[speechmatics](https://github.com/speechmatics/ctranslate2_triton_backend)
backend's translation-model names (`INPUT_IDS`/`OUTPUT_IDS`). Whichever backend
build is deployed, **the tensor names must be reconciled with it** — this config
cannot be checked here.

## Known gaps / risks

* **No end-to-end Triton run.** Everything about Triton here is static.
* **No accuracy evaluation set.** There is no labelled corpus in this project, so
  the promotion gates measure *numerical equivalence* and *regression*, not model
  quality. Stated rather than invented.
* **Laya confidence is uncalibrated** (see `ai-engine/README.md`); the export does
  not change that.
* **`tts_frontend/1/model.py` is unverified code.** It is a reference
  implementation of the espeak-ng phonemisation step and has not been executed.
* **The Laya encoder graph is 1.58 GB fp32.** No quantisation is applied here
  (fp16/int8 would shrink it substantially, at an accuracy cost this repo has no
  way to measure). Left as an explicit follow-up.
* **`piper_tts` batching is off** (`max_batch_size: 0`) because the `scales` input
  has no batch dimension — see that config's comment. Server-side batching for
  TTS would need a client/front-end change, not a config tweak.
