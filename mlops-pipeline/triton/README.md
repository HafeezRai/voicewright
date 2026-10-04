# mlops-pipeline/triton

The Triton Inference Server model repository.

```
model_repository/
├── whisper_ct2/     ctranslate2   STT  — no ONNX export (see below)
├── laya_encoder/    onnxruntime   NLU  — ModernBERT encoder, exported
├── minilm_nlu/      onnxruntime   NLU  — fallback encoder, exported
├── piper_tts/       onnxruntime   TTS  — already ONNX
└── tts_frontend/    python        TTS  — text → phoneme ids (reference impl)
```

Each `<model>/` holds a committed `config.pbtxt`; the weights live in
`<model>/1/` and are **gitignored** — they are produced by
`mlops-pipeline/export/populate_triton_repo.py` (the Laya graph alone is 1.58 GB).

## ⚠️ Static validation only

**Triton has never been started on this host** (arm64 macOS, no NVIDIA GPU — Docker
reports only the `runc` runtime, no `nvidia`). What *has* been done:

```bash
python mlops-pipeline/triton/validate_configs.py
#   OK  laya_encoder / minilm_nlu / piper_tts / tts_frontend / whisper_ct2
#   5/5 configs valid
```

`validate_configs.py` parses each file with the genuine
`tritonclient.grpc.model_config_pb2.ModelConfig` — the same protobuf Triton
itself uses — and then checks the invariants Triton enforces at load: model name
must equal the directory name, the backend must be one Triton has, a
`dynamic_batching` block is rejected when `max_batch_size` is 0, version dirs
must be numeric. This matters because Triton's own failure mode is silent: a
typo'd config leaves the model `UNAVAILABLE` with no fatal log line.

A config that parses is **not** a model that serves. Nothing here has been served.

## Why `whisper_ct2` is CTranslate2, not ONNX

`faster-whisper`'s artifact is a CTranslate2 model directory. Converting it to
ONNX would discard the int8 quantisation. See `../README.md`.

## Running it (GPU host)

```bash
python mlops-pipeline/export/export_onnx.py --model all
python mlops-pipeline/export/populate_triton_repo.py
docker compose -f mlops-pipeline/triton/docker-compose.triton.yml up
```

`docker-compose.triton.yml` uses `--strict-model-config=true` so a bad config
fails the server at startup rather than silently.

## Known gap

`whisper_ct2/config.pbtxt` uses the tensor names of the Whisper-supporting
CTranslate2 backend fork (`FEATURES`, `PROMPTS_IDS`, `OUTPUT_IDS`). The base
speechmatics backend expects different names (`INPUT_IDS`/`OUTPUT_IDS`) and only
handles translation models. **Reconcile the tensor names with the backend build
you actually deploy** — that cannot be checked without the backend present.
