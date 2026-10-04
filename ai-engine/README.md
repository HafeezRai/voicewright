# ai-engine — STT → NLU → TTS

The Phase 3 speech pipeline for the CloudPSO Voice Agent. It transcribes audio,
classifies the transcript into an intent, and synthesises a spoken reply.

Everything runs **real inference** on interchangeable, permissively licensed
models. Nothing in the test suite is mocked.

```
audio ──► STT (faster-whisper / CTranslate2)
           │  text
           ▼
          NLU (Laya  ──or──  HuggingFace embeddings)
           │  intent + slots
           ▼
          reply text  (app/pipeline/responder.py)
           │
           ▼
          TTS (sherpa-onnx / Piper-VITS) ──► PCM chunks
```

## Layout

| Path | Purpose |
|---|---|
| `app/config.py` | Validated settings; `models_root`, `hf_home`, `hf_hub_cache` |
| `app/metrics.py` | Prometheus series (`cloudpso_ai_*`) |
| `app/audio_io.py` | WAV / PCM helpers (no scipy) |
| `app/stt/` | `STTEngine` + faster-whisper backend |
| `app/nlu/` | `NLUEngine`, intent schema, slots, Laya + HF backends, factory |
| `app/tts/` | `TTSEngine`, sherpa-onnx backend, `TTSPlaybackSink` (barge-in) |
| `app/pipeline/` | Batch + streaming orchestrator, response templater |
| `app/grpc/` | `VoiceGateway` servicer |
| `app/gen/` | Committed gRPC stubs (regenerate with `make proto`) |
| `scripts/gen_proto.sh` | Regenerate stubs |
| `scripts/fetch_models.sh` | Download model weights into `./models` |

## Stage choices

| Stage | Model | Params / size | Licence | Why |
|---|---|---|---|---|
| STT | `Systran/faster-whisper-base.en` | ~74 M / ~150 MB | MIT (weights) | Whisper accuracy without PyTorch; CTranslate2 int8 on CPU; same artifact family Phase 5 can export |
| NLU (primary) | `convaiinnovations/laya` | 421 M / ~840 MB | Apache-2.0 | Single forward pass, typed decisions, calibrated-*intent* probabilities |
| NLU (fallback) | `sentence-transformers/all-MiniLM-L6-v2` | 22 M / ~90 MB | Apache-2.0 | No in-domain data needed; keeps the pipeline working if Laya is unavailable |
| TTS | `vits-piper-en_US-amy-low` (sherpa-onnx) | ~20 M / ~80 MB | MIT voice, Apache-2.0 runtime | ONNX-native, cancelable mid-generation. **`piper-tts` 1.8 was rejected: GPL-3.0-or-later.** |

`AI_ENGINE_NLU_BACKEND` selects the NLU backend: `laya`, `hf`, or `auto`
(default — tries Laya on first use and degrades permanently to the HF fallback,
logging which one is answering). Laya never leaks past `app/nlu/base.py`: the
dependency is young and effectively single-author, so it stays swappable.

## Configuration

All settings are documented in the root `.env.example`. The ones that matter:

| Variable | Default | Notes |
|---|---|---|
| `AI_ENGINE_DEVICE` | `cpu` | `mps` on Apple Silicon is ~4x faster for Laya |
| `AI_ENGINE_NLU_BACKEND` | `auto` | `laya` / `hf` / `auto` |
| `AI_ENGINE_NLU_MODEL` | `convaiinnovations/laya` | Laya checkpoint id |
| `AI_ENGINE_MODEL_CACHE` | `/models` | Falls back to `<repo>/ai-engine/models` when absent/unwritable |

Models are loaded **lazily**: the process starts, and `GET /health` answers, in
milliseconds with no weights on disk.

## Serving

* `GET /health` → `{"status":"ok","service":"ai-engine"}` (frozen since Phase 1)
* `GET /metrics` → Prometheus exposition, `cloudpso_ai_*` series
* `POST /stt` (WAV body) · `POST /nlu` (JSON) · `POST /tts` (JSON → WAV) ·
  `POST /pipeline` (WAV → transcript + intent + reply audio)
* gRPC `VoiceGateway/StreamAudio` + `Health` on port 50052

## Getting the models

```bash
make models                 # STT + TTS (~230 MB)
make models -- --with-laya  # ...plus the 840 MB Laya checkpoint
```

`./models` is gitignored (`models/` in the root `.gitignore`); in Docker it is
the `ai-engine-models` named volume.

## Verified versions (this machine)

Python 3.11.5 · macOS arm64 · torch 2.14.1 · transformers 4.57.6 · laya 0.3.26 ·
faster-whisper 1.2.1 · ctranslate2 4.8.2 · sherpa-onnx 1.13.8 · onnxruntime
1.30.0 · numpy 2.4.6 · grpcio 1.84.0 · prometheus-client 0.26.0

## Phase 5 (Triton) export paths

Phase 5 owns the Triton model repository; this section only records what each
stage needs, so the models chosen here can actually get there.

* **STT** — fastest path, no export needed. faster-whisper's artifact *is* a
  CTranslate2 model directory (`model.bin` + `config.json` + tokenizer). Triton
  serves it via the
  [CTranslate2 backend](https://github.com/triton-inference-server/ctranslate2_backend)
  with `model.bin` as the model file. An ONNX route also exists
  (`openai/whisper` → `optimum`/`onnxruntime` export), but CTranslate2 is the
  lower-effort, already-benchmarked path.
* **NLU** — Laya ships an ONNX **runtime** (`laya.onnx_agent.ONNXAgent`, the
  `laya[onnx]` extra) and `laya.load(..., backend="onnx", onnx_path=...)`, so a
  Triton ONNX Runtime backend will work. Caveat: the export *script*
  (`export_onnx.py`) is **not bundled** with the wheel — it lives in the upstream
  GitHub repo — so the export step must be sourced from upstream or written.
  The HF fallback is a plain `AutoModel` and exports with `optimum`/`torch.onnx`
  trivially.
* **TTS** — already an ONNX file (`en_US-amy-low.onnx` + `tokens.txt` +
  `espeak-ng-data`). The VITS graph has dynamic input length, so it maps to a
  Triton ONNX Runtime backend with minimal work; the front-end (espeak-ng
  phonemisation) stays in Python and would become an ensemble/BLS step or a
  pre-processing client.

## Known gaps / risks

* **Laya confidence is not calibrated.** The checkpoint prints
  `RuntimeWarning: this checkpoint ships invalid temperatures or values outside
  [0.5, 5] ... Treat confidence from the affected entries as uncalibrated` — so
  the vendor's "mathematically calibrated probabilities" claim does not hold for
  the shipped `choice` head. The *intent label* is reliable; the *confidence
  number* should be treated as a ranking score, not a probability.
* **Language coverage** is English-only by default (`base.en`, English Laya
  checkpoint, `en_US` voice). `laya-multilingual` is a config change
  (`AI_ENGINE_NLU_MODEL=.../laya-multilingual`); STT/TTS would need matching
  models.
* **CPU-only latency.** Laya on CPU measured ~136 ms/decision vs ~31 ms on MPS.
  The default device is `cpu` for portability; set `AI_ENGINE_DEVICE=mps` on
  Apple Silicon.
* **Response text is templated**, not generated (`app/pipeline/responder.py`).
  There is no LLM in this phase, and the code says so rather than faking one.
* `faster-whisper`'s language field is set to `en` for `*.en` models by
  construction, so `language_probability` is not meaningful there.
