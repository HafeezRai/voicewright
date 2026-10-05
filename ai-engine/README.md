# ai-engine — STT → LLM → TTS

The Phase 3 speech pipeline for the CloudPSO Voice Agent. It transcribes audio,
*generates* a spoken-style reply with a generative LLM, and synthesises it to
speech.

The STT and TTS stages run **real inference** on interchangeable, permissively
licensed models. The LLM stage is a real HTTP call to an OpenAI-compatible
provider (local Ollama by default): the default test suite drives it with a
deterministic fake so it needs no model, and a separate opt-in test exercises the
live wire protocol against a running Ollama.

```
audio ──► STT (faster-whisper / CTranslate2)
           │  text
           ▼
          LLM (OpenAI-compatible: local Ollama by default, DeepSeek, vLLM, ...)
           │  reply text, streamed sentence by sentence
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
| `app/llm/` | `LLMEngine` + the OpenAI-compatible streaming client, factory, prompts, bounded history |
| `app/knowledge/` | `EmbeddingModel` — the MiniLM encoder kept for step 2's knowledge base |
| `app/cancellation.py` | `CancelToken` shared by LLM generation and TTS playback (barge-in) |
| `app/tts/` | `TTSEngine`, sherpa-onnx backend, `TTSPlaybackSink` (barge-in) |
| `app/pipeline/` | Batch + streaming orchestrator, sentence segmentation |
| `app/grpc/` | `VoiceGateway` servicer |
| `app/gen/` | Committed gRPC stubs (regenerate with `make proto`) |
| `scripts/gen_proto.sh` | Regenerate stubs |
| `scripts/fetch_models.sh` | Download model weights into `./models`; `ollama pull`s the LLM |

## Stage choices

| Stage | Model | Params / size | Licence | Why |
|---|---|---|---|---|
| STT | `Systran/faster-whisper-base.en` | ~74 M / ~150 MB | MIT (weights) | Whisper accuracy without PyTorch; CTranslate2 int8 on CPU; same artifact family Phase 5 can export |
| LLM | `llama3:8b` via Ollama (configurable) | 8 B / ~4.7 GB | Meta Llama 3 Community | Free-form replies from a system prompt: the agent can hold a persona, answer from context and improvise. Local by default; OpenAI-compatible, so DeepSeek/vLLM is a config change |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 22 M / ~90 MB | Apache-2.0 | The retrieval encoder for the knowledge base (step 2). Not on the reply path today |
| TTS | `vits-piper-en_US-amy-low` (sherpa-onnx) | ~20 M / ~80 MB | MIT voice, Apache-2.0 runtime | ONNX-native, cancelable mid-generation. **`piper-tts` 1.8 was rejected: GPL-3.0-or-later.** |

There is no intent classifier any more. A reply is *generated*, not selected from
a template table, so a new "intent" is now a prompt rather than code. One
OpenAI-compatible client covers every provider: `AI_ENGINE_LLM_PROVIDER` selects
a default base URL (`ollama`, `deepseek`, `openai`, ...), `AI_ENGINE_LLM_BASE_URL`
overrides it, and `AI_ENGINE_LLM_MODEL` names the model. The provider is a *base
URL and model* choice, not a code path — a new OpenAI-compatible host needs no
code change.

## Configuration

All settings are documented in the root `.env.example`. The ones that matter:

| Variable | Default | Notes |
|---|---|---|
| `AI_ENGINE_DEVICE` | `cpu` | `mps` on Apple Silicon speeds up STT + the embedder. It does **not** affect the LLM: Ollama picks its own backend and does its own offload |
| `AI_ENGINE_LLM_PROVIDER` | `ollama` | `ollama` / `deepseek` / `openai` / any OpenAI-compatible host |
| `AI_ENGINE_LLM_MODEL` | `llama3:8b` | Exact provider tag. **Bare `llama3` does not exist and 404s** |
| `AI_ENGINE_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | MiniLM encoder for the knowledge base |
| `AI_ENGINE_MODEL_CACHE` | `/models` | Falls back to `<repo>/ai-engine/models` when absent/unwritable |

Models are loaded **lazily**: the process starts, and `GET /health` answers, in
milliseconds with no weights on disk. The one exception is deliberate —
`AI_ENGINE_LLM_WARMUP` warms the LLM on a *background thread*, so a cold
multi-gigabyte load is paid before the first caller arrives without ever delaying
`/health`.

## Serving

* `GET /health` → `{"status":"ok","service":"ai-engine"}` (frozen since Phase 1)
* `GET /metrics` → Prometheus exposition, `cloudpso_ai_*` series
* `POST /stt` (WAV body) · `POST /llm` (JSON) · `POST /tts` (JSON → WAV) ·
  `POST /pipeline` (WAV → transcript + generated reply + reply audio)
* gRPC `VoiceGateway/StreamAudio` + `Health` on port 50052

## Getting the models

```bash
make models                 # STT + TTS (~230 MB); also `ollama pull`s the LLM
                            # when Ollama is on PATH, and skips cleanly if not
ollama pull llama3:8b       # ...or fetch the ~4.7 GB model yourself
```

`./models` is gitignored (`models/` in the root `.gitignore`); in Docker it is
the `ai-engine-models` named volume. The LLM is **not** part of this cache: it is
served by Ollama, which keeps its own model store.

## Verified versions (this machine)

Python 3.11.5 · macOS arm64 · torch 2.14.1 · transformers 4.57.6 · httpx 0.28.1 ·
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
* **LLM** — not exported. The generative stage is served over HTTP by Ollama (or
  another OpenAI-compatible host) at request time; there is no PyTorch checkpoint
  in this repository to export, and it does not participate in the Triton graph.
* **Embeddings** — the MiniLM encoder is a plain `AutoModel` and exports with
  `optimum`/`torch.onnx` trivially (see `mlops-pipeline/`), for a future
  Triton-served retrieval path.
* **TTS** — already an ONNX file (`en_US-amy-low.onnx` + `tokens.txt` +
  `espeak-ng-data`). The VITS graph has dynamic input length, so it maps to a
  Triton ONNX Runtime backend with minimal work; the front-end (espeak-ng
  phonemisation) stays in Python and would become an ensemble/BLS step or a
  pre-processing client.

## Known gaps / risks

* **The first turn pays a cold load.** A local `llama3:8b` is ~4.7 GB; a cold
  Ollama load of it is a handful of seconds. `AI_ENGINE_LLM_WARMUP` and the
  `keep_alive` extra-body field keep it resident, but the very first turn after a
  restart is still slower than the rest.
* **The model must be pulled first.** If `AI_ENGINE_LLM_MODEL` is absent from
  Ollama — or the tag is wrong, since bare `llama3` does not exist — the first
  request fails with a 404 from the provider. `make models` pulls it when Ollama
  is installed.
* **`AI_ENGINE_DEVICE=mps` does not affect the LLM.** It selects the torch device
  for STT and the embedder only; Ollama chooses its own backend and offload.
* **Reasoning models are a footgun.** Models that emit chain-of-thought (e.g.
  `deepseek-r1`, `qwq`) would have the TTS read their thinking aloud.
  `AI_ENGINE_LLM_ALLOW_REASONING_MODELS=false` (the default) refuses them rather
  than only documenting the hazard.
* **Language coverage** is English-only by default (`base.en` STT, `en_US`
  voice). Multi-language is *configurable* — point `AI_ENGINE_LLM_MODEL` at a
  multilingual model and swap the STT/TTS models — but is not claimed to work out
  of the box.
* **The LLM is the latency budget.** A generative reply is seconds, not the tens
  of milliseconds the old intent classifier cost. `AI_ENGINE_LLM_MAX_TOKENS`
  (default 160) is the cheapest lever; `cloudpso_ai_llm_first_token_seconds` is
  the series that says whether the agent feels responsive.
* `faster-whisper`'s language field is set to `en` for `*.en` models by
  construction, so `language_probability` is not meaningful there.
