# Voicewright

**Build a voice agent by configuration, then talk to it.**

Every agent here is role, prompt, voice, knowledge and tools — typed into a
form. Standing up a second one is a new row, not a new deployment.

Self-hosted: your audio, your models, your infrastructure. The LLM is any
OpenAI-compatible endpoint, so it runs entirely on your machine or against a
hosted API, and switching between them is a config value.

```
Configure an agent  ->  talk to it  ->  hear it answer  ->  see why it answered that way
```

---

## What it does

| | |
|---|---|
| **Speech in** | `faster-whisper` (CTranslate2), CPU int8 |
| **Thinking** | Any OpenAI-compatible LLM — Ollama, DeepSeek, vLLM, LM Studio |
| **Speech out** | `sherpa-onnx` running Piper VITS voices |
| **Knowledge** | Documents chunked, embedded with MiniLM, retrieved per turn and injected into the prompt |
| **Tools** | Function calling with a bounded call → execute → answer loop |
| **Turn-taking** | Silence detection, tunable; barge-in by talking over it |
| **Streaming** | Server-sent events, synthesised sentence by sentence |

The console shows the transcript as a conversation, and under each reply the
evidence: which chunks grounded it, which tools ran, and how long each stage
took.

---

## Quick start

Requires **Python 3.11**, **Node 22**, and **Docker** (for the compose stack).
An LLM endpoint is needed — local Ollama is the default and free.

```bash
# 1. Models (~230 MB: Whisper base.en + a Piper voice)
cd ai-engine && ./scripts/fetch_models.sh

# 2. ai-engine  (STT -> LLM -> TTS)
./.venv/bin/python -m uvicorn app.main:app --port 8001

# 3. backend-core  (WebRTC gateway, VAD, barge-in)
cd ../backend-core && ./.venv/bin/python -m uvicorn app.main:app --port 18000

# 4. Frontend
cd ../frontend-dashboard && npm install && npx vite --port 5173
```

Open `http://127.0.0.1:5173`.

**Point it at an LLM.** Default is a local Ollama:

```bash
ollama pull llama3:8b
```

For a hosted model, put a key in `ai-engine/.env` (gitignored) and pick it per
agent in the console:

```
DEEPSEEK_API_KEY=sk-...
```

Nothing else changes. Agents that do not name a provider keep using the local
default, so a local setup still needs no key and no network.

**Or run the whole stack in containers:** `docker compose up`.

---

## Architecture

Three services, one job each:

```
browser ──WebRTC──> backend-core ──┐
   │             (gateway, VAD,    │
   │              barge-in)        │
   │                               v
   └─────── HTTP ───────────>  ai-engine
                              STT -> LLM -> TTS
                                 │      │      │
                          faster-  OpenAI-  sherpa-
                          whisper  compat   onnx
                                     │
                              tools + knowledge
```

**`backend-core`** — the real-time audio path: WebRTC sessions, signalling over
a WebSocket, Silero VAD, and the barge-in controller.

**`ai-engine`** — the pipeline, plus the control plane: agents, documents,
retrieval, tools, and streaming turns over SSE.

**`frontend-dashboard`** — the console, and a live observability view parsed
from both services' Prometheus endpoints.

**State:** agents and knowledge in SQLite; tool and session state in-process
(see [Limitations](#limitations) — this matters).

---

## What I measured

Most of this project's real work was measurement, and several results changed
the design. These are the ones worth reading.

### Tool calling is a model property, not a feature you add

Probed before building anything on top of it:

```
llama3:8b  (the default)   ERROR: does not support tools
mistral:latest             ERROR: does not support tools
qwen3.5:0.8b               works
```

Two of three local models **cannot call tools at all** — Ollama rejects the
request outright. And the one that works is a *thinking* model, which bills its
hidden reasoning against `max_tokens`: at a 160-token reply budget it called a
tool **once in five**, returning `finish_reason=length` otherwise. It reasoned
until it ran out of room and never acted. At 512+ it managed four in five.

So tool turns use their own much larger budget, and the editor warns when tools
are configured on a model that cannot call them — a configuration that
otherwise looks correct and silently never does anything.

### The first benchmark was wrong, and the correction was the finding

Head-to-head on identical input:

| | exact date | vague date |
|---|---|---|
| local `qwen3.5:0.8b` | 4/5 · **4.45 s** | 4/5 tool, 1/5 asked |
| `deepseek-chat` | **5/5 · 0.97 s** | 3/5 tool, 2/5 asked |

The first run of this said DeepSeek was *worse* — 2/5 against 4/5. It wasn't:
that run was counting *"asked which Saturday you mean"* as a failure, and on a
vague question asking is the correct answer. The local model's habit of
inventing a date was the actual failure, and it was being scored a success.

The honest number is 4.6× lower latency, not "unreliable versus reliable".

### Vendor claims, checked

A model chosen earlier for the classification stage advertised "32.8 ms
latency, state-of-the-art calibration".

- Latency: **holds only on a GPU.** 31 ms on Apple Silicon vs **136 ms on CPU** — a 4.4× gap the headline does not mention.
- Calibration: **false.** The model's own runtime warns that the checkpoint ships invalid temperatures and that "confidence from the affected entries is uncalibrated". Anything built on those numbers as probabilities would have been built on nothing.

That model is no longer in the pipeline. Measuring is why.

### Streaming is worth a lot, and only on long replies

Synthesising sentence by sentence, as the reply is generated, rather than
waiting for the whole thing:

| reply shape | batched | streamed |
|---|---|---|
| one sentence | ~1.1 s | ~1.0 s |
| four to five sentences | ~2.0–2.9 s | **~0.9–1.1 s** |

Flat regardless of reply length, because the first sentence is enough to start
speaking. On a short reply it buys nothing — worth knowing before selling it as
a feature.

### Bugs that only a real call finds

Four, none of which reading the code would have turned up:

- **The call went permanently deaf about twelve seconds in.** On "no speech for 8s" the detector set itself idle and nothing re-armed it. You pause to think; the timer expires; the agent never hears anything again while still looking attentive.
- **Every utterance after the first failed to decode.** One recorder ran for the whole call and each utterance was sliced out of it — but a WebM/Opus stream carries its container header in the *opening* chunks, so a mid-stream slice is not a decodable file.
- **The agent interrupted itself.** Barge-in fired on a single frame above threshold, so its own voice through the speakers cut it off mid-word.
- **Interrupting deleted what had been said.** Aborting the request client-side meant the server's terminal event never arrived, the read loop threw, and the error handler replaced the whole turn — transcript and reply — with an error message.

---

## Design decisions worth knowing

**MediaRecorder was the wrong tool.** It hands back a compressed container that
has to be decoded back into samples — and the pipeline only ever wanted PCM. In
Safari that round trip fails outright: `decodeAudioData` rejects the fragmented
MP4 that Safari's own `MediaRecorder` produced. Capture now taps the audio
graph directly: no codec, no container, no header to lose.

**Echo rejection is a text filter, and its tuning is a trade-off.** The
microphone picks up the agent's own voice; comparing each transcript against
what the agent just said drops those turns before a reply is generated. But a
four-word question made of the agent's own words — *"39% of the wealth?"* —
scores a **perfect** match, higher than real echo. No threshold separates them,
because at that length the words genuinely are all the agent's. Length is the
only signal left, so short transcripts are never judged: answering an echo is
one odd turn, ignoring a caller is the agent failing at its job.

**The tool loop is bounded, and withdraws its tools.** A small model can ask
for a tool, be given the result, and ask again. On the final round the tools
are removed, so the model has to produce speech rather than another call.

**Connect timeouts scale to the network.** Two seconds is right for a local
model — it answers in about a millisecond, so a refusal means it is down. It is
far too tight across the public internet, where a three-second TLS handshake is
healthy. Loopback keeps the tight budget; anything else gets ten seconds.

---

## Limitations

Read this before putting it in front of anyone.

**There is no authentication.** Twelve endpoints, zero security schemes. Anyone
who can reach the port can read and edit every agent, its prompt, and its
entire knowledge base. The multi-tenant schema exists and the queries are
scoped, but `org_id` is **client-asserted** — a routing label, not a credential.
Do not expose this publicly without putting something in front of it.

**Tool and session state live in the process.** A restart clears every booking;
two replicas would disagree. Conversation history is an in-process LRU, so a
caller loses context if the next turn lands on another pod. The agent and
knowledge stores are SQLite — a single writer.

**English only.** The STT and TTS models are English; the language field is
stored and does not change the voices.

**The VAD thresholds are tuned against a synthetic fixture**, not a real room
or a phone line. `silenceMs`, `interruptAfterMs` and the echo `minWords` are
plausible, not verified. This is the most likely thing to disappoint in a live
demo.

**No evaluation set.** The tests verify that the pipeline runs, that retrieval
returns the right chunks, and that tool calls round-trip. Nothing measures
whether the *answers* are good, so nothing catches a regression in quality.

---

## Development

```bash
# ai-engine                             # backend-core
cd ai-engine                           cd backend-core
make test                              make test
make lint                              make lint
make proto                             make proto

# frontend
cd frontend-dashboard
npm run typecheck   # tsc
npm run test        # vitest
npm run build
npm run e2e         # Playwright, real browser + fake mic
```

**426 tests** — 264 ai-engine, 115 frontend, 47 backend-core — plus 4 browser
end-to-end specs that drive a real Chromium with a fake microphone: a WebRTC
call against the aiortc gateway, a browser-initiated barge-in, creating and
grounding an agent through the UI, and a hands-free call that must answer
without any button being pressed.

The end-to-end specs skip cleanly when no LLM is reachable, so the suite runs
on a machine with no model.

---

## Layout

```
ai-engine/            STT -> LLM -> TTS, agents, knowledge, tools
  app/llm/            provider-agnostic client (any OpenAI-compatible host)
  app/knowledge/      embeddings, chunking, retrieval
  app/tools/          tool specs, registry, the shipped tools
  app/pipeline/       orchestration, sentence segmentation, streaming
backend-core/         WebRTC gateway, signalling, VAD, barge-in
frontend-dashboard/   the console + observability UI
infra/                Kubernetes, Helm, Prometheus, Grafana
mlops-pipeline/       ONNX export with numerical-parity tests, Triton configs
```

---

## License

Not yet chosen — add one before publishing if you intend others to use this.
