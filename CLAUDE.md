# Voicewright — working notes for Claude

A self-hosted voice agent you **configure rather than deploy**: an agent is a row
(role, prompt, voice, knowledge, tools), and standing up a second one is a new row,
not a new service. STT → LLM → TTS over WebRTC/WebSocket.

`README.md` is the human-facing pitch, and it is accurate — read it for the design
rationale and the benchmarks. This file holds the operational context that does not
belong in a README.

## Layout

| Path | What it is |
|---|---|
| `ai-engine/` | STT → LLM → TTS, plus the control plane: agents, documents, retrieval, tools, SSE turns |
| `backend-core/` | WebRTC gateway — sessions, WS signalling, Silero VAD, barge-in |
| `frontend-dashboard/` | React console + observability UI; Playwright e2e |
| `infra/` | k8s manifests, Helm chart, Prometheus config/rules, Grafana dashboards |
| `mlops-pipeline/` | ONNX export with numerical-parity tests, Triton model repo |

The gRPC contract is duplicated by design: `ai-engine/proto/voice_gateway.proto` and
`backend-core/proto/voice_gateway.proto`. Stubs generate into the **committed**
`app/gen/`; `make proto` (needs `grpcio-tools`) regenerates them.

## Commands

Each Python service owns its own venv and its own pytest config. Run from inside the
service directory, never from the repo root. Both `.venv/` directories and
`node_modules/` already exist and are populated.

```bash
cd ai-engine
make test          # .venv/bin/python -m pytest
make lint          # ruff — see the lint gotcha below before trusting this
make models        # fetch Whisper + Piper weights (~230 MB) into ./models

cd ../backend-core
make test
make lint

cd ../frontend-dashboard
npm run typecheck  # tsc --noEmit
npm test           # vitest run
npm run e2e        # Playwright — real Chromium with a fake microphone
```

## Verified test counts

Collected locally on 2026-10-10:

| Suite | Tests | Local state |
|---|---|---|
| `ai-engine` | 264 | collects clean |
| `frontend-dashboard` (vitest) | 115 | 115 passed |
| `backend-core` | 47 | collects clean |
| `infra` + `mlops-pipeline` | 14 | 3 pass, 2 fail — see below |

`.github/workflows/ci.yml`'s header comment claims "ai-engine 182, frontend 41".
Those numbers are stale; the table above is what the suites actually contain.

## CI is red — 4 of 6 jobs fail

Run `38029846031` on `master`:

| Job | Result | Cause |
|---|---|---|
| ai-engine (lint + test) | pass | — |
| frontend-dashboard (build + test) | pass | — |
| backend-core (lint + test) | **fail** | `ruff check app tests` → 7 errors; fails before the tests run |
| dashboards + alert rules | **fail** | 2 tests read a rules file that is not there |
| k8s + helm + prometheus | **fail** | the same missing file, via `promtool` |
| mlops (ONNX export parity) | **fail** | export needs model weights CI never downloads |
| build images | skipped | depends on the failing jobs |

**Two jobs share one cause: an unfinished rename.** Commit `5ccfece` renamed the
project to Voicewright, but four files kept the old `cloudpso` name:

```
infra/prometheus/rules/cloudpso.rules.yml           <- the one that breaks CI
infra/grafana/dashboards/cloudpso-overview.json
infra/grafana/dashboards/cloudpso-ai-pipeline.json
infra/grafana/provisioning/dashboards/cloudpso.yml
```

`infra/tests/test_observability.py:31` hardcodes
`prometheus/rules/voicewright.rules.yml`, and `.github/workflows/ci.yml:167` hands
the same path to `promtool check rules`. Renaming the first file fixes both jobs.
The other three are cosmetic — the dashboard test globs `*.json`, so names do not
matter to it.

The **mlops** failure is separate and pre-existing: `export_onnx.py --model all`
needs cached MiniLM / faster-whisper / Piper weights, and the workflow installs
torch but never runs `scripts/fetch_models.sh`. It cannot pass as written.

## Gotchas

**`make lint` cannot fail — and in backend-core it cannot even run.** Both Makefiles
end the target with `|| true`, so a lint failure reads as a pass. Worse: `ruff` is
installed in `ai-engine/.venv` but **not** in `backend-core/.venv`, so backend-core's
`make lint` dies on `No module named ruff` and the `|| true` swallows it — that
target has never linted anything. CI invokes `python -m ruff check app tests`
directly, which is why only CI notices. To reproduce locally, from `backend-core/`:

```bash
/opt/homebrew/bin/ruff check app tests
```

**The working directory is still `cloudpso`**, while the project, the GitHub repo and
most config say `voicewright`. Do not infer the project name from the directory path.

**Config is env-driven.** Copy `.env.example` to `.env` (gitignored; only the template
is tracked). Every variable is commented there, and it is the single best description
of the system's knobs.

**One LLM client covers every provider.** Any OpenAI-compatible endpoint works — the
provider name only selects a default base URL, so adding a host needs no code. The
default is a local Ollama (`llama3:8b`): no key, no network, no cost.

**Tool calling is a model property, not a feature you add.** `llama3:8b` (the default)
and `mistral` cannot call tools at all — Ollama refuses the request outright. Tool
turns also get a much larger `max_tokens` budget, because a thinking model will reason
until it runs out of room and never act. The console warns when tools are configured
against a model that cannot call them.

**There is no authentication.** Twelve endpoints, zero security schemes, and `org_id`
is client-asserted — a routing label, not a credential. Do not expose this publicly.
Note the GitHub repo is currently **public**.

**State is split.** Agents and knowledge live in SQLite (single writer); tool and
session state live in-process, so a restart clears them and two replicas would
disagree.

## The old phased plan is obsolete

This file used to describe a five-phase greenfield build — "Phase 1: scaffold repo",
spawn one sub-agent per phase, stop and report between phases. **The project is built.**
Do not apply that framing: the code exists, the suites exist, and the real remaining
work is the red CI above, not scaffolding.

Residue of that plan still labels things "Phase 2/3/4" — in `.env.example` and several
`pyproject.toml` description fields. Those labels are meaningless now; read them as
historical.

## Branches

Default branch is `master`, both locally and on `origin`. There is no `main`.
