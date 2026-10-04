# CloudPSO Voice Agent

An enterprise real-time AI voice assistant: speech-to-text, natural-language
understanding and text-to-speech over WebRTC/WebSocket, with barge-in support and
an observability dashboard. This repository is a monorepo holding the backend
gateway, the AI engine, the frontend dashboard, the MLOps pipeline and the
deployment infrastructure.

> **Status: Phase 1 complete.** This is a scaffold. Services expose `/health` and
> the stack is wired end-to-end, but no voice pipeline logic exists yet.

## Tech stack

| Layer     | Technology                                                              |
| --------- | ----------------------------------------------------------------------- |
| Backend   | Python 3.11, FastAPI, WebSockets, gRPC                                   |
| AI/ML     | PyTorch, HuggingFace, ONNX Runtime, Triton Inference Server             |
| Audio     | WebRTC (aiortc), PyAV, Silero VAD                                        |
| Frontend  | React 18, TypeScript, Tailwind CSS, Vite                                 |
| Infra     | Docker, Kubernetes, Helm, Prometheus                                     |

## Build phases

1. **Phase 1 — Scaffold** *(done)*: monorepo layout, docker-compose, base config
   layer, importable health-check stubs.
2. **Phase 2 — backend-core**: WebRTC gateway, WebSocket management, VAD and
   barge-in.
3. **Phase 3 — ai-engine**: the STT → NLU → TTS pipeline.
4. **Phase 4 — frontend-dashboard**: live voice demo + observability UI.
5. **Phase 5 — mlops-pipeline + infra**: ONNX export, Triton configs, K8s,
   Helm and Prometheus.

Each phase is executed in order and must be complete before the next begins.

## Monorepo layout

```
cloudpso/
├── backend-core/        # WebRTC gateway, WebSocket mgmt, barge-in (Phase 2)
│   ├── app/             #   config.py, main.py (GET /health)
│   ├── proto/           #   gRPC contract placeholder (not compiled yet)
│   └── tests/           #   pytest health + config tests
├── ai-engine/           # STT → NLU → TTS pipeline (Phase 3)
│   ├── app/             #   config.py, main.py (GET /health)
│   └── tests/           #   pytest health + config tests
├── mlops-pipeline/      # ONNX export, Triton repo, pipeline configs (Phase 5)
├── frontend-dashboard/  # React 18 + TS + Tailwind + Vite (Phase 4)
├── infra/               # k8s/, helm/, prometheus/ (Phase 5)
├── docker-compose.yml   # Local stack: backend-core, ai-engine, frontend, prometheus
└── .env.example         # Every env var the stack reads — copy to .env
```

## Getting started

```bash
cp .env.example .env          # optional: override ports, log level, model paths
docker compose up --build     # backend-core :8000, ai-engine :8001,
                              # frontend :5173, prometheus :9090
```

Verify:

```bash
curl http://localhost:8000/health   # {"status":"ok","service":"backend-core"}
curl http://localhost:8001/health   # {"status":"ok","service":"ai-engine"}
```

Validate the compose file without building or starting anything:

```bash
docker compose config
```

> **Note:** the `ai-engine` image is built with `INSTALL_ML_STACK=false` by
> default, so the multi-GB ML stack (torch, transformers, onnxruntime) is *not*
> installed in Phase 1. Flip the build arg when Phase 3 lands the pipeline.

## Development

### backend-core

```bash
cd backend-core
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt pytest httpx
pytest
```

### ai-engine

Tests run against the light dev manifest — no torch required, because the
Phase 1 app imports none of the ML stack.

```bash
cd ai-engine
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

### frontend-dashboard

```bash
cd frontend-dashboard
npm install
npm run typecheck    # tsc --noEmit
npm run build        # tsc --noEmit && vite build
npm run dev          # Vite dev server on :5173
```
