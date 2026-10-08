# Voicewright Voice Agent - Master Context

## Project Overview
Enterprise real-time AI voice assistant. STT + NLU + TTS over WebRTC/WebSocket.

## Tech Stack (NON-NEGOTIABLE)
- Backend: Python 3.11, FastAPI, WebSockets, gRPC
- AI/ML: PyTorch, HuggingFace, ONNX Runtime, Triton Inference Server
- Audio: WebRTC (aiortc), PyAV, Silero VAD
- Frontend: React 18 + TypeScript + Tailwind
- Infra: Docker, Kubernetes, Helm, Prometheus

## Directory Structure
/voicewright-voice-agent
  /backend-core         # WebRTC gateway, WebSocket mgmt, barge-in logic
  /ai-engine            # STT, NLU, TTS microservices
  /mlops-pipeline       # ONNX export, Triton configs
  /frontend-dashboard   # React demo + observability UI
  /infra                # K8s manifests, Helm charts, CI/CD

## Build Phases (EXECUTE IN ORDER)
1. Phase 1: Scaffold repo, docker-compose, base configs
2. Phase 2: backend-core (WebRTC gateway + VAD + barge-in)
3. Phase 3: ai-engine (STT → NLU → TTS pipeline)
4. Phase 4: frontend-dashboard (React demo + dashboard)
5. Phase 5: mlops-pipeline + infra (ONNX, Triton, K8s, Prometheus)

## Orchestration Rules
- NEVER skip phases. NEVER combine phases.
- After each phase, STOP and report status to me.
- Use the Task tool to spawn sub-agents for each phase.
- Each sub-agent must receive ONLY the context it needs.
- Verify each phase's output before spawning the next.
- If a sub-agent fails or produces broken code, FIX IT before moving on.
