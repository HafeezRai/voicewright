"""Prometheus metrics for ai-engine.

Mirrors backend-core's conventions: every series is namespaced ``cloudpso_`` and
declared once at import time. The FastAPI app exposes them via ``GET /metrics``
(see ``app.main``), which ``infra/prometheus/prometheus.yml`` already scrapes.

The three pipeline stages each get a request counter and a latency histogram;
audio stages additionally record the number of *audio seconds* processed, which
is the honest denominator for STT/TTS throughput (request counts alone make a
1-second and a 30-second utterance look alike).
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# Latency buckets in seconds. Fine-grained under 250 ms (the interactive budget
# this service is built for), coarser above it so a cold model load does not
# dominate the histogram's dynamic range.
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

# --- STT ---------------------------------------------------------------------
STT_REQUESTS = Counter(
    "cloudpso_ai_stt_requests_total",
    "STT transcription requests, labelled by outcome.",
    ["status"],
)
STT_LATENCY = Histogram(
    "cloudpso_ai_stt_latency_seconds",
    "Wall-clock time to transcribe one utterance.",
    buckets=_LATENCY_BUCKETS,
)
STT_AUDIO_SECONDS = Histogram(
    "cloudpso_ai_stt_audio_seconds",
    "Duration of audio submitted to STT, per request.",
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120),
)

# --- LLM ---------------------------------------------------------------------
#: First-token latency on a *cold* local model is dominated by loading several
#: gigabytes of weights, which the shared buckets (topping out at 10s) would
#: report as +Inf -- hiding the very thing the series exists to separate.
_LLM_FIRST_TOKEN_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)

LLM_REQUESTS = Counter(
    "cloudpso_ai_llm_requests_total",
    "LLM generation requests, labelled by provider and outcome.",
    ["provider", "status"],
)
LLM_LATENCY = Histogram(
    "cloudpso_ai_llm_latency_seconds",
    "Wall-clock time to generate one reply.",
    ["provider"],
    buckets=_LATENCY_BUCKETS,
)
#: The single most valuable new series: it separates "the model is loading"
#: (first token late, then fast) from "the model is slow" (first token late AND
#: total late). Its own coarse buckets, for the reason above.
LLM_FIRST_TOKEN = Histogram(
    "cloudpso_ai_llm_first_token_seconds",
    "Time from request to the first streamed token, labelled by provider.",
    ["provider"],
    buckets=_LLM_FIRST_TOKEN_BUCKETS,
)
LLM_TOKENS = Counter(
    "cloudpso_ai_llm_tokens_total",
    "Tokens reported by the provider, labelled by provider and kind (prompt|completion).",
    ["provider", "kind"],
)
#: Emitted when barge-in stops generation before the reply was finished.
LLM_CANCELLED = Counter(
    "cloudpso_ai_llm_cancelled_total",
    "Generations abandoned early by a barge-in cancel.",
    ["provider"],
)

# --- TTS ---------------------------------------------------------------------
TTS_REQUESTS = Counter(
    "cloudpso_ai_tts_requests_total",
    "TTS synthesis requests, labelled by outcome (ok|cancelled|error).",
    ["status"],
)
TTS_LATENCY = Histogram(
    "cloudpso_ai_tts_latency_seconds",
    "Wall-clock time to synthesise one utterance.",
    buckets=_LATENCY_BUCKETS,
)
TTS_AUDIO_SECONDS = Histogram(
    "cloudpso_ai_tts_audio_seconds",
    "Duration of audio produced by TTS, per request.",
    buckets=(0.5, 1, 2, 5, 10, 30, 60),
)
#: Emitted when barge-in (or an explicit cancel) stops playback mid-utterance.
TTS_CANCELLED = Counter(
    "cloudpso_ai_tts_cancelled_total",
    "Synthesis/playback runs stopped early by a barge-in cancel.",
)

# --- end-to-end pipeline -----------------------------------------------------
PIPELINE_REQUESTS = Counter(
    "cloudpso_ai_pipeline_requests_total",
    "Full STT -> LLM -> TTS turns, labelled by outcome and LLM provider.",
    ["status", "llm_provider"],
)
PIPELINE_LATENCY = Histogram(
    "cloudpso_ai_pipeline_latency_seconds",
    "Wall-clock time for one full STT -> LLM -> TTS turn.",
    buckets=_LATENCY_BUCKETS,
)

# --- knowledge base / retrieval ---------------------------------------------
#: One increment per turn that *looked*: ``status`` is ``ok`` (retrieval ran,
#: injected or not), ``empty`` (the agent has no documents, so it was skipped
#: entirely) or ``error`` (embedder unavailable; the turn proceeded ungrounded).
#: Worth its own counter because "how often is grounding actually happening" is
#: otherwise invisible -- a silent ``empty`` on every turn looks identical to a
#: knowledge base that works.
KNOWLEDGE_RETRIEVALS = Counter(
    "cloudpso_ai_knowledge_retrievals_total",
    "Retrieval attempts per turn, labelled by outcome (ok|empty|error).",
    ["status"],
)
KNOWLEDGE_RETRIEVAL_LATENCY = Histogram(
    "cloudpso_ai_knowledge_retrieval_latency_seconds",
    "Wall-clock time to embed the utterance and rank the agent's chunks.",
    buckets=_LATENCY_BUCKETS,
)
KNOWLEDGE_CHUNKS_RETURNED = Histogram(
    "cloudpso_ai_knowledge_chunks_returned",
    "Chunks that cleared the similarity threshold and were injected, per turn.",
    buckets=(0, 1, 2, 3, 4, 5, 8, 16),
)

# --- gRPC --------------------------------------------------------------------
GRPC_FRAMES = Counter(
    "cloudpso_ai_grpc_audio_frames_total",
    "Audio frames received on the gRPC StreamAudio endpoint.",
)
GRPC_EVENTS = Counter(
    "cloudpso_ai_grpc_events_total",
    "VoiceEvents emitted downstream, labelled by event kind.",
    ["kind"],
)
