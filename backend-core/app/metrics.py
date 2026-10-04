"""Prometheus metrics for backend-core.

Metrics are declared once, at import time, and mutated from the WebRTC/WS/VAD
code paths. The FastAPI app exposes them via ``GET /metrics`` (see
``app.main``), which is the endpoint Phase 1's ``infra/prometheus/prometheus.yml``
already scrapes.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge

# --- WebRTC sessions ---------------------------------------------------------
ACTIVE_WEBRTC_SESSIONS = Gauge(
    "cloudpso_active_webrtc_sessions",
    "Number of WebRTC peer sessions currently established.",
)

TOTAL_WEBRTC_SESSIONS = Counter(
    "cloudpso_webrtc_sessions_total",
    "Total WebRTC peer sessions created since process start.",
)

# --- barge-in ---------------------------------------------------------------
BARGEIN_EVENTS = Counter(
    "cloudpso_bargein_events_total",
    "Total barge-in (user interrupt) events emitted while the agent was speaking.",
)

# --- VAD --------------------------------------------------------------------
VAD_FRAMES_PROCESSED = Counter(
    "cloudpso_vad_frames_processed_total",
    "Audio frames classified by the VAD, labelled by backend and decision.",
    ["backend", "speech"],
)

# --- signalling -------------------------------------------------------------
WS_CONNECTIONS = Gauge(
    "cloudpso_signaling_connections",
    "Number of WebSocket signalling connections currently open.",
)
