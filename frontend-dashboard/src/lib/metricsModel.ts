/**
 * Turns the raw Prometheus exposition text of both services into the single
 * `DashboardSnapshot` the UI renders.
 *
 * The metric *names* live here, next to the arithmetic that consumes them, so
 * the tile definitions in the UI stay declarative. Every lookup tolerates a
 * missing series (a service that has not recorded that metric yet, or is
 * offline entirely) and degrades to `NaN`/`0` rather than throwing.
 */

import {
  type ParsedMetrics,
  histogram,
  histogramQuantile,
  parse,
  scalar,
  sum,
  value,
} from "./prometheus";

export interface ServiceMetrics {
  online: boolean;
  /** Human-readable reason when `online` is false. */
  error?: string;
  fetchedAt: number;
  parsed?: ParsedMetrics;
}

export interface DashboardSnapshot {
  backend: ServiceMetrics;
  aiEngine: ServiceMetrics;

  // --- backend-core (WebRTC / VAD / barge-in) ------------------------------
  activeSessions: number;
  totalSessions: number;
  bargeInEvents: number;
  wsConnections: number;
  vadFrames: number;
  vadSpeechFrames: number;

  // --- ai-engine: request counts ------------------------------------------
  sttRequests: number;
  llmRequests: number;
  ttsRequests: number;
  pipelineRequests: number;
  ttsCancelled: number;

  // --- ai-engine: latency percentiles (milliseconds) ----------------------
  sttP50: number;
  sttP95: number;
  llmP50: number;
  llmP95: number;
  ttsP50: number;
  ttsP95: number;
  pipelineP50: number;
  pipelineP95: number;

  /** Total failed requests across both services (status="error"). */
  errorCount: number;
}

const METRIC = {
  activeSessions: "cloudpso_active_webrtc_sessions",
  totalSessions: "cloudpso_webrtc_sessions_total",
  bargeInEvents: "cloudpso_bargein_events_total",
  wsConnections: "cloudpso_signaling_connections",
  vadFrames: "cloudpso_vad_frames_processed_total",

  sttRequests: "cloudpso_ai_stt_requests_total",
  sttLatency: "cloudpso_ai_stt_latency_seconds",
  llmRequests: "cloudpso_ai_llm_requests_total",
  llmLatency: "cloudpso_ai_llm_latency_seconds",
  ttsRequests: "cloudpso_ai_tts_requests_total",
  ttsLatency: "cloudpso_ai_tts_latency_seconds",
  ttsCancelled: "cloudpso_ai_tts_cancelled_total",
  pipelineRequests: "cloudpso_ai_pipeline_requests_total",
  pipelineLatency: "cloudpso_ai_pipeline_latency_seconds",
} as const;

export const METRIC_NAMES = METRIC;

function offline(error: string): ServiceMetrics {
  return { online: false, error, fetchedAt: Date.now() };
}

/** Sum of a counter across every label set whose `status` is not a success. */
function errorTotal(...parsed: (ParsedMetrics | undefined)[]): number {
  let total = 0;
  for (const p of parsed) {
    if (!p) continue;
    for (const sample of p.samples) {
      const status = sample.labels.status;
      if (status !== undefined && status !== "ok" && status !== "success") {
        total += sample.value;
      }
    }
  }
  return total;
}

/** p50/p95 in ms for a `*_seconds` histogram, read from the given service. */
function percentile(
  parsed: ParsedMetrics | undefined,
  name: string,
  labels?: Record<string, string>,
): { p50: number; p95: number } {
  if (!parsed) return { p50: Number.NaN, p95: Number.NaN };
  const h = histogram(parsed, name, labels);
  return {
    p50: histogramQuantile(h, 0.5) * 1000,
    p95: histogramQuantile(h, 0.95) * 1000,
  };
}

export function parseService(text: string): ServiceMetrics {
  return { online: true, fetchedAt: Date.now(), parsed: parse(text) };
}

export function offlineService(error: string): ServiceMetrics {
  return offline(error);
}

export function summarise(backend: ServiceMetrics, aiEngine: ServiceMetrics): DashboardSnapshot {
  const b = backend.parsed;
  const a = aiEngine.parsed;

  const stt = percentile(a, METRIC.sttLatency);
  const llm = percentile(a, METRIC.llmLatency);
  const tts = percentile(a, METRIC.ttsLatency);
  const pipeline = percentile(a, METRIC.pipelineLatency);

  return {
    backend,
    aiEngine,

    activeSessions: b ? scalar(b, METRIC.activeSessions) : 0,
    totalSessions: b ? sum(b, METRIC.totalSessions) : 0,
    bargeInEvents: b ? sum(b, METRIC.bargeInEvents) : 0,
    wsConnections: b ? scalar(b, METRIC.wsConnections) : 0,
    vadFrames: b ? sum(b, METRIC.vadFrames) : 0,
    vadSpeechFrames: b ? sum(b, METRIC.vadFrames, { speech: "true" }) : 0,

    sttRequests: a ? sum(a, METRIC.sttRequests) : 0,
    llmRequests: a ? sum(a, METRIC.llmRequests) : 0,
    ttsRequests: a ? sum(a, METRIC.ttsRequests) : 0,
    pipelineRequests: a ? sum(a, METRIC.pipelineRequests) : 0,
    ttsCancelled: a ? sum(a, METRIC.ttsCancelled) : 0,

    sttP50: stt.p50,
    sttP95: stt.p95,
    llmP50: llm.p50,
    llmP95: llm.p95,
    ttsP50: tts.p50,
    ttsP95: tts.p95,
    pipelineP50: pipeline.p50,
    pipelineP95: pipeline.p95,

    errorCount: errorTotal(b, a),
  };
}

/** Read a single labelled value straight out of a service's parsed text. */
export function labelledValue(
  service: ServiceMetrics,
  name: string,
  labels?: Record<string, string>,
): number | undefined {
  return service.parsed ? value(service.parsed, name, labels) : undefined;
}
