/**
 * Single, env-driven configuration module.
 *
 * Every network location the dashboard talks to is resolved here so there is
 * exactly one place to change when the stack moves off localhost. Values come
 * from Vite's `import.meta.env` (i.e. build-time `VITE_*` variables) with
 * localhost defaults, so a bare `npm run dev` works against the compose stack
 * with no `.env` at all.
 *
 * `VITE_BACKEND_CORE_URL` is honoured for continuity with the Phase 1 scaffold
 * (docker-compose.yml / Dockerfile already set it); `VITE_BACKEND_URL` is the
 * preferred Phase 4 name and wins when both are present. `VITE_AI_ENGINE_URL`
 * points at the STT/NLU/TTS service.
 */

// NOTE: 127.0.0.1 rather than `localhost` on purpose. On a machine running
// Docker Desktop, `localhost` resolves to ::1 first and a published container
// can squat on the IPv6 wildcard, so `localhost:8000` reaches the wrong service.
// The explicit IPv4 literal is unambiguous for local development.
const DEFAULT_BACKEND = "http://127.0.0.1:8000";
const DEFAULT_AI_ENGINE = "http://127.0.0.1:8001";

/**
 * Vite performs a *textual* substitution of `import.meta.env.VITE_FOO` at build
 * time, so every variable must be read through a static property access — a
 * computed `import.meta.env[name]` lookup is left untouched and silently falls
 * back to the defaults. Hence this explicit table.
 */
const RAW_ENV = {
  backendUrl: import.meta.env.VITE_BACKEND_URL,
  backendCoreUrl: import.meta.env.VITE_BACKEND_CORE_URL,
  aiEngineUrl: import.meta.env.VITE_AI_ENGINE_URL,
  metricsPollMs: import.meta.env.VITE_METRICS_POLL_MS,
  metricsHistory: import.meta.env.VITE_METRICS_HISTORY,
} as const;

function env(value: string | undefined): string | undefined {
  return value && value.trim().length > 0 ? value.trim() : undefined;
}

/** Strip a trailing slash so URL joins are predictable. */
function normaliseBase(url: string): string {
  return url.replace(/\/+$/, "");
}

/**
 * Derive the WebSocket signalling URL from an HTTP base.
 *
 * `http://host:8000` -> `ws://host:8000/ws/signaling`
 * `https://host`     -> `wss://host/ws/signaling`
 */
export function toSignalingUrl(backendBaseUrl: string): string {
  const base = normaliseBase(backendBaseUrl);
  const ws = base.replace(/^http:/i, "ws:").replace(/^https:/i, "wss:");
  return `${ws}/ws/signaling`;
}

export interface AppConfig {
  /** backend-core HTTP base (health, /metrics). */
  backendUrl: string;
  /** backend-core WebSocket signalling endpoint. */
  signalingUrl: string;
  /** ai-engine HTTP base (stt, nlu, tts, health, /metrics). */
  aiEngineUrl: string;
  /** How often the dashboard polls both /metrics endpoints, in ms. */
  metricsPollMs: number;
  /** How many samples of history the live time-series keeps. */
  metricsHistory: number;
}

export function readConfig(): AppConfig {
  const backendUrl = normaliseBase(
    env(RAW_ENV.backendUrl) ?? env(RAW_ENV.backendCoreUrl) ?? DEFAULT_BACKEND,
  );
  const aiEngineUrl = normaliseBase(env(RAW_ENV.aiEngineUrl) ?? DEFAULT_AI_ENGINE);
  const poll = Number(env(RAW_ENV.metricsPollMs) ?? "2000");
  const history = Number(env(RAW_ENV.metricsHistory) ?? "60");

  return {
    backendUrl,
    signalingUrl: toSignalingUrl(backendUrl),
    aiEngineUrl,
    metricsPollMs: Number.isFinite(poll) && poll >= 250 ? poll : 2000,
    metricsHistory: Number.isFinite(history) && history >= 5 ? history : 60,
  };
}

/** Process-wide config singleton (read once at module load). */
export const config: AppConfig = readConfig();
