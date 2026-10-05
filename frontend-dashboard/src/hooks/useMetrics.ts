/**
 * Polls both services' `/metrics` endpoints and folds the results into a
 * `DashboardSnapshot` plus a rolling history for the live time-series.
 *
 * The whole point of this hook is *graceful degradation*: if one (or both)
 * services are down, the dashboard must show a clear "offline" state and keep
 * the existing chart history rather than throwing or white-screening.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { config } from "../config";
import { fetchMetricsText } from "../lib/metricsClient";
import {
  type DashboardSnapshot,
  type ServiceMetrics,
  offlineService,
  parseService,
  summarise,
} from "../lib/metricsModel";

export interface HistoryPoint {
  t: number;
  activeSessions: number;
  bargeInEvents: number;
  wsConnections: number;
  sttP95: number;
  llmP95: number;
  ttsP95: number;
}

export interface UseMetrics {
  snapshot: DashboardSnapshot;
  history: HistoryPoint[];
  /** Epoch ms of the last completed poll attempt, or null before the first. */
  lastPoll: number | null;
  refresh(): void;
}

function blank(): DashboardSnapshot {
  return summarise(offlineService("not polled yet"), offlineService("not polled yet"));
}

export function useMetrics(
  pollMs: number = config.metricsPollMs,
  historySize: number = config.metricsHistory,
): UseMetrics {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot>(blank);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [lastPoll, setLastPoll] = useState<number | null>(null);
  const inFlight = useRef(false);
  const mounted = useRef(true);

  const poll = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    try {
      const backend: ServiceMetrics = await fetchMetricsText(`${config.backendUrl}/metrics`)
        .then(parseService)
        .catch((error: unknown) => offlineService(messageOf(error)));
      const aiEngine: ServiceMetrics = await fetchMetricsText(`${config.aiEngineUrl}/metrics`)
        .then(parseService)
        .catch((error: unknown) => offlineService(messageOf(error)));

      if (!mounted.current) return;

      const next = summarise(backend, aiEngine);
      setSnapshot(next);
      setLastPoll(Date.now());
      setHistory((prev) => {
        const point: HistoryPoint = {
          t: Date.now(),
          activeSessions: next.activeSessions,
          bargeInEvents: next.bargeInEvents,
          wsConnections: next.wsConnections,
          sttP95: finite(next.sttP95),
          llmP95: finite(next.llmP95),
          ttsP95: finite(next.ttsP95),
        };
        const merged = [...prev, point];
        return merged.length > historySize ? merged.slice(merged.length - historySize) : merged;
      });
    } finally {
      inFlight.current = false;
    }
  }, [historySize]);

  useEffect(() => {
    mounted.current = true;
    void poll();
    const timer = window.setInterval(() => void poll(), pollMs);
    return () => {
      mounted.current = false;
      window.clearInterval(timer);
    };
  }, [poll, pollMs]);

  const refresh = useCallback(() => {
    void poll();
  }, [poll]);

  return { snapshot, history, lastPoll, refresh };
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Replace NaN/Infinity with 0 so the chart's y-scale stays finite. */
function finite(value: number): number {
  return Number.isFinite(value) ? value : 0;
}
