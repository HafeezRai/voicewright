import type { UseMetrics } from "../hooks/useMetrics";
import { formatMetric, MetricTile } from "./MetricTile";
import { StatusPill } from "./StatusPill";
import { TimeSeriesChart } from "./TimeSeriesChart";

const COLORS = {
  accent: "#4f8cff",
  ok: "#34d399",
  warn: "#fbbf24",
  err: "#f87171",
  violet: "#a78bfa",
};

function ms(value: number): string {
  if (!Number.isFinite(value) || value === 0) return "—";
  if (value < 1) return value.toFixed(2);
  if (value < 100) return value.toFixed(1);
  return value.toFixed(0);
}

export function DashboardPanel({ metrics }: { metrics: UseMetrics }) {
  const { snapshot, history, lastPoll, refresh } = metrics;
  const online = (flag: boolean, error?: string) =>
    flag ? (
      <StatusPill label="online" tone="ok" testId="svc-online" />
    ) : (
      <StatusPill label={error ? `offline — ${error}` : "offline"} tone="err" testId="svc-offline" />
    );

  const sessions = history.map((h) => h.activeSessions);
  const ws = history.map((h) => h.wsConnections);
  const bargeIns = history.map((h) => h.bargeInEvents);
  const stt = history.map((h) => h.sttP95);
  const nlu = history.map((h) => h.nluP95);
  const tts = history.map((h) => h.ttsP95);

  return (
    <section className="flex flex-col gap-4" data-testid="dashboard">
      <div className="rounded-2xl border border-cloudpso-line bg-cloudpso-panel p-5">
        <header className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-lg font-semibold text-slate-100">Observability</h2>
          <div className="flex items-center gap-2">
            <span className="text-xs text-cloudpso-muted">
              {lastPoll ? `updated ${new Date(lastPoll).toLocaleTimeString()}` : "polling…"}
            </span>
            <button
              type="button"
              onClick={refresh}
              className="rounded-md border border-cloudpso-line bg-cloudpso-panel2 px-2.5 py-1 text-xs text-slate-200 transition hover:border-cloudpso-accent/60"
            >
              Refresh
            </button>
          </div>
        </header>

        <div className="mt-3 grid gap-2 sm:grid-cols-2">
          <div className="flex items-center justify-between rounded-lg border border-cloudpso-line bg-cloudpso-panel2 px-3 py-2 text-sm">
            <span className="text-slate-200">backend-core</span>
            {online(snapshot.backend.online, snapshot.backend.error)}
          </div>
          <div className="flex items-center justify-between rounded-lg border border-cloudpso-line bg-cloudpso-panel2 px-3 py-2 text-sm">
            <span className="text-slate-200">ai-engine</span>
            {online(snapshot.aiEngine.online, snapshot.aiEngine.error)}
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-3">
          <MetricTile
            label="Active sessions"
            value={formatMetric(snapshot.activeSessions)}
            tone={snapshot.activeSessions > 0 ? "ok" : "idle"}
            testId="tile-active-sessions"
          />
          <MetricTile label="Total sessions" value={formatMetric(snapshot.totalSessions)} testId="tile-total-sessions" />
          <MetricTile label="WS connections" value={formatMetric(snapshot.wsConnections)} testId="tile-ws" />
          <MetricTile
            label="Barge-in events"
            value={formatMetric(snapshot.bargeInEvents)}
            tone={snapshot.bargeInEvents > 0 ? "warn" : "idle"}
            testId="tile-bargein"
          />
          <MetricTile
            label="VAD frames"
            value={formatMetric(snapshot.vadFrames)}
            sub={`${formatMetric(snapshot.vadSpeechFrames)} speech`}
            testId="tile-vad"
          />
          <MetricTile
            label="Errors"
            value={formatMetric(snapshot.errorCount)}
            tone={snapshot.errorCount > 0 ? "err" : "idle"}
            sub={`${formatMetric(snapshot.ttsCancelled)} TTS cancelled`}
            testId="tile-errors"
          />
        </div>
      </div>

      <div className="rounded-2xl border border-cloudpso-line bg-cloudpso-panel p-5">
        <h3 className="text-sm font-semibold text-slate-200">Pipeline latency (p50 / p95, ms)</h3>
        <div className="mt-3 grid grid-cols-2 gap-3 lg:grid-cols-4">
          <MetricTile label="STT p50" value={ms(snapshot.sttP50)} unit="ms" sub={`p95 ${ms(snapshot.sttP95)}`} testId="tile-stt" />
          <MetricTile label="NLU p50" value={ms(snapshot.nluP50)} unit="ms" sub={`p95 ${ms(snapshot.nluP95)}`} testId="tile-nlu" />
          <MetricTile label="TTS p50" value={ms(snapshot.ttsP50)} unit="ms" sub={`p95 ${ms(snapshot.ttsP95)}`} testId="tile-tts" />
          <MetricTile label="Pipeline p50" value={ms(snapshot.pipelineP50)} unit="ms" sub={`p95 ${ms(snapshot.pipelineP95)}`} testId="tile-pipeline" />
        </div>
        <div className="mt-3 grid grid-cols-2 gap-3 text-xs text-cloudpso-muted lg:grid-cols-4">
          <span>STT reqs: {formatMetric(snapshot.sttRequests)}</span>
          <span>NLU reqs: {formatMetric(snapshot.nluRequests)}</span>
          <span>TTS reqs: {formatMetric(snapshot.ttsRequests)}</span>
          <span>Pipeline turns: {formatMetric(snapshot.pipelineRequests)}</span>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="rounded-2xl border border-cloudpso-line bg-cloudpso-panel p-5">
          <h3 className="text-sm font-semibold text-slate-200">Sessions &amp; connections</h3>
          <div className="mt-3">
            <TimeSeriesChart
              testId="chart-sessions"
              zeroBased
              series={[
                { key: "sessions", label: "active sessions", color: COLORS.ok, points: sessions },
                { key: "ws", label: "ws connections", color: COLORS.accent, points: ws },
              ]}
            />
          </div>
        </div>
        <div className="rounded-2xl border border-cloudpso-line bg-cloudpso-panel p-5">
          <h3 className="text-sm font-semibold text-slate-200">Barge-in events (cumulative)</h3>
          <div className="mt-3">
            <TimeSeriesChart
              testId="chart-bargein"
              zeroBased
              series={[{ key: "bargein", label: "barge-in total", color: COLORS.warn, points: bargeIns }]}
            />
          </div>
        </div>
        <div className="rounded-2xl border border-cloudpso-line bg-cloudpso-panel p-5 xl:col-span-2">
          <h3 className="text-sm font-semibold text-slate-200">Latency p95 (ms)</h3>
          <div className="mt-3">
            <TimeSeriesChart
              testId="chart-latency"
              series={[
                { key: "stt", label: "STT", color: COLORS.accent, points: stt },
                { key: "nlu", label: "NLU", color: COLORS.violet, points: nlu },
                { key: "tts", label: "TTS", color: COLORS.ok, points: tts },
              ]}
            />
          </div>
        </div>
      </div>
    </section>
  );
}
