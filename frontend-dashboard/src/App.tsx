import { AgentTurnPanel } from "./components/AgentTurnPanel";
import { DashboardPanel } from "./components/DashboardPanel";
import { VoicePanel } from "./components/VoicePanel";
import { config } from "./config";
import { useMetrics } from "./hooks/useMetrics";
import { useVoiceClient } from "./hooks/useVoiceClient";

/**
 * CloudPSO Voice Agent dashboard (Phase 4).
 *
 * Two halves, deliberately side by side: the left drives a real browser voice
 * session (mic -> WebRTC -> backend-core signalling) and the ai-engine turn
 * endpoints; the right is a live observability view parsed straight from both
 * services' Prometheus endpoints.
 */
export default function App() {
  const voice = useVoiceClient();
  const metrics = useMetrics();

  return (
    <div className="min-h-screen bg-cloudpso-bg text-slate-100">
      <header className="border-b border-cloudpso-line bg-cloudpso-panel/70 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-4">
          <div>
            <h1 className="text-xl font-semibold text-cloudpso-accent">CloudPSO Voice Agent</h1>
            <p className="text-xs text-cloudpso-muted">
              Phase 4 — browser WebRTC voice client + live observability
            </p>
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px] text-cloudpso-muted">
            <span title="backend-core base URL">backend {config.backendUrl}</span>
            <span title="ai-engine base URL">ai {config.aiEngineUrl}</span>
            <span title="signalling endpoint">{config.signalingUrl}</span>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-5 px-4 py-6 lg:grid-cols-2">
        <div className="flex flex-col gap-5">
          <VoicePanel voice={voice} />
          <AgentTurnPanel onAgentSpeaking={voice.setAgentSpeaking} />
        </div>
        <DashboardPanel metrics={metrics} />
      </main>

      <footer className="mx-auto max-w-7xl px-4 pb-8 text-[11px] text-cloudpso-muted">
        Media note: backend-core (Phase 2) is receive-only, so no remote agent audio
        arrives over the peer connection — the agent's audible voice is synthesised
        via ai-engine <code>/tts</code>.
      </footer>
    </div>
  );
}
