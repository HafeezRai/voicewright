import { useState } from "react";

import { AgentBuilder } from "./components/AgentBuilder";
import { AgentTurnPanel } from "./components/AgentTurnPanel";
import { DashboardPanel } from "./components/DashboardPanel";
import { VoicePanel } from "./components/VoicePanel";
import { config } from "./config";
import { useMetrics } from "./hooks/useMetrics";
import { useVoiceClient } from "./hooks/useVoiceClient";

type View = "builder" | "session";

const TABS: { id: View; label: string; hint: string }[] = [
  { id: "builder", label: "Agent builder", hint: "configure agents and talk to them" },
  { id: "session", label: "Live session", hint: "browser WebRTC + observability" },
];

/**
 * CloudPSO — the agent-builder demo.
 *
 * The builder is the product: an agent is a role, a prompt, a voice and a
 * knowledge base, all of it configuration, so standing up a second one is
 * typing into a form. The live-session view is the older Phase 4 surface — a
 * real browser WebRTC call plus metrics parsed from both services' Prometheus
 * endpoints.
 */
export default function App() {
  const [view, setView] = useState<View>("builder");
  const voice = useVoiceClient();
  const metrics = useMetrics();

  return (
    <div className="min-h-screen bg-cloudpso-bg text-slate-100">
      <header className="border-b border-cloudpso-line bg-cloudpso-panel/70 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-4">
          <div>
            <h1 className="text-xl font-semibold text-cloudpso-accent">CloudPSO Voice Agent</h1>
            <p className="text-xs text-cloudpso-muted">
              Build a voice agent by configuration — role, prompt, knowledge, voice.
            </p>
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px] text-cloudpso-muted">
            <span title="backend-core base URL">backend {config.backendUrl}</span>
            <span title="ai-engine base URL">ai {config.aiEngineUrl}</span>
          </div>
        </div>
        <nav className="mx-auto flex max-w-7xl gap-1 px-4" role="tablist">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              role="tab"
              aria-selected={view === tab.id}
              data-testid={`tab-${tab.id}`}
              title={tab.hint}
              onClick={() => setView(tab.id)}
              className={`-mb-px rounded-t-lg border-x border-t px-3 py-2 text-xs font-medium transition ${
                view === tab.id
                  ? "border-cloudpso-line bg-cloudpso-panel text-cloudpso-accent"
                  : "border-transparent text-cloudpso-muted hover:text-slate-100"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6">
        {view === "builder" ? (
          <div className="grid gap-5 lg:grid-cols-2">
            <AgentBuilder />
            <div className="flex flex-col gap-5">
              <section className="rounded-xl border border-cloudpso-line bg-cloudpso-panel p-4">
                <h2 className="text-sm font-semibold">Why grounding matters</h2>
                <p className="mt-2 text-[11px] leading-relaxed text-cloudpso-muted">
                  An agent with no documents still answers — it just answers from the model's
                  priors. Ask it about a policy it has never seen and it will often invent one,
                  fluently and confidently. Attaching the document changes the answer, and the{" "}
                  <em>grounded / not grounded</em> pill on each reply shows whether the
                  knowledge base actually fired for that turn.
                </p>
              </section>
              <DashboardPanel metrics={metrics} />
            </div>
          </div>
        ) : (
          <div className="grid gap-5 lg:grid-cols-2">
            <div className="flex flex-col gap-5">
              <VoicePanel voice={voice} />
              <AgentTurnPanel onAgentSpeaking={voice.setAgentSpeaking} />
            </div>
            <DashboardPanel metrics={metrics} />
          </div>
        )}
      </main>

      <footer className="mx-auto max-w-7xl px-4 pb-8 text-[11px] text-cloudpso-muted">
        The builder talks to ai-engine's <code>/pipeline</code> directly with an{" "}
        <code>agent_id</code>. backend-core's WebRTC gateway has no notion of agents, so the
        record → reply → play loop is the route that exercises a configured agent end to end.
      </footer>
    </div>
  );
}
