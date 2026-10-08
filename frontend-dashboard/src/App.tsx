import { useState } from "react";

import { AgentBuilder } from "./components/AgentBuilder";
import { AgentTurnPanel } from "./components/AgentTurnPanel";
import { DashboardPanel } from "./components/DashboardPanel";
import { TalkToAgent } from "./components/TalkToAgent";
import { VoicePanel } from "./components/VoicePanel";
import { config } from "./config";
import type { Agent } from "./lib/agents";
import { useMetrics } from "./hooks/useMetrics";
import { useVoiceClient } from "./hooks/useVoiceClient";

type View = "console" | "session";

const TABS: { id: View; label: string; hint: string }[] = [
  { id: "console", label: "Agent console", hint: "configure an agent and talk to it" },
  { id: "session", label: "Telephony session", hint: "browser WebRTC + observability" },
];

/**
 * Voicewright — a voice agent console.
 *
 * Laid out the way a playground has to be: the conversation is the page, and
 * everything you can change about the agent sits to its left. The alternative —
 * a configuration form with a "talk" button somewhere below the fold — makes
 * the product look like a settings screen when the entire point is the talking.
 */
export default function App() {
  const [view, setView] = useState<View>("console");
  const [agent, setAgent] = useState<Agent | null>(null);
  const voice = useVoiceClient();
  const metrics = useMetrics();

  return (
    <div className="flex h-screen flex-col bg-voicewright-bg text-slate-100">
      <header className="shrink-0 border-b border-voicewright-line bg-voicewright-panel/70 backdrop-blur">
        <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3">
          <div className="flex items-baseline gap-3">
            <h1 className="text-base font-semibold text-voicewright-accent">Voicewright</h1>
            <p className="text-xs text-voicewright-muted">
              Build a voice agent by configuration, then talk to it.
            </p>
          </div>
          <nav className="flex gap-1" role="tablist">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                role="tab"
                aria-selected={view === tab.id}
                data-testid={`tab-${tab.id}`}
                title={tab.hint}
                onClick={() => setView(tab.id)}
                className={`rounded-full px-3 py-1.5 text-xs font-medium transition ${
                  view === tab.id
                    ? "bg-voicewright-accent/15 text-voicewright-accent"
                    : "text-voicewright-muted hover:text-slate-100"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
          <div className="flex gap-4 font-mono text-[10px] text-voicewright-muted">
            <span title="backend-core base URL">backend {config.backendUrl}</span>
            <span title="ai-engine base URL">ai {config.aiEngineUrl}</span>
          </div>
        </div>
      </header>

      <main className="min-h-0 flex-1">
        {view === "console" ? (
          <div className="grid h-full min-h-0 gap-4 p-4 lg:grid-cols-[minmax(320px,400px)_1fr]">
            <aside className="min-h-0 overflow-y-auto pr-1">
              <AgentBuilder onSelect={setAgent} />
            </aside>

            {agent ? (
              // Remount per agent: an in-flight call belongs to the agent that
              // started it, and switching mid-call must not carry state over.
              <TalkToAgent key={agent.id} agent={agent} />
            ) : (
              <section className="flex items-center justify-center rounded-2xl border border-dashed border-voicewright-line">
                <p className="max-w-xs text-center text-sm text-voicewright-muted">
                  Create an agent on the left — or start from an example — and the
                  conversation appears here.
                </p>
              </section>
            )}
          </div>
        ) : (
          <div className="grid h-full min-h-0 gap-4 overflow-y-auto p-4 lg:grid-cols-2">
            <div className="flex flex-col gap-4">
              <VoicePanel voice={voice} />
              <AgentTurnPanel onAgentSpeaking={voice.setAgentSpeaking} />
            </div>
            <DashboardPanel metrics={metrics} />
          </div>
        )}
      </main>
    </div>
  );
}
