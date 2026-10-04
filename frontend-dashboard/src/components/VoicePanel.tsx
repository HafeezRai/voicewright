import { useEffect, useRef } from "react";

import type { UseVoiceClient } from "../hooks/useVoiceClient";
import { isLive, statusLabel, type VoiceStatus } from "../state/voiceMachine";
import { StatusPill, type Tone } from "./StatusPill";

const STATUS_TONE: Record<VoiceStatus, Tone> = {
  idle: "idle",
  requesting: "warn",
  connecting: "warn",
  connected: "ok",
  speaking: "info",
  listening: "ok",
  closing: "warn",
  closed: "idle",
  failed: "err",
};

export function VoicePanel({ voice }: { voice: UseVoiceClient }) {
  const { state, remoteStream, start, stop, setAgentSpeaking } = voice;
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const live = isLive(state.status);
  const canSpeak = state.status === "connected" || state.status === "speaking" || state.status === "listening";

  // Attach the remote (agent) audio track when one arrives.
  useEffect(() => {
    const element = audioRef.current;
    if (!element) return;
    element.srcObject = remoteStream;
    if (remoteStream) void element.play().catch(() => undefined);
  }, [remoteStream]);

  return (
    <section className="rounded-2xl border border-cloudpso-line bg-cloudpso-panel p-5">
      <header className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-100">Voice session</h2>
        <div className="flex items-center gap-2">
          <StatusPill
            label={statusLabel(state.status)}
            tone={STATUS_TONE[state.status]}
            pulse={state.status === "requesting" || state.status === "connecting"}
            testId="voice-status"
          />
          <StatusPill label={`ICE: ${state.ice}`} tone={state.ice === "connected" || state.ice === "completed" ? "ok" : state.ice === "failed" ? "err" : "idle"} testId="ice-status" />
        </div>
      </header>

      <div className="mt-4 flex flex-wrap gap-2">
        <button
          type="button"
          data-testid="start-call"
          onClick={() => void start()}
          disabled={live}
          className="rounded-lg bg-cloudpso-accent px-4 py-2 text-sm font-medium text-white transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Start call
        </button>
        <button
          type="button"
          data-testid="stop-call"
          onClick={stop}
          disabled={!live}
          className="rounded-lg border border-cloudpso-line bg-cloudpso-panel2 px-4 py-2 text-sm font-medium text-slate-200 transition hover:border-cloudpso-err/60 disabled:cursor-not-allowed disabled:opacity-40"
        >
          End call
        </button>
        <button
          type="button"
          data-testid="toggle-agent-speaking"
          onClick={() => setAgentSpeaking(!state.agentSpeaking)}
          disabled={!canSpeak}
          className="rounded-lg border border-cloudpso-line bg-cloudpso-panel2 px-4 py-2 text-sm font-medium text-slate-200 transition hover:border-cloudpso-accent/60 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {state.agentSpeaking ? "Agent: speaking — stop" : "Simulate agent speaking"}
        </button>
      </div>

      <div
        data-testid="barge-in-indicator"
        data-active={state.bargeInActive ? "true" : "false"}
        className={`mt-4 flex items-center justify-between rounded-xl border px-4 py-3 text-sm transition ${
          state.bargeInActive
            ? "border-cloudpso-err/60 bg-cloudpso-err/10 text-cloudpso-err animate-pulse-ring"
            : "border-cloudpso-line bg-cloudpso-panel2 text-cloudpso-muted"
        }`}
      >
        <span className="font-medium">
          {state.bargeInActive ? "Barge-in — agent interrupted" : "Barge-in monitor"}
        </span>
        <span className="tabular-nums text-xs">
          {state.bargeInCount} event{state.bargeInCount === 1 ? "" : "s"}
          {state.lastBargeInMs !== null ? ` · last ${state.lastBargeInMs} ms` : ""}
        </span>
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-3 text-xs">
        <div className="rounded-lg border border-cloudpso-line bg-cloudpso-panel2 px-3 py-2">
          <dt className="text-cloudpso-muted">Session</dt>
          <dd className="mt-0.5 truncate font-mono text-slate-200" data-testid="session-id">
            {state.sessionId ?? "—"}
          </dd>
        </div>
        <div className="rounded-lg border border-cloudpso-line bg-cloudpso-panel2 px-3 py-2">
          <dt className="text-cloudpso-muted">Microphone</dt>
          <dd className="mt-0.5 text-slate-200">{state.micGranted ? "granted" : "—"}</dd>
        </div>
      </dl>

      {state.error ? (
        <p className="mt-3 rounded-lg border border-cloudpso-err/50 bg-cloudpso-err/10 px-3 py-2 text-xs text-cloudpso-err" data-testid="voice-error">
          {state.error}
        </p>
      ) : null}

      <div className="mt-4">
        <h3 className="text-xs uppercase tracking-wide text-cloudpso-muted">Signalling log</h3>
        <ul className="mt-2 max-h-40 space-y-1 overflow-y-auto font-mono text-[11px] leading-relaxed">
          {state.log.length === 0 ? (
            <li className="text-cloudpso-muted">no activity yet</li>
          ) : (
            state.log.map((entry, index) => (
              <li key={`${entry.at}-${index}`} className="text-slate-300">
                <span className={entry.direction === "in" ? "text-cloudpso-ok" : entry.direction === "out" ? "text-cloudpso-accent" : "text-cloudpso-muted"}>
                  {entry.direction === "in" ? "◀ in " : entry.direction === "out" ? "▶ out" : "• info"}
                </span>{" "}
                {entry.message}
              </li>
            ))
          )}
        </ul>
      </div>

      <audio ref={audioRef} autoPlay playsInline className="hidden" data-testid="remote-audio" />
    </section>
  );
}
