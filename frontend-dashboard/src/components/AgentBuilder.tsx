import { useCallback, useEffect, useState } from "react";

import {
  AgentApiError,
  type Agent,
  type AgentInput,
  createAgent,
  deleteAgent,
  listAgents,
  updateAgent,
} from "../lib/agents";
import { KnowledgePanel } from "./KnowledgePanel";
import { StatusPill } from "./StatusPill";
import { TalkToAgent } from "./TalkToAgent";

/**
 * Which voices the TTS backend actually has loaded.
 *
 * The agent's `voice` field is free text on the server, but only one Piper
 * voice is shipped, so anything else would be stored and silently ignored —
 * i.e. the picker would lie. We expose the wired voice plus an explicit escape
 * hatch rather than a long dropdown of voices that do nothing.
 */
const WIRED_VOICE = "vits-piper-en_US-amy-low";

const LANGUAGES: { value: string; label: string; wired: boolean }[] = [
  { value: "en", label: "English (en)", wired: true },
  { value: "es", label: "Spanish (es)", wired: false },
  { value: "ur", label: "Urdu (ur)", wired: false },
  { value: "ar", label: "Arabic (ar)", wired: false },
];

const EMPTY_FORM: AgentInput = {
  name: "",
  role: "",
  system_prompt: "",
  voice: WIRED_VOICE,
  language: "en",
  greeting: "",
};

function describe(error: unknown): string {
  if (error instanceof AgentApiError) return error.message;
  return error instanceof Error ? error.message : String(error);
}

/**
 * The agent builder — the screen the product is sold from.
 *
 * Everything an agent *is* lives here as configuration: a role, a prompt, a
 * voice, a language, and the knowledge that grounds it. Creating a second agent
 * is typing into a form, not shipping code, and that is the claim the demo has
 * to make good on.
 */
export function AgentBuilder() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [form, setForm] = useState<AgentInput>(EMPTY_FORM);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selected = agents.find((agent) => agent.id === selectedId) ?? null;

  const refresh = useCallback(async (signal?: AbortSignal) => {
    try {
      const next = await listAgents(signal);
      setAgents(next);
      setError(null);
      return next;
    } catch (cause) {
      if ((cause as Error)?.name === "AbortError") return [];
      setError(describe(cause));
      return [];
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void refresh(controller.signal);
    return () => controller.abort();
  }, [refresh]);

  const set = useCallback(<K extends keyof AgentInput>(key: K, value: AgentInput[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  }, []);

  const resetForm = useCallback(() => {
    setForm(EMPTY_FORM);
    setEditingId(null);
  }, []);

  const onSubmit = useCallback(async () => {
    const name = (form.name ?? "").trim();
    if (!name) {
      setError("An agent needs a name.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const saved = editingId
        ? await updateAgent(editingId, form)
        : await createAgent({ ...form, name });
      await refresh();
      setSelectedId(saved.id);
      resetForm();
    } catch (cause) {
      setError(describe(cause));
    } finally {
      setBusy(false);
    }
  }, [editingId, form, refresh, resetForm]);

  const onEdit = useCallback((agent: Agent) => {
    setEditingId(agent.id);
    setForm({
      name: agent.name,
      role: agent.role,
      system_prompt: agent.system_prompt,
      voice: agent.voice || WIRED_VOICE,
      language: agent.language || "en",
      greeting: agent.greeting,
    });
  }, []);

  const onDelete = useCallback(
    async (agent: Agent) => {
      if (!window.confirm(`Delete "${agent.name}" and its knowledge base?`)) return;
      setBusy(true);
      try {
        await deleteAgent(agent.id);
        if (selectedId === agent.id) setSelectedId(null);
        if (editingId === agent.id) resetForm();
        await refresh();
      } catch (cause) {
        setError(describe(cause));
      } finally {
        setBusy(false);
      }
    },
    [editingId, refresh, resetForm, selectedId],
  );

  return (
    <div className="flex flex-col gap-5" data-testid="agent-builder">
      {/* ---- create / edit ---- */}
      <section className="rounded-xl border border-cloudpso-line bg-cloudpso-panel p-4">
        <header className="mb-3">
          <h2 className="text-sm font-semibold">
            {editingId ? "Edit agent" : "Create an agent"}
          </h2>
          <p className="text-[11px] text-cloudpso-muted">
            A role, a prompt, a voice and some knowledge — that is the whole agent.
          </p>
        </header>

        <div className="grid gap-3 sm:grid-cols-2">
          <label className="flex flex-col gap-1 text-[11px] text-cloudpso-muted">
            Name
            <input
              value={form.name}
              onChange={(event) => set("name", event.target.value)}
              placeholder="Downtown clinic receptionist"
              data-testid="agent-name"
              className="rounded-lg border border-cloudpso-line bg-cloudpso-panel2 px-3 py-1.5 text-sm text-slate-100 outline-none focus:border-cloudpso-accent"
            />
          </label>
          <label className="flex flex-col gap-1 text-[11px] text-cloudpso-muted">
            Role
            <input
              value={form.role}
              onChange={(event) => set("role", event.target.value)}
              placeholder="clinic receptionist"
              data-testid="agent-role"
              className="rounded-lg border border-cloudpso-line bg-cloudpso-panel2 px-3 py-1.5 text-sm text-slate-100 outline-none focus:border-cloudpso-accent"
            />
          </label>
        </div>

        <label className="mt-3 flex flex-col gap-1 text-[11px] text-cloudpso-muted">
          System prompt — who the agent is and how it speaks
          <textarea
            value={form.system_prompt}
            onChange={(event) => set("system_prompt", event.target.value)}
            rows={4}
            placeholder="You are a friendly clinic receptionist. Reply in one short spoken sentence, plain prose. Never use markdown."
            data-testid="agent-prompt"
            className="resize-y rounded-lg border border-cloudpso-line bg-cloudpso-panel2 px-3 py-2 text-sm text-slate-100 outline-none focus:border-cloudpso-accent"
          />
        </label>

        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <label className="flex flex-col gap-1 text-[11px] text-cloudpso-muted">
            Voice
            <select
              value={form.voice}
              onChange={(event) => set("voice", event.target.value)}
              data-testid="agent-voice"
              className="rounded-lg border border-cloudpso-line bg-cloudpso-panel2 px-3 py-1.5 text-sm text-slate-100 outline-none focus:border-cloudpso-accent"
            >
              <option value={WIRED_VOICE}>Piper en_US · amy (low) — wired</option>
            </select>
            <span className="text-[10px] text-cloudpso-muted">
              Only this voice is loaded today; more are a TTS-side addition, not a form change.
            </span>
          </label>
          <label className="flex flex-col gap-1 text-[11px] text-cloudpso-muted">
            Language
            <select
              value={form.language}
              onChange={(event) => set("language", event.target.value)}
              data-testid="agent-language"
              className="rounded-lg border border-cloudpso-line bg-cloudpso-panel2 px-3 py-1.5 text-sm text-slate-100 outline-none focus:border-cloudpso-accent"
            >
              {LANGUAGES.map((language) => (
                <option key={language.value} value={language.value}>
                  {language.label}
                  {language.wired ? " — wired" : " — not wired yet"}
                </option>
              ))}
            </select>
            <span className="text-[10px] text-cloudpso-muted">
              Stored per agent. Non-English speech is not wired end to end yet.
            </span>
          </label>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => void onSubmit()}
            disabled={busy}
            data-testid="save-agent"
            className="rounded-lg bg-cloudpso-accent px-3 py-1.5 text-sm font-medium text-white transition hover:bg-cloudpso-accent/90 disabled:opacity-40"
          >
            {editingId ? "Save changes" : "Create agent"}
          </button>
          {editingId && (
            <button
              type="button"
              onClick={resetForm}
              className="rounded-lg border border-cloudpso-line px-3 py-1.5 text-sm text-cloudpso-muted transition hover:text-slate-100"
            >
              Cancel
            </button>
          )}
          {error && <span className="text-[11px] text-cloudpso-err">{error}</span>}
        </div>
      </section>

      {/* ---- list ---- */}
      <section className="rounded-xl border border-cloudpso-line bg-cloudpso-panel p-4">
        <header className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold">Agents</h2>
          <StatusPill label={`${agents.length}`} tone="info" testId="agent-count" />
        </header>
        {agents.length === 0 ? (
          <p className="text-[11px] text-cloudpso-muted">
            No agents yet — create one above, or start from the example below.
          </p>
        ) : (
          <ul className="flex flex-col gap-2" data-testid="agent-list">
            {agents.map((agent) => {
              const active = agent.id === selectedId;
              return (
                <li
                  key={agent.id}
                  className={`rounded-lg border p-3 transition ${
                    active
                      ? "border-cloudpso-accent/60 bg-cloudpso-accent/10"
                      : "border-cloudpso-line bg-cloudpso-panel2"
                  }`}
                >
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <button
                      type="button"
                      onClick={() => setSelectedId(agent.id)}
                      className="min-w-0 text-left"
                    >
                      <p className="truncate text-sm font-medium">{agent.name}</p>
                      <p className="truncate text-[11px] text-cloudpso-muted">
                        {agent.role || "no role set"}
                        {agent.system_prompt ? "" : " · no prompt"}
                      </p>
                    </button>
                    <div className="flex shrink-0 gap-2 text-[11px]">
                      <button
                        type="button"
                        onClick={() => onEdit(agent)}
                        className="text-cloudpso-muted transition hover:text-slate-100"
                      >
                        Edit
                      </button>
                      <button
                        type="button"
                        onClick={() => void onDelete(agent)}
                        className="text-cloudpso-muted transition hover:text-cloudpso-err"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      {/* ---- selected agent: knowledge + talk ---- */}
      {selected ? (
        <>
          <KnowledgePanel agent={selected} />
          <TalkToAgent agent={selected} />
        </>
      ) : (
        <section className="rounded-xl border border-dashed border-cloudpso-line p-6 text-center">
          <p className="text-sm text-cloudpso-muted">
            Select an agent above to attach knowledge and talk to it.
          </p>
        </section>
      )}
    </div>
  );
}
