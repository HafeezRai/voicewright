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
import { AGENT_PRESETS } from "../lib/agentPresets";
import { listModels, listTools, type ModelCatalogue, type ToolSpec } from "../lib/agents";
import { KnowledgePanel } from "./KnowledgePanel";
import { StatusPill } from "./StatusPill";

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
  model: "",
  provider: "",
  tools: [],
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
export function AgentBuilder({ onSelect }: { onSelect?: (agent: Agent | null) => void } = {}) {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [form, setForm] = useState<AgentInput>(EMPTY_FORM);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [catalogue, setCatalogue] = useState<ModelCatalogue | null>(null);
  const [available, setAvailable] = useState<ToolSpec[]>([]);

  const selected = agents.find((agent) => agent.id === selectedId) ?? null;

  useEffect(() => {
    onSelect?.(selected);
    // `onSelect` is intentionally not a dependency: it is a render callback, and
    // depending on it would re-fire on every parent render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected]);

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
    void listModels(controller.signal).then(setCatalogue).catch(() => undefined);
    void listTools(controller.signal).then(setAvailable).catch(() => undefined);
    return () => controller.abort();
  }, [refresh]);

  /**
   * The picker's value for the form's current provider/model.
   *
   * An agent saved before `provider` existed has a model but no provider, so
   * the provider is recovered from the catalogue by model name. Without this
   * the select matches no option and silently renders blank -- which looks like
   * the model was never set at all.
   */
  const selectedOptionKey = (() => {
    if (!form.model) return "";
    const options = catalogue?.options ?? [];
    const exact = options.find(
      (option) => option.model === form.model && option.provider === (form.provider || "ollama"),
    );
    if (exact) return `${exact.provider}:${exact.model}`;
    const byModel = options.find((option) => option.model === form.model);
    return byModel ? `${byModel.provider}:${byModel.model}` : "";
  })();

  /** Names of the tools currently ticked in the form. */
  const chosenTools = (form.tools ?? []).map((entry) => entry.name);

  // The failure this guards against is silent: an agent with tools on a model
  // that cannot call them looks configured and simply never does anything.
  // The failure this guards against is silent: an agent with tools on a model
  // that cannot call them looks configured and simply never does anything.
  const activeOption = catalogue
    ? (catalogue.options.find(
        (option) => option.model === (form.model || catalogue.default.model) &&
          option.provider === (form.provider || catalogue.default.provider),
      ) ?? catalogue.options.find((option) => option.model === form.model))
    : undefined;
  const toolWarning =
    chosenTools.length > 0 && activeOption && !activeOption.tools
      ? `\`${activeOption.model}\` cannot call tools — it will answer without using them. Pick a model marked “tools”.`
      : null;

  const toggleTool = useCallback((name: string) => {
    setForm((prev) => {
      const current = (prev.tools ?? []).map((entry) => entry.name);
      const next = current.includes(name)
        ? current.filter((entry) => entry !== name)
        : [...current, name];
      return { ...prev, tools: next.map((entry) => ({ name: entry })) };
    });
  }, []);

  const set = useCallback(<K extends keyof AgentInput>(key: K, value: AgentInput[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  }, []);

  const resetForm = useCallback(() => {
    setForm(EMPTY_FORM);
    setEditingId(null);
  }, []);

  /** Fill the form from a preset, so the first step is talking, not writing. */
  const applyPreset = useCallback((presetId: string) => {
    const preset = AGENT_PRESETS.find((entry) => entry.id === presetId);
    if (!preset) return;
    setEditingId(null);
    setForm({
      name: preset.label,
      role: preset.role,
      system_prompt: preset.systemPrompt,
      voice: WIRED_VOICE,
      language: "en",
      greeting: preset.greeting,
    });
  }, []);

  const onEdit = useCallback((agent: Agent) => {
    setEditingId(agent.id);
    setForm({
      name: agent.name,
      role: agent.role,
      system_prompt: agent.system_prompt,
      voice: agent.voice || WIRED_VOICE,
      language: agent.language || "en",
      greeting: agent.greeting,
      model: agent.model || "",
      provider: agent.provider || "",
      tools: (agent.tools ?? []).map((entry) => ({ name: entry.name })),
    });
  }, []);

  /**
   * Load an agent into the form, and make it the one the console talks to.
   *
   * These used to be two separate selections -- pick in the list to talk, click
   * Edit to change -- which meant changing the model in the form could silently
   * affect a different agent than the conversation was running against.
   */
  const openAgent = useCallback((agent: Agent) => {
    setSelectedId(agent.id);
    onEdit(agent);
  }, [onEdit]);

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
      // Keep it open. Blanking the form after a save leaves the sidebar empty
      // while the console is still talking to that agent.
      openAgent(saved);
    } catch (cause) {
      setError(describe(cause));
    } finally {
      setBusy(false);
    }
  }, [editingId, form, openAgent, refresh]);

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

        {!editingId ? (
          <div className="mb-4">
            <p className="mb-1.5 text-[10px] uppercase tracking-wide text-cloudpso-muted">
              Start from an example
            </p>
            <div className="flex flex-wrap gap-1.5">
              {AGENT_PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  type="button"
                  title={preset.blurb}
                  data-testid={`preset-${preset.id}`}
                  onClick={() => applyPreset(preset.id)}
                  className="rounded-full border border-cloudpso-line px-2.5 py-1 text-[11px] text-cloudpso-muted transition hover:border-cloudpso-accent/60 hover:text-slate-100"
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>
        ) : null}

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

        <label className="mt-3 flex flex-col gap-1 text-[11px] text-cloudpso-muted">
          Greeting — spoken before the caller says anything (optional)
          <input
            value={form.greeting}
            onChange={(event) => set("greeting", event.target.value)}
            placeholder="Hello! How may I help you today?"
            data-testid="agent-greeting"
            className="rounded-lg border border-cloudpso-line bg-cloudpso-panel2 px-3 py-1.5 text-sm text-slate-100 outline-none focus:border-cloudpso-accent"
          />
        </label>

        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <label className="flex flex-col gap-1 text-[11px] text-cloudpso-muted">
            Model
            <select
              value={selectedOptionKey}
              onChange={(event) => {
                const [provider, ...rest] = event.target.value.split(":");
                setForm((prev) => ({
                  ...prev,
                  provider: event.target.value ? provider : "",
                  model: rest.join(":"),
                }));
              }}
              data-testid="agent-model"
              className="rounded-lg border border-cloudpso-line bg-cloudpso-panel2 px-3 py-1.5 text-sm text-slate-100 outline-none focus:border-cloudpso-accent"
            >
              <option value="">
                Service default
                {catalogue ? ` (${catalogue.default.provider}/${catalogue.default.model})` : ""}
              </option>
              {(catalogue?.options ?? []).map((option) => (
                <option
                  key={`${option.provider}:${option.model}`}
                  value={`${option.provider}:${option.model}`}
                >
                  {option.provider} · {option.model}
                  {option.tools ? " — tools" : " — no tools"}
                </option>
              ))}
            </select>
          </label>
        </div>

        <fieldset className="mt-3">
          <legend className="text-[11px] text-cloudpso-muted">
            Tools this agent may call
          </legend>
          {toolWarning ? (
            <p className="mt-1 text-[11px] text-cloudpso-warn" data-testid="tool-warning">
              {toolWarning}
            </p>
          ) : null}
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {available.length === 0 ? (
              <span className="text-[11px] text-cloudpso-muted">None available.</span>
            ) : null}
            {available.map((tool) => {
              const on = chosenTools.includes(tool.name);
              return (
                <button
                  key={tool.name}
                  type="button"
                  title={tool.description}
                  data-testid={`tool-${tool.name}`}
                  aria-pressed={on}
                  onClick={() => toggleTool(tool.name)}
                  className={`rounded-full border px-2.5 py-1 font-mono text-[11px] transition ${
                    on
                      ? "border-cloudpso-accent/60 bg-cloudpso-accent/15 text-cloudpso-accent"
                      : "border-cloudpso-line text-cloudpso-muted hover:text-slate-100"
                  }`}
                >
                  {tool.name}
                </button>
              );
            })}
          </div>
        </fieldset>

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
          <div className="flex items-center gap-2">
            <StatusPill label={`${agents.length}`} tone="info" testId="agent-count" />
            <button
              type="button"
              onClick={resetForm}
              data-testid="new-agent"
              className="rounded-full border border-cloudpso-line px-2.5 py-1 text-[11px] text-cloudpso-muted transition hover:border-cloudpso-accent/60 hover:text-slate-100"
            >
              + New
            </button>
          </div>
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
                      onClick={() => openAgent(agent)}
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
                        title="Reset the form to this agent's saved values"
                      >
                        Revert
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

      {/* Per-agent knowledge: still settings, still the sidebar. */}
      {selected ? <KnowledgePanel agent={selected} /> : null}
    </div>
  );
}
