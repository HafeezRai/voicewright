import { useCallback, useEffect, useRef, useState } from "react";

import {
  AgentApiError,
  type Agent,
  type AgentDocument,
  addFileDocument,
  addTextDocument,
  deleteDocument,
  listDocuments,
} from "../lib/agents";
import { StatusPill } from "./StatusPill";

function describe(error: unknown): string {
  if (error instanceof AgentApiError) return error.message;
  return error instanceof Error ? error.message : String(error);
}

/**
 * The knowledge that grounds one agent's answers.
 *
 * Whatever lives here is chunked, embedded and retrieved into the system prompt
 * at answer time. An agent with no documents is not "empty" — it is
 * *ungrounded*, which means it will answer from the model's priors and can
 * confidently invent policy. That distinction is the reason this panel exists.
 */
export function KnowledgePanel({ agent }: { agent: Agent }) {
  const [documents, setDocuments] = useState<AgentDocument[]>([]);
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const refresh = useCallback(
    async (signal?: AbortSignal) => {
      try {
        setDocuments(await listDocuments(agent.id, signal));
        setError(null);
      } catch (cause) {
        if ((cause as Error)?.name === "AbortError") return;
        setError(describe(cause));
      }
    },
    [agent.id],
  );

  useEffect(() => {
    const controller = new AbortController();
    setDocuments([]);
    setTitle("");
    setText("");
    void refresh(controller.signal);
    return () => controller.abort();
  }, [refresh]);

  const run = useCallback(
    async (work: () => Promise<unknown>) => {
      setBusy(true);
      setError(null);
      try {
        await work();
        await refresh();
      } catch (cause) {
        setError(describe(cause));
      } finally {
        setBusy(false);
      }
    },
    [refresh],
  );

  const onAddText = useCallback(() => {
    const trimmed = text.trim();
    if (!trimmed) return;
    void run(async () => {
      await addTextDocument(agent.id, title.trim() || "Pasted note", trimmed);
      setText("");
      setTitle("");
    });
  }, [agent.id, run, text, title]);

  const onAddFile = useCallback(
    (file: File) => {
      void run(() => addFileDocument(agent.id, file));
    },
    [agent.id, run],
  );

  const onFilePicked = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) onAddFile(file);
      event.target.value = "";
    },
    [onAddFile],
  );

  return (
    <section className="rounded-xl border border-voicewright-line bg-voicewright-panel p-4" data-testid="knowledge-panel">
      <header className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold">Knowledge base</h3>
          <p className="text-[11px] text-voicewright-muted">
            Chunked, embedded and retrieved into this agent's prompt at answer time.
          </p>
        </div>
        <StatusPill
          label={documents.length === 0 ? "ungrounded" : `${documents.length} document${documents.length === 1 ? "" : "s"}`}
          tone={documents.length === 0 ? "warn" : "ok"}
          testId="knowledge-count"
        />
      </header>

      <div className="flex flex-col gap-2">
        <input
          value={title}
          onChange={(event) => setTitle(event.target.value)}
          placeholder="Title (optional)"
          aria-label="Document title"
          className="rounded-lg border border-voicewright-line bg-voicewright-panel2 px-3 py-1.5 text-sm outline-none focus:border-voicewright-accent"
        />
        <textarea
          value={text}
          onChange={(event) => setText(event.target.value)}
          rows={4}
          placeholder="Paste policy, FAQs, pricing — anything this agent should know."
          aria-label="Document text"
          className="resize-y rounded-lg border border-voicewright-line bg-voicewright-panel2 px-3 py-2 text-sm outline-none focus:border-voicewright-accent"
        />
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            data-testid="add-text-document"
            disabled={busy || text.trim().length === 0}
            onClick={onAddText}
            className="rounded-lg bg-voicewright-accent px-3 py-1.5 text-sm font-medium text-white transition hover:bg-voicewright-accent/90 disabled:opacity-40"
          >
            Add text
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => fileRef.current?.click()}
            className="rounded-lg border border-voicewright-line px-3 py-1.5 text-sm text-voicewright-muted transition hover:text-slate-100 disabled:opacity-40"
          >
            Upload .txt / .md
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".txt,.md,text/plain,text/markdown"
            className="hidden"
            onChange={onFilePicked}
          />
        </div>
      </div>

      {error && <p className="mt-2 text-[11px] text-voicewright-err">{error}</p>}

      <ul className="mt-3 flex flex-col gap-1.5">
        {documents.map((document) => (
          <li
            key={document.id}
            className="flex items-center justify-between gap-3 rounded-lg border border-voicewright-line bg-voicewright-panel2 px-3 py-2"
          >
            <div className="min-w-0">
              <p className="truncate text-sm">{document.title}</p>
              <p className="font-mono text-[10px] text-voicewright-muted">
                {document.char_count} chars · {document.chunk_count} chunk
                {document.chunk_count === 1 ? "" : "s"} · {document.source}
              </p>
            </div>
            <button
              type="button"
              onClick={() => void run(() => deleteDocument(document.id))}
              disabled={busy}
              aria-label={`Delete ${document.title}`}
              className="shrink-0 text-[11px] text-voicewright-muted transition hover:text-voicewright-err disabled:opacity-40"
            >
              Delete
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
