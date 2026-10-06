import { useEffect, useRef } from "react";

import type { ConversationTurn } from "../lib/conversation";

/**
 * The conversation, as a conversation.
 *
 * The previous UI listed turns as rows of result data — mode, grounding,
 * timings, the reply text — which reads like a log because it *was* one. A
 * call is two people talking, so it is rendered as speech: the agent on the
 * left, the caller on the right, in the order it happened.
 *
 * Timings and grounding are still there, demoted to a caption under each agent
 * turn. They are evidence, not the content.
 */
export function Transcript({
  turns,
  greeting,
  live,
  emptyHint,
}: {
  turns: ConversationTurn[];
  /** Spoken by the agent before the caller says anything. */
  greeting?: string;
  /** True while a call is running, so the caption can say who is waiting. */
  live?: boolean;
  emptyHint?: React.ReactNode;
}) {
  const endRef = useRef<HTMLDivElement | null>(null);

  // Newest at the bottom, like every conversation anyone has ever read.
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [turns, greeting]);

  const ordered = [...turns].reverse(); // state holds newest-first
  const nothingYet = ordered.length === 0 && !greeting;

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto px-5 py-4" data-testid="transcript">
      {/* `mt-auto` anchors a short conversation to the bottom, the way every
          messaging app does. Without it the first exchange floats at the top of
          a mostly-empty panel and the product looks unfinished. */}
      <div className="mt-auto flex flex-col gap-4">
        {greeting ? <AgentBubble text={greeting} greeting /> : null}

        {ordered.map((turn) => (
          <div key={turn.id} className="flex flex-col gap-2">
            {turn.transcript ? <UserBubble turn={turn} /> : null}
            {turn.reply ? <AgentBubble turn={turn} /> : null}
            {turn.error ? <Note turn={turn} /> : null}
          </div>
        ))}
        {/* A quiet "still here" marker, the way a chat shows a typing indicator. */}
        {live ? (
          <p className="flex items-center gap-1.5 text-[11px] text-cloudpso-muted">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-cloudpso-ok" />
            listening
          </p>
        ) : null}
      </div>

      {nothingYet ? (
        <div className="m-auto max-w-sm text-center text-xs leading-relaxed text-cloudpso-muted">
          {emptyHint ?? "Start the call and just talk. The agent replies when you stop."}
        </div>
      ) : null}

      <div ref={endRef} />
    </div>
  );
}

function UserBubble({ turn }: { turn: ConversationTurn }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%]">
        <p className="mb-1 text-right text-[10px] uppercase tracking-wide text-cloudpso-muted">
          You
        </p>
        <p
          className="rounded-2xl rounded-br-sm bg-cloudpso-accent/20 px-4 py-2 text-sm text-slate-100"
          data-testid="turn-transcript"
        >
          {turn.transcript}
        </p>
      </div>
    </div>
  );
}

function AgentBubble({
  text,
  turn,
  greeting,
}: {
  text?: string;
  turn?: ConversationTurn;
  greeting?: boolean;
}) {
  const body = text ?? turn?.reply ?? "";
  const streaming = turn ? turn.totalMs === null && !turn.error : false;

  return (
    <div className="flex justify-start">
      <div className="max-w-[80%]">
        <p className="mb-1 text-[10px] uppercase tracking-wide text-cloudpso-muted">Agent</p>
        <p className="rounded-2xl rounded-bl-sm bg-cloudpso-panel2 px-4 py-2 text-sm text-slate-100">
          {body}
          {streaming ? <span className="ml-0.5 animate-pulse text-cloudpso-accent">▍</span> : null}
        </p>
        {turn && !greeting ? <Caption turn={turn} /> : null}
      </div>
    </div>
  );
}

/** The evidence line: model, grounding, latency. */
function Caption({ turn }: { turn: ConversationTurn }) {
  if (turn.totalMs === null && !turn.model && turn.knowledgeChunks === 0) return null;
  return (
    <p className="mt-1 font-mono text-[10px] text-cloudpso-muted">
      {turn.knowledgeChunks > 0 ? (
        <span className="text-cloudpso-ok">grounded · {turn.knowledgeChunks}</span>
      ) : (
        <span className="text-cloudpso-warn">ungrounded</span>
      )}
      {turn.firstAudioMs !== null ? <> · first audio {turn.firstAudioMs}ms</> : null}
      {turn.totalMs !== null ? <> · turn {turn.totalMs}ms</> : null}
      {turn.model ? <> · {turn.model}</> : null}
    </p>
  );
}

function Note({ turn }: { turn: ConversationTurn }) {
  return (
    <p
      data-testid="turn-note"
      className={`text-[11px] ${turn.reply ? "text-cloudpso-warn" : "text-cloudpso-err"}`}
    >
      {turn.error}
    </p>
  );
}
