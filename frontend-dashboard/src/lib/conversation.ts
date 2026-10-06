/**
 * One exchange in a call, as the UI needs it.
 *
 * Kept separate from the wire types (`AgentTurn`, the SSE callbacks) because
 * the transcript accumulates a turn *while* it is still happening: the reply
 * grows token by token and the timings are filled in later. The wire types
 * describe a finished request; this describes a row on screen.
 */
import type { ToolInvocation } from "./agents";

export interface ConversationTurn {
  id: number;
  /** What the caller said. Empty until speech-to-text returns. */
  transcript: string;
  /** What the agent is saying. Grows token by token while streaming. */
  reply: string;
  /** Knowledge chunks that grounded this turn. 0 means it answered ungrounded. */
  knowledgeChunks: number;
  model: string;
  /** Milliseconds from sending the clip to the first audible audio. */
  firstAudioMs: number | null;
  /** Milliseconds for the whole turn. null while still in flight. */
  totalMs: number | null;
  /**
   * Tools the agent ran to produce this reply, in order.
   *
   * Shown because the difference between an agent that *claims* to have booked
   * something and one that did is whether you can see the call and its result.
   */
  tools: ToolInvocation[];
  /** A note shown under the turn — an interruption, a failure, a rejected echo. */
  error?: string;
  /**
   * Set when the turn was discarded as the agent hearing itself. Rendered
   * differently: it is not something the caller said, so it must not look like
   * a contribution to the conversation.
   */
  rejectedAsEcho?: boolean;
}

export function emptyTurn(id: number): ConversationTurn {
  return {
    id,
    transcript: "",
    reply: "",
    knowledgeChunks: 0,
    model: "",
    firstAudioMs: null,
    totalMs: null,
    tools: [],
  };
}
