/**
 * The voice session as an explicit, pure state machine.
 *
 * All the messy imperative work (getUserMedia, RTCPeerConnection, the
 * signalling socket) lives in `lib/voiceClient.ts`; it reports what happened by
 * dispatching the events below. Keeping the transitions here — pure and
 * dependency-free — is what makes the lifecycle testable without a browser.
 *
 * Status vocabulary (as surfaced in the UI):
 *   idle        — no call
 *   requesting  — asking for microphone permission
 *   connecting  — mic granted, offer sent, ICE not yet up
 *   connected   — session up, agent not speaking (listening)
 *   speaking    — agent is talking
 *   listening   — the user has just barged in; agent yielded the floor
 *   closing     — teardown in progress
 *   closed      — call ended
 *   failed      — mic denied or the connection failed
 */

export type VoiceStatus =
  | "idle"
  | "requesting"
  | "connecting"
  | "connected"
  | "speaking"
  | "listening"
  | "closing"
  | "closed"
  | "failed";

export type IceState =
  | "none"
  | "new"
  | "checking"
  | "connected"
  | "completed"
  | "disconnected"
  | "failed"
  | "closed";

export interface LogEntry {
  at: number;
  direction: "in" | "out" | "info";
  message: string;
}

export interface VoiceState {
  status: VoiceStatus;
  ice: IceState;
  sessionId: string | null;
  micGranted: boolean;
  /** True while the agent is producing audio. */
  agentSpeaking: boolean;
  /** True from the moment barge-in fires until the agent speaks again. */
  bargeInActive: boolean;
  bargeInCount: number;
  lastBargeInMs: number | null;
  error: string | null;
  /** Newest-last signalling/lifecycle log, capped for the UI. */
  log: LogEntry[];
}

export const initialVoiceState: VoiceState = {
  status: "idle",
  ice: "none",
  sessionId: null,
  micGranted: false,
  agentSpeaking: false,
  bargeInActive: false,
  bargeInCount: 0,
  lastBargeInMs: null,
  error: null,
  log: [],
};

export type VoiceEvent =
  | { type: "start" }
  | { type: "mic-granted" }
  | { type: "mic-denied"; error: string }
  | { type: "signaling-open" }
  | { type: "answer"; sessionId: string }
  | { type: "ice"; state: IceState }
  | { type: "agent-speaking"; speaking: boolean }
  | { type: "barge-in"; durationMs: number }
  | { type: "stop" }
  | { type: "closed" }
  | { type: "error"; message: string };

const MAX_LOG = 50;

function logged(state: VoiceState, direction: LogEntry["direction"], message: string): VoiceState {
  const log = [...state.log, { at: Date.now(), direction, message }];
  if (log.length > MAX_LOG) log.splice(0, log.length - MAX_LOG);
  return { ...state, log };
}

/**
 * Where the machine sits when the agent is not speaking and nothing is wrong.
 * Once the transport is up, the agent being silent means the user has the
 * floor — so we report `listening`. `connected` is the narrow window between
 * ICE coming up and the first agent utterance.
 */
function idleStatusFor(state: VoiceState): VoiceStatus {
  if (state.ice === "connected" || state.ice === "completed") {
    return "listening";
  }
  return state.status;
}

export function voiceReducer(state: VoiceState, event: VoiceEvent): VoiceState {
  switch (event.type) {
    case "start":
      if (state.status !== "idle" && state.status !== "closed" && state.status !== "failed") {
        return state;
      }
      return logged(
        {
          ...initialVoiceState,
          status: "requesting",
          log: state.log,
        },
        "info",
        "requesting microphone",
      );

    case "mic-granted":
      return logged(
        { ...state, micGranted: true, status: "connecting", error: null },
        "info",
        "microphone granted; opening signaling",
      );

    case "mic-denied":
      return logged(
        { ...state, micGranted: false, status: "failed", error: event.error },
        "info",
        `microphone denied: ${event.error}`,
      );

    case "signaling-open":
      return logged(state, "info", "signaling socket open; sending offer");

    case "answer":
      return logged(
        { ...state, sessionId: event.sessionId, status: "connecting" },
        "in",
        `answer received (session ${event.sessionId})`,
      );

    case "ice": {
      const next: VoiceState = { ...state, ice: event.state };
      if (event.state === "connected" || event.state === "completed") {
        if (next.status === "connecting") next.status = "connected";
      } else if (event.state === "failed") {
        next.status = "failed";
        next.error = next.error ?? "ICE connection failed";
      } else if (event.state === "disconnected" && next.status === "connected") {
        next.status = "connecting";
      }
      return logged(next, "in", `ice: ${event.state}`);
    }

    case "agent-speaking":
      if (event.speaking) {
        if (state.status === "closing" || state.status === "closed" || state.status === "failed") {
          return state;
        }
        return logged(
          { ...state, agentSpeaking: true, bargeInActive: false, status: "speaking" },
          "info",
          "agent speaking",
        );
      }
      return logged(
        {
          ...state,
          agentSpeaking: false,
          status: idleStatusFor({ ...state, agentSpeaking: false }),
        },
        "info",
        "agent stopped",
      );

    case "barge-in":
      return logged(
        {
          ...state,
          agentSpeaking: false,
          bargeInActive: true,
          bargeInCount: state.bargeInCount + 1,
          lastBargeInMs: event.durationMs,
          status: "listening",
        },
        "in",
        `barge-in (${event.durationMs} ms)`,
      );

    case "stop":
      return logged({ ...state, status: "closing", agentSpeaking: false }, "info", "stopping call");

    case "closed":
      return logged(
        {
          ...state,
          status: "closed",
          ice: "closed",
          agentSpeaking: false,
          bargeInActive: false,
        },
        "info",
        "call closed",
      );

    case "error":
      return logged(
        { ...state, error: event.message, status: "failed" },
        "in",
        `error: ${event.message}`,
      );

    default:
      return state;
  }
}

/** True while a call is live (used to disable the start button). */
export function isLive(status: VoiceStatus): boolean {
  return status === "requesting" || status === "connecting" || status === "connected"
    || status === "speaking" || status === "listening";
}

export function statusLabel(status: VoiceStatus): string {
  switch (status) {
    case "idle":
      return "Idle";
    case "requesting":
      return "Requesting mic";
    case "connecting":
      return "Connecting";
    case "connected":
      return "Connected";
    case "speaking":
      return "Agent speaking";
    case "listening":
      return "Listening";
    case "closing":
      return "Closing";
    case "closed":
      return "Closed";
    case "failed":
      return "Failed";
  }
}
