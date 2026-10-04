import { describe, expect, it } from "vitest";

import {
  type VoiceEvent,
  type VoiceState,
  initialVoiceState,
  isLive,
  statusLabel,
  voiceReducer,
} from "./voiceMachine";

function run(events: VoiceEvent[], from: VoiceState = initialVoiceState): VoiceState {
  return events.reduce(voiceReducer, from);
}

describe("voiceReducer lifecycle", () => {
  it("walks idle -> requesting -> connecting -> connected", () => {
    let state = run([{ type: "start" }]);
    expect(state.status).toBe("requesting");

    state = voiceReducer(state, { type: "mic-granted" });
    expect(state.status).toBe("connecting");
    expect(state.micGranted).toBe(true);

    state = voiceReducer(state, { type: "signaling-open" });
    state = voiceReducer(state, { type: "answer", sessionId: "abc123" });
    expect(state.sessionId).toBe("abc123");
    expect(state.status).toBe("connecting"); // still connecting until ICE is up

    state = voiceReducer(state, { type: "ice", state: "connected" });
    expect(state.status).toBe("connected");
    expect(state.ice).toBe("connected");
  });

  it("ignores start while already live", () => {
    const state = run([{ type: "start" }, { type: "mic-granted" }]);
    expect(voiceReducer(state, { type: "start" }).status).toBe("connecting");
  });

  it("fails cleanly when the microphone is denied", () => {
    const state = run([{ type: "start" }, { type: "mic-denied", error: "NotAllowedError" }]);
    expect(state.status).toBe("failed");
    expect(state.micGranted).toBe(false);
    expect(state.error).toBe("NotAllowedError");
  });

  it("returns to connecting when ICE drops", () => {
    const connected = run([
      { type: "start" },
      { type: "mic-granted" },
      { type: "ice", state: "connected" },
    ]);
    const dropped = voiceReducer(connected, { type: "ice", state: "disconnected" });
    expect(dropped.status).toBe("connecting");
  });
});

describe("voiceReducer speaking and barge-in", () => {
  const connected = run([
    { type: "start" },
    { type: "mic-granted" },
    { type: "ice", state: "connected" },
  ]);

  it("tracks the agent speaking and returning the floor to the user", () => {
    const speaking = voiceReducer(connected, { type: "agent-speaking", speaking: true });
    expect(speaking.status).toBe("speaking");
    expect(speaking.agentSpeaking).toBe(true);

    const stopped = voiceReducer(speaking, { type: "agent-speaking", speaking: false });
    expect(stopped.status).toBe("listening");
  });

  it("records a barge-in, counts it and hands the floor to the listener", () => {
    const speaking = voiceReducer(connected, { type: "agent-speaking", speaking: true });
    const interrupted = voiceReducer(speaking, { type: "barge-in", durationMs: 224 });

    expect(interrupted.status).toBe("listening");
    expect(interrupted.bargeInActive).toBe(true);
    expect(interrupted.bargeInCount).toBe(1);
    expect(interrupted.lastBargeInMs).toBe(224);
    expect(interrupted.agentSpeaking).toBe(false);
  });

  it("stays in listening after the agent resumes and stops again", () => {
    let state = voiceReducer(connected, { type: "barge-in", durationMs: 224 });
    state = voiceReducer(state, { type: "agent-speaking", speaking: true });
    expect(state.status).toBe("speaking");
    expect(state.bargeInActive).toBe(false);
    state = voiceReducer(state, { type: "agent-speaking", speaking: false });
    expect(state.status).toBe("listening");
  });
});

describe("voiceReducer teardown and errors", () => {
  it("closes the session and clears the speaking flags", () => {
    const state = run([
      { type: "start" },
      { type: "mic-granted" },
      { type: "ice", state: "connected" },
      { type: "agent-speaking", speaking: true },
      { type: "stop" },
      { type: "closed" },
    ]);
    expect(state.status).toBe("closed");
    expect(state.ice).toBe("closed");
    expect(state.agentSpeaking).toBe(false);
    expect(state.bargeInActive).toBe(false);
  });

  it("surfaces server errors as a failed status", () => {
    const state = run([{ type: "start" }, { type: "error", message: "boom" }]);
    expect(state.status).toBe("failed");
    expect(state.error).toBe("boom");
  });
});

describe("helpers", () => {
  it("classifies liveness", () => {
    expect(isLive("idle")).toBe(false);
    expect(isLive("connecting")).toBe(true);
    expect(isLive("speaking")).toBe(true);
    expect(isLive("closed")).toBe(false);
  });

  it("labels every status", () => {
    for (const status of [
      "idle",
      "requesting",
      "connecting",
      "connected",
      "speaking",
      "listening",
      "closing",
      "closed",
      "failed",
    ] as const) {
      expect(statusLabel(status)).toBeTruthy();
    }
  });
});
