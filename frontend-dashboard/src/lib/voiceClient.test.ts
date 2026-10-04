import { describe, expect, it, vi } from "vitest";

import type { VoiceEvent } from "../state/voiceMachine";
import { VoiceClient, type VoiceClientDeps } from "./voiceClient";

// --- fakes -----------------------------------------------------------------

class FakeTrack {
  kind = "audio";
  stopped = false;
  stop(): void {
    this.stopped = true;
  }
}

class FakeStream {
  constructor(readonly tracks: FakeTrack[] = [new FakeTrack()]) {}
  getTracks(): FakeTrack[] {
    return this.tracks;
  }
  getAudioTracks(): FakeTrack[] {
    return this.tracks;
  }
}

class FakePC {
  onconnectionstatechange: (() => void) | null = null;
  oniceconnectionstatechange: (() => void) | null = null;
  ontrack: ((e: { track: unknown; streams: unknown[] }) => void) | null = null;

  connectionState = "new";
  iceConnectionState = "new";
  /** "complete" so the vanilla-ICE gather wait resolves immediately. */
  iceGatheringState = "complete";
  localDescription: { type: string; sdp: string } | null = null;
  addedTracks: unknown[] = [];
  remoteDescriptions: unknown[] = [];
  candidates: unknown[] = [];
  closed = false;

  addTrack(track: unknown): void {
    this.addedTracks.push(track);
  }
  addEventListener(): void {}
  removeEventListener(): void {}
  async createOffer(): Promise<{ type: string; sdp: string }> {
    return { type: "offer", sdp: "v=0\r\nfake-offer" };
  }
  async setLocalDescription(description: { type: string; sdp: string }): Promise<void> {
    this.localDescription = { type: description.type, sdp: `${description.sdp}\r\na=candidate:host` };
  }
  async setRemoteDescription(description: unknown): Promise<void> {
    this.remoteDescriptions.push(description);
  }
  async addIceCandidate(candidate: unknown): Promise<void> {
    this.candidates.push(candidate);
  }
  close(): void {
    this.closed = true;
  }
}

class FakeWS {
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;

  readyState = 0;
  sent: Record<string, unknown>[] = [];
  closedExplicitly = false;

  send(data: string): void {
    this.sent.push(JSON.parse(data) as Record<string, unknown>);
  }
  close(): void {
    this.closedExplicitly = true;
    this.readyState = 3;
    this.onclose?.();
  }
  open(): void {
    this.readyState = 1;
    this.onopen?.();
  }
  deliver(message: Record<string, unknown>): void {
    this.onmessage?.({ data: JSON.stringify(message) });
  }
}

const flush = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

interface Harness {
  client: VoiceClient;
  events: VoiceEvent[];
  pc: FakePC;
  ws: FakeWS;
  stream: FakeStream;
}

function makeHarness(overrides: Partial<VoiceClientDeps> = {}): Harness {
  const events: VoiceEvent[] = [];
  const pc = new FakePC();
  const ws = new FakeWS();
  const stream = new FakeStream();
  const deps: VoiceClientDeps = {
    getUserMedia: async () => stream as unknown as MediaStream,
    createPeerConnection: () => pc as unknown as RTCPeerConnection,
    createWebSocket: () => ws as unknown as WebSocket,
    ...overrides,
  };
  const client = new VoiceClient({ dispatch: (e) => events.push(e), deps });
  return { client, events, pc, ws, stream };
}

// --- tests -----------------------------------------------------------------

describe("VoiceClient signalling", () => {
  it("captures the mic, sends an offer on socket open, and applies the answer", async () => {
    const { client, events, pc, ws, stream } = makeHarness();

    await client.start();
    expect(events.map((e) => e.type)).toContain("mic-granted");
    expect(client.microphone).toBe(stream);
    expect(pc.addedTracks).toHaveLength(1);

    ws.open();
    await flush();
    expect(events.map((e) => e.type)).toContain("signaling-open");
    // Vanilla ICE: the offer SDP already carries the gathered candidates.
    expect(ws.sent[0]).toEqual({ type: "offer", sdp: "v=0\r\nfake-offer\r\na=candidate:host" });

    ws.deliver({ type: "answer", sdp: "v=0\r\nanswer", session_id: "s1" });
    await flush();
    expect(events).toContainEqual({ type: "answer", sessionId: "s1" });
    expect(pc.remoteDescriptions[0]).toEqual({ type: "answer", sdp: "v=0\r\nanswer" });
  });

  it("applies the server's ICE candidates and reports ICE state changes", async () => {
    const { client, events, pc, ws } = makeHarness();
    await client.start();
    ws.open();
    await flush();

    ws.deliver({ type: "ice-candidate", candidate: { candidate: "remote-cand" } });
    await flush();
    expect(pc.candidates).toContainEqual({ candidate: "remote-cand" });

    pc.connectionState = "connected";
    pc.onconnectionstatechange?.();
    expect(events).toContainEqual({ type: "ice", state: "connected" });
  });

  it("never tricks ICE candidates to the server (vanilla ICE only)", async () => {
    // Regression guard: backend-core's add_ice_candidate raises TypeError on any
    // candidate, so trickling would tear the session down. See voiceClient.ts.
    const { client, ws } = makeHarness();
    await client.start();
    ws.open();
    await flush();

    ws.deliver({ type: "answer", sdp: "v=0 answer", session_id: "s1" });
    await flush();

    expect(ws.sent.some((m) => m.type === "ice-candidate")).toBe(false);
    expect(ws.sent.filter((m) => m.type === "offer")).toHaveLength(1);
  });

  it("surfaces barge-in events from the server", async () => {
    const { client, events, ws } = makeHarness();
    await client.start();
    ws.open();
    await flush();

    ws.deliver({ type: "barge-in", duration_ms: 224 });
    await flush();
    expect(events).toContainEqual({ type: "barge-in", durationMs: 224 });
  });

  it("drives the playback sink when the agent speaks", async () => {
    const { client, events, ws } = makeHarness();
    await client.start();
    ws.open();
    await flush();

    client.setAgentSpeaking(true);
    client.setAgentSpeaking(false);

    expect(ws.sent).toContainEqual({ type: "playback", state: "start" });
    expect(ws.sent).toContainEqual({ type: "playback", state: "stop" });
    expect(events).toContainEqual({ type: "agent-speaking", speaking: true });
    expect(events).toContainEqual({ type: "agent-speaking", speaking: false });
  });

  it("reports a microphone denial without opening a socket", async () => {
    const events: VoiceEvent[] = [];
    const createWebSocket = vi.fn();
    const client = new VoiceClient({
      dispatch: (e) => events.push(e),
      deps: {
        getUserMedia: async () => {
          throw new DOMException("denied", "NotAllowedError");
        },
        createPeerConnection: () => new FakePC() as unknown as RTCPeerConnection,
        createWebSocket: createWebSocket as unknown as VoiceClientDeps["createWebSocket"],
      },
    });

    await client.start();
    expect(events[0]?.type).toBe("mic-denied");
    expect(createWebSocket).not.toHaveBeenCalled();
  });
});

describe("VoiceClient teardown", () => {
  it("releases the microphone, closes the peer connection and the socket", async () => {
    const { client, events, pc, ws, stream } = makeHarness();
    await client.start();
    ws.open();
    await flush();

    client.close();

    expect(ws.sent).toContainEqual({ type: "close" });
    expect(stream.getTracks().every((t) => t.stopped)).toBe(true);
    expect(pc.closed).toBe(true);
    expect(ws.closedExplicitly).toBe(true);
    expect(client.microphone).toBeNull();
    expect(client.peerConnection).toBeNull();
    expect(events).toContainEqual({ type: "closed" });
  });

  it("is idempotent — a second close dispatches no further events", async () => {
    const { client, events } = makeHarness();
    await client.start();

    client.close();
    const count = events.filter((e) => e.type === "closed").length;
    client.close();
    expect(events.filter((e) => e.type === "closed").length).toBe(count);
  });
});
