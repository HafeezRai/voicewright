import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import aiEngineFixture from "./test/fixtures/ai-engine-metrics.txt?raw";
import backendFixture from "./test/fixtures/backend-core-metrics.txt?raw";

// --- browser API fakes ------------------------------------------------------

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
  static instances: FakePC[] = [];
  onconnectionstatechange: (() => void) | null = null;
  oniceconnectionstatechange: (() => void) | null = null;
  ontrack: unknown = null;
  connectionState = "new";
  iceConnectionState = "new";
  iceGatheringState = "complete";
  localDescription = { type: "offer", sdp: "v=0 fake" };
  closed = false;

  constructor(public config?: unknown) {
    FakePC.instances.push(this);
  }
  addEventListener(): void {}
  removeEventListener(): void {}
  addTrack(): void {}
  async createOffer(): Promise<{ type: string; sdp: string }> {
    return { type: "offer", sdp: "v=0 fake" };
  }
  async setLocalDescription(): Promise<void> {}
  async setRemoteDescription(): Promise<void> {}
  async addIceCandidate(): Promise<void> {}
  close(): void {
    this.closed = true;
  }
}

class FakeWS {
  static instances: FakeWS[] = [];
  static OPEN = 1;
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  readyState = 0;

  constructor(public url: string) {
    FakeWS.instances.push(this);
  }
  send(): void {}
  close(): void {
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

function stubBrowserApis(stream: FakeStream): void {
  vi.stubGlobal("RTCPeerConnection", FakePC);
  vi.stubGlobal("WebSocket", FakeWS);
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia: vi.fn(async () => stream) },
  });
}

function stubFetch(handler: (url: string) => { ok: boolean; text?: string; status?: number }) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const result = handler(url);
      return {
        ok: result.ok,
        status: result.status ?? (result.ok ? 200 : 500),
        statusText: result.ok ? "OK" : "Error",
        text: async () => result.text ?? "",
        json: async () => JSON.parse(result.text ?? "{}"),
        blob: async () => new Blob(),
      } as unknown as Response;
    }),
  );
}

beforeEach(() => {
  FakePC.instances = [];
  FakeWS.instances = [];
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("App — observability dashboard", () => {
  it("renders live tiles parsed from both real /metrics fixtures", async () => {
    stubBrowserApis(new FakeStream());
    stubFetch((url) => ({ ok: true, text: url.includes(":8001") ? aiEngineFixture : backendFixture }));

    render(<App />);
    // Observability moved to its own tab when the console became the default view.
    await userEvent.setup().click(screen.getByTestId("tab-session"));

    await waitFor(() => expect(screen.getAllByTestId("svc-online")).toHaveLength(2));
    expect(screen.getByTestId("tile-active-sessions")).toHaveTextContent("1");
    expect(screen.getByTestId("tile-ws")).toHaveTextContent("1");
    expect(screen.getByTestId("tile-llm")).toHaveTextContent("p95");
    // No error series in the fixtures.
    expect(screen.getByTestId("tile-errors")).toHaveTextContent("0");
  });

  it("shows a clear offline state instead of white-screening when the backends are down", async () => {
    stubBrowserApis(new FakeStream());
    stubFetch(() => ({ ok: false, status: 502 }));

    render(<App />);
    await userEvent.setup().click(screen.getByTestId("tab-session"));

    await waitFor(() => expect(screen.getAllByTestId("svc-offline")).toHaveLength(2));
    // The page is still fully rendered.
    expect(screen.getByText("CloudPSO")).toBeInTheDocument();
    expect(screen.getByTestId("dashboard")).toBeInTheDocument();
  });
});

describe("App — voice client integration", () => {
  it("walks the session from idle to connected through the real signalling client", async () => {
    const stream = new FakeStream();
    stubBrowserApis(stream);
    stubFetch((url) => ({ ok: true, text: url.includes(":8001") ? aiEngineFixture : backendFixture }));

    const user = userEvent.setup();
    render(<App />);

    // The builder is the default view; the WebRTC session lives on its own tab.
    await user.click(screen.getByTestId("tab-session"));

    expect(screen.getByTestId("voice-status")).toHaveTextContent("Idle");

    await user.click(screen.getByTestId("start-call"));

    await waitFor(() => expect(screen.getByTestId("voice-status")).toHaveTextContent("Connecting"));

    // The client opened a WebSocket to the signalling endpoint.
    expect(FakeWS.instances).toHaveLength(1);
    expect(FakeWS.instances[0].url).toContain("/ws/signaling");

    // Drive the server side of the handshake.
    await act(async () => {
      FakeWS.instances[0].open();
      await Promise.resolve();
    });
    await act(async () => {
      FakeWS.instances[0].deliver({ type: "answer", sdp: "v=0 answer", session_id: "sess-42" });
      await Promise.resolve();
    });
    await act(async () => {
      const pc = FakePC.instances[0];
      pc.connectionState = "connected";
      pc.onconnectionstatechange?.();
      await Promise.resolve();
    });

    await waitFor(() => expect(screen.getByTestId("voice-status")).toHaveTextContent("Connected"));
    expect(screen.getByTestId("ice-status")).toHaveTextContent("connected");
    expect(screen.getByTestId("session-id")).toHaveTextContent("sess-42");

    // Ending the call releases the microphone track.
    await user.click(screen.getByTestId("stop-call"));
    await waitFor(() => expect(screen.getByTestId("voice-status")).toHaveTextContent("Closed"));
    expect(stream.getTracks().every((t) => t.stopped)).toBe(true);
  });
});
