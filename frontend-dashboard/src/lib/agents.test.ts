import { afterEach, describe, expect, it, vi } from "vitest";

import {
  AgentApiError,
  addTextDocument,
  createAgent,
  deleteAgent,
  listAgents,
  listDocuments,
  runAgentTurn,
  streamAgentTurn,
} from "./agents";

interface Call {
  url: string;
  init?: RequestInit;
}

const calls: Call[] = [];

function stub(handler: (call: Call) => { status?: number; body?: unknown }): void {
  vi.stubGlobal("fetch", (url: string | URL, init?: RequestInit) => {
    const call = { url: String(url), init };
    calls.push(call);
    const { status = 200, body = {} } = handler(call);
    return Promise.resolve(
      new Response(JSON.stringify(body), {
        status,
        headers: { "Content-Type": "application/json" },
      }),
    );
  });
}

afterEach(() => {
  calls.length = 0;
  vi.unstubAllGlobals();
});

describe("agents client", () => {
  it("lists agents from a bare array response", async () => {
    stub(() => ({ body: [{ id: "a1", name: "Clinic" }] }));
    const agents = await listAgents();
    expect(agents).toHaveLength(1);
    expect(agents[0].name).toBe("Clinic");
    expect(calls[0].url).toContain("/agents");
  });

  it("lists agents from a wrapped response too", async () => {
    // The endpoint returned a bare list when last checked, but tolerating the
    // wrapped shape costs nothing and avoids a silent [] if that ever changes.
    stub(() => ({ body: { agents: [{ id: "a1", name: "Clinic" }] } }));
    await expect(listAgents()).resolves.toHaveLength(1);
  });

  it("creates an agent with a JSON body", async () => {
    stub(() => ({ body: { id: "a2", name: "Sales" } }));
    await createAgent({ name: "Sales", role: "sdr" });
    const init = calls[0].init!;
    expect(init.method).toBe("POST");
    expect((init.headers as Record<string, string>)["Content-Type"]).toBe("application/json");
    expect(JSON.parse(String(init.body))).toEqual({ name: "Sales", role: "sdr" });
  });

  it("surfaces the server's detail message on failure", async () => {
    stub(() => ({ status: 404, body: { detail: "no such agent" } }));
    await expect(listAgents()).rejects.toBeInstanceOf(AgentApiError);
    await expect(listAgents()).rejects.toThrow("no such agent");
  });

  it("sends agent_id and include_audio on a pipeline turn", async () => {
    stub(() => ({ body: { transcript: "hi", reply_text: "hello", knowledge_chunks: 2 } }));
    const turn = await runAgentTurn("agent-7", new Blob(["x"], { type: "audio/wav" }));
    expect(calls[0].url).toContain("/pipeline?agent_id=agent-7");
    expect(calls[0].url).toContain("include_audio=true");
    expect((calls[0].init!.headers as Record<string, string>)["Content-Type"]).toBe("audio/wav");
    expect(turn.knowledge_chunks).toBe(2);
  });

  it("URL-encodes an agent id so a query-shaped id cannot escape its param", async () => {
    stub(() => ({ body: [] }));
    await listDocuments("a&org_id=other");
    expect(calls[0].url).toContain("a%26org_id%3Dother");
    expect(calls[0].url).not.toContain("org_id=other");
  });

  it("posts a pasted document as JSON", async () => {
    stub(() => ({ body: { id: "d1", chunk_count: 3 } }));
    await addTextDocument("a1", "Policy", "Saturday is existing patients only.");
    expect(JSON.parse(String(calls[0].init!.body))).toEqual({
      title: "Policy",
      text: "Saturday is existing patients only.",
    });
  });

  it("deletes with no body", async () => {
    stub(() => ({ body: "" }));
    await deleteAgent("a1");
    expect(calls[0].init!.method).toBe("DELETE");
    expect(calls[0].init!.body).toBeUndefined();
  });

  it("reports an unreachable ai-engine distinctly from an HTTP error", async () => {
    vi.stubGlobal("fetch", () => Promise.reject(new TypeError("Failed to fetch")));
    await expect(listAgents()).rejects.toThrow(/unreachable/);
  });
});

/** Build an SSE response whose body arrives in awkwardly small pieces. */
function sseResponse(frames: string[], chunkSize = 7): Response {
  const bytes = new TextEncoder().encode(frames.join(""));
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (let i = 0; i < bytes.length; i += chunkSize) {
        controller.enqueue(bytes.slice(i, i + chunkSize));
      }
      controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

function sse(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

describe("streamAgentTurn", () => {
  const wav = () => new Blob(["x"], { type: "audio/wav" });

  it("parses frames that straddle network reads", async () => {
    // 7-byte chunks guarantee every frame boundary falls mid-frame at least
    // once, which is what the incremental buffer exists to survive.
    const frames = [
      sse("transcript", { text: "how do i book" }),
      sse("reply_delta", { text: "Sure" }),
      sse("reply_delta", { text: ", let me help." }),
      sse("done", { reply_text: "Sure, let me help.", model: "llama3:8b", knowledge_chunks: 2 }),
    ];
    vi.stubGlobal("fetch", () => Promise.resolve(sseResponse(frames)));

    const transcript: string[] = [];
    const deltas: string[] = [];
    let done: unknown = null;

    await streamAgentTurn("a1", wav(), {
      onTranscript: (t) => transcript.push(t),
      onDelta: (t) => deltas.push(t),
      onDone: (r) => (done = r),
    });

    expect(transcript).toEqual(["how do i book"]);
    expect(deltas.join("")).toBe("Sure, let me help.");
    expect(done).toEqual({ replyText: "Sure, let me help.", model: "llama3:8b", knowledgeChunks: 2 });
  });

  it("decodes audio frames to PCM and reports the sample rate", async () => {
    const pcm = btoa(String.fromCharCode(0x01, 0x00, 0xff, 0xff));
    vi.stubGlobal("fetch", () =>
      Promise.resolve(
        sseResponse([
          sse("audio", { sample_rate: 16000, pcm16: pcm }),
          sse("done", { reply_text: "hi", model: "m", knowledge_chunks: 0 }),
        ]),
      ),
    );

    const heard: { samples: number[]; rate: number }[] = [];
    await streamAgentTurn("a1", wav(), {
      onAudio: (samples, rate) => heard.push({ samples: Array.from(samples), rate }),
    });

    expect(heard).toEqual([{ samples: [1, -1], rate: 16000 }]);
  });

  it("stops at the terminal event instead of reading the rest of the stream", async () => {
    // A server that keeps the connection open after `done` must not hang the
    // caller -- the turn is over when the terminal frame says so.
    const frames = [
      sse("done", { reply_text: "early", model: "m", knowledge_chunks: 0 }),
      sse("reply_delta", { text: "should never be seen" }),
    ];
    vi.stubGlobal("fetch", () => Promise.resolve(sseResponse(frames)));

    const deltas: string[] = [];
    await streamAgentTurn("a1", wav(), { onDelta: (t) => deltas.push(t) });
    expect(deltas).toEqual([]);
  });

  it("surfaces a mid-stream error event through onError", async () => {
    vi.stubGlobal("fetch", () =>
      Promise.resolve(sseResponse([sse("error", { detail: "model exploded" })])),
    );

    const errors: string[] = [];
    await streamAgentTurn("a1", wav(), { onError: (d) => errors.push(d) });
    expect(errors).toEqual(["model exploded"]);
  });

  it("reports a cancelled turn through onCancelled, not onDone", async () => {
    vi.stubGlobal("fetch", () =>
      Promise.resolve(
        sseResponse([sse("cancelled", { reply_text: "We don't", model: "m", knowledge_chunks: 1 })]),
      ),
    );

    let done = false;
    let cancelled: unknown = null;
    await streamAgentTurn("a1", wav(), {
      onDone: () => (done = true),
      onCancelled: (r) => (cancelled = r),
    });

    expect(done).toBe(false);
    expect(cancelled).toEqual({ replyText: "We don't", model: "m", knowledgeChunks: 1 });
  });

  it("throws before streaming if the agent does not exist", async () => {
    vi.stubGlobal("fetch", () =>
      Promise.resolve(
        new Response(JSON.stringify({ detail: "agent not found: nope" }), { status: 404 }),
      ),
    );

    await expect(streamAgentTurn("nope", wav())).rejects.toThrow("agent not found: nope");
  });

  it("accepts a stream with no terminal frame at all", async () => {
    vi.stubGlobal("fetch", () =>
      Promise.resolve(sseResponse([sse("reply_delta", { text: "cut off" })])),
    );
    const deltas: string[] = [];
    await expect(streamAgentTurn("a1", wav(), { onDelta: (t) => deltas.push(t) })).resolves.toBeUndefined();
    expect(deltas).toEqual(["cut off"]);
  });
});
