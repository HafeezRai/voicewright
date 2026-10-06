import { afterEach, describe, expect, it, vi } from "vitest";

import {
  AgentApiError,
  addTextDocument,
  createAgent,
  deleteAgent,
  listAgents,
  listDocuments,
  runAgentTurn,
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
