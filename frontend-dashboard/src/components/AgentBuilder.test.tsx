import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AgentBuilder } from "./AgentBuilder";

interface Call {
  url: string;
  method: string;
  body: unknown;
}

const calls: Call[] = [];

/** Route by method + path so the component's refreshes land where expected. */
function stubApi(initial: unknown[] = []) {
  let agents = initial as Record<string, unknown>[];
  vi.stubGlobal("fetch", (url: string | URL, init?: RequestInit) => {
    const href = String(url);
    const method = init?.method ?? "GET";
    calls.push({ url: href, method, body: init?.body });

    if (href.includes("/documents")) {
      return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }));
    }
    if (href.includes("/agents")) {
      if (method === "POST") {
        const created = { id: "new-1", ...(JSON.parse(String(init!.body)) as object) };
        agents = [...agents, created];
        return Promise.resolve(new Response(JSON.stringify(created), { status: 201 }));
      }
      return Promise.resolve(new Response(JSON.stringify(agents), { status: 200 }));
    }
    return Promise.resolve(new Response("{}", { status: 200 }));
  });
}

const AGENT = {
  id: "a1",
  name: "Downtown clinic",
  role: "clinic receptionist",
  system_prompt: "You are a clinic receptionist.",
  voice: "vits-piper-en_US-amy-low",
  language: "en",
  greeting: "",
  created_at: "",
  updated_at: "",
};

afterEach(() => {
  calls.length = 0;
  vi.unstubAllGlobals();
});

describe("AgentBuilder", () => {
  it("lists the agents it loads from the API", async () => {
    stubApi([AGENT]);
    render(<AgentBuilder />);
    expect(await screen.findByText("Downtown clinic")).toBeInTheDocument();
    expect(screen.getByText(/clinic receptionist/)).toBeInTheDocument();
  });

  it("says so plainly when there are no agents yet", async () => {
    stubApi([]);
    render(<AgentBuilder />);
    expect(await screen.findByText(/No agents yet/i)).toBeInTheDocument();
  });

  it("creates an agent and shows it in the list", async () => {
    stubApi([]);
    const user = userEvent.setup();
    render(<AgentBuilder />);
    await screen.findByText(/No agents yet/i);

    await user.type(screen.getByTestId("agent-name"), "Sales SDR");
    await user.type(screen.getByTestId("agent-role"), "sales development");
    await user.click(screen.getByTestId("save-agent"));

    await waitFor(() => {
      const post = calls.find((call) => call.method === "POST" && call.url.includes("/agents"));
      expect(post).toBeDefined();
      expect(JSON.parse(String(post!.body))).toMatchObject({ name: "Sales SDR", role: "sales development" });
    });
    expect(await screen.findByText("Sales SDR")).toBeInTheDocument();
  });

  it("refuses to create an agent with no name", async () => {
    stubApi([]);
    const user = userEvent.setup();
    render(<AgentBuilder />);
    await screen.findByText(/No agents yet/i);

    await user.click(screen.getByTestId("save-agent"));

    expect(await screen.findByText(/needs a name/i)).toBeInTheDocument();
    expect(calls.some((call) => call.method === "POST")).toBe(false);
  });

  it("reveals the knowledge base and talk panels once an agent is selected", async () => {
    stubApi([AGENT]);
    const user = userEvent.setup();
    render(<AgentBuilder />);

    await user.click(await screen.findByText("Downtown clinic"));

    expect(await screen.findByTestId("knowledge-panel")).toBeInTheDocument();
    expect(screen.getByTestId("talk-to-agent")).toBeInTheDocument();
    // No documents attached yet -> the agent is ungrounded, and the UI says so.
    expect(screen.getByTestId("knowledge-count")).toHaveTextContent(/ungrounded/i);
  });

  it("only offers a voice the TTS backend actually ships", async () => {
    stubApi([]);
    render(<AgentBuilder />);
    await screen.findByText(/No agents yet/i);

    const options = Array.from(screen.getByTestId("agent-voice").querySelectorAll("option")).map(
      (option) => option.value,
    );
    expect(options).toEqual(["vits-piper-en_US-amy-low"]);
  });

  it("marks non-English languages as not wired rather than pretending", async () => {
    stubApi([]);
    render(<AgentBuilder />);
    await screen.findByText(/No agents yet/i);

    const labels = Array.from(screen.getByTestId("agent-language").querySelectorAll("option")).map(
      (option) => option.textContent ?? "",
    );
    expect(labels.some((label) => /English.*wired(?! yet)/i.test(label))).toBe(true);
    expect(labels.some((label) => /not wired yet/i.test(label))).toBe(true);
  });
});
