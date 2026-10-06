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

    if (href.includes("/models")) {
      return Promise.resolve(
        new Response(
          JSON.stringify({
            default: { provider: "ollama", model: "llama3:8b" },
            options: [
              { provider: "ollama", model: "llama3:8b", tools: false },
              { provider: "ollama", model: "qwen3.5:0.8b", tools: true },
              { provider: "deepseek", model: "deepseek-chat", tools: true },
            ],
          }),
          { status: 200 },
        ),
      );
    }
    if (href.includes("/tools")) {
      return Promise.resolve(
        new Response(
          JSON.stringify([
            { name: "check_availability", description: "List free slots", parameters: {} },
            { name: "book_appointment", description: "Book one", parameters: {} },
          ]),
          { status: 200 },
        ),
      );
    }
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
  model: "",
  provider: "",
  tools: [],
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

  it("reveals the knowledge base once an agent is selected", async () => {
    stubApi([AGENT]);
    const user = userEvent.setup();
    render(<AgentBuilder />);

    await user.click(await screen.findByText("Downtown clinic"));

    expect(await screen.findByTestId("knowledge-panel")).toBeInTheDocument();
    // No documents attached yet -> the agent is ungrounded, and the UI says so.
    expect(screen.getByTestId("knowledge-count")).toHaveTextContent(/ungrounded/i);
  });

  it("reports the selected agent upward so the console can render it", async () => {
    stubApi([AGENT]);
    const onSelect = vi.fn();
    const user = userEvent.setup();
    render(<AgentBuilder onSelect={onSelect} />);

    await user.click(await screen.findByText("Downtown clinic"));
    await waitFor(() => expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ id: "a1" })));
  });

  it("fills the form from an example preset", async () => {
    stubApi([]);
    const user = userEvent.setup();
    render(<AgentBuilder />);
    await screen.findByText(/No agents yet/i);

    await user.click(screen.getByTestId("preset-clinic"));

    // Prefilled, so the first step is talking rather than writing a prompt.
    expect(screen.getByTestId("agent-name")).toHaveValue("Clinic reception");
    expect(screen.getByTestId("agent-greeting")).toHaveValue(
      "Thanks for calling the clinic. How can I help?",
    );
    expect((screen.getByTestId("agent-prompt") as HTMLTextAreaElement).value).toMatch(
      /clinic receptionist/i,
    );
  });

  it("warns when tools are selected on a model that cannot call them", async () => {
    // The failure this prevents is silent: the agent looks configured and
    // simply never calls anything.
    stubApi([]);
    const user = userEvent.setup();
    render(<AgentBuilder />);
    await screen.findByText(/No agents yet/i);

    await user.click(await screen.findByTestId("tool-check_availability"));

    // Default model cannot call tools -> warned.
    expect(await screen.findByTestId("tool-warning")).toHaveTextContent(/cannot call tools/i);

    // Switching to a tool-capable model clears it.
    await user.selectOptions(screen.getByTestId("agent-model"), "ollama:qwen3.5:0.8b");
    await waitFor(() =>
      expect(screen.queryByTestId("tool-warning")).not.toBeInTheDocument(),
    );
  });

  it("only offers tools this deployment can actually run", async () => {
    stubApi([]);
    render(<AgentBuilder />);
    await screen.findByText(/No agents yet/i);

    const names = (await screen.findAllByTestId(/^tool-/)).map((el) => el.textContent);
    expect(names).toEqual(["check_availability", "book_appointment"]);
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
