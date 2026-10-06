/**
 * Client for ai-engine's agent-builder surface:
 *   GET/POST        /agents
 *   GET/PUT/DELETE  /agents/{id}
 *   POST            /agents/{id}/documents      (JSON text, or multipart file)
 *   GET             /agents/{id}/documents
 *   DELETE          /documents/{id}
 *   POST            /pipeline?agent_id=...      (WAV body -> grounded turn)
 *
 * An agent is pure configuration: a role, a system prompt, a voice and a
 * language, plus whatever documents ground its answers. Nothing here is
 * compiled in — that is the whole point of the demo.
 *
 * Tenancy: every endpoint accepts an optional `org_id` query param or
 * `X-Org-Id` header and falls back to the seeded default org. The UI does not
 * expose an org switcher yet, so we send nothing and take the default.
 */

import { config } from "../config";

export interface Agent {
  id: string;
  name: string;
  role: string;
  system_prompt: string;
  voice: string;
  language: string;
  greeting: string;
  created_at: string;
  updated_at: string;
}

/** The writable subset — what the form actually edits. */
export interface AgentInput {
  name: string;
  role?: string;
  system_prompt?: string;
  voice?: string;
  language?: string;
  greeting?: string;
}

export interface AgentDocument {
  id: string;
  agent_id: string;
  title: string;
  source: string;
  char_count: number;
  chunk_count: number;
  created_at: string;
}

export interface TurnTimings {
  stt: number;
  llm: number;
  tts: number;
  total: number;
}

/** One grounded STT -> LLM -> TTS turn, as `POST /pipeline` returns it. */
export interface AgentTurn {
  transcript: string;
  reply_text: string;
  model: string;
  provider: string;
  /** Echoed back so a stale response can be attributed to the right agent. */
  agent_id?: string;
  /**
   * How many knowledge-base chunks were retrieved and injected into the system
   * prompt for this turn. 0 means the answer was NOT grounded — which is the
   * single most useful number for showing whether a knowledge base is earning
   * its keep.
   */
  knowledge_chunks?: number;
  timings_ms: TurnTimings;
  audio?: { sample_rate: number; duration_s: number; cancelled: boolean; wav_base64?: string };
}

export class AgentApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "AgentApiError";
  }
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    // `no-store` because these endpoints serve mutable state and send no cache
    // directives of their own. Browsers are then free to reuse a stale GET, and
    // a deleted agent would linger in the list with no error to explain it.
    response = await fetch(`${config.aiEngineUrl}${path}`, { cache: "no-store", ...init });
  } catch (error) {
    throw new AgentApiError(
      `ai-engine unreachable at ${config.aiEngineUrl} (${error instanceof Error ? error.message : String(error)})`,
    );
  }
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (typeof payload?.detail === "string") detail = payload.detail;
      else if (payload?.detail) detail = JSON.stringify(payload.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new AgentApiError(`${path} failed: ${detail}`, response.status);
  }
  return response;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export async function listAgents(signal?: AbortSignal): Promise<Agent[]> {
  const response = await request("/agents", { signal });
  const payload = (await response.json()) as Agent[] | { agents: Agent[] };
  return Array.isArray(payload) ? payload : payload.agents;
}

export async function createAgent(input: AgentInput): Promise<Agent> {
  return (await (await request("/agents", json(input))).json()) as Agent;
}

export async function updateAgent(id: string, input: Partial<AgentInput>): Promise<Agent> {
  const response = await request(`/agents/${encodeURIComponent(id)}`, {
    ...json(input),
    method: "PUT",
  });
  return (await response.json()) as Agent;
}

export async function deleteAgent(id: string): Promise<void> {
  await request(`/agents/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export async function listDocuments(agentId: string, signal?: AbortSignal): Promise<AgentDocument[]> {
  const response = await request(`/agents/${encodeURIComponent(agentId)}/documents`, { signal });
  const payload = (await response.json()) as AgentDocument[] | { documents: AgentDocument[] };
  return Array.isArray(payload) ? payload : payload.documents;
}

/** Paste text as a document. */
export async function addTextDocument(
  agentId: string,
  title: string,
  text: string,
): Promise<AgentDocument> {
  const response = await request(`/agents/${encodeURIComponent(agentId)}/documents`, json({ title, text }));
  return (await response.json()) as AgentDocument;
}

/** Upload a .txt/.md file. Multipart, so no manual Content-Type. */
export async function addFileDocument(agentId: string, file: File): Promise<AgentDocument> {
  const form = new FormData();
  form.append("file", file);
  form.append("title", file.name);
  const response = await request(`/agents/${encodeURIComponent(agentId)}/documents`, {
    method: "POST",
    body: form,
  });
  return (await response.json()) as AgentDocument;
}

export async function deleteDocument(documentId: string): Promise<void> {
  await request(`/documents/${encodeURIComponent(documentId)}`, { method: "DELETE" });
}

/**
 * Run one full grounded turn for an agent: the recorded WAV in, the spoken
 * reply (plus transcript and timings) back.
 *
 * This deliberately does NOT go through backend-core's WebRTC gateway: that
 * service has no notion of agents and threading `agent_id` through the
 * signalling path is out of scope. The record -> /pipeline -> play loop is the
 * route that exercises the configured agent end to end.
 */
export async function runAgentTurn(agentId: string, wav: Blob): Promise<AgentTurn> {
  const path = `/pipeline?agent_id=${encodeURIComponent(agentId)}&include_audio=true`;
  const response = await request(path, {
    method: "POST",
    headers: { "Content-Type": "audio/wav" },
    body: wav,
  });
  return (await response.json()) as AgentTurn;
}
