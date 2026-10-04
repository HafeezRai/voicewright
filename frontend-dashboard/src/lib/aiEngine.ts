/**
 * Thin client for ai-engine's REST surface (Phase 3):
 *   GET  /health
 *   POST /stt       (WAV body -> transcript)
 *   POST /nlu       ({"text"} -> intent)
 *   POST /tts       ({"text"} -> WAV)
 *
 * These drive the real STT/NLU/TTS backends from the browser, which is how the
 * demo shows a transcript + intent without the WebRTC media path carrying them
 * yet (Phase 2's gateway is receive-only, so nothing is relayed over the peer
 * connection).
 */

import { config } from "../config";

export interface IntentResult {
  intent: string;
  confidence: number;
  backend: string;
  slots: Record<string, string>;
  scores: Record<string, number>;
}

export interface TranscriptResult {
  text: string;
  language: string;
  duration_s: number;
  backend: string;
}

export class AiEngineError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "AiEngineError";
  }
}

async function post(path: string, body: BodyInit, headers: Record<string, string>): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(`${config.aiEngineUrl}${path}`, { method: "POST", body, headers });
  } catch (error) {
    throw new AiEngineError(
      `ai-engine unreachable at ${config.aiEngineUrl} (${error instanceof Error ? error.message : String(error)})`,
    );
  }
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload?.detail) detail = payload.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new AiEngineError(`${path} failed: ${detail}`, response.status);
  }
  return response;
}

export async function aiHealth(signal?: AbortSignal): Promise<boolean> {
  try {
    const response = await fetch(`${config.aiEngineUrl}/health`, { signal });
    return response.ok;
  } catch {
    return false;
  }
}

export async function classify(text: string): Promise<IntentResult> {
  const response = await post("/nlu", JSON.stringify({ text }), { "Content-Type": "application/json" });
  return (await response.json()) as IntentResult;
}

export async function synthesize(text: string, speed?: number): Promise<Blob> {
  const body: Record<string, unknown> = { text };
  if (speed !== undefined) body.speed = speed;
  const response = await post("/tts", JSON.stringify(body), { "Content-Type": "application/json" });
  return response.blob();
}

export async function transcribe(wav: Blob): Promise<TranscriptResult> {
  const response = await post("/stt", wav, { "Content-Type": "audio/wav" });
  return (await response.json()) as TranscriptResult;
}
