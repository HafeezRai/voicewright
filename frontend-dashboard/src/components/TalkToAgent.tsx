import { useCallback, useEffect, useRef, useState } from "react";

import { AgentApiError, type Agent, type AgentTurn, runAgentTurn } from "../lib/agents";
import { blobToMonoWav, pickRecorderMimeType } from "../lib/audio";
import { StatusPill } from "./StatusPill";

interface Result {
  id: number;
  transcript: string;
  reply: string;
  knowledgeChunks: number;
  model: string;
  timings: AgentTurn["timings_ms"];
  error?: string;
}

function describe(error: unknown): string {
  if (error instanceof AgentApiError) return error.message;
  return error instanceof Error ? error.message : String(error);
}

/**
 * Speak to the selected agent: record a clip, send it to
 * `POST /pipeline?agent_id=...`, then play the grounded reply back.
 *
 * This runs the browser -> ai-engine loop directly rather than through
 * backend-core's WebRTC gateway, which has no notion of agents. Swapping the
 * selected agent and asking the same question again is the demo's punchline:
 * same voice, same model, different answer.
 */
export function TalkToAgent({ agent }: { agent: Agent }) {
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState<Result[]>([]);
  const [notice, setNotice] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const micRef = useRef<MediaStream | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const nextId = useRef(1);

  // Stop the mic and free any object URL when the agent changes or we unmount.
  useEffect(() => {
    return () => {
      const recorder = recorderRef.current;
      if (recorder && recorder.state === "recording") recorder.stop();
      micRef.current?.getTracks().forEach((track) => track.stop());
      if (audioRef.current?.src) URL.revokeObjectURL(audioRef.current.src);
    };
  }, [agent.id]);

  const play = useCallback((base64: string) => {
    const bytes = atob(base64);
    const buffer = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i += 1) buffer[i] = bytes.charCodeAt(i);
    const url = URL.createObjectURL(new Blob([buffer], { type: "audio/wav" }));
    if (audioRef.current?.src) URL.revokeObjectURL(audioRef.current.src);
    const audio = new Audio(url);
    audioRef.current = audio;
    void audio.play().catch(() => setNotice("Reply audio is ready but the browser blocked autoplay."));
  }, []);

  const send = useCallback(
    async (wav: Blob) => {
      setBusy(true);
      setNotice(null);
      const id = nextId.current++;
      try {
        const turn = await runAgentTurn(agent.id, wav);
        setResults((prev) =>
          [
            {
              id,
              transcript: turn.transcript,
              reply: turn.reply_text,
              knowledgeChunks: turn.knowledge_chunks ?? 0,
              model: turn.model,
              timings: turn.timings_ms,
            },
            ...prev,
          ].slice(0, 6),
        );
        if (turn.audio?.wav_base64) play(turn.audio.wav_base64);
      } catch (error) {
        setResults((prev) =>
          [
            {
              id,
              transcript: "—",
              reply: "",
              knowledgeChunks: 0,
              model: "",
              timings: { stt: 0, llm: 0, tts: 0, total: 0 },
              error: describe(error),
            },
            ...prev,
          ].slice(0, 6),
        );
      } finally {
        setBusy(false);
      }
    },
    [agent.id, play],
  );

  const startRecording = useCallback(async () => {
    setNotice(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micRef.current = stream;
      chunksRef.current = [];
      const mimeType = pickRecorderMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        micRef.current = null;
        if (chunksRef.current.length === 0) return;
        const raw = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
        try {
          await send(await blobToMonoWav(raw));
        } catch (error) {
          setNotice(`Could not prepare the recording: ${describe(error)}`);
        }
      };

      recorder.start();
      setRecording(true);
    } catch (error) {
      setNotice(`Microphone unavailable: ${describe(error)}`);
    }
  }, [send]);

  const stopRecording = useCallback(() => {
    const recorder = recorderRef.current;
    if (recorder && recorder.state === "recording") recorder.stop();
    setRecording(false);
  }, []);

  return (
    <section className="rounded-xl border border-cloudpso-line bg-cloudpso-panel p-4" data-testid="talk-to-agent">
      <header className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold">Talk to this agent</h3>
          <p className="text-[11px] text-cloudpso-muted">
            Record a question — it goes to ai-engine's <code>/pipeline</code> with{" "}
            <code>agent_id</code>, grounded by this agent's knowledge.
          </p>
        </div>
        <StatusPill
          label={recording ? "Recording…" : busy ? "Thinking…" : "Idle"}
          tone={recording ? "err" : busy ? "warn" : "idle"}
          pulse={recording || busy}
        />
      </header>

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          data-testid="record-button"
          disabled={busy}
          onClick={recording ? stopRecording : () => void startRecording()}
          className={`rounded-lg px-3 py-1.5 text-sm font-medium transition disabled:opacity-50 ${
            recording
              ? "bg-cloudpso-err/20 text-cloudpso-err ring-1 ring-cloudpso-err/50"
              : "bg-cloudpso-accent text-white hover:bg-cloudpso-accent/90"
          }`}
        >
          {recording ? "Stop & send" : "Record question"}
        </button>
      </div>

      {notice && <p className="mt-2 text-[11px] text-cloudpso-warn">{notice}</p>}

      <ul className="mt-3 flex flex-col gap-2">
        {results.map((result) => (
          <li key={result.id} className="rounded-lg border border-cloudpso-line bg-cloudpso-panel2 p-3">
            <div className="flex items-start justify-between gap-3">
              <p className="text-[11px] text-cloudpso-muted">Heard: {result.transcript}</p>
              {!result.error && (
                <StatusPill
                  label={
                    result.knowledgeChunks > 0
                      ? `grounded · ${result.knowledgeChunks} chunk${result.knowledgeChunks === 1 ? "" : "s"}`
                      : "not grounded"
                  }
                  tone={result.knowledgeChunks > 0 ? "ok" : "warn"}
                />
              )}
            </div>
            {result.error ? (
              <p className="mt-1 text-sm text-cloudpso-err">{result.error}</p>
            ) : (
              <>
                <p className="mt-1 text-sm text-slate-100">{result.reply}</p>
                <p className="mt-1 font-mono text-[10px] text-cloudpso-muted">
                  {result.model} · stt {Math.round(result.timings.stt)}ms · llm{" "}
                  {Math.round(result.timings.llm)}ms · tts {Math.round(result.timings.tts)}ms
                </p>
              </>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
