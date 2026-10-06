import { useCallback, useEffect, useRef, useState } from "react";

import {
  AgentApiError,
  type Agent,
  type AgentTurn,
  runAgentTurn,
  streamAgentTurn,
} from "../lib/agents";
import { blobToMonoWav, pickRecorderMimeType } from "../lib/audio";
import { ProgressivePlayer } from "../lib/player";
import { StatusPill } from "./StatusPill";

/**
 * How the reply is delivered.
 *
 * Both are kept because the difference is the demo: the batch path returns one
 * blob once the whole reply exists, while the streaming path starts speaking
 * after the first sentence. Same audio, same agent, very different feel.
 */
type Mode = "stream" | "batch";

interface Result {
  id: number;
  mode: Mode;
  transcript: string;
  /** Grows token by token in streaming mode. */
  reply: string;
  knowledgeChunks: number;
  model: string;
  /** Milliseconds from sending the clip to the first audible audio. */
  firstAudioMs: number | null;
  totalMs: number | null;
  error?: string;
}

function describe(error: unknown): string {
  if (error instanceof AgentApiError) return error.message;
  return error instanceof Error ? error.message : String(error);
}

/**
 * Speak to the selected agent: record a clip, send it to this agent's pipeline,
 * and play the answer back.
 *
 * In streaming mode the reply is played as it is generated — audio chunks are
 * scheduled onto the Web Audio clock the moment they arrive, so the agent
 * starts speaking while it is still writing the rest of the sentence. Time to
 * first audio is measured and shown, because "it feels faster" is not an
 * argument anyone should have to take on faith.
 */
export function TalkToAgent({ agent }: { agent: Agent }) {
  const [mode, setMode] = useState<Mode>("stream");
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState<Result[]>([]);
  const [notice, setNotice] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const micRef = useRef<MediaStream | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const playerRef = useRef<ProgressivePlayer | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const nextId = useRef(1);

  useEffect(() => {
    return () => {
      const recorder = recorderRef.current;
      if (recorder && recorder.state === "recording") recorder.stop();
      micRef.current?.getTracks().forEach((track) => track.stop());
      if (audioRef.current?.src) URL.revokeObjectURL(audioRef.current.src);
      abortRef.current?.abort();
      void playerRef.current?.close();
    };
  }, [agent.id]);

  const patch = useCallback((id: number, update: Partial<Result>) => {
    setResults((prev) => prev.map((row) => (row.id === id ? { ...row, ...update } : row)));
  }, []);

  /** Play a complete WAV blob (the batch path). */
  const playBlob = useCallback(
    async (base64: string) => {
      const bytes = atob(base64);
      const buffer = new Uint8Array(bytes.length);
      for (let i = 0; i < bytes.length; i += 1) buffer[i] = bytes.charCodeAt(i);
      const url = URL.createObjectURL(new Blob([buffer], { type: "audio/wav" }));
      if (audioRef.current?.src) URL.revokeObjectURL(audioRef.current.src);
      const audio = new Audio(url);
      audioRef.current = audio;
      await audio.play().catch(() => setNotice("Reply audio is ready but the browser blocked autoplay."));
    },
    [],
  );

  /** Reset (or create) the progressive player for a fresh turn. */
  const startPlayer = useCallback((): ProgressivePlayer => {
    const player = playerRef.current ?? new ProgressivePlayer();
    playerRef.current = player;
    player.stop();
    return player;
  }, []);

  const sendStreaming = useCallback(
    async (id: number, wav: Blob) => {
      const player = startPlayer();
      const controller = new AbortController();
      abortRef.current = controller;
      const t0 = performance.now();
      let firstAudio = -1;
      let transcript = "";
      let reply = "";

      await streamAgentTurn(
        agent.id,
        wav,
        {
          onTranscript: (text) => {
            transcript = text;
            patch(id, { transcript: text });
          },
          onDelta: (text) => {
            reply += text;
            patch(id, { reply });
          },
          onAudio: (pcm16, sampleRate) => {
            if (firstAudio < 0) {
              firstAudio = performance.now() - t0;
              patch(id, { firstAudioMs: Math.round(firstAudio) });
            }
            try {
              player.push(pcm16, sampleRate || 16000);
            } catch (error) {
              setNotice(`Playback failed: ${describe(error)}`);
            }
          },
          onDone: (result) => {
            patch(id, {
              reply: result.replyText || reply,
              model: result.model,
              knowledgeChunks: result.knowledgeChunks,
              firstAudioMs: firstAudio < 0 ? null : Math.round(firstAudio),
              totalMs: Math.round(performance.now() - t0),
            });
          },
          onCancelled: (result) => {
            patch(id, {
              reply: result.replyText || reply,
              model: result.model,
              knowledgeChunks: result.knowledgeChunks,
              firstAudioMs: firstAudio < 0 ? null : Math.round(firstAudio),
              totalMs: Math.round(performance.now() - t0),
              error: "turn cancelled",
            });
          },
          onError: (detail) => patch(id, { error: detail }),
        },
        controller.signal,
      );
      void transcript;
    },
    [agent.id, patch, startPlayer],
  );

  const sendBatch = useCallback(
    async (id: number, wav: Blob) => {
      const t0 = performance.now();
      const turn: AgentTurn = await runAgentTurn(agent.id, wav);
      const elapsed = Math.round(performance.now() - t0);
      patch(id, {
        transcript: turn.transcript,
        reply: turn.reply_text,
        model: turn.model,
        knowledgeChunks: turn.knowledge_chunks ?? 0,
        // Nothing is audible until the whole turn lands -- that is the point
        // of the comparison, so first audio == total here, by construction.
        firstAudioMs: elapsed,
        totalMs: elapsed,
      });
      if (turn.audio?.wav_base64) await playBlob(turn.audio.wav_base64);
    },
    [agent.id, patch, playBlob],
  );

  const send = useCallback(
    async (wav: Blob) => {
      setBusy(true);
      setNotice(null);
      const id = nextId.current++;
      const blank: Result = {
        id,
        mode,
        transcript: "…",
        reply: "",
        knowledgeChunks: 0,
        model: "",
        firstAudioMs: null,
        totalMs: null,
      };
      setResults((prev) => [blank, ...prev].slice(0, 6));
      try {
        if (mode === "stream") await sendStreaming(id, wav);
        else await sendBatch(id, wav);
      } catch (error) {
        patch(id, { error: describe(error) });
      } finally {
        setBusy(false);
      }
    },
    [mode, patch, sendBatch, sendStreaming],
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

  const stopSpeaking = useCallback(() => {
    playerRef.current?.stop();
    audioRef.current?.pause();
    abortRef.current?.abort();
  }, []);

  return (
    <section className="rounded-xl border border-cloudpso-line bg-cloudpso-panel p-4" data-testid="talk-to-agent">
      <header className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold">Talk to this agent</h3>
          <p className="text-[11px] text-cloudpso-muted">
            Record a question — it goes to this agent's pipeline, grounded by its knowledge.
          </p>
        </div>
        <StatusPill
          label={recording ? "Recording…" : busy ? "Thinking…" : "Idle"}
          tone={recording ? "err" : busy ? "warn" : "idle"}
          pulse={recording || busy}
        />
      </header>

      <div className="mb-3 flex flex-wrap items-center gap-2" role="radiogroup" aria-label="Reply mode">
        {(
          [
            { id: "stream", label: "Live", hint: "speaks as it generates" },
            { id: "batch", label: "Wait for full reply", hint: "one blob at the end" },
          ] as const
        ).map((option) => (
          <button
            key={option.id}
            type="button"
            role="radio"
            aria-checked={mode === option.id}
            title={option.hint}
            data-testid={`mode-${option.id}`}
            onClick={() => setMode(option.id)}
            className={`rounded-full border px-3 py-1 text-[11px] font-medium transition ${
              mode === option.id
                ? "border-cloudpso-accent/60 bg-cloudpso-accent/15 text-cloudpso-accent"
                : "border-cloudpso-line text-cloudpso-muted hover:text-slate-100"
            }`}
          >
            {option.label}
          </button>
        ))}
      </div>

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
        <button
          type="button"
          onClick={stopSpeaking}
          data-testid="stop-speaking"
          className="rounded-lg border border-cloudpso-line px-3 py-1.5 text-sm text-cloudpso-muted transition hover:text-slate-100"
        >
          Stop speaking
        </button>
      </div>

      {notice && <p className="mt-2 text-[11px] text-cloudpso-warn">{notice}</p>}

      <ul className="mt-3 flex flex-col gap-2">
        {results.map((result) => (
          <li key={result.id} className="rounded-lg border border-cloudpso-line bg-cloudpso-panel2 p-3">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <p className="text-[11px] text-cloudpso-muted">Heard: {result.transcript}</p>
              {!result.error && (
                <div className="flex flex-wrap items-center gap-2">
                  <StatusPill
                    label={
                      result.knowledgeChunks > 0
                        ? `grounded · ${result.knowledgeChunks} chunk${result.knowledgeChunks === 1 ? "" : "s"}`
                        : "not grounded"
                    }
                    tone={result.knowledgeChunks > 0 ? "ok" : "warn"}
                  />
                  <StatusPill
                    label={result.mode === "stream" ? "live" : "batch"}
                    tone={result.mode === "stream" ? "info" : "idle"}
                  />
                </div>
              )}
            </div>

            {result.error ? (
              <p className="mt-1 text-sm text-cloudpso-err">{result.error}</p>
            ) : (
              <>
                <p className="mt-1 text-sm text-slate-100">
                  {result.reply}
                  {result.totalMs === null && <span className="animate-pulse text-cloudpso-accent">▍</span>}
                </p>
                <p className="mt-1 font-mono text-[10px] text-cloudpso-muted">
                  {result.firstAudioMs !== null && <>first audio {result.firstAudioMs}ms · </>}
                  {result.totalMs !== null && <>turn {result.totalMs}ms</>}
                  {result.model && <> · {result.model}</>}
                </p>
              </>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
