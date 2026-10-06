import { useCallback, useEffect, useRef, useState } from "react";

import { AiEngineError, type AgentReply, ask, synthesize, transcribe } from "../lib/aiEngine";
import { MIC_CONSTRAINTS, PcmCapture, resampleToMonoWav } from "../lib/audio";
import { StatusPill } from "./StatusPill";

interface Turn {
  id: number;
  /** Where the utterance text came from. */
  source: "typed" | "stt";
  text: string;
  /** The agent's generated reply, once it has arrived. */
  reply?: string;
  error?: string;
  pending?: boolean;
}

/**
 * The transcript + reply pane. It drives ai-engine's real `/stt`, `/llm` and
 * `/tts` endpoints from the browser: capture a clip as raw PCM (see
 * `PcmCapture`), transcribe it, ask the agent for a reply, and synthesise
 * that reply to the speakers.
 *
 * `onAgentSpeaking` is called while TTS audio plays so the voice panel can tell
 * backend-core the agent is "speaking" — which is what makes barge-in
 * observable end-to-end.
 */
export function AgentTurnPanel({ onAgentSpeaking }: { onAgentSpeaking?: (speaking: boolean) => void }) {
  const [text, setText] = useState("I'd like to book an appointment for tomorrow");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [recording, setRecording] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const captureRef = useRef<PcmCapture | null>(null);
  const micRef = useRef<MediaStream | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const nextId = useRef(1);

  useEffect(() => {
    return () => {
      captureRef.current?.stop();
      captureRef.current = null;
      micRef.current?.getTracks().forEach((track) => track.stop());
      if (audioRef.current?.src) URL.revokeObjectURL(audioRef.current.src);
    };
  }, []);

  const addTurn = useCallback((turn: Omit<Turn, "id">): number => {
    const id = nextId.current++;
    setTurns((prev) => [{ id, ...turn }, ...prev].slice(0, 12));
    return id;
  }, []);

  const updateTurn = useCallback((id: number, patch: Partial<Turn>) => {
    setTurns((prev) => prev.map((turn) => (turn.id === id ? { ...turn, ...patch } : turn)));
  }, []);

  const runAsk = useCallback(
    async (value: string, turnId: number) => {
      try {
        const reply: AgentReply = await ask(value);
        updateTurn(turnId, { reply: reply.reply_text, pending: false });
      } catch (error) {
        updateTurn(turnId, { pending: false, error: describe(error) });
      }
    },
    [updateTurn],
  );

  const onAsk = useCallback(async () => {
    const value = text.trim();
    if (!value) return;
    setNotice(null);
    const id = addTurn({ source: "typed", text: value, pending: true });
    await runAsk(value, id);
  }, [addTurn, runAsk, text]);

  const onSpeak = useCallback(async () => {
    const value = text.trim();
    if (!value) return;
    setBusy(true);
    setNotice(null);
    try {
      const wav = await synthesize(value);
      const url = URL.createObjectURL(wav);
      const element = audioRef.current;
      if (element) {
        if (element.src) URL.revokeObjectURL(element.src);
        element.src = url;
        onAgentSpeaking?.(true);
        element.onended = () => onAgentSpeaking?.(false);
        await element.play().catch(() => undefined);
      }
    } catch (error) {
      setNotice(describe(error));
    } finally {
      setBusy(false);
    }
  }, [onAgentSpeaking, text]);

  const stopRecording = useCallback(() => {
    setRecording(false);
    const capture = captureRef.current;
    captureRef.current = null;
    micRef.current?.getTracks().forEach((track) => track.stop());
    micRef.current = null;
    const recorded = capture?.finish();
    capture?.stop();
    if (!recorded || recorded.samples.length === 0) return;

    void (async () => {
      setBusy(true);
      const id = addTurn({ source: "stt", text: "transcribing…", pending: true });
      try {
        const wav = await resampleToMonoWav(recorded.samples, recorded.sampleRate, 16000);
        const result = await transcribe(wav);
        const heard = result.text.trim() || "(no speech detected)";
        updateTurn(id, { text: heard, pending: true });
        await runAsk(heard, id);
      } catch (error) {
        updateTurn(id, { pending: false, error: describe(error), text: "(transcription failed)" });
      } finally {
        setBusy(false);
      }
    })();
  }, [addTurn, runAsk, updateTurn]);

  const startRecording = useCallback(async () => {
    setNotice(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);
      micRef.current = stream;
      // Raw PCM, not MediaRecorder: see PcmCapture. MediaRecorder's output is a
      // compressed container that has to be decoded back to samples, and Safari
      // refuses to decode the fragmented MP4 it produced itself.
      const capture = new PcmCapture();
      await capture.start(stream);
      captureRef.current = capture;
      capture.begin();
      setRecording(true);
    } catch (error) {
      setNotice(describe(error));
    }
  }, []);

  return (
    <section className="rounded-2xl border border-cloudpso-line bg-cloudpso-panel p-5">
      <header className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-100">Agent turn — STT / LLM / TTS</h2>
        <StatusPill label="ai-engine /stt /llm /tts" tone="info" />
      </header>

      <div className="mt-4 flex flex-col gap-3 sm:flex-row">
        <input
          data-testid="turn-text"
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Type an utterance…"
          className="flex-1 rounded-lg border border-cloudpso-line bg-cloudpso-panel2 px-3 py-2 text-sm text-slate-100 outline-none focus:border-cloudpso-accent"
        />
        <div className="flex gap-2">
          <button
            type="button"
            data-testid="ask"
            onClick={() => void onAsk()}
            disabled={busy}
            className="rounded-lg bg-cloudpso-accent px-4 py-2 text-sm font-medium text-white transition hover:brightness-110 disabled:opacity-40"
          >
            Ask the agent
          </button>
          <button
            type="button"
            data-testid="speak"
            onClick={() => void onSpeak()}
            disabled={busy}
            className="rounded-lg border border-cloudpso-line bg-cloudpso-panel2 px-4 py-2 text-sm font-medium text-slate-200 transition hover:border-cloudpso-accent/60 disabled:opacity-40"
          >
            Speak reply
          </button>
          <button
            type="button"
            data-testid="record"
            onClick={() => void (recording ? stopRecording() : startRecording())}
            className={`rounded-lg border px-4 py-2 text-sm font-medium transition ${
              recording
                ? "border-cloudpso-err/60 bg-cloudpso-err/10 text-cloudpso-err"
                : "border-cloudpso-line bg-cloudpso-panel2 text-slate-200 hover:border-cloudpso-accent/60"
            }`}
          >
            {recording ? "Stop & transcribe" : "Record & transcribe"}
          </button>
        </div>
      </div>

      {notice ? (
        <p className="mt-3 rounded-lg border border-cloudpso-warn/50 bg-cloudpso-warn/10 px-3 py-2 text-xs text-cloudpso-warn" data-testid="turn-notice">
          {notice}
        </p>
      ) : null}

      <ul className="mt-4 space-y-2" data-testid="conversation">
        {turns.length === 0 ? (
          <li className="text-xs text-cloudpso-muted">
            Ask the agent about a typed utterance, or record a clip to exercise real STT.
          </li>
        ) : (
          turns.map((turn) => (
            <li key={turn.id} className="rounded-xl border border-cloudpso-line bg-cloudpso-panel2 px-3 py-2">
              <div className="flex items-center gap-2 text-[11px] uppercase tracking-wide text-cloudpso-muted">
                <span>{turn.source === "stt" ? "STT heard" : "Input"}</span>
                {turn.pending ? <span className="text-cloudpso-warn">processing…</span> : null}
              </div>
              <p className="mt-1 text-sm text-slate-100">{turn.text}</p>
              {turn.reply ? (
                <p className="mt-2 rounded-md bg-cloudpso-accent/15 px-2 py-1 text-sm text-cloudpso-accent" data-testid="turn-reply">
                  {turn.reply}
                </p>
              ) : null}
              {turn.error ? <p className="mt-2 text-xs text-cloudpso-err">{turn.error}</p> : null}
            </li>
          ))
        )}
      </ul>

      <audio ref={audioRef} className="hidden" data-testid="tts-audio" />
    </section>
  );
}

function describe(error: unknown): string {
  if (error instanceof AiEngineError) return error.message;
  if (error instanceof Error) return error.message;
  return String(error);
}
