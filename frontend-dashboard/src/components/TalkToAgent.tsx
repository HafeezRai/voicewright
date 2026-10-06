import { useCallback, useEffect, useRef, useState } from "react";

import {
  AgentApiError,
  type Agent,
  type AgentTurn,
  runAgentTurn,
  streamAgentTurn,
} from "../lib/agents";
import { PcmCapture, resampleToMonoWav } from "../lib/audio";
import { ProgressivePlayer } from "../lib/player";
import { TurnDetector } from "../lib/vad";
import { StatusPill } from "./StatusPill";

/**
 * How the caller talks to the agent.
 *
 * `call` is hands-free: the microphone stays open, silence detection decides
 * where the turn ends, and the agent answers on its own — talking over it cuts
 * it off. `batch` is the older push-to-talk shape, kept because a fixed-length
 * clip is a useful control when something sounds wrong.
 */
type Mode = "call" | "batch";

/** Where a live call currently is. */
type CallState = "off" | "listening" | "thinking" | "speaking";

interface Result {
  id: number;
  transcript: string;
  /** Grows token by token while streaming. */
  reply: string;
  knowledgeChunks: number;
  model: string;
  firstAudioMs: number | null;
  totalMs: number | null;
  error?: string;
}

function describe(error: unknown): string {
  if (error instanceof AgentApiError) return error.message;
  return error instanceof Error ? error.message : String(error);
}

const CALL_LABELS: Record<CallState, string> = {
  off: "Ready",
  listening: "Listening…",
  thinking: "Thinking…",
  speaking: "Speaking — talk over me to interrupt",
};

/**
 * Talk to the selected agent.
 *
 * In call mode the loop runs itself: listen → detect the end of the caller's
 * turn → stream the reply → play it → listen again. The caller never presses
 * a button mid-conversation, which is the difference between a call and a
 * walkie-talkie.
 */
export function TalkToAgent({ agent }: { agent: Agent }) {
  const [mode, setMode] = useState<Mode>("call");
  const [callState, setCallState] = useState<CallState>("off");
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState<Result[]>([]);
  const [notice, setNotice] = useState<string | null>(null);
  const [level, setLevel] = useState<{ rms: number; phase: string }>({ rms: 0, phase: "idle" });

  const captureRef = useRef<PcmCapture | null>(null);
  const micRef = useRef<MediaStream | null>(null);
  const detectorRef = useRef<TurnDetector | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const playerRef = useRef<ProgressivePlayer | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const nextId = useRef(1);
  const inCallRef = useRef(false);

  const patch = useCallback((id: number, update: Partial<Result>) => {
    setResults((prev) => prev.map((row) => (row.id === id ? { ...row, ...update } : row)));
  }, []);

  const pushResult = useCallback((row: Result) => {
    setResults((prev) => [row, ...prev].slice(0, 6));
  }, []);

  /** Stop every source of sound and input. Safe to call repeatedly. */
  const teardown = useCallback(() => {
    inCallRef.current = false;
    detectorRef.current?.stop();
    detectorRef.current = null;
    captureRef.current?.stop();
    captureRef.current = null;
    micRef.current?.getTracks().forEach((track) => track.stop());
    micRef.current = null;
    playerRef.current?.stop();
    audioRef.current?.pause();
    abortRef.current?.abort();
    abortRef.current = null;
    setRecording(false);
    setCallState("off");
  }, []);

  useEffect(() => () => {
    teardown();
    void playerRef.current?.close();
    if (audioRef.current?.src) URL.revokeObjectURL(audioRef.current.src);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agent.id]);

  const ensurePlayer = useCallback((): ProgressivePlayer => {
    const player = playerRef.current ?? new ProgressivePlayer();
    playerRef.current = player;
    return player;
  }, []);

  /**
   * Stream one turn and play it as it arrives.
   *
   * Returns when the turn is over. The caller stays live throughout: turn
   * detection is re-armed by the detector itself, so a caller who talks over
   * the reply cuts it off rather than waiting their turn.
   */
  const streamTurn = useCallback(
    async (wav: Blob, announce: boolean): Promise<void> => {
      const id = nextId.current++;
      if (announce) {
        pushResult({
          id,
          transcript: "…",
          reply: "",
          knowledgeChunks: 0,
          model: "",
          firstAudioMs: null,
          totalMs: null,
        });
      }

      const player = ensurePlayer();
      player.stop();
      const controller = new AbortController();
      abortRef.current = controller;
      const t0 = performance.now();
      let firstAudio = -1;
      let reply = "";
      let rowCreated = announce;
      const ensureRow = () => {
        if (rowCreated) return;
        rowCreated = true;
        pushResult({
          id,
          transcript: "",
          reply: "",
          knowledgeChunks: 0,
          model: "",
          firstAudioMs: null,
          totalMs: null,
        });
      };

      try {
        await streamAgentTurn(
          agent.id,
          wav,
          {
            onTranscript: (text) => {
              ensureRow();
              patch(id, { transcript: text });
            },
            onDelta: (text) => {
              ensureRow();
              reply += text;
              patch(id, { reply });
            },
            onAudio: (pcm16, sampleRate) => {
              ensureRow();
              if (firstAudio < 0) {
                firstAudio = performance.now() - t0;
                patch(id, { firstAudioMs: Math.round(firstAudio) });
                setCallState("speaking");
              }
              try {
                player.push(pcm16, sampleRate || 16000);
              } catch (error) {
                setNotice(`Playback failed: ${describe(error)}`);
              }
            },
            onDone: (result) => {
              ensureRow();
              patch(id, {
                reply: result.replyText || reply,
                model: result.model,
                knowledgeChunks: result.knowledgeChunks,
                firstAudioMs: firstAudio < 0 ? null : Math.round(firstAudio),
                totalMs: Math.round(performance.now() - t0),
              });
            },
            onCancelled: (result) => {
              ensureRow();
              patch(id, {
                reply: result.replyText || reply,
                model: result.model,
                knowledgeChunks: result.knowledgeChunks,
                firstAudioMs: firstAudio < 0 ? null : Math.round(firstAudio),
                totalMs: Math.round(performance.now() - t0),
                error: "interrupted",
              });
            },
            onError: (detail) => {
              ensureRow();
              patch(id, { error: detail });
            },
          },
          controller.signal,
        );
      } catch (error) {
        ensureRow();
        patch(id, { error: describe(error) });
      }
    },
    [agent.id, ensurePlayer, patch, pushResult],
  );

  /** Mark the start of an utterance; anything captured before it is dropped. */
  const beginCapture = useCallback(() => {
    captureRef.current?.begin();
  }, []);

  /** End the utterance and send what was captured. */
  const endCapture = useCallback(() => {
    const capture = captureRef.current;
    if (!capture) return;
    const recorded = capture.finish();
    if (!recorded || recorded.samples.length === 0) {
      if (inCallRef.current) setCallState("listening");
      return;
    }
    void (async () => {
      try {
        const wav = await resampleToMonoWav(recorded.samples, recorded.sampleRate);
        setCallState("thinking");
        await streamTurn(wav, true);
      } catch (error) {
        setNotice(`Could not prepare the recording: ${describe(error)}`);
      } finally {
        if (inCallRef.current) setCallState("listening");
      }
    })();
  }, [streamTurn]);

  const startCall = useCallback(async () => {
    setNotice(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          // The agent's own voice comes out of the speakers and straight back
          // into the mic. Browser AEC is what stops the agent interrupting
          // itself; it is not perfect, so a headset still sounds better.
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      micRef.current = stream;
      inCallRef.current = true;

      // Raw PCM, not MediaRecorder: see PcmCapture for why.
      const capture = new PcmCapture();
      await capture.start(stream);
      captureRef.current = capture;

      const detector = new TurnDetector({
        onSpeechStart: () => {
          // Talking over the agent is how you interrupt it. Gated on audio
          // actually playing, not merely having played -- otherwise the next
          // utterance would abort a turn that already finished.
          if (playerRef.current?.isPlaying) {
            playerRef.current.stop();
            abortRef.current?.abort();
          }
          setCallState("listening");
          beginCapture();
        },
        onUtteranceEnd: () => endCapture(),
        onNoSpeech: () => setNotice("Didn't catch anything — still listening."),
        onLevel: (rms, phase) => setLevel({ rms, phase }),
      });
      detectorRef.current = detector;
      await detector.start(stream);

      setCallState("listening");
    } catch (error) {
      setNotice(`Microphone unavailable: ${describe(error)}`);
      teardown();
    }
  }, [beginCapture, endCapture, teardown]);

  // --- push-to-talk (batch mode) -------------------------------------------

  const playBlob = useCallback(async (base64: string) => {
    const bytes = atob(base64);
    const buffer = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i += 1) buffer[i] = bytes.charCodeAt(i);
    const url = URL.createObjectURL(new Blob([buffer], { type: "audio/wav" }));
    if (audioRef.current?.src) URL.revokeObjectURL(audioRef.current.src);
    const audio = new Audio(url);
    audioRef.current = audio;
    await audio.play().catch(() => setNotice("Reply audio is ready but the browser blocked autoplay."));
  }, []);

  const sendBatch = useCallback(
    async (wav: Blob) => {
      const id = nextId.current++;
      const t0 = performance.now();
      pushResult({
        id,
        transcript: "…",
        reply: "",
        knowledgeChunks: 0,
        model: "",
        firstAudioMs: null,
        totalMs: null,
      });
      try {
        const turn: AgentTurn = await runAgentTurn(agent.id, wav);
        const elapsed = Math.round(performance.now() - t0);
        patch(id, {
          transcript: turn.transcript,
          reply: turn.reply_text,
          model: turn.model,
          knowledgeChunks: turn.knowledge_chunks ?? 0,
          // Nothing is audible until the whole turn lands. That is precisely
          // the comparison, so first audio == total here by construction.
          firstAudioMs: elapsed,
          totalMs: elapsed,
        });
        if (turn.audio?.wav_base64) await playBlob(turn.audio.wav_base64);
      } catch (error) {
        patch(id, { error: describe(error) });
      }
    },
    [agent.id, patch, playBlob, pushResult],
  );

  const startRecording = useCallback(async () => {
    setNotice(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micRef.current = stream;
      // Same raw-PCM capture as call mode, for the same reason: MediaRecorder
      // hands back a container that Safari cannot decode back into samples.
      const capture = new PcmCapture();
      await capture.start(stream);
      captureRef.current = capture;
      capture.begin();
      setRecording(true);
    } catch (error) {
      setNotice(`Microphone unavailable: ${describe(error)}`);
    }
  }, []);

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
      try {
        await sendBatch(await resampleToMonoWav(recorded.samples, recorded.sampleRate));
      } catch (error) {
        setNotice(`Could not prepare the recording: ${describe(error)}`);
      } finally {
        setBusy(false);
      }
    })();
  }, [sendBatch]);

  const inCall = callState !== "off";

  return (
    <section className="rounded-xl border border-cloudpso-line bg-cloudpso-panel p-4" data-testid="talk-to-agent">
      <header className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold">Talk to this agent</h3>
          <p className="text-[11px] text-cloudpso-muted">
            Grounded by this agent's knowledge. In call mode it answers when you stop talking.
          </p>
        </div>
        <StatusPill
          label={mode === "call" ? CALL_LABELS[callState] : recording ? "Recording…" : busy ? "Thinking…" : "Idle"}
          tone={
            callState === "speaking" || recording
              ? "err"
              : callState === "thinking" || busy
                ? "warn"
                : callState === "listening"
                  ? "ok"
                  : "idle"
          }
          pulse={callState === "listening" || callState === "thinking" || recording || busy}
          testId="call-status"
        />
      </header>

      <div className="mb-3 flex flex-wrap items-center gap-2" role="radiogroup" aria-label="Conversation mode">
        {(
          [
            { id: "call", label: "Live call", hint: "hands-free; answers when you stop talking" },
            { id: "batch", label: "Push to talk", hint: "record a clip, wait for the full reply" },
          ] as const
        ).map((option) => (
          <button
            key={option.id}
            type="button"
            role="radio"
            aria-checked={mode === option.id}
            title={option.hint}
            data-testid={`mode-${option.id}`}
            disabled={inCall}
            onClick={() => setMode(option.id)}
            className={`rounded-full border px-3 py-1 text-[11px] font-medium transition disabled:opacity-40 ${
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
        {mode === "call" ? (
          <button
            type="button"
            data-testid="call-button"
            onClick={inCall ? teardown : () => void startCall()}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition ${
              inCall
                ? "bg-cloudpso-err/20 text-cloudpso-err ring-1 ring-cloudpso-err/50"
                : "bg-cloudpso-accent text-white hover:bg-cloudpso-accent/90"
            }`}
          >
            {inCall ? "End call" : "Start call"}
          </button>
        ) : (
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
        )}
        <button
          type="button"
          onClick={() => {
            playerRef.current?.stop();
            audioRef.current?.pause();
            abortRef.current?.abort();
          }}
          className="rounded-lg border border-cloudpso-line px-3 py-1.5 text-sm text-cloudpso-muted transition hover:text-slate-100"
        >
          Stop speaking
        </button>
      </div>

      {inCall && (
        <div
          className="mt-3 flex items-center gap-2"
          data-testid="mic-meter"
          data-phase={level.phase}
          data-track={micRef.current?.getAudioTracks()[0]?.readyState ?? "none"}
        >
          <span className="w-14 text-[10px] text-cloudpso-muted">mic</span>
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-cloudpso-panel2">
            <div
              className={`h-full rounded-full transition-[width] duration-75 ${
                level.phase === "speech" ? "bg-cloudpso-ok" : "bg-cloudpso-muted"
              }`}
              // RMS is small even for loud speech (~0.4); scale it so the bar
              // reads as a level rather than sitting near zero.
              style={{ width: `${Math.min(100, Math.round(level.rms * 250))}%` }}
            />
          </div>
          <span className="w-20 font-mono text-[10px] text-cloudpso-muted" data-testid="mic-level">
            {level.phase} {level.rms.toFixed(3)}
          </span>
        </div>
      )}

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
                  <StatusPill label={mode === "call" ? "live" : "batch"} tone={mode === "call" ? "info" : "idle"} />
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
