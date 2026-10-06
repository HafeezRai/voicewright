import { useCallback, useEffect, useRef, useState } from "react";

import {
  AgentApiError,
  type Agent,
  type AgentTurn,
  runAgentTurn,
  streamAgentTurn,
} from "../lib/agents";
import { synthesize } from "../lib/aiEngine";
import type { ToolInvocation } from "../lib/agents";
import { MIC_CONSTRAINTS, PcmCapture, resampleToMonoWav } from "../lib/audio";
import { type ConversationTurn as Result, emptyTurn } from "../lib/conversation";
import { looksLikeEcho } from "../lib/echo";
import { ProgressivePlayer } from "../lib/player";
import { TurnDetector } from "../lib/vad";
import { StatusPill } from "./StatusPill";
import { Transcript } from "./Transcript";

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
  const [turns, setTurns] = useState<Result[]>([]);
  const [notice, setNotice] = useState<string | null>(null);
  const [level, setLevel] = useState<{ rms: number; phase: string }>({ rms: 0, phase: "idle" });
  //: What the browser actually granted, which is not always what was asked for.
  const [processing, setProcessing] = useState<{
    echoCancellation: boolean;
    noiseSuppression: boolean;
    autoGainControl: boolean;
  } | null>(null);

  const captureRef = useRef<PcmCapture | null>(null);
  const micRef = useRef<MediaStream | null>(null);
  const detectorRef = useRef<TurnDetector | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const playerRef = useRef<ProgressivePlayer | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const nextId = useRef(1);
  const inCallRef = useRef(false);
  //: The agent's most recent utterance -- what the microphone might pick back up.
  const agentSpeechRef = useRef<{ text: string; endedAt: number | null }>({
    text: "",
    endedAt: null,
  });

  const patch = useCallback((id: number, update: Partial<Result>) => {
    setTurns((prev) => prev.map((row) => (row.id === id ? { ...row, ...update } : row)));
  }, []);

  const pushResult = useCallback((row: Result) => {
    setTurns((prev) => [row, ...prev].slice(0, 6));
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
        pushResult(emptyTurn(id));
      }

      const player = ensurePlayer();
      player.stop();
      const controller = new AbortController();
      abortRef.current = controller;
      const t0 = performance.now();
      let firstAudio = -1;
      let reply = "";
      let rowCreated = announce;
      //: Set when the transcript turned out to be the agent's own voice, so the
      //: resulting abort is not reported as an interruption.
      let rejectedAsEcho = false;
      //: Tools this turn has run, in order. Grows as the stream reports them.
      let toolCalls: ToolInvocation[] = [];
      const ensureRow = () => {
        if (rowCreated) return;
        rowCreated = true;
        pushResult(emptyTurn(id));
      };

      try {
        await streamAgentTurn(
          agent.id,
          wav,
          {
            onTranscript: (text) => {
              ensureRow();
              // The agent hearing itself is not a user turn. Recognising it
              // here means the reply to it is never generated or spoken.
              if (looksLikeEcho(text, agentSpeechRef.current, performance.now())) {
                rejectedAsEcho = true;
                patch(id, {
                  transcript: text,
                  reply: "",
                  error: "ignored — that was my own voice coming back through the mic",
                  rejectedAsEcho: true,
                  totalMs: Math.round(performance.now() - t0),
                });
                controller.abort();
                return;
              }
              patch(id, { transcript: text });
            },
            onDelta: (text) => {
              ensureRow();
              reply += text;
              patch(id, { reply });
              // Track what is about to be spoken, so the next transcript can be
              // checked against it.
              agentSpeechRef.current = { text: reply, endedAt: null };
            },
            onToolCall: (call) => {
              ensureRow();
              toolCalls = [...toolCalls, call];
              patch(id, { tools: toolCalls });
            },
            onToolResult: (outcome) => {
              ensureRow();
              toolCalls = toolCalls.map((entry) =>
                entry.id === outcome.id
                  ? { ...entry, ok: outcome.ok, content: outcome.content }
                  : entry,
              );
              patch(id, { tools: toolCalls });
            },
            onAudio: (pcm16, sampleRate) => {
              ensureRow();
              if (firstAudio < 0) {
                firstAudio = performance.now() - t0;
                patch(id, { firstAudioMs: Math.round(firstAudio) });
                setCallState("speaking");
              }
              // Audio is going out, so the echo window is open again.
              agentSpeechRef.current = { text: agentSpeechRef.current.text, endedAt: null };
              try {
                player.push(pcm16, sampleRate || 16000);
              } catch (error) {
                setNotice(`Playback failed: ${describe(error)}`);
              }
            },
            onDone: (result) => {
              ensureRow();
              agentSpeechRef.current = { text: result.replyText || reply, endedAt: null };
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
              if (rejectedAsEcho) return;
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
      } finally {
        // No longer in flight: an interruption arriving after this point has
        // nothing to cancel.
        if (abortRef.current === controller) abortRef.current = null;
      }
    },
    [agent.id, ensurePlayer, patch, pushResult],
  );

  /**
   * Speak the agent's greeting before the caller says anything.
   *
   * This is what makes it feel like answering a phone rather than operating a
   * tool: the agent opens the conversation, so the caller only has to respond.
   * A greeting that fails to synthesise is not worth failing a call over.
   */
  const speakGreeting = useCallback(
    async (text: string) => {
      agentSpeechRef.current = { text, endedAt: null };
      setCallState("speaking");
      try {
        const wav = await synthesize(text);
        const url = URL.createObjectURL(wav);
        if (audioRef.current?.src) URL.revokeObjectURL(audioRef.current.src);
        const audio = new Audio(url);
        audioRef.current = audio;
        audio.onended = () => {
          // Closes the echo window: after this, matching words are the caller's.
          agentSpeechRef.current = { text, endedAt: performance.now() };
          if (inCallRef.current) setCallState("listening");
        };
        await audio.play().catch(() => undefined);
      } catch {
        agentSpeechRef.current = { text, endedAt: performance.now() };
        if (inCallRef.current) setCallState("listening");
      }
    },
    [],
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
      // Echo cancellation and noise suppression are requested here because
      // without them the agent hears itself and treats room noise as speech.
      const stream = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);

      // Ask what was actually granted. Browsers are free to ignore a constraint
      // and say nothing, so a machine that silently refused noise suppression
      // looks identical to one where it is on -- until the agent starts
      // answering its own voice.
      const applied = stream.getAudioTracks()[0]?.getSettings();
      setProcessing({
        echoCancellation: applied?.echoCancellation ?? false,
        noiseSuppression: applied?.noiseSuppression ?? false,
        autoGainControl: applied?.autoGainControl ?? false,
      });
      const player = ensurePlayer();
      player.onIdle = () => {
        if (agentSpeechRef.current.endedAt === null) {
          agentSpeechRef.current = { ...agentSpeechRef.current, endedAt: performance.now() };
        }
      };
      micRef.current = stream;
      inCallRef.current = true;

      // Raw PCM, not MediaRecorder: see PcmCapture for why.
      const capture = new PcmCapture();
      await capture.start(stream);
      captureRef.current = capture;

      const detector = new TurnDetector({
        onSpeechStart: () => {
          beginCapture();
        },
        onInterrupt: () => {
          // Sustained speech over the agent is how you interrupt it.
          //
          // Gated on a turn being *in flight*, not on audio already playing.
          // The agent spends a second or two thinking before its first word,
          // and during that window `isPlaying` is false -- so gating on
          // playback meant talking over the thinking was ignored and the agent
          // then started speaking a reply to a question the caller had already
          // moved on from.
          const controller = abortRef.current;
          if (!controller) return;
          playerRef.current?.stop();
          controller.abort();
          setCallState("listening");
        },
        onUtteranceEnd: () => endCapture(),
        onNoSpeech: () => setNotice("Didn't catch anything — still listening."),
        onLevel: (rms, phase) => setLevel({ rms, phase }),
      });
      detectorRef.current = detector;
      await detector.start(stream);

      setCallState("listening");
      if (agent.greeting.trim()) void speakGreeting(agent.greeting.trim());
    } catch (error) {
      setNotice(`Microphone unavailable: ${describe(error)}`);
      teardown();
    }
  }, [agent.greeting, beginCapture, endCapture, ensurePlayer, speakGreeting, teardown]);

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
      pushResult(emptyTurn(id));
      try {
        const turn: AgentTurn = await runAgentTurn(agent.id, wav);
        const elapsed = Math.round(performance.now() - t0);
        patch(id, {
          transcript: turn.transcript,
          reply: turn.reply_text,
          model: turn.model,
          knowledgeChunks: turn.knowledge_chunks ?? 0,
          tools: (turn.tool_calls ?? []).map((call) => ({
            ...call,
            ok: (turn.tool_results ?? []).find((r) => r.id === call.id)?.ok,
            content: (turn.tool_results ?? []).find((r) => r.id === call.id)?.content,
          })),
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
      const stream = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);
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

  // Last completed turn, for the footer readout.
  const lastDone = turns.find((turn) => turn.totalMs !== null);
  const stats = lastDone
    ? `last turn ${lastDone.totalMs}ms · first audio ${lastDone.firstAudioMs ?? "—"}ms${
        lastDone.model ? ` · ${lastDone.model}` : ""
      }`
    : null;

  return (
    <section
      className="flex h-full min-h-[560px] flex-col overflow-hidden rounded-2xl border border-cloudpso-line bg-cloudpso-panel"
      data-testid="talk-to-agent"
    >
      {/* ---- status bar ---- */}
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-cloudpso-line px-4 py-3">
        <div className="flex items-center gap-2">
          <StatusPill
            label={
              mode === "call"
                ? CALL_LABELS[callState]
                : recording
                  ? "Recording…"
                  : busy
                    ? "Thinking…"
                    : "Idle"
            }
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
          {processing ? (
            <span
              className="font-mono text-[10px]"
              data-testid="mic-processing"
              title="Echo cancellation / noise suppression / auto gain, as granted by the browser"
            >
              {[
                ["ec", processing.echoCancellation],
                ["ns", processing.noiseSuppression],
                ["agc", processing.autoGainControl],
              ].map(([name, granted], index) => (
                <span key={String(name)}>
                  {index > 0 ? " " : null}
                  <span className={granted ? "text-cloudpso-ok" : "text-cloudpso-err"}>
                    {granted ? "✓" : "✗"}
                    {name}
                  </span>
                </span>
              ))}
            </span>
          ) : null}
        </div>

        <div className="flex items-center gap-1" role="radiogroup" aria-label="Conversation mode">
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
              className={`rounded-full px-2.5 py-1 text-[11px] transition disabled:opacity-40 ${
                mode === option.id
                  ? "bg-cloudpso-accent/15 text-cloudpso-accent"
                  : "text-cloudpso-muted hover:text-slate-100"
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
      </header>

      {/* ---- the conversation ---- */}
      <div className="min-h-0 flex-1">
        <Transcript
          turns={turns}
          greeting={agent.greeting || undefined}
          live={inCall}
          emptyHint={
            mode === "call" ? (
              <div className="flex flex-col items-center gap-1.5">
                <p>Press start and just talk — the agent replies when you stop.</p>
              </div>
            ) : (
              <p>Record a clip, then wait for the reply.</p>
            )
          }
        />
      </div>

      {/* ---- talk control ---- */}
      <footer className="border-t border-cloudpso-line px-4 py-3">
        {notice ? <p className="mb-2 text-[11px] text-cloudpso-warn">{notice}</p> : null}

        <div className="flex items-center gap-3">
          {mode === "call" ? (
            <button
              type="button"
              data-testid="call-button"
              onClick={inCall ? teardown : () => void startCall()}
              className={`flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-medium transition ${
                inCall
                  ? "bg-cloudpso-err/20 text-cloudpso-err ring-1 ring-cloudpso-err/50"
                  : "bg-cloudpso-accent text-white hover:bg-cloudpso-accent/90"
              }`}
            >
              <span
                className={`h-2.5 w-2.5 rounded-full ${
                  inCall ? "bg-cloudpso-err" : "bg-white"
                } ${callState === "listening" ? "animate-pulse-ring" : ""}`}
              />
              {inCall ? "Stop listening" : "Start listening"}
            </button>
          ) : (
            <button
              type="button"
              data-testid="record-button"
              disabled={busy}
              onClick={recording ? stopRecording : () => void startRecording()}
              className={`flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-medium transition disabled:opacity-50 ${
                recording
                  ? "bg-cloudpso-err/20 text-cloudpso-err ring-1 ring-cloudpso-err/50"
                  : "bg-cloudpso-accent text-white hover:bg-cloudpso-accent/90"
              }`}
            >
              <span className="h-2.5 w-2.5 rounded-full bg-white" />
              {recording ? "Stop & send" : "Record question"}
            </button>
          )}

          {/* Live level, so a muted mic cannot be mistaken for a dead agent. */}
          {inCall ? (
            <div className="flex flex-1 items-center gap-2" data-testid="mic-meter">
              <span className="w-7 text-[10px] text-cloudpso-muted">mic</span>
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-cloudpso-panel2">
                <div
                  className={`h-full rounded-full transition-[width] duration-75 ${
                    level.phase === "speech" ? "bg-cloudpso-ok" : "bg-cloudpso-muted/50"
                  }`}
                  style={{ width: `${Math.min(100, Math.round(level.rms * 250))}%` }}
                />
              </div>
            </div>
          ) : (
            <p className="flex-1 text-[11px] text-cloudpso-muted">
              {mode === "call"
                ? "The agent answers when you stop speaking. Talk over it to interrupt."
                : "Record a clip and wait for the full reply."}
            </p>
          )}

          <button
            type="button"
            onClick={() => {
              playerRef.current?.stop();
              audioRef.current?.pause();
              abortRef.current?.abort();
            }}
            className="text-[11px] text-cloudpso-muted transition hover:text-slate-100"
          >
            Stop speaking
          </button>
        </div>

        {stats ? (
          <p className="mt-2 font-mono text-[10px] text-cloudpso-muted" data-testid="call-stats">
            {stats}
          </p>
        ) : null}
      </footer>
    </section>
  );
}
