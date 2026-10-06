/**
 * End-of-speech detection for hands-free turns.
 *
 * Push-to-talk is not a call. A live call has to notice that the caller stopped
 * talking and answer on its own, which means deciding where the turn ends from
 * the audio itself.
 *
 * The decision logic lives in `TurnStateMachine`, which is pure: it takes RMS
 * samples and timestamps and emits events. Everything browser-shaped (the
 * AudioContext, the analyser, the polling loop) is in `TurnDetector` below it.
 * Splitting them that way is what makes the hard part -- the thresholds and the
 * timing -- testable without a microphone.
 */

export type TurnEventType = "speech_start" | "utterance_end" | "no_speech";

export interface TurnEvent {
  type: TurnEventType;
  /** Only on `utterance_end`. */
  reason?: "silence" | "max_length";
  /** Milliseconds of speech in the utterance, on `utterance_end`. */
  speechMs?: number;
}

export interface TurnStateMachineOptions {
  /** RMS above which a frame counts as speech, before the noise floor is added. */
  threshold?: number;
  /** Continuous quiet that ends a turn. */
  silenceMs?: number;
  /** Speech shorter than this is a cough, not an utterance. */
  minSpeechMs?: number;
  /** Hard cap, so a noisy room cannot hold the turn open forever. */
  maxUtteranceMs?: number;
  /** If nobody speaks within this, give up rather than listening forever. */
  noSpeechMs?: number;
}

const DEFAULTS = {
  threshold: 0.012,
  // 700ms, not 900: people pause mid-sentence, but a caller who has genuinely
  // finished will not wait much longer than this before wondering whether they
  // were heard. It is the most noticeable number in the whole component.
  silenceMs: 700,
  minSpeechMs: 300,
  maxUtteranceMs: 15_000,
  noSpeechMs: 8_000,
};

/**
 * Ceiling on how far the measured noise floor may raise the threshold.
 *
 * Without this, a noisy room (or a bad floor estimate) could push the bar above
 * the caller's own voice and the detector would go deaf.
 */
const MAX_ADAPTIVE_THRESHOLD = 0.05;

/**
 * Whether a frame of audio is speech.
 *
 * A fixed threshold is fragile -- laptop mics and room noise vary by an order
 * of magnitude -- so the tracked floor raises it. The multiplier is deliberately
 * conservative: missing the very start of a word is recoverable (the caller
 * keeps talking), whereas treating room hum as speech makes the agent interrupt
 * itself.
 */
export function isSpeechFrame(rms: number, threshold: number, noiseFloor: number): boolean {
  const adaptive = Math.min(noiseFloor * 3, MAX_ADAPTIVE_THRESHOLD);
  return rms > Math.max(threshold, adaptive);
}

/**
 * Tracks the room's noise level as the call goes on.
 *
 * Snaps *down* to any quieter frame immediately, and only creeps up when every
 * recent frame is louder. The asymmetry is the point: silence is ground truth
 * about the room, whereas a loud frame is more likely to be the caller talking
 * than a sudden change in the environment.
 *
 * This replaces an up-front calibration window, which had a fatal flaw -- a
 * caller who starts talking the moment they click "Start call" is measured
 * *while speaking*, so the "noise floor" comes back as their own voice and the
 * detector spends the rest of the call unable to hear them.
 */
export class NoiseFloorTracker {
  constructor(
    private floor = 0,
    /** How fast the estimate is allowed to rise. Slow, so speech cannot drag it. */
    private readonly rise = 0.002,
  ) {}

  push(rms: number): void {
    if (rms < this.floor) this.floor = rms;
    else this.floor += (rms - this.floor) * this.rise;
  }

  get value(): number {
    return this.floor;
  }
}

export class TurnStateMachine {
  readonly opts: Required<TurnStateMachineOptions>;

  private phase: "idle" | "waiting" | "speech" = "idle";
  private speechStartedAt = 0;
  private lastSpeechAt = 0;
  private startedAt = 0;
  private speechAccum = 0;
  private lastTick: number | null = null;
  private noiseFloor = 0;

  constructor(options: TurnStateMachineOptions = {}) {
    this.opts = { ...DEFAULTS, ...options };
  }

  /** Begin waiting for a caller to say something. */
  start(now: number): void {
    this.phase = "waiting";
    this.startedAt = now;
    this.lastTick = now;
    this.speechAccum = 0;
    this.speechStartedAt = 0;
    this.lastSpeechAt = 0;
  }

  /** Update the current noise-floor estimate. */
  setNoiseFloor(rms: number): void {
    this.noiseFloor = Math.max(0, rms);
  }

  get currentPhase(): "idle" | "waiting" | "speech" {
    return this.phase;
  }

  /** Milliseconds of speech accumulated in the current utterance. */
  get speechMs(): number {
    return this.speechAccum;
  }

  /**
   * Feed one RMS sample. Returns an event when that sample decided something.
   *
   * `now` is a monotonic millisecond clock from the caller, so the machine
   * never reads the wall clock itself and a test can drive it deterministically.
   */
  push(rms: number, now: number): TurnEvent | null {
    if (this.phase === "idle") return null;

    const delta = this.lastTick === null ? 0 : now - this.lastTick;
    this.lastTick = now;
    const speech = isSpeechFrame(rms, this.opts.threshold, this.noiseFloor);

    if (this.phase === "waiting") {
      if (speech) {
        this.phase = "speech";
        this.speechStartedAt = now;
        this.lastSpeechAt = now;
        this.speechAccum = delta;
        return { type: "speech_start" };
      }
      if (now - this.startedAt >= this.opts.noSpeechMs) {
        this.phase = "idle";
        return { type: "no_speech" };
      }
      return null;
    }

    // Speaking.
    if (speech) {
      this.speechAccum += delta;
      this.lastSpeechAt = now;
    }

    const quietFor = now - this.lastSpeechAt;
    const spokenFor = this.speechAccum;
    const totalFor = now - this.speechStartedAt;

    if (totalFor >= this.opts.maxUtteranceMs) {
      this.phase = "idle";
      // Too short to have been an utterance even though it ran long -- treat it
      // as noise rather than sending the model half a word.
      if (spokenFor < this.opts.minSpeechMs) return { type: "no_speech" };
      return { type: "utterance_end", reason: "max_length", speechMs: spokenFor };
    }

    if (quietFor >= this.opts.silenceMs) {
      if (spokenFor < this.opts.minSpeechMs) {
        // A click or a cough. Go back to waiting rather than sending it.
        this.phase = "waiting";
        this.startedAt = now;
        this.speechAccum = 0;
        return null;
      }
      this.phase = "idle";
      return { type: "utterance_end", reason: "silence", speechMs: spokenFor };
    }

    return null;
  }
}

// ---------------------------------------------------------------------------
// Browser wiring
// ---------------------------------------------------------------------------

export interface TurnDetectorOptions extends TurnStateMachineOptions {
  /** Fires on the first frame above threshold. Use it to start capturing. */
  onSpeechStart?: () => void;
  /**
   * Fires once the caller has been speaking for `interruptAfterMs` without a
   * gap. Use it to cut off playback — *not* `onSpeechStart`.
   *
   * Interrupting on the first loud frame means a door slam, a cough, or the
   * agent's own voice leaking through the speakers all cut it off mid-word.
   * Requiring a short run of continuous speech is the cheapest thing that
   * distinguishes "someone is talking to me" from "something was loud".
   */
  onInterrupt?: () => void;
  /** Sustained speech needed before `onInterrupt` fires. */
  interruptAfterMs?: number;
  /** Called once, when the caller has finished their turn. */
  onUtteranceEnd: (event: TurnEvent) => void;
  onNoSpeech?: () => void;
  /**
   * Every measured frame, with the phase it landed in.
   *
   * Drives the microphone meter. Without a visible level there is no way for a
   * caller to tell "the agent isn't answering" apart from "my mic is muted",
   * which is the most common reason a call appears to do nothing at all.
   */
  onLevel?: (rms: number, phase: "waiting" | "speech") => void;
}

/** Poll interval. 40ms is fine-grained enough for sub-second turn ends. */
const TICK_MS = 40;

/**
 * Watches a microphone stream and reports whole utterances.
 *
 * Reads RMS from an `AnalyserNode` on a timer rather than `requestAnimationFrame`
 * because rAF stops in a backgrounded tab, which would strand the call listening
 * forever with no way to finish a turn.
 */
export class TurnDetector {
  private ctx: AudioContext | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private analyser: AnalyserNode | null = null;
  // Typed with an explicit ArrayBuffer parameter: the DOM lib's
  // `getFloatTimeDomainData` will not accept the wider `ArrayBufferLike`.
  private samples: Float32Array<ArrayBuffer> | null = null;
  private timer: ReturnType<typeof setInterval> | null = null;
  private readonly machine: TurnStateMachine;
  private readonly tracker = new NoiseFloorTracker();
  private started = false;
  private stopped = false;
  private interrupted = false;

  constructor(private readonly options: TurnDetectorOptions) {
    this.machine = new TurnStateMachine(options);
  }

  async start(stream: MediaStream): Promise<void> {
    const Ctor =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) throw new Error("Web Audio is not available in this browser");

    this.ctx = new Ctor();
    if (this.ctx.state === "suspended") await this.ctx.resume();
    this.source = this.ctx.createMediaStreamSource(stream);
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 2048;
    this.source.connect(this.analyser);
    this.samples = new Float32Array(this.analyser.fftSize);

    this.stopped = false;
    this.started = false;
    this.interrupted = false;
    this.timer = setInterval(() => this.tick(), TICK_MS);
  }

  stop(): void {
    this.stopped = true;
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
    this.source?.disconnect();
    this.analyser?.disconnect();
    void this.ctx?.close().catch(() => undefined);
    this.ctx = null;
  }

  private tick(): void {
    if (this.stopped || !this.analyser || !this.samples) return;
    const now = performance.now();

    // The analyser is only read, never connected to a destination, so this
    // cannot echo the caller back into their own speakers.
    this.analyser.getFloatTimeDomainData(this.samples);
    const rms = rootMeanSquare(this.samples);

    this.tracker.push(rms);
    this.machine.setNoiseFloor(this.tracker.value);

    if (!this.started) {
      this.started = true;
      this.machine.start(now);
    }

    const event = this.machine.push(rms, now);
    this.options.onLevel?.(rms, this.machine.currentPhase === "speech" ? "speech" : "waiting");

    // Sustained speech, checked every frame rather than only on transitions.
    const interruptAfter = this.options.interruptAfterMs ?? 150;
    if (
      !this.interrupted &&
      this.machine.currentPhase === "speech" &&
      this.machine.speechMs >= interruptAfter
    ) {
      this.interrupted = true;
      this.options.onInterrupt?.();
    }

    if (!event) return;

    if (event.type === "speech_start") {
      this.interrupted = false;
      this.options.onSpeechStart?.();
      return;
    }

    // Both terminal events below leave the machine "idle", and an idle machine
    // ignores every subsequent frame -- so it must be re-armed here or the call
    // goes permanently deaf. That failure is silent and delayed: the caller
    // pauses to think, the no-speech timer expires, and from then on the agent
    // never hears anything again while still looking attentive.
    this.machine.start(now);
    this.interrupted = false;

    if (event.type === "utterance_end") this.options.onUtteranceEnd(event);
    else this.options.onNoSpeech?.();
  }
}

export function rootMeanSquare(samples: Float32Array): number {
  if (samples.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** The value at `fraction` through the sorted samples, 0..1. */
export function percentile(values: number[], fraction: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.round(fraction * (sorted.length - 1))),
  );
  return sorted[index];
}
