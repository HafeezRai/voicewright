/**
 * Progressive PCM playback.
 *
 * A streamed turn delivers audio in chunks as the agent generates, so playback
 * has to start on the *first* chunk rather than after the last. That rules out
 * an <audio> element (which needs a complete, seekable file) and rules out
 * `decodeAudioData` on a growing buffer.
 *
 * Instead each chunk becomes an `AudioBuffer` scheduled back-to-back on a
 * running clock. `currentTime` is the audio hardware clock, so scheduling the
 * next chunk at `max(currentTime, nextStart)` gives gapless playback without
 * drifting -- and if the model stalls, `currentTime` catches up to `nextStart`
 * on its own and the next chunk simply starts immediately.
 */
export class ProgressivePlayer {
  /**
   * Fired when the queue drains and the agent has stopped speaking.
   *
   * The end of playback is what closes the window in which the microphone can
   * pick up the agent's own voice — see `looksLikeEcho`. Not fired by `stop()`,
   * which is a deliberate cut-off the caller already knows about.
   */
  onIdle?: () => void;

  private ctx: AudioContext | null = null;
  private nextStart = 0;
  private sources = new Set<AudioBufferSourceNode>();
  private scheduled = 0;

  /** True once at least one chunk has been scheduled this turn. */
  get hasAudio(): boolean {
    return this.scheduled > 0;
  }

  /**
   * True while audio is queued or actually coming out of the speakers.
   *
   * This is the one to gate barge-in on. `hasAudio` stays true for the rest of
   * the turn once the first chunk lands, so using it would have the *next*
   * utterance count as an interruption of a reply that already finished.
   */
  get isPlaying(): boolean {
    return this.sources.size > 0;
  }

  /** Seconds of audio queued but not yet played. */
  get backlogSeconds(): number {
    if (!this.ctx) return 0;
    return Math.max(0, this.nextStart - this.ctx.currentTime);
  }

  /**
   * Schedule one chunk of signed 16-bit PCM.
   *
   * The AudioContext is created lazily and resumed here: browsers only allow
   * that from a user gesture, and this is always called in the wake of one
   * (the click that started the recording).
   */
  push(pcm16: Int16Array, sampleRate: number): void {
    if (pcm16.length === 0) return;
    const ctx = this.ensureContext();
    const buffer = ctx.createBuffer(1, pcm16.length, sampleRate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < pcm16.length; i += 1) {
      // 32768, not 32767: the minimum sample is -32768, so this maps the range
      // symmetrically onto [-1, 1) with no positive-side clipping.
      channel[i] = pcm16[i] / 32768;
    }

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    const startAt = Math.max(ctx.currentTime, this.nextStart);
    source.start(startAt);
    this.nextStart = startAt + buffer.duration;
    this.scheduled += 1;

    this.sources.add(source);
    source.onended = () => {
      this.sources.delete(source);
      if (this.sources.size === 0) this.onIdle?.();
    };
  }

  /** Stop everything immediately (barge-in, or the user leaving). */
  stop(): void {
    for (const source of this.sources) {
      try {
        source.stop();
      } catch {
        // Already ended; nothing to stop.
      }
    }
    this.sources.clear();
    this.nextStart = 0;
    this.scheduled = 0;
  }

  /** Release the AudioContext. Call on unmount. */
  async close(): Promise<void> {
    this.stop();
    const ctx = this.ctx;
    this.ctx = null;
    if (ctx) await ctx.close().catch(() => undefined);
  }

  private ensureContext(): AudioContext {
    if (!this.ctx) {
      const Ctor =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!Ctor) throw new Error("Web Audio is not available in this browser");
      this.ctx = new Ctor();
    }
    if (this.ctx.state === "suspended") void this.ctx.resume();
    return this.ctx;
  }
}

/** Decode the base64 PCM16 an SSE `audio` frame carries. */
export function decodePcm16(base64: string): Int16Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  // The payload is little-endian signed 16-bit, which is what the backend's
  // `to_pcm16_bytes` writes and what `Int16Array` reads on every platform we
  // ship to. Guard the odd trailing byte rather than producing a NaN sample.
  const usable = bytes.length - (bytes.length % 2);
  return new Int16Array(bytes.buffer, 0, usable / 2);
}
