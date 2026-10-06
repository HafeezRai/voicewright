/**
 * Browser-side audio helpers: WAV encoding, resampling, and raw PCM capture.
 *
 * ai-engine's `/stt` expects a WAV body (`soundfile` decodes it) and its VAD
 * stack is tuned for 16 kHz mono. `OfflineAudioContext` is the standard,
 * dependency-free way to get there from whatever the microphone produced.
 */

/** Encode mono float samples in [-1, 1] as a 16-bit PCM WAV `Blob`. */
export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample; // mono
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset: number, text: string): void => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 8 * bytesPerSample, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += bytesPerSample;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

/**
 * Decode an arbitrary browser audio `Blob` (WebM/Opus from MediaRecorder) and
 * re-encode it as 16 kHz mono WAV. Downmixes stereo and resamples.
 */
export async function blobToMonoWav(blob: Blob, targetRate = 16000): Promise<Blob> {
  const arrayBuffer = await blob.arrayBuffer();
  const AudioCtx: typeof AudioContext =
    window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const decodeCtx = new AudioCtx();
  let decoded: AudioBuffer;
  try {
    decoded = await decodeCtx.decodeAudioData(arrayBuffer);
  } finally {
    void decodeCtx.close();
  }

  const frames = Math.max(1, Math.ceil(decoded.duration * targetRate));
  const offline = new OfflineAudioContext(1, frames, targetRate);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return encodeWav(rendered.getChannelData(0), targetRate);
}

/** Pick a MediaRecorder mime type the current browser actually supports. */
export function pickRecorderMimeType(): string | undefined {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"];
  if (typeof MediaRecorder === "undefined" || !MediaRecorder.isTypeSupported) return undefined;
  return candidates.find((type) => MediaRecorder.isTypeSupported(type));
}

/**
 * Resample raw mono samples to `targetRate` and encode as a WAV.
 *
 * Uses `OfflineAudioContext` so the browser's own resampler does the work,
 * rather than decimating by hand and aliasing everything above 8 kHz.
 */
export async function resampleToMonoWav(
  // Explicit ArrayBuffer parameter: the DOM lib's `copyToChannel` will not
  // accept the wider `ArrayBufferLike`.
  samples: Float32Array<ArrayBuffer>,
  sourceRate: number,
  targetRate = 16000,
): Promise<Blob> {
  if (sourceRate === targetRate || samples.length === 0) return encodeWav(samples, sourceRate);

  const frames = Math.max(1, Math.round((samples.length / sourceRate) * targetRate));
  const offline = new OfflineAudioContext(1, frames, targetRate);
  const buffer = offline.createBuffer(1, samples.length, sourceRate);
  buffer.copyToChannel(samples, 0);
  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return encodeWav(rendered.getChannelData(0), targetRate);
}

/**
 * Capture raw PCM straight off the microphone graph.
 *
 * `MediaRecorder` was the obvious tool and the wrong one. It hands back a
 * *compressed container* -- WebM/Opus in Chrome, fragmented MP4/AAC in Safari --
 * which then has to be decoded back into samples before the pipeline, which
 * only ever wanted PCM, can use them. That round trip through a lossy codec is
 * pure loss, and in Safari it fails outright: `decodeAudioData` rejects the
 * fragmented MP4 that Safari's own `MediaRecorder` produced, so every recording
 * died with "unknown content type". Chrome merely failed differently, on
 * mid-stream slices whose WebM header had been cut off.
 *
 * Tapping the graph directly has neither problem: no codec, no container, no
 * header, and identical behaviour in every browser.
 */
export class PcmCapture {
  private ctx: AudioContext | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private processor: ScriptProcessorNode | null = null;
  private sink: GainNode | null = null;
  private chunks: Float32Array<ArrayBuffer>[] = [];
  private active = false;

  /** Sample rate of the captured audio, once started. */
  get sampleRate(): number {
    return this.ctx?.sampleRate ?? 0;
  }

  async start(stream: MediaStream): Promise<void> {
    const Ctor: typeof AudioContext =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    this.ctx = new Ctor();
    if (this.ctx.state === "suspended") await this.ctx.resume();
    this.source = this.ctx.createMediaStreamSource(stream);

    // ScriptProcessorNode is deprecated in favour of AudioWorklet, but it needs
    // no separate module fetch, works identically in every browser including
    // Safari, and capture does not have the realtime constraints that motivated
    // the replacement. Revisit if it ever glitches.
    this.processor = this.ctx.createScriptProcessor(4096, 1, 1);
    this.processor.onaudioprocess = (event) => {
      if (!this.active) return;
      // Copy: the input buffer is reused by the audio thread after this returns.
      this.chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    };
    this.source.connect(this.processor);

    // A ScriptProcessorNode is only pulled if it reaches the destination, so
    // route it through a silent gain node -- otherwise it never fires, and it
    // must not be audible.
    this.sink = this.ctx.createGain();
    this.sink.gain.value = 0;
    this.processor.connect(this.sink);
    this.sink.connect(this.ctx.destination);
  }

  /** Begin a fresh utterance. Anything captured before this is discarded. */
  begin(): void {
    this.chunks = [];
    this.active = true;
  }

  /** End the utterance and hand back its samples. */
  finish(): { samples: Float32Array<ArrayBuffer>; sampleRate: number } | null {
    this.active = false;
    const sampleRate = this.sampleRate;
    if (this.chunks.length === 0 || sampleRate === 0) return null;

    let total = 0;
    for (const chunk of this.chunks) total += chunk.length;
    const samples = new Float32Array(total);
    let offset = 0;
    for (const chunk of this.chunks) {
      samples.set(chunk, offset);
      offset += chunk.length;
    }
    this.chunks = [];
    return { samples, sampleRate };
  }

  stop(): void {
    this.active = false;
    this.chunks = [];
    this.processor?.disconnect();
    this.source?.disconnect();
    this.sink?.disconnect();
    this.processor = null;
    this.source = null;
    this.sink = null;
    void this.ctx?.close().catch(() => undefined);
    this.ctx = null;
  }
}
