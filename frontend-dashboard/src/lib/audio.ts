/**
 * Browser-side audio helpers: a minimal WAV encoder and a "whatever the
 * MediaRecorder produced" -> 16 kHz mono WAV converter.
 *
 * ai-engine's `/stt` expects a WAV body (`soundfile` decodes it) and its VAD
 * stack is tuned for 16 kHz mono, while `MediaRecorder` gives us Opus-in-WebM.
 * `OfflineAudioContext` is the standard, dependency-free way to bridge the two.
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
