import { describe, expect, it } from "vitest";

import { encodeWav } from "./audio";

/**
 * The WAV encoder is the last thing standing between captured samples and the
 * server, which parses this header with `soundfile`. A wrong field there is a
 * 400 that looks like a microphone problem, so the header is worth pinning
 * down byte by byte.
 *
 * `PcmCapture` and `resampleToMonoWav` need a real Web Audio implementation and
 * are covered end-to-end by the browser specs instead of being stubbed here.
 */
// jsdom's Blob has no `arrayBuffer()`, so read it the way the browser would.
function readArrayBuffer(blob: Blob): Promise<ArrayBuffer> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as ArrayBuffer);
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(blob);
  });
}

async function bytesOf(blob: Blob): Promise<DataView> {
  return new DataView(await readArrayBuffer(blob));
}

function textAt(view: DataView, offset: number, length: number): string {
  let text = "";
  for (let i = 0; i < length; i += 1) text += String.fromCharCode(view.getUint8(offset + i));
  return text;
}

describe("encodeWav", () => {
  const samples = Float32Array.from([0, 0.5, -0.5, 1, -1]);

  it("writes a RIFF/WAVE header soundfile will accept", async () => {
    const blob = encodeWav(samples, 16000);
    expect(blob.type).toBe("audio/wav");

    const view = await bytesOf(blob);
    expect(textAt(view, 0, 4)).toBe("RIFF");
    expect(textAt(view, 8, 4)).toBe("WAVE");
    expect(textAt(view, 12, 4)).toBe("fmt ");
    expect(textAt(view, 36, 4)).toBe("data");
  });

  it("declares mono 16-bit PCM at the requested rate", async () => {
    const view = await bytesOf(encodeWav(samples, 16000));
    expect(view.getUint16(20, true)).toBe(1); // PCM
    expect(view.getUint16(22, true)).toBe(1); // mono
    expect(view.getUint32(24, true)).toBe(16000);
    expect(view.getUint16(34, true)).toBe(16); // bits per sample
    // byte rate and block align must agree with the above or players disagree
    expect(view.getUint32(28, true)).toBe(16000 * 2);
    expect(view.getUint16(32, true)).toBe(2);
  });

  it("sizes the RIFF and data chunks to the payload", async () => {
    const view = await bytesOf(encodeWav(samples, 16000));
    const dataSize = samples.length * 2;
    expect(view.getUint32(4, true)).toBe(36 + dataSize);
    expect(view.getUint32(40, true)).toBe(dataSize);
    expect(view.byteLength).toBe(44 + dataSize);
  });

  it("round-trips sample values as signed 16-bit", async () => {
    const view = await bytesOf(encodeWav(Float32Array.from([0, 1, -1]), 16000));
    expect(view.getInt16(44, true)).toBe(0);
    expect(view.getInt16(46, true)).toBe(0x7fff); // +1 maps to 32767
    // -1 maps to the full negative extreme, not -32767
    expect(view.getInt16(48, true)).toBe(-0x8000);
  });

  it("clamps samples outside [-1, 1] instead of wrapping", async () => {
    // Wrapping here would turn a loud passage into noise rather than clipping.
    const view = await bytesOf(encodeWav(Float32Array.from([2, -2]), 16000));
    expect(view.getInt16(44, true)).toBe(0x7fff);
    expect(view.getInt16(46, true)).toBe(-0x8000);
  });

  it("encodes an empty recording as a valid, zero-length WAV", async () => {
    const view = await bytesOf(encodeWav(new Float32Array(0), 16000));
    expect(view.byteLength).toBe(44);
    expect(view.getUint32(40, true)).toBe(0);
    expect(textAt(view, 0, 4)).toBe("RIFF");
  });
});
