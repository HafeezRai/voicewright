import { afterEach, describe, expect, it, vi } from "vitest";

import { ProgressivePlayer, decodePcm16 } from "./player";

/** Minimal AudioContext stand-in that records what got scheduled, and when. */
class FakeAudioContext {
  static instances: FakeAudioContext[] = [];
  currentTime = 0;
  state = "running";
  destination = { name: "destination" };
  readonly scheduled: { startAt: number; duration: number }[] = [];

  constructor() {
    FakeAudioContext.instances.push(this);
  }

  createBuffer(_channels: number, length: number, sampleRate: number) {
    return {
      duration: length / sampleRate,
      getChannelData: () => new Float32Array(length),
    };
  }

  createBufferSource() {
    const ctx = this;
    return {
      buffer: null as { duration: number } | null,
      connect: () => undefined,
      onended: null as (() => void) | null,
      start(startAt: number) {
        ctx.scheduled.push({ startAt, duration: (this.buffer as { duration: number })?.duration ?? 0 });
      },
      stop: () => undefined,
    };
  }

  resume() {
    return Promise.resolve();
  }
  close() {
    return Promise.resolve();
  }
}

function pcm16(...samples: number[]): Int16Array {
  return Int16Array.from(samples);
}

afterEach(() => {
  FakeAudioContext.instances = [];
  vi.unstubAllGlobals();
});

describe("decodePcm16", () => {
  it("round-trips little-endian signed 16-bit samples", () => {
    // 0x0001 -> 1, 0xFFFF -> -1 (as a signed 16-bit read)
    const base64 = btoa(String.fromCharCode(0x01, 0x00, 0xff, 0xff));
    const decoded = decodePcm16(base64);
    expect(Array.from(decoded)).toEqual([1, -1]);
  });

  it("ignores a trailing odd byte rather than emitting a broken sample", () => {
    const base64 = btoa(String.fromCharCode(0x01, 0x00, 0x7f));
    expect(Array.from(decodePcm16(base64))).toEqual([1]);
  });

  it("returns nothing for an empty payload", () => {
    expect(decodePcm16("").length).toBe(0);
  });
});

describe("ProgressivePlayer", () => {
  it("does nothing until audio actually arrives", () => {
    const player = new ProgressivePlayer();
    expect(player.hasAudio).toBe(false);
    expect(player.backlogSeconds).toBe(0);
    player.push(pcm16(), 16000); // empty chunk
    expect(player.hasAudio).toBe(false);
  });

  it("schedules the first chunk immediately and the next back-to-back", () => {
    vi.stubGlobal("AudioContext", FakeAudioContext);
    const player = new ProgressivePlayer();

    player.push(pcm16(...new Array(1600).fill(0)), 16000); // 0.1s
    player.push(pcm16(...new Array(1600).fill(0)), 16000);

    const ctx = FakeAudioContext.instances[0];
    expect(ctx.scheduled).toHaveLength(2);
    // First at the head of the clock, second immediately after it -- gapless.
    expect(ctx.scheduled[0].startAt).toBe(0);
    expect(ctx.scheduled[1].startAt).toBeCloseTo(0.1, 5);
    expect(player.hasAudio).toBe(true);
  });

  it("does not schedule in the past when the model stalls mid-reply", () => {
    vi.stubGlobal("AudioContext", FakeAudioContext);
    const player = new ProgressivePlayer();

    player.push(pcm16(...new Array(1600).fill(0)), 16000); // scheduled 0 -> 0.1
    const ctx = FakeAudioContext.instances[0];
    // The clock runs past the queued audio while we wait for the next chunk.
    ctx.currentTime = 5;

    player.push(pcm16(...new Array(1600).fill(0)), 16000);
    // Starts "now", not at 0.1 in the past -- otherwise the buffer would be
    // scheduled behind the clock and silently skipped.
    expect(ctx.scheduled[1].startAt).toBe(5);
  });

  it("reports how much audio is queued but unplayed", () => {
    vi.stubGlobal("AudioContext", FakeAudioContext);
    const player = new ProgressivePlayer();
    player.push(pcm16(...new Array(16000).fill(0)), 16000); // 1s
    const ctx = FakeAudioContext.instances[0];
    ctx.currentTime = 0.25;
    expect(player.backlogSeconds).toBeCloseTo(0.75, 5);
  });

  it("forgets everything on stop, so a barge-in does not leave audio queued", () => {
    vi.stubGlobal("AudioContext", FakeAudioContext);
    const player = new ProgressivePlayer();
    player.push(pcm16(...new Array(1600).fill(0)), 16000);
    player.stop();
    expect(player.hasAudio).toBe(false);

    // The next turn starts from the head of the clock again.
    player.push(pcm16(...new Array(1600).fill(0)), 16000);
    const ctx = FakeAudioContext.instances[0];
    expect(ctx.scheduled[1].startAt).toBe(0);
  });
});
