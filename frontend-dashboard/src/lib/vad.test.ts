import { describe, expect, it, vi } from "vitest";

import { TurnStateMachine, isSpeechFrame, median, rootMeanSquare } from "./vad";

const SPEECH = 0.2; // comfortably above any sane threshold
const QUIET = 0.0;

/** Drive the machine with a run of frames, returning every event it emitted. */
function feed(
  machine: TurnStateMachine,
  frames: { rms: number; until: number }[],
  from = 0,
): { events: { type: string; reason?: string; at: number }[]; last: number } {
  const events: { type: string; reason?: string; at: number }[] = [];
  let t = from;
  for (const frame of frames) {
    while (t < frame.until) {
      t += 40;
      const event = machine.push(frame.rms, t);
      if (event) events.push({ type: event.type, reason: event.reason, at: t });
    }
  }
  return { events, last: t };
}

describe("isSpeechFrame", () => {
  it("uses the fixed threshold when the room is silent", () => {
    expect(isSpeechFrame(0.05, 0.012, 0)).toBe(true);
    expect(isSpeechFrame(0.001, 0.012, 0)).toBe(false);
  });

  it("raises the bar with the measured noise floor", () => {
    // In a noisy room, 0.05 would otherwise read as speech forever.
    expect(isSpeechFrame(0.05, 0.012, 0.02)).toBe(false);
    expect(isSpeechFrame(0.1, 0.012, 0.02)).toBe(true);
  });
});

describe("TurnStateMachine", () => {
  it("emits nothing until the caller actually speaks", () => {
    const machine = new TurnStateMachine();
    machine.start(0);
    const { events } = feed(machine, [{ rms: QUIET, until: 400 }]);
    expect(events).toEqual([]);
    expect(machine.currentPhase).toBe("waiting");
  });

  it("starts a turn on speech and ends it after the silence", () => {
    const machine = new TurnStateMachine();
    machine.start(0);
    feed(machine, [{ rms: SPEECH, until: 600 }]); // ~600ms of talking
    const { events } = feed(machine, [{ rms: QUIET, until: 2000 }], 600);

    expect(events.map((e) => e.type)).toEqual(["utterance_end"]);
    expect(events[0].reason).toBe("silence");
    // Fires one silence-window after speech stopped (700ms default), not
    // instantly and not at the end of the feed.
    expect(events[0].at).toBeGreaterThanOrEqual(600 + 700 - 40);
    expect(events[0].at).toBeLessThanOrEqual(600 + 700 + 80);
  });

  it("reports speech_start so playback can be cut off", () => {
    const machine = new TurnStateMachine();
    machine.start(0);
    const { events } = feed(machine, [{ rms: QUIET, until: 200 }, { rms: SPEECH, until: 400 }]);
    expect(events.map((e) => e.type)).toEqual(["speech_start"]);
  });

  it("ignores a cough instead of sending it to the model", () => {
    const machine = new TurnStateMachine({ minSpeechMs: 300 });
    machine.start(0);
    // 120ms of noise, then quiet for well past the silence window.
    feed(machine, [{ rms: SPEECH, until: 120 }]);
    const { events } = feed(machine, [{ rms: QUIET, until: 2000 }], 120);

    expect(events).toEqual([]);
    // Back to listening, so the caller's real sentence still gets a turn.
    expect(machine.currentPhase).toBe("waiting");
  });

  it("keeps a turn open across a pause shorter than the silence window", () => {
    const machine = new TurnStateMachine({ silenceMs: 900 });
    machine.start(0);
    const first = feed(machine, [{ rms: SPEECH, until: 500 }]);
    const second = feed(machine, [{ rms: QUIET, until: 900 }], 500); // a 400ms breath
    const third = feed(machine, [{ rms: SPEECH, until: 1400 }], 900);

    // One speech_start and *no* utterance_end: pausing mid-sentence to think is
    // not the caller finishing, and cutting them off there is the classic
    // voice-agent annoyance.
    expect([...first.events, ...second.events, ...third.events].map((e) => e.type)).toEqual([
      "speech_start",
    ]);
    expect(machine.currentPhase).toBe("speech");
  });

  it("gives up when nobody speaks at all", () => {
    const machine = new TurnStateMachine({ noSpeechMs: 2000 });
    machine.start(0);
    const { events } = feed(machine, [{ rms: QUIET, until: 3000 }]);
    expect(events.map((e) => e.type)).toEqual(["no_speech"]);
    expect(machine.currentPhase).toBe("idle");
  });

  it("caps a turn so a noisy room cannot hold it open forever", () => {
    const machine = new TurnStateMachine({ maxUtteranceMs: 2000 });
    machine.start(0);
    // Continuous "speech" -- e.g. consistent background noise.
    const { events } = feed(machine, [{ rms: SPEECH, until: 3000 }]);
    expect(events.map((e) => e.type)).toEqual(["speech_start", "utterance_end"]);
    expect(events[1].reason).toBe("max_length");
  });

  it("treats a long stretch of sub-minimum noise as nothing, not an utterance", () => {
    const machine = new TurnStateMachine({ maxUtteranceMs: 500, minSpeechMs: 400 });
    machine.start(0);
    const { events } = feed(machine, [{ rms: SPEECH, until: 100 }, { rms: QUIET, until: 900 }]);
    // Speech started, but only 40ms accumulated before the cap: not worth sending.
    expect(events.map((e) => e.type)).toEqual(["speech_start", "no_speech"]);
  });

  it("does nothing at all until start() is called", () => {
    const machine = new TurnStateMachine();
    expect(machine.push(SPEECH, 100)).toBeNull();
    expect(machine.currentPhase).toBe("idle");
  });
});

describe("signal helpers", () => {
  it("computes RMS of a known signal", () => {
    // A constant 0.5 has RMS 0.5.
    expect(rootMeanSquare(Float32Array.from([0.5, 0.5, 0.5, 0.5]))).toBeCloseTo(0.5, 6);
    expect(rootMeanSquare(Float32Array.from([0, 0, 0]))).toBe(0);
    expect(rootMeanSquare(new Float32Array(0))).toBe(0);
  });

  it("takes the median of even and odd counts", () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 3, 2])).toBe(2.5);
    expect(median([])).toBe(0);
  });

  it("does not let one loud outlier drag the calibrated noise floor up", () => {
    // A slammed door during calibration would skew a mean; the median ignores it.
    const floor = median([0.001, 0.001, 0.002, 0.001, 0.9]);
    vi.spyOn(Math, "max").mockRestore();
    expect(floor).toBeCloseTo(0.001, 6);
    expect(isSpeechFrame(SPEECH, 0.012, floor)).toBe(true);
  });
});
