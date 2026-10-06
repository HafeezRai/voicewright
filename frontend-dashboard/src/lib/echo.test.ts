import { describe, expect, it } from "vitest";

import { echoScore, looksLikeEcho, normalizeWords } from "./echo";

/** A real reply, used as the reference the microphone might pick back up. */
const REPLY =
  "According to our records, the top 1% of global earners hold around 39% of the world's total wealth.";

describe("normalizeWords", () => {
  it("lowercases and strips punctuation", () => {
    expect(normalizeWords("Hello, world! It's 39%.")).toEqual(["hello", "world", "it", "s", "39"]);
  });

  it("keeps non-Latin text instead of deleting it", () => {
    // A range-based strip that only knows a-z would silently erase this.
    expect(normalizeWords("ہیلو دنیا")).toEqual(["ہیلو", "دنیا"]);
  });

  it("returns nothing for empty or punctuation-only input", () => {
    expect(normalizeWords("")).toEqual([]);
    expect(normalizeWords("... --- ...")).toEqual([]);
  });
});

describe("echoScore", () => {
  it("scores a verbatim pickup near 1", () => {
    expect(echoScore(REPLY, REPLY)).toBe(1);
  });

  it("scores the tail of a reply highly", () => {
    // Containment, not Jaccard: the rest of the reply was never captured, but
    // every word that *was* captured is the agent's.
    expect(echoScore("hold around 39% of the world's total wealth", REPLY)).toBeGreaterThan(0.85);
  });

  it("scores an unrelated question near zero", () => {
    expect(echoScore("can you book me an appointment tomorrow", REPLY)).toBe(0);
  });

  it("scores nothing for an empty transcript", () => {
    expect(echoScore("", REPLY)).toBe(0);
  });
});

describe("looksLikeEcho", () => {
  const now = 10_000;
  const speaking = { text: REPLY, endedAt: null };

  it("rejects the agent's own reply coming back through the mic", () => {
    expect(looksLikeEcho(REPLY, speaking, now)).toBe(true);
  });

  it("rejects a captured tail of the reply", () => {
    expect(looksLikeEcho("hold around 39% of the world's total wealth", speaking, now)).toBe(true);
  });

  it("accepts a genuine question", () => {
    expect(looksLikeEcho("can you book me an appointment for tomorrow", speaking, now)).toBe(false);
  });

  it("accepts a caller quoting the agent back rather than repeating it", () => {
    // Shares plenty of words with the reply, but not *only* the reply's words --
    // and this caller deserves an answer.
    expect(
      looksLikeEcho("so you said the top 1% of earners hold 39% of the wealth, is that right?", speaking, now),
    ).toBe(false);
  });

  it("leaves a short quote alone, even though it scores a perfect match", () => {
    // "39% of the wealth?" is 100% the agent's words. So is real echo. At four
    // words there is no signal that separates them, and ignoring a caller is
    // worse than answering an echo -- so short transcripts are never judged.
    expect(echoScore("39% of the wealth", REPLY)).toBeGreaterThan(0.9);
    expect(looksLikeEcho("39% of the wealth?", speaking, now)).toBe(false);
  });

  it("stops rejecting once the agent has been quiet for a while", () => {
    // A perfect match long after playback is someone quoting the agent, not echo.
    const longAgo = { text: REPLY, endedAt: now - 10_000 };
    expect(looksLikeEcho(REPLY, longAgo, now)).toBe(false);
  });

  it("still rejects in the window just after playback ends", () => {
    const justStopped = { text: REPLY, endedAt: now - 400 };
    expect(looksLikeEcho(REPLY, justStopped, now)).toBe(true);
  });

  it("does nothing before the agent has said anything", () => {
    expect(looksLikeEcho(REPLY, { text: "", endedAt: null }, now)).toBe(false);
    expect(looksLikeEcho(REPLY, null, now)).toBe(false);
  });

  it("honours overrides", () => {
    // Lowering the floor makes a caller's question look like echo -- the reason
    // the defaults are not lower.
    const question = "so you said the top 1% of earners hold 39% of the wealth, is that right?";
    expect(looksLikeEcho(question, speaking, now)).toBe(false);
    expect(looksLikeEcho(question, speaking, now, { threshold: 0.5 })).toBe(true);
  });
});
