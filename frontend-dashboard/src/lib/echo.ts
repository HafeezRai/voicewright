/**
 * Rejecting the agent's own voice.
 *
 * When the agent speaks through the speakers, some of it lands back in the
 * microphone. Browser echo cancellation removes most of it, but never all of
 * it -- and what survives gets transcribed, arrives looking exactly like a user
 * utterance, and the agent answers its own words. On open speakers it is the
 * single most confusing thing that happens in a demo.
 *
 * This is the cheap, reliable half of the fix: after speech-to-text, compare
 * what was "heard" against what the agent had just said. Echo transcribes as
 * the agent's own words, so the overlap is high; a person asking a question
 * almost never reproduces the reply word for word.
 *
 * It is deliberately conservative -- see `looksLikeEcho` -- because the failure
 * modes are asymmetric. Missing some echo means one odd turn; discarding a real
 * utterance means the agent ignores the person, which is much worse.
 */

/**
 * Lowercase word tokens with punctuation removed.
 *
 * Uses a Unicode-aware strip so accented and non-Latin text is not silently
 * deleted into nothing.
 */
export function normalizeWords(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter(Boolean);
}

/**
 * How much of `transcript` the agent had just said, 0..1.
 *
 * Containment, not symmetric similarity: echo is often only the tail of a
 * reply (the rest was cancelled, or ran before the mic opened), so every word
 * the transcript contains can be the agent's even when most of the agent's
 * sentence is missing. Jaccard would mark that as a poor match and let it
 * through.
 */
export function echoScore(transcript: string, spoken: string): number {
  const heard = normalizeWords(transcript);
  if (heard.length === 0) return 0;
  const said = new Set(normalizeWords(spoken));
  let hits = 0;
  for (const word of heard) if (said.has(word)) hits += 1;
  return hits / heard.length;
}

export interface AgentSpeech {
  text: string;
  /** When the agent's audio stopped playing, or null if still speaking. */
  endedAt: number | null;
}

export interface EchoOptions {
  /** Fraction of the transcript that must be the agent's own words. */
  threshold?: number;
  /**
   * Transcripts shorter than this are never judged.
   *
   * This does more work than the threshold does. Measured against a real reply:
   *
   *   echo of the reply      0.94   15 words
   *   echo of the tail       0.88    8 words
   *   caller quoting back    0.63   15 words
   *   caller short quote     1.00    4 words   <-- indistinguishable by score
   *
   * A four-word question made of the agent's own words -- "39% of the
   * wealth?" -- scores a *perfect* 1.0, higher than real echo. No threshold
   * separates them, because at that length there is nothing to separate: the
   * words really are all the agent's. Length is the only signal left, so short
   * utterances are left alone and the occasional short echo is tolerated.
   */
  minWords?: number;
  /** How long after the agent stops speaking echo can still arrive. */
  windowMs?: number;
}

export const ECHO_DEFAULTS = { threshold: 0.85, minWords: 5, windowMs: 1500 };

/**
 * Whether a transcript is probably the agent hearing itself.
 *
 * Three guards, all of which must hold, because the cost of a false positive
 * is the agent ignoring its caller:
 *
 *  - **Time.** Echo only exists while, or just after, the agent was speaking.
 *    A perfect word match thirty seconds later is a person quoting the agent
 *    back at it, and they deserve an answer.
 *  - **Length.** A two-word transcript ("yes", "okay") is contained in almost
 *    anything. Short utterances are never judged.
 *  - **Overlap.** Nearly every word must be one the agent just used. A caller
 *    repeating the agent's phrasing in a question -- "you said Saturdays are
 *    closed?" -- shares plenty of words, but not *only* the agent's.
 */
export function looksLikeEcho(
  transcript: string,
  speech: AgentSpeech | null,
  now: number,
  options: EchoOptions = {},
): boolean {
  const { threshold, minWords, windowMs } = { ...ECHO_DEFAULTS, ...options };
  if (!speech || !speech.text) return false;

  // `endedAt === null` means the agent is still speaking right now.
  if (speech.endedAt !== null && now - speech.endedAt > windowMs) return false;

  if (normalizeWords(transcript).length < minWords) return false;

  return echoScore(transcript, speech.text) >= threshold;
}
