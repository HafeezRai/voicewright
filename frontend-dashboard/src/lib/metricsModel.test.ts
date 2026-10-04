import { describe, expect, it } from "vitest";

import aiEngineFixture from "../test/fixtures/ai-engine-metrics.txt?raw";
import backendFixture from "../test/fixtures/backend-core-metrics.txt?raw";
import { labelledValue, offlineService, parseService, summarise } from "./metricsModel";

describe("summarise against the real captured fixtures", () => {
  const snapshot = summarise(parseService(backendFixture), parseService(aiEngineFixture));

  it("reports both services online", () => {
    expect(snapshot.backend.online).toBe(true);
    expect(snapshot.aiEngine.online).toBe(true);
  });

  it("pulls the WebRTC/barge-in figures from backend-core", () => {
    expect(snapshot.activeSessions).toBe(1);
    expect(snapshot.totalSessions).toBe(1);
    expect(snapshot.wsConnections).toBe(1);
    expect(snapshot.errorCount).toBe(0);
  });

  it("pulls the request counts from ai-engine", () => {
    expect(snapshot.nluRequests).toBe(2);
    expect(snapshot.ttsRequests).toBe(1);
    // No TTS was cancelled, and STT was never called, in the capture.
    expect(snapshot.ttsCancelled).toBe(0);
    expect(snapshot.sttRequests).toBe(0);
  });

  it("yields a usable NLU percentile and a graceful NaN where there is no data", () => {
    expect(snapshot.nluP50).toBeGreaterThan(0);
    expect(snapshot.nluP95).toBeGreaterThanOrEqual(snapshot.nluP50);
    // STT histogram has no observations -> percentile is NaN, not a crash.
    expect(snapshot.sttP95).toBeNaN();
  });

  it("exposes labelled values for arbitrary series lookups", () => {
    const okCount = labelledValue(parseService(aiEngineFixture), "cloudpso_ai_nlu_requests_total", {
      status: "ok",
    });
    expect(okCount).toBe(2);
  });
});

describe("offline services degrade gracefully", () => {
  const snapshot = summarise(offlineService("ECONNREFUSED"), offlineService("ECONNREFUSED"));

  it("marks both offline and zeroes every numeric field", () => {
    expect(snapshot.backend.online).toBe(false);
    expect(snapshot.backend.error).toBe("ECONNREFUSED");
    expect(snapshot.aiEngine.online).toBe(false);
    for (const key of [
      "activeSessions",
      "bargeInEvents",
      "sttRequests",
      "nluRequests",
      "ttsRequests",
      "errorCount",
    ] as const) {
      expect(snapshot[key]).toBe(0);
    }
  });

  it("reports NaN percentiles rather than throwing", () => {
    expect(snapshot.sttP50).toBeNaN();
    expect(snapshot.pipelineP95).toBeNaN();
  });
});
