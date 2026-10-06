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

  it("carries the renamed LLM series and none of the retired NLU ones", () => {
    // The worst failure mode here is silent: a fixture left on the old names
    // makes the dashboard show 0 requests and NaN latencies with no error, and
    // every other assertion still passes. Pin the names directly.
    expect(aiEngineFixture).toContain("cloudpso_ai_llm_requests_total");
    expect(aiEngineFixture).toContain("cloudpso_ai_llm_latency_seconds");
    expect(aiEngineFixture).not.toContain("cloudpso_ai_nlu");
  });

  it("pulls the WebRTC/barge-in figures from backend-core", () => {
    expect(snapshot.activeSessions).toBe(1);
    expect(snapshot.totalSessions).toBe(1);
    expect(snapshot.wsConnections).toBe(1);
    expect(snapshot.errorCount).toBe(0);
  });

  it("pulls the request counts from ai-engine", () => {
    expect(snapshot.llmRequests).toBe(2);
    expect(snapshot.ttsRequests).toBe(1);
    // No TTS was cancelled, and STT was never called, in the capture.
    expect(snapshot.ttsCancelled).toBe(0);
    expect(snapshot.sttRequests).toBe(0);
  });

  it("does not count _created metadata or benign statuses as errors", () => {
    // Regression: prometheus_client exports a `*_created` series per counter
    // whose value is a Unix timestamp. Counting every sample with a non-"ok"
    // status picked one of those up and reported ~1.8 billion errors on a
    // healthy system. `empty` (retrieval found nothing) and `cancelled` (the
    // user barged in) are outcomes, not failures.
    const text = [
      '# TYPE cloudpso_ai_knowledge_retrievals_total counter',
      'cloudpso_ai_knowledge_retrievals_total{status="ok"} 3.0',
      'cloudpso_ai_knowledge_retrievals_total{status="empty"} 2.0',
      'cloudpso_ai_knowledge_retrievals_created{status="empty"} 1.7912637259712079e+09',
      'cloudpso_ai_pipeline_requests_total{status="cancelled"} 4.0',
      'cloudpso_ai_pipeline_requests_total{status="fallback"} 5.0',
      'cloudpso_ai_pipeline_requests_total{status="error"} 2.0',
    ].join("\n");

    const empty = parseService("");
    // Only `error` (2) and `fallback` (5) are real failures.
    expect(summarise(parseService(text), empty).errorCount).toBe(7);

    // The timestamp series alone would have produced a nine-digit "error" count.
    expect(
      summarise(
        parseService('cloudpso_ai_knowledge_retrievals_created{status="empty"} 1.7912637259712079e+09'),
        empty,
      ).errorCount,
    ).toBe(0);
  });

  it("yields a usable LLM percentile and a graceful NaN where there is no data", () => {
    expect(snapshot.llmP50).toBeGreaterThan(0);
    expect(snapshot.llmP95).toBeGreaterThanOrEqual(snapshot.llmP50);
    // STT histogram has no observations -> percentile is NaN, not a crash.
    expect(snapshot.sttP95).toBeNaN();
  });

  it("exposes labelled values for arbitrary series lookups", () => {
    const okCount = labelledValue(parseService(aiEngineFixture), "cloudpso_ai_llm_requests_total", {
      provider: "ollama",
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
      "llmRequests",
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
