import { describe, expect, it } from "vitest";

import aiEngineFixture from "../test/fixtures/ai-engine-metrics.txt?raw";
import backendFixture from "../test/fixtures/backend-core-metrics.txt?raw";
import {
  histogram,
  histogramQuantile,
  parse,
  scalar,
  select,
  sum,
  value,
} from "./prometheus";

describe("prometheus text parser", () => {
  it("parses unlabelled gauges and counters", () => {
    const parsed = parse(
      [
        "# HELP up Whether the target is up",
        "# TYPE up gauge",
        "up 1.0",
        "# TYPE hits_total counter",
        "hits_total 42",
      ].join("\n"),
    );
    expect(scalar(parsed, "up")).toBe(1);
    expect(scalar(parsed, "hits_total")).toBe(42);
    expect(parsed.families.get("up")?.type).toBe("gauge");
    expect(parsed.families.get("up")?.help).toBe("Whether the target is up");
  });

  it("parses labels, including escaped characters inside values", () => {
    const parsed = parse('http_requests_total{method="GET",path="/a\\"b",note="line\\n1"} 7');
    const [sample] = select(parsed, "http_requests_total");
    expect(sample.labels.method).toBe("GET");
    expect(sample.labels.path).toBe('/a"b');
    expect(sample.labels.note).toBe("line\n1");
    expect(sample.value).toBe(7);
  });

  it("sums across label sets and filters by label", () => {
    const parsed = parse(
      [
        'req_total{status="ok"} 10',
        'req_total{status="error"} 2',
        'req_total{status="ok"} 5',
      ].join("\n"),
    );
    expect(sum(parsed, "req_total")).toBe(17);
    expect(sum(parsed, "req_total", { status: "ok" })).toBe(15);
    expect(sum(parsed, "req_total", { status: "error" })).toBe(2);
    expect(value(parsed, "req_total", { status: "missing" })).toBeUndefined();
  });

  it("groups histogram buckets under the family and keeps +Inf", () => {
    const parsed = parse(
      [
        "# TYPE latency_seconds histogram",
        'latency_seconds_bucket{le="0.1"} 5',
        'latency_seconds_bucket{le="0.5"} 9',
        'latency_seconds_bucket{le="+Inf"} 10',
        "latency_seconds_sum 1.7",
        "latency_seconds_count 10",
      ].join("\n"),
    );
    const family = parsed.families.get("latency_seconds");
    expect(family?.samples).toHaveLength(5);

    const h = histogram(parsed, "latency_seconds");
    expect(h.buckets.map((b) => b.le)).toEqual([0.1, 0.5, Number.POSITIVE_INFINITY]);
    expect(h.count).toBe(10);
    expect(h.sum).toBeCloseTo(1.7);
  });

  it("merges histogram label dimensions that are not requested", () => {
    const parsed = parse(
      [
        'lat_bucket{backend="a",le="1"} 3',
        'lat_bucket{backend="b",le="1"} 4',
        'lat_bucket{backend="a",le="+Inf"} 5',
        'lat_bucket{backend="b",le="+Inf"} 6',
        "lat_count 11",
      ].join("\n"),
    );
    const h = histogram(parsed, "lat");
    expect(h.buckets[0]).toEqual({ le: 1, cumulative: 7 });
    expect(h.buckets[1]).toEqual({ le: Number.POSITIVE_INFINITY, cumulative: 11 });
  });

  it("computes quantiles by linear interpolation", () => {
    const parsed = parse(
      [
        "# TYPE h histogram",
        'h_bucket{le="1"} 0',
        'h_bucket{le="2"} 5',
        'h_bucket{le="3"} 10',
        'h_bucket{le="+Inf"} 10',
        "h_count 10",
      ].join("\n"),
    );
    const h = histogram(parsed, "h");
    expect(histogramQuantile(h, 0.5)).toBeCloseTo(2.0, 5);
    expect(histogramQuantile(h, 0.95)).toBeCloseTo(2.9, 5);
    expect(histogramQuantile(h, 1)).toBeCloseTo(3.0, 5);
  });

  it("returns NaN for an empty histogram", () => {
    const parsed = parse("# TYPE h histogram\nh_count 0");
    expect(histogramQuantile(histogram(parsed, "h"), 0.5)).toBeNaN();
  });
});

// ---------------------------------------------------------------------------
// Real fixtures: captured from the running services (`GET /metrics`) while a
// live signalling session and real NLU/TTS calls were in flight.
// ---------------------------------------------------------------------------

describe("parsing real backend-core /metrics output", () => {
  const parsed = parse(backendFixture);

  it("reads the live session gauges captured mid-session", () => {
    expect(scalar(parsed, "cloudpso_active_webrtc_sessions")).toBe(1);
    expect(sum(parsed, "cloudpso_webrtc_sessions_total")).toBe(1);
    expect(scalar(parsed, "cloudpso_signaling_connections")).toBe(1);
  });

  it("exposes every metric backend-core promises", () => {
    for (const name of [
      "cloudpso_active_webrtc_sessions",
      "cloudpso_webrtc_sessions_total",
      "cloudpso_bargein_events_total",
      "cloudpso_vad_frames_processed_total",
      "cloudpso_signaling_connections",
    ]) {
      expect(parsed.families.has(name), `missing ${name}`).toBe(true);
    }
  });
});

describe("parsing real ai-engine /metrics output", () => {
  const parsed = parse(aiEngineFixture);

  it("counts the real NLU/TTS requests made against the service", () => {
    expect(sum(parsed, "cloudpso_ai_nlu_requests_total", { status: "ok" })).toBe(2);
    expect(sum(parsed, "cloudpso_ai_tts_requests_total", { status: "ok" })).toBe(1);
  });

  it("produces finite latency percentiles from the real histograms", () => {
    const h = histogram(parsed, "cloudpso_ai_nlu_latency_seconds", { backend: "hf" });
    expect(h.count).toBe(2);
    const p50 = histogramQuantile(h, 0.5);
    const p95 = histogramQuantile(h, 0.95);
    expect(p50).toBeGreaterThan(0);
    expect(p95).toBeGreaterThanOrEqual(p50);
    expect(p95).toBeLessThanOrEqual(10); // highest finite bucket is 10 s
  });
});
