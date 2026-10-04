/**
 * A small, dependency-free parser for the Prometheus text exposition format,
 * plus the histogram helpers the dashboard needs (percentiles).
 *
 * The demo deliberately does *not* stand up a Prometheus server just to render
 * a few tiles: both services already expose the canonical text format at
 * `GET /metrics`, and parsing it directly keeps the dashboard a static bundle
 * with no extra runtime. The format is described at
 * https://prometheus.io/docs/instrumenting/exposition_formats/ and the subset
 * implemented here is exactly what `prometheus_client` emits.
 *
 * Supported:
 *   # HELP <name> <text>
 *   # TYPE <name> counter|gauge|histogram|summary|untyped
 *   <name>[{<label>="<value>",...}] <value> [<timestamp>]
 * with backslash escaping (`\\`, `\"`, `\n`) inside label values and the
 * `+Inf` / `-Inf` / `NaN` float spellings prometheus_client produces.
 */

export interface Sample {
  /** Metric name as it appears on the wire (`..._bucket`, `..._sum`, `..._count` included). */
  name: string;
  labels: Record<string, string>;
  value: number;
}

export interface MetricFamily {
  name: string;
  type?: string;
  help?: string;
  samples: Sample[];
}

export interface ParsedMetrics {
  families: Map<string, MetricFamily>;
  /** Flat list of every sample, in file order. */
  samples: Sample[];
  /** Metric families in the order their `# TYPE`/`# HELP` first appeared. */
  order: string[];
}

const HISTOGRAM_SUFFIXES = ["_bucket", "_sum", "_count", "_created"] as const;
const SUMMARY_SUFFIXES = ["_sum", "_count", "_created"] as const;

function parseValue(raw: string): number {
  switch (raw) {
    case "+Inf":
      return Number.POSITIVE_INFINITY;
    case "-Inf":
      return Number.NEGATIVE_INFINITY;
    case "NaN":
      return Number.NaN;
    default:
      return Number(raw);
  }
}

/** Parse the `{a="1",b="2"}` label block. `body` excludes the braces. */
function parseLabels(body: string): Record<string, string> {
  const labels: Record<string, string> = {};
  let i = 0;
  const n = body.length;
  while (i < n) {
    // name
    const nameStart = i;
    while (i < n && body[i] !== "=") i++;
    const name = body.slice(nameStart, i).trim();
    if (i >= n) break;
    i++; // skip '='
    if (body[i] !== '"') {
      // Malformed; skip to next comma to avoid an infinite loop.
      while (i < n && body[i] !== ",") i++;
      i++;
      continue;
    }
    i++; // skip opening quote
    let value = "";
    while (i < n) {
      const ch = body[i];
      if (ch === "\\" && i + 1 < n) {
        const next = body[i + 1];
        value += next === "n" ? "\n" : next;
        i += 2;
        continue;
      }
      if (ch === '"') {
        i++;
        break;
      }
      value += ch;
      i++;
    }
    if (name) labels[name] = value;
    // skip to ',' or end
    while (i < n && body[i] !== ",") i++;
    i++;
  }
  return labels;
}

/** Split a sample line into its name+labels part and its value part. */
function splitSampleLine(line: string): { head: string; rest: string } | null {
  // The value is the last whitespace-separated token (optionally followed by a
  // timestamp). Label values may contain spaces, so scan for the first space
  // that is *outside* braces.
  let depth = 0;
  let inString = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inString) {
      if (ch === "\\") i++;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === "{") depth++;
    else if (ch === "}") depth--;
    else if ((ch === " " || ch === "\t") && depth === 0) {
      return { head: line.slice(0, i), rest: line.slice(i + 1).trim() };
    }
  }
  return null;
}

export function parse(text: string): ParsedMetrics {
  const families = new Map<string, MetricFamily>();
  const order: string[] = [];
  const samples: Sample[] = [];

  const ensureFamily = (name: string): MetricFamily => {
    let family = families.get(name);
    if (!family) {
      family = { name, samples: [] };
      families.set(name, family);
      order.push(name);
    }
    return family;
  };

  // Names for which a `# TYPE` line has been seen, so histogram samples can be
  // attributed to the right family rather than a `_bucket`-suffixed one.
  const typed = new Set<string>();

  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();
    if (line.length === 0) continue;

    if (line.startsWith("#")) {
      const helpMatch = /^#\s+HELP\s+(\S+)\s?(.*)$/.exec(line);
      if (helpMatch) {
        ensureFamily(helpMatch[1]).help = helpMatch[2];
        continue;
      }
      const typeMatch = /^#\s+TYPE\s+(\S+)\s+(\S+)$/.exec(line);
      if (typeMatch) {
        ensureFamily(typeMatch[1]).type = typeMatch[2];
        typed.add(typeMatch[1]);
        continue;
      }
      continue; // other comments
    }

    const split = splitSampleLine(line);
    if (!split) continue;
    const { head, rest } = split;

    let name = head;
    let labels: Record<string, string> = {};
    const brace = head.indexOf("{");
    if (brace !== -1) {
      name = head.slice(0, brace);
      const end = head.lastIndexOf("}");
      labels = parseLabels(head.slice(brace + 1, end === -1 ? undefined : end));
    }

    const valueToken = rest.split(/\s+/)[0];
    const value = parseValue(valueToken);
    if (Number.isNaN(value) && valueToken !== "NaN") continue; // unparseable

    const sample: Sample = { name, labels, value };
    samples.push(sample);

    // Attribute the sample to its declared family (a histogram's `_bucket`
    // samples belong to the bare family name), else its own.
    let familyName = name;
    if (!typed.has(name)) {
      for (const suffix of [...HISTOGRAM_SUFFIXES, ...SUMMARY_SUFFIXES]) {
        if (name.endsWith(suffix)) {
          const base = name.slice(0, -suffix.length);
          if (typed.has(base)) {
            familyName = base;
            break;
          }
        }
      }
    }
    ensureFamily(familyName).samples.push(sample);
  }

  return { families, samples, order };
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

function labelsMatch(
  sampleLabels: Record<string, string>,
  wanted?: Record<string, string>,
): boolean {
  if (!wanted) return true;
  for (const key of Object.keys(wanted)) {
    if (sampleLabels[key] !== wanted[key]) return false;
  }
  return true;
}

/** Every sample whose name is exactly `name` (optionally filtered by labels). */
export function select(
  parsed: ParsedMetrics,
  name: string,
  labels?: Record<string, string>,
): Sample[] {
  return parsed.samples.filter((s) => s.name === name && labelsMatch(s.labels, labels));
}

/** Sum of matching samples; `0` when the series is absent. */
export function sum(parsed: ParsedMetrics, name: string, labels?: Record<string, string>): number {
  return select(parsed, name, labels).reduce((acc, s) => acc + s.value, 0);
}

/** Value of the first matching sample; `undefined` when absent. */
export function value(
  parsed: ParsedMetrics,
  name: string,
  labels?: Record<string, string>,
): number | undefined {
  return select(parsed, name, labels)[0]?.value;
}

/** Gauge/counter with no labels, read as a bare value (0 if absent). */
export function scalar(parsed: ParsedMetrics, name: string): number {
  return value(parsed, name) ?? 0;
}

export interface Histogram {
  /** Cumulative buckets, ascending by upper bound, `+Inf` last. */
  buckets: { le: number; cumulative: number }[];
  sum: number;
  count: number;
}

/**
 * Assemble a cumulative histogram from its `_bucket` / `_sum` / `_count`
 * samples, summing across any label dimensions not named in `labels`.
 */
export function histogram(
  parsed: ParsedMetrics,
  name: string,
  labels?: Record<string, string>,
): Histogram {
  const byLe = new Map<number, number>();
  for (const sample of select(parsed, `${name}_bucket`, labels)) {
    const leRaw = sample.labels.le;
    if (leRaw === undefined) continue;
    const le = leRaw === "+Inf" ? Number.POSITIVE_INFINITY : Number(leRaw);
    byLe.set(le, (byLe.get(le) ?? 0) + sample.value);
  }
  const buckets = [...byLe.entries()]
    .map(([le, cumulative]) => ({ le, cumulative }))
    .sort((a, b) => a.le - b.le);

  const sumSamples = select(parsed, `${name}_sum`, labels);
  const countSamples = select(parsed, `${name}_count`, labels);
  const count =
    countSamples.length > 0
      ? countSamples.reduce((acc, s) => acc + s.value, 0)
      : buckets.length > 0
        ? buckets[buckets.length - 1].cumulative
        : 0;
  return {
    buckets,
    sum: sumSamples.reduce((acc, s) => acc + s.value, 0),
    count,
  };
}

/**
 * Estimate the `q`-quantile (0..1) from a cumulative histogram using
 * Prometheus' `histogram_quantile` algorithm: locate the bucket the rank falls
 * in and linearly interpolate within its width. Extrapolates linearly above the
 * highest finite bucket when the rank lands in `+Inf`.
 *
 * Returns `NaN` when the histogram is empty (no observations).
 */
export function histogramQuantile(h: Histogram, q: number): number {
  const total = h.count || (h.buckets.length ? h.buckets[h.buckets.length - 1].cumulative : 0);
  if (total <= 0 || h.buckets.length === 0) return Number.NaN;
  if (q <= 0) return 0;
  if (q > 1) q = 1;

  const rank = q * total;
  let prevUpper = 0;
  let prevCumulative = 0;

  for (const bucket of h.buckets) {
    if (bucket.cumulative >= rank) {
      if (!Number.isFinite(bucket.le)) {
        // Rank lands in the +Inf bucket: the true value is above the highest
        // finite bound, which is the best estimate a histogram can offer.
        return prevUpper;
      }
      const width = bucket.cumulative - prevCumulative;
      if (width <= 0) return bucket.le;
      const frac = (rank - prevCumulative) / width;
      return prevUpper + frac * (bucket.le - prevUpper);
    }
    prevUpper = Number.isFinite(bucket.le) ? bucket.le : prevUpper;
    prevCumulative = bucket.cumulative;
  }
  return prevUpper;
}
