/**
 * A dependency-free, responsive multi-series line chart.
 *
 * The dashboard needs live-updating time series but not a charting library:
 * pulling in Recharts/Chart.js would add hundreds of KB to the bundle for one
 * polyline. The SVG is drawn in a normalised `0 0 100 100` viewBox stretched to
 * the container, with `vector-effect="non-scaling-stroke"` so the line weight
 * stays crisp regardless of the stretch.
 */

export interface Series {
  key: string;
  label: string;
  color: string;
  points: number[];
}

export interface TimeSeriesChartProps {
  series: Series[];
  height?: number;
  /** Force the y-axis to include zero (right for counts). */
  zeroBased?: boolean;
  testId?: string;
  emptyHint?: string;
}

function pathFor(points: number[], min: number, max: number): string {
  if (points.length === 0) return "";
  const span = max - min || 1;
  const n = points.length;
  const coords = points.map((value, i) => {
    const x = n === 1 ? 0 : (i / (n - 1)) * 100;
    const y = 100 - ((value - min) / span) * 100;
    const clampedY = Math.max(0, Math.min(100, y));
    return `${x.toFixed(2)},${clampedY.toFixed(2)}`;
  });
  return `M${coords.join(" L")}`;
}

export function TimeSeriesChart({
  series,
  height = 120,
  zeroBased = false,
  testId,
  emptyHint = "waiting for samples…",
}: TimeSeriesChartProps) {
  const all = series.flatMap((s) => s.points).filter((v) => Number.isFinite(v));
  const hasData = all.length > 0 && all.some((v) => v !== 0);
  const rawMin = all.length ? Math.min(...all) : 0;
  const rawMax = all.length ? Math.max(...all) : 0;
  const min = zeroBased ? Math.min(0, rawMin) : rawMin;
  const max = rawMax === min ? rawMin + 1 : rawMax;

  return (
    <div data-testid={testId}>
      <div className="flex items-center gap-3 pb-1 text-[11px] text-voicewright-muted">
        {series.map((s) => (
          <span key={s.key} className="inline-flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: s.color }} />
            {s.label}
            <span className="tabular-nums text-slate-200">
              {Number.isFinite(s.points[s.points.length - 1])
                ? s.points[s.points.length - 1].toFixed(s.points[s.points.length - 1] < 10 ? 1 : 0)
                : "—"}
            </span>
          </span>
        ))}
        <span className="ml-auto tabular-nums">
          max {max.toFixed(max < 10 ? 1 : 0)}
        </span>
      </div>
      <div
        className="relative rounded-lg border border-voicewright-line bg-voicewright-panel2"
        style={{ height }}
      >
        {hasData ? (
          <svg
            className="h-full w-full"
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
            role="img"
            aria-label="time series"
          >
            {[25, 50, 75].map((y) => (
              <line
                key={y}
                x1={0}
                x2={100}
                y1={y}
                y2={y}
                stroke="#2a3352"
                strokeWidth={1}
                vectorEffect="non-scaling-stroke"
              />
            ))}
            {series.map((s) => (
              <path
                key={s.key}
                d={pathFor(s.points, min, max)}
                fill="none"
                stroke={s.color}
                strokeWidth={2}
                strokeLinejoin="round"
                strokeLinecap="round"
                vectorEffect="non-scaling-stroke"
              />
            ))}
          </svg>
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-voicewright-muted">
            {emptyHint}
          </div>
        )}
      </div>
    </div>
  );
}
