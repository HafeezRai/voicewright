import type { Tone } from "./StatusPill";

const VALUE_TONE: Record<Tone, string> = {
  ok: "text-cloudpso-ok",
  warn: "text-cloudpso-warn",
  err: "text-cloudpso-err",
  idle: "text-slate-100",
  info: "text-cloudpso-accent",
};

/** Format a number for a tile, tolerating NaN/undefined from missing series. */
export function formatMetric(value: number | undefined, digits = 0): string {
  if (value === undefined || Number.isNaN(value)) return "—";
  if (!Number.isFinite(value)) return "∞";
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function MetricTile({
  label,
  value,
  unit,
  sub,
  tone = "idle",
  testId,
}: {
  label: string;
  value: string;
  unit?: string;
  sub?: string;
  tone?: Tone;
  testId?: string;
}) {
  return (
    <div
      data-testid={testId}
      className="rounded-xl border border-cloudpso-line bg-cloudpso-panel2 px-4 py-3"
    >
      <div className="text-[11px] uppercase tracking-wide text-cloudpso-muted">{label}</div>
      <div className={`mt-1 text-2xl font-semibold tabular-nums ${VALUE_TONE[tone]}`}>
        {value}
        {unit ? <span className="ml-1 text-sm font-normal text-cloudpso-muted">{unit}</span> : null}
      </div>
      {sub ? <div className="mt-0.5 text-xs text-cloudpso-muted">{sub}</div> : null}
    </div>
  );
}
