export type Tone = "ok" | "warn" | "err" | "idle" | "info";

const TONE_CLASSES: Record<Tone, string> = {
  ok: "bg-cloudpso-ok/15 text-cloudpso-ok border-cloudpso-ok/40",
  warn: "bg-cloudpso-warn/15 text-cloudpso-warn border-cloudpso-warn/40",
  err: "bg-cloudpso-err/15 text-cloudpso-err border-cloudpso-err/40",
  idle: "bg-slate-500/10 text-cloudpso-muted border-cloudpso-line",
  info: "bg-cloudpso-accent/15 text-cloudpso-accent border-cloudpso-accent/40",
};

const DOT_CLASSES: Record<Tone, string> = {
  ok: "bg-cloudpso-ok",
  warn: "bg-cloudpso-warn",
  err: "bg-cloudpso-err",
  idle: "bg-cloudpso-muted",
  info: "bg-cloudpso-accent",
};

export function StatusPill({
  label,
  tone = "idle",
  pulse = false,
  testId,
}: {
  label: string;
  tone?: Tone;
  pulse?: boolean;
  testId?: string;
}) {
  return (
    <span
      data-testid={testId}
      className={`inline-flex items-center gap-2 rounded-full border px-2.5 py-1 text-xs font-medium ${TONE_CLASSES[tone]}`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${DOT_CLASSES[tone]} ${pulse ? "animate-pulse-ring" : ""}`}
      />
      {label}
    </span>
  );
}
