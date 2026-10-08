export type Tone = "ok" | "warn" | "err" | "idle" | "info";

const TONE_CLASSES: Record<Tone, string> = {
  ok: "bg-voicewright-ok/15 text-voicewright-ok border-voicewright-ok/40",
  warn: "bg-voicewright-warn/15 text-voicewright-warn border-voicewright-warn/40",
  err: "bg-voicewright-err/15 text-voicewright-err border-voicewright-err/40",
  idle: "bg-slate-500/10 text-voicewright-muted border-voicewright-line",
  info: "bg-voicewright-accent/15 text-voicewright-accent border-voicewright-accent/40",
};

const DOT_CLASSES: Record<Tone, string> = {
  ok: "bg-voicewright-ok",
  warn: "bg-voicewright-warn",
  err: "bg-voicewright-err",
  idle: "bg-voicewright-muted",
  info: "bg-voicewright-accent",
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
