/**
 * Phase 1 placeholder. The real voice demo + observability dashboard is Phase 4.
 * This intentionally renders static markup and makes no network calls.
 */
export default function App() {
  return (
    <main className="min-h-screen bg-cloudpso-bg text-slate-100 flex items-center justify-center p-8">
      <section className="max-w-xl w-full rounded-2xl bg-cloudpso-panel p-8 shadow-lg">
        <h1 className="text-2xl font-semibold text-cloudpso-accent">
          CloudPSO Voice Agent
        </h1>
        <p className="mt-3 text-slate-300">
          Frontend scaffold is in place. The live voice demo and observability
          dashboard arrive in Phase 4.
        </p>
        <p className="mt-4 text-sm text-slate-400">
          Backend endpoint:{" "}
          <code className="rounded bg-black/30 px-1.5 py-0.5">
            {import.meta.env.VITE_BACKEND_CORE_URL ?? "http://localhost:8000"}
          </code>
        </p>
      </section>
    </main>
  );
}
