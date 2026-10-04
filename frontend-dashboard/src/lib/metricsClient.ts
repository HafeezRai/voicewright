/** Fetch helpers for the Prometheus text exposition endpoints. */

export async function fetchMetricsText(url: string, signal?: AbortSignal): Promise<string> {
  let response: Response;
  try {
    response = await fetch(url, { signal, headers: { Accept: "text/plain" } });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new Error(`unreachable (${error instanceof Error ? error.message : String(error)})`);
  }
  if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`.trim());
  return response.text();
}
