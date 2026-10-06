import { expect, test, type Page } from "@playwright/test";

/**
 * The headline Phase 4 verification: a real browser captures the microphone,
 * negotiates WebRTC against backend-core's AIORTC gateway over the real
 * `/ws/signaling` protocol, and reaches a connected ICE state.
 *
 * The fake media device produces a periodic tone, so the second test can also
 * drive the VAD -> barge-in path end to end.
 */

interface VoiceHook {
  status: string;
  ice: string;
  sessionId: string | null;
  bargeInCount: number;
}

const hook = (page: Page): Promise<VoiceHook> =>
  page.evaluate(() => (window as unknown as { __cloudpsoVoice: VoiceHook }).__cloudpsoVoice);

test("browser mic negotiates WebRTC with backend-core and reaches connected", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });

  await page.goto("/");
  // The app opens on the agent builder; the WebRTC session is its own view.
  await page.getByTestId("tab-session").click();
  await expect(page.getByTestId("voice-status")).toHaveText("Idle");

  // The dashboard reaches BOTH services cross-origin from the browser — this is
  // the real proof that the CORS middleware works, not just curl.
  await expect(page.getByTestId("svc-online")).toHaveCount(2);

  await page.getByTestId("start-call").click();

  // The signalling handshake must complete: a session id appears.
  await expect
    .poll(async () => (await hook(page)).sessionId, { timeout: 30_000 })
    .toBeTruthy();
  await expect(page.getByTestId("session-id")).not.toHaveText("—");
  await expect(page.getByTestId("voice-error")).toHaveCount(0);

  // The answer really arrived over the socket.
  await expect(page.getByText(/answer received/)).toBeVisible();

  // And media negotiated: ICE reaches connected/completed.
  await expect
    .poll(async () => (await hook(page)).ice, { timeout: 30_000 })
    .toMatch(/connected|completed/);
  await expect(page.getByTestId("voice-status")).toHaveText(/Connected|Listening/);
  await expect(page.getByTestId("ice-status")).toHaveText(/connected|completed/);

  // Teardown must release everything and settle on Closed.
  await page.getByTestId("stop-call").click();
  await expect(page.getByTestId("voice-status")).toHaveText("Closed");

  expect(errors, `console errors: ${errors.join(" | ")}`).toHaveLength(0);
});

test("browser speech interrupts the agent (barge-in over the media path)", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("tab-session").click();
  await page.getByTestId("start-call").click();

  await expect
    .poll(async () => (await hook(page)).ice, { timeout: 30_000 })
    .toMatch(/connected|completed/);

  // Put the agent "in playback" so backend-core arms barge-in, then let the
  // fake microphone's tone trip the VAD.
  await page.getByTestId("toggle-agent-speaking").click();
  await expect(page.getByTestId("voice-status")).toHaveText("Agent speaking");

  await expect
    .poll(async () => (await hook(page)).bargeInCount, { timeout: 30_000 })
    .toBeGreaterThanOrEqual(1);

  await expect(page.getByTestId("barge-in-indicator")).toHaveAttribute("data-active", "true");
  await expect(page.getByText(/barge-in \(/)).toBeVisible();

  await page.getByTestId("stop-call").click();
});
