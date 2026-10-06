import { expect, test } from "@playwright/test";

/**
 * Browser end-to-end for the streaming ("live") reply mode.
 *
 * This is the one spec that needs a working LLM: unlike the WebRTC specs, a
 * streamed reply cannot be faked at the transport layer without testing nothing.
 * It skips cleanly when Ollama is not answering, so the suite still runs on a
 * machine without a model.
 */
const AI_ENGINE = "http://127.0.0.1:18001";

test.beforeAll(async ({ request }) => {
  let ok = false;
  try {
    const response = await request.post(`${AI_ENGINE}/llm`, {
      data: { text: "hi" },
      timeout: 90_000,
    });
    ok = response.ok();
  } catch {
    ok = false;
  }
  test.skip(!ok, `no LLM available at ${AI_ENGINE}; start ollama to run this spec`);
});

test("live mode speaks before the reply is finished", async ({ page }) => {
  const name = `E2E streaming ${Date.now()}`;

  await page.goto("/");
  await page.getByTestId("agent-name").fill(name);
  await page
    .getByTestId("agent-prompt")
    .fill("You are a helpful assistant. Answer in two short spoken sentences, plain prose.");
  await page.getByTestId("save-agent").click();
  await page.getByTestId("agent-list").getByText(name).click();
  await expect(page.getByTestId("talk-to-agent")).toBeVisible();

  // Live is the default; assert it rather than assuming.
  await page.getByTestId("mode-stream").click();
  await expect(page.getByTestId("mode-stream")).toHaveAttribute("aria-checked", "true");

  await page.getByTestId("record-button").click();
  await page.waitForTimeout(4000); // let the fake mic produce a few seconds
  await page.getByTestId("record-button").click();

  const row = page.getByTestId("talk-to-agent").locator("li").first();

  // The reply must render incrementally. `first audio` is published the moment
  // the first chunk is scheduled -- i.e. while the reply is still growing --
  // which is precisely what separates this mode from the batch one.
  await expect(row).toContainText("first audio", { timeout: 90_000 });
  // A reply arrived, and it is captioned with its grounding state. This agent
  // has no documents, so "not grounded" is the *correct* caption -- the check
  // is that the caption exists at all.
  await expect(row).toContainText(/grounded/);
  await expect(row).toContainText("live");
});
