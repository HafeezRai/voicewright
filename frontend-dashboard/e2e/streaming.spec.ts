import { expect, test } from "@playwright/test";

/**
 * Browser end-to-end for hands-free call mode.
 *
 * The claim: a caller presses "Start call" once, talks, and the agent answers
 * on its own -- no stop button, no send button. Everything after the single
 * click in this spec is the product working by itself.
 *
 * This is the one spec that needs a working LLM: a streamed, turn-detected
 * reply cannot be faked at the transport layer without testing nothing. It
 * skips cleanly when Ollama is not answering, so the suite still runs on a
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

test("a call answers on its own, with no stop or send button", async ({ page }) => {
  const name = `E2E call ${Date.now()}`;

  await page.goto("/");
  await page.getByTestId("agent-name").fill(name);
  await page
    .getByTestId("agent-prompt")
    .fill("You are a helpful assistant. Answer in two short spoken sentences, plain prose.");
  await page.getByTestId("save-agent").click();
  await page.getByTestId("agent-list").getByText(name).click();
  await expect(page.getByTestId("talk-to-agent")).toBeVisible();

  // Call mode is the default, and it has no send button at all -- assert that,
  // because a stray stop button is exactly the regression being guarded here.
  await expect(page.getByTestId("mode-call")).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("record-button")).toHaveCount(0);

  await page.getByTestId("call-button").click();
  await expect(page.getByTestId("call-button")).toHaveText("End call");
  await expect(page.getByTestId("call-status")).toContainText(/Listening|Speaking/);

  // From here on the page is never touched. A completed turn can only appear if
  // silence detection ended the caller's turn by itself.
  const row = page.getByTestId("talk-to-agent").locator("li").first();
  await expect(row).toContainText("first audio", { timeout: 90_000 });
  await expect(row).toContainText(/grounded/);
  await expect(page.getByTestId("call-button")).toHaveText("End call");

  await page.getByTestId("call-button").click();
  await expect(page.getByTestId("call-button")).toHaveText("Start call");
});
