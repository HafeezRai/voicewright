import { expect, test } from "@playwright/test";

/**
 * Browser end-to-end for the agent builder.
 *
 * The claim this proves is the product claim: an agent is configuration you
 * type into a form, not code someone ships. It drives the real UI against the
 * real ai-engine (started by playwright.config.ts, pointed at a throwaway agent
 * DB so it never touches a developer's agents).
 */
test("an agent can be created, grounded and deleted entirely from the UI", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));

  const name = `E2E concierge ${Date.now()}`;

  // Deletion is gated behind window.confirm. Stub it at page-init rather than
  // racing a dialog listener against the click: a dialog that gets dismissed
  // returns false, the handler early-returns, and the test then fails for a
  // reason that has nothing to do with the code under test.
  await page.addInitScript(() => {
    window.confirm = () => true;
  });

  await page.goto("/");
  await expect(page.getByTestId("agent-builder")).toBeVisible();

  // --- create ---
  await page.getByTestId("agent-name").fill(name);
  await page.getByTestId("agent-role").fill("hotel concierge");
  await page
    .getByTestId("agent-prompt")
    .fill("You are a warm hotel concierge. Reply in one short spoken sentence, plain prose.");
  await page.getByTestId("save-agent").click();

  await expect(page.getByTestId("agent-list")).toContainText(name);

  // --- select it: knowledge + talk panels appear, and it starts ungrounded ---
  await page.getByTestId("agent-list").getByText(name).click();
  await expect(page.getByTestId("knowledge-panel")).toBeVisible();
  await expect(page.getByTestId("talk-to-agent")).toBeVisible();
  await expect(page.getByTestId("knowledge-count")).toHaveText(/ungrounded/i);

  // --- ground it with a document ---
  await page.getByLabel("Document title").fill("Weekend policy");
  await page
    .getByLabel("Document text")
    .fill(
      "Clinic policy: Saturday appointments are reserved exclusively for existing patients. New patients may only book Monday to Friday.",
    );
  await page.getByTestId("add-text-document").click();

  // The agent is no longer ungrounded — retrieval has something to find now.
  await expect(page.getByTestId("knowledge-count")).toHaveText(/1 document(?!s)/i);
  await expect(page.getByTestId("knowledge-panel")).toContainText("Weekend policy");

  // --- delete ---
  await page
    .getByTestId("agent-list")
    .locator("li", { hasText: name })
    .getByRole("button", { name: "Delete" })
    .click();

  // Assert on the agent, NOT on `agent-list`. When the last agent goes the
  // component stops rendering the <ul> at all, so a locator scoped to it
  // disappears and the assertion would hang for its full timeout instead of
  // passing — a failure that looks like a product bug and is not one.
  await expect(page.getByText(name)).toHaveCount(0);

  expect(errors, `console errors: ${errors.join(" | ")}`).toHaveLength(0);
});
