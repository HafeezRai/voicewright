/** Vitest global setup: jest-dom matchers and the browser stubs jsdom lacks. */

import "@testing-library/jest-dom/vitest";

import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

// Unmount anything a test rendered so DOM state never leaks between cases.
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

// jsdom implements neither of these; components call them defensively.
if (!window.matchMedia) {
  window.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => undefined,
    removeListener: () => undefined,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia;
}

if (!HTMLMediaElement.prototype.play) {
  HTMLMediaElement.prototype.play = () => Promise.resolve();
} else {
  // jsdom's play() throws "Not implemented"; silence it for the demo panels.
  HTMLMediaElement.prototype.play = () => Promise.resolve();
}
