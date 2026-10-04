import { defineConfig, devices } from "@playwright/test";
import { fileURLToPath } from "node:url";

/**
 * Browser end-to-end configuration.
 *
 * Ports are deliberately NON-default (18000/18001/15173 rather than
 * 8000/8001/5173). On a machine running Docker Desktop, a published container
 * can hold an IPv4-mapped IPv6 wildcard on a port such as :8000, so
 * `127.0.0.1:8000/health` answers 200 even when backend-core is not running —
 * Playwright's `reuseExistingServer` probe would then "reuse" a foreign service
 * and every request would go to the wrong process. Dedicated ports make the
 * probe trustworthy. `npm run e2e` builds the bundle against these ports, and
 * the CORS allow-list is set to match the E2E frontend origin.
 *
 * Chromium runs with a *fake* media device so the test has a real microphone
 * track with no hardware and no permission prompt. The flags that matter:
 *
 *  - `--use-file-for-fake-audio-capture=<speech.wav>` — Chromium's built-in fake
 *    device emits a beep whose speech runs are too short to trip the barge-in
 *    hysteresis; feeding it a real utterance makes the VAD -> barge-in path
 *    exercisable. We reuse backend-core's own test fixture.
 *  - `--disable-features=WebRtcHideLocalIpsWithMdns` — by default Chromium
 *    replaces host ICE candidates with mDNS `.local` names, which the AIORTC
 *    server (backend-core) cannot resolve; without this ICE never connects on
 *    loopback.
 *  - `--use-fake-ui-for-media-stream` — auto-grants the getUserMedia prompt.
 */

const BACKEND_PORT = 18000;
const AI_ENGINE_PORT = 18001;
const FRONTEND_PORT = 15173;

const BACKEND = `http://127.0.0.1:${BACKEND_PORT}`;
const AI_ENGINE = `http://127.0.0.1:${AI_ENGINE_PORT}`;
const FRONTEND = `http://127.0.0.1:${FRONTEND_PORT}`;
/** The origin the browser loads the dashboard from — backend CORS must allow it. */
const CORS_ORIGIN = FRONTEND;

const SPEECH_WAV = fileURLToPath(
  new URL("../backend-core/tests/data/speech_16k.wav", import.meta.url),
);

export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: FRONTEND,
    permissions: ["microphone"],
    trace: "retain-on-failure",
    launchOptions: {
      args: [
        "--use-fake-device-for-media-stream",
        "--use-fake-ui-for-media-stream",
        `--use-file-for-fake-audio-capture=${SPEECH_WAV}`,
        "--disable-features=WebRtcHideLocalIpsWithMdns",
        "--autoplay-policy=no-user-gesture-required",
      ],
    },
    ...devices["Desktop Chrome"],
  },
  webServer: [
    {
      command: `./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port ${BACKEND_PORT}`,
      cwd: "../backend-core",
      url: `${BACKEND}/health`,
      reuseExistingServer: true,
      timeout: 60_000,
      env: {
        BACKEND_CORE_GRPC_ENABLED: "false",
        BACKEND_CORE_VAD_BACKEND: "energy",
        BACKEND_CORE_LOG_LEVEL: "WARNING",
        CLOUDEPSO_CORS_ALLOW_ORIGINS: CORS_ORIGIN,
      },
    },
    {
      command: `./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port ${AI_ENGINE_PORT}`,
      cwd: "../ai-engine",
      url: `${AI_ENGINE}/health`,
      reuseExistingServer: true,
      timeout: 120_000,
      env: {
        AI_ENGINE_GRPC_ENABLED: "false",
        AI_ENGINE_LOG_LEVEL: "WARNING",
        CLOUDEPSO_CORS_ALLOW_ORIGINS: CORS_ORIGIN,
      },
    },
    {
      command: `npx vite preview --host 127.0.0.1 --port ${FRONTEND_PORT}`,
      url: FRONTEND,
      reuseExistingServer: true,
      timeout: 60_000,
    },
  ],
});
