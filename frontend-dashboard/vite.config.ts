/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Build + dev-server config, plus the Vitest project settings (Vitest reads the
// `test` block from this file when run via `vitest`).
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_BACKEND_CORE_URL ?? "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  preview: {
    host: "0.0.0.0",
    port: 5173,
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    // Playwright specs live in e2e/ and must not be collected by Vitest.
    exclude: ["e2e/**", "node_modules/**", "dist/**"],
    css: false,
  },
});
