import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Phase 1 skeleton: build config only, plus a dev proxy so the dashboard can
// reach backend-core without CORS once Phase 4 wires up real calls.
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
});
