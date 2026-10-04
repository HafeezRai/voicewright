/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // CloudPSO dashboard palette (Phase 4).
        cloudpso: {
          bg: "#0b1020",
          panel: "#151b30",
          panel2: "#1b2340",
          line: "#2a3352",
          accent: "#4f8cff",
          ok: "#34d399",
          warn: "#fbbf24",
          err: "#f87171",
          muted: "#93a1c4",
        },
      },
      keyframes: {
        "pulse-ring": {
          "0%, 100%": { opacity: "1", transform: "scale(1)" },
          "50%": { opacity: "0.4", transform: "scale(1.06)" },
        },
      },
      animation: {
        "pulse-ring": "pulse-ring 1s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
