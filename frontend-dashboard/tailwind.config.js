/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Placeholder brand tokens; Phase 4 replaces with the real palette.
        cloudpso: {
          bg: "#0b1020",
          panel: "#151b30",
          accent: "#4f8cff",
        },
      },
    },
  },
  plugins: [],
};
