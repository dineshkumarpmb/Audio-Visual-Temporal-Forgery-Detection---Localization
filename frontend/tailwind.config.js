/** @type {import('tailwindcss').Config} */
// Colours are roles backed by CSS variables (src/index.css), so light and dark swap in one
// place. Status colours follow the dataviz reference palette and are never reused as series.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  // ScoreChart derives its hover-dot class from the line class at runtime.
  safelist: ["fill-accent", "fill-series-audio", "fill-series-visual"],
  theme: {
    extend: {
      colors: {
        page: "var(--page)",
        surface: "var(--surface)",
        ink: "var(--ink)",
        "ink-2": "var(--ink-2)",
        muted: "var(--muted)",
        grid: "var(--grid)",
        axis: "var(--axis)",
        border: "var(--border)",
        accent: "var(--accent)",
        fake: "var(--fake)",
        "fake-wash": "var(--fake-wash)",
        real: "var(--real)",
        warn: "var(--warn)",
        track: "var(--track)",
        "series-audio": "var(--series-audio)",
        "series-visual": "var(--series-visual)",
      },
      fontFamily: {
        sans: ["system-ui", "-apple-system", '"Segoe UI"', "sans-serif"],
      },
    },
  },
  plugins: [],
};
