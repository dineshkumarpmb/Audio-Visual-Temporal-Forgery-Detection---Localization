/**
 * Dev-only visual check (`npm run dev`, then /preview.html?fixture=dev_004623&theme=dark):
 * the real result view on a real API response from src/test/fixtures, no upload needed.
 * Not part of the production build (vite builds index.html only).
 */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { ErrorPanel } from "./components/StatusPanels";
import { ResultView } from "./components/ResultView";
import "./index.css";
import { FIXTURES } from "./test/fixtures";

const params = new URLSearchParams(location.search);
const theme = params.get("theme");
if (theme) document.documentElement.dataset.theme = theme;
const error = params.get("error");
// Headless browsers on Windows will not go below ~500 px; ?width=390 frames a phone instead.
const width = params.get("width");
const job = FIXTURES[params.get("fixture") ?? "dev_004623"];

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <div className="min-h-screen bg-page font-sans text-ink">
      <main className="mx-auto max-w-4xl px-4 py-8" style={width ? { maxWidth: Number(width), outline: "1px dashed #898781" } : undefined}>
        {error ? (
          <ErrorPanel code={error} onReset={() => {}} />
        ) : (
          <ResultView job={job} videoUrl={null} segmentThreshold={0.2} onReset={() => {}} />
        )}
      </main>
    </div>
  </StrictMode>,
);
