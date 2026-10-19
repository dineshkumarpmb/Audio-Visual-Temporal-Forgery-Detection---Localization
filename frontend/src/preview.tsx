/**
 * Dev-only visual check (`npm run dev`), not part of the production build (vite builds
 * index.html only):
 *
 *   /preview.html?fixture=dev_004623&theme=dark&width=390   recorded API response
 *   /preview.html?job=<job_id>                               a live job from `make api`
 *
 * The live mode shows keyframes (they live with the job on the server); fixtures cannot.
 */
import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

import { getJob, getModelInfo } from "./api/client";
import type { Job, ModelInfo } from "./api/types";
import { ResultView } from "./components/ResultView";
import { ErrorPanel } from "./components/StatusPanels";
import "./index.css";
import { FIXTURES } from "./test/fixtures";

const params = new URLSearchParams(location.search);
const theme = params.get("theme");
if (theme) document.documentElement.dataset.theme = theme;
const error = params.get("error");
const liveJob = params.get("job");
// Headless browsers on Windows will not go below ~500 px; ?width=390 frames a phone instead.
const width = params.get("width");

function Preview() {
  const [job, setJob] = useState<Job | null>(
    liveJob ? null : FIXTURES[params.get("fixture") ?? "dev_004623"],
  );
  const [info, setInfo] = useState<ModelInfo | null>(null);
  useEffect(() => {
    getModelInfo().then(setInfo, () => setInfo(null));
    if (liveJob) getJob(liveJob).then(setJob, () => setJob(null));
  }, []);
  if (error) return <ErrorPanel code={error} onReset={() => {}} />;
  if (!job?.result) return <p className="text-ink-2">Loading…</p>;
  return <ResultView job={job} videoUrl={null} info={info} segmentThreshold={0.2} onReset={() => {}} />;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <div className="min-h-screen bg-page font-sans text-ink">
      <main
        className="mx-auto max-w-4xl px-4 py-8"
        style={width ? { maxWidth: Number(width), outline: "1px dashed #898781" } : undefined}
      >
        <Preview />
      </main>
    </div>
  </StrictMode>,
);
