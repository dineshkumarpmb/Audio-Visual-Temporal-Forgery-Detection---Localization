import { useState } from "react";

import { ApiError, fetchReport, reportUrl } from "../api/client";
import type { Job } from "../api/types";
import { friendlyError } from "../lib/errors";

function save(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Not on the next tick: Firefox and Safari can cancel a large download whose URL is gone.
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}

const stem = (job: Job) => job.filename.replace(/\.[^.]+$/, "");

export function downloadJson(job: Job): void {
  save(new Blob([JSON.stringify(job, null, 2)], { type: "application/json" }), `${stem(job)}.analysis.json`);
}

/** One row per frame: every per-frame trace the API returned, ready for a spreadsheet. */
export function framesCsv(job: Job): string {
  const r = job.result!;
  const cols: [string, (i: number) => string][] = [
    ["frame", (i) => String(i)],
    ["time_s", (i) => (i / r.fps).toFixed(3)],
    ["score", (i) => r.frame_scores[i].toFixed(5)],
  ];
  const optional: [string, number[] | null | undefined][] = [
    ["voice_only", r.audio_only_scores],
    ["face_only", r.visual_only_scores],
    ["boundary_start", r.boundary_start_scores],
    ["boundary_end", r.boundary_end_scores],
    ["face_found", r.face_found],
    ["attention", r.attention],
  ];
  for (const [name, arr] of optional) {
    if (arr) cols.push([name, (i) => String(arr[i])]);
  }
  const inSegment = (i: number) =>
    r.segments.some((s) => i / r.fps >= s.start_s - 1e-9 && i / r.fps < s.end_s - 1e-9) ? "1" : "0";
  cols.push(["in_segment", inSegment]);
  const lines = [cols.map(([n]) => n).join(",")];
  for (let i = 0; i < r.frame_scores.length; i++) lines.push(cols.map(([, f]) => f(i)).join(","));
  return lines.join("\n") + "\n";
}

export function Downloads({ job }: { job: Job }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const button =
    "rounded-lg border border-border px-4 py-2 text-ink outline-none hover:border-accent focus-visible:ring-2 focus-visible:ring-accent";

  const downloadPdf = async () => {
    setBusy(true);
    setError(null);
    try {
      const { blob, filename } = await fetchReport(job.job_id);
      save(blob, filename ?? `${stem(job)}-forgery-report.pdf`);
    } catch (e) {
      setError(e instanceof ApiError ? e.code : "unexpected_response");
    } finally {
      setBusy(false);
    }
  };

  const problem = error ? friendlyError(error) : null;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-3" role="group" aria-label="Downloads">
        <button
          type="button"
          onClick={downloadPdf}
          disabled={busy}
          aria-busy={busy}
          className="rounded-lg bg-ink px-4 py-2 font-medium text-page outline-none hover:opacity-90 focus-visible:ring-2 focus-visible:ring-accent disabled:cursor-wait disabled:opacity-70"
          data-testid="download-pdf"
          data-href={reportUrl(job.job_id)}
        >
          {busy ? "Preparing PDF…" : "Download PDF report"}
        </button>
        <button type="button" className={button} onClick={() => downloadJson(job)}>
          Full result (JSON)
        </button>
        <button
          type="button"
          className={button}
          onClick={() => save(new Blob([framesCsv(job)], { type: "text/csv" }), `${stem(job)}.frames.csv`)}
        >
          Frame scores (CSV)
        </button>
      </div>
      {problem && (
        <div role="alert" className="rounded-lg border border-fake bg-fake-wash px-3 py-2 text-sm" data-testid="pdf-error">
          <p className="font-medium text-ink">{problem.title}</p>
          <p className="text-ink-2">{problem.advice}</p>
          <p className="mt-1 text-xs text-muted">error code: {error}</p>
        </div>
      )}
    </div>
  );
}
