import { useEffect, useState } from "react";

import type { Job } from "../api/types";
import { friendlyError } from "../lib/errors";

export function JobProgress({ job, reconnecting }: { job: Job | null; reconnecting: boolean }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const t0 = Date.now();
    const id = setInterval(() => setElapsed((Date.now() - t0) / 1000), 500);
    return () => clearInterval(id);
  }, []);

  const queued = job?.status === "queued";
  const ahead = job?.queue_position ?? 0;
  return (
    <div className="rounded-xl border border-border bg-surface p-6" aria-live="polite">
      <div className="flex items-center gap-3">
        <span
          aria-hidden="true"
          className="h-5 w-5 animate-spin rounded-full border-2 border-track border-t-accent"
        />
        <p className="text-lg font-medium text-ink">
          {queued
            ? `Waiting for the server — ${ahead} video${ahead === 1 ? "" : "s"} ahead`
            : "Analysing the video…"}
        </p>
      </div>
      <p className="mt-2 text-sm text-ink-2">
        {queued
          ? "One video is analysed at a time. Yours starts automatically."
          : "Finding the face, reading the soundtrack and checking every frame. A 10-second clip takes about 5 seconds."}
      </p>
      <p className="mt-1 text-sm text-muted tabular-nums">{Math.floor(elapsed)} s elapsed</p>
      {reconnecting && (
        <p className="mt-2 text-sm text-warn">Connection hiccup — retrying…</p>
      )}
    </div>
  );
}

export function ErrorPanel({ code, onReset }: { code: string; onReset: () => void }) {
  const { title, advice } = friendlyError(code);
  return (
    <div role="alert" className="rounded-xl border-2 border-fake bg-fake-wash p-6">
      <h2 className="text-xl font-semibold text-ink">{title}</h2>
      <p className="mt-1 text-ink">{advice}</p>
      <p className="mt-2 text-xs text-muted">error code: {code}</p>
      <button
        type="button"
        onClick={onReset}
        className="mt-4 rounded-lg bg-accent px-4 py-2 font-medium text-white outline-none hover:opacity-90 focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-accent"
      >
        Try another video
      </button>
    </div>
  );
}
