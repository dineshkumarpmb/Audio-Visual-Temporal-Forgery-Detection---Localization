import type { AnalysisResult } from "../api/types";
import { formatDuration, formatPercent } from "../lib/format";

/** The headline sentence, in words a non-expert reads correctly (P14-6, P14-12). */
export function summarise(r: AnalysisResult): string {
  const forged = r.segments.reduce((acc, s) => acc + (s.end_s - s.start_s), 0);
  if (!r.is_fake) {
    return r.segments.length === 0
      ? "No part of this video was flagged as manipulated."
      : `Overall the video looks genuine, though ${r.segments.length} short moment${r.segments.length > 1 ? "s were" : " was"} flagged — see the timeline.`;
  }
  if (r.segments.length === 0) {
    return "The video as a whole looks manipulated, but no specific moment stood out.";
  }
  const n = r.segments.length;
  return (
    `${n} manipulated segment${n > 1 ? "s" : ""} found, ` +
    `${formatDuration(forged)} of the ${formatDuration(r.duration_s)} video.`
  );
}

export function Verdict({ result }: { result: AnalysisResult }) {
  const fake = result.is_fake;
  return (
    <section
      aria-labelledby="verdict-title"
      className={[
        "rounded-xl border-2 p-5",
        fake ? "border-fake bg-fake-wash" : "border-real bg-surface",
      ].join(" ")}
    >
      <div className="flex items-start gap-4">
        <span
          aria-hidden="true"
          className={[
            "mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-white",
            fake ? "bg-fake" : "bg-real",
          ].join(" ")}
        >
          {fake ? (
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M12 7v6m0 4h.01" strokeLinecap="round" />
            </svg>
          ) : (
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M5 12.5l4.5 4.5L19 7.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </span>
        <div className="min-w-0">
          <h2 id="verdict-title" className="text-2xl font-semibold text-ink">
            {fake ? "Likely manipulated" : "No manipulation detected"}
          </h2>
          <p className="mt-1 text-ink">{summarise(result)}</p>
          <p className="mt-2 text-sm text-ink-2" data-testid="confidence">
            Manipulation score for the whole video:{" "}
            <strong className="text-ink tabular-nums">{formatPercent(result.clip_score, 1)}</strong>{" "}
            (flagged above {formatPercent(result.threshold)})
          </p>
        </div>
      </div>
      {result.warnings.length > 0 && (
        <ul className="mt-4 space-y-1 border-t border-border pt-3 text-sm text-ink-2">
          {result.warnings.map((w) => (
            <li key={w} className="flex gap-2">
              <span aria-hidden="true" className="text-warn">▲</span>
              <span>
                <span className="sr-only">Warning: </span>
                {w.charAt(0).toUpperCase() + w.slice(1)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
