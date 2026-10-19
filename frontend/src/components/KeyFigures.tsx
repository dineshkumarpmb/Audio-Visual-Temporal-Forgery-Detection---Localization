import type { AnalysisResult } from "../api/types";
import { formatPercent } from "../lib/format";

/**
 * The result in eight numbers -- the same eight the PDF's first page carries. Each tile is a
 * value and a plain label; no colour carries meaning here (the verdict badge does that).
 */
export function KeyFigures({ result }: { result: AnalysisResult }) {
  const kf = result.evidence?.key_figures;
  const forged = kf?.forged_s ?? result.segments.reduce((a, s) => a + s.end_s - s.start_s, 0);
  const tiles: [string, string][] = [
    [formatPercent(result.clip_score, 1), "manipulation score"],
    [String(result.segments.length), "manipulated segments"],
    [`${forged.toFixed(2)} s`, "flagged time"],
    [formatPercent(kf?.forged_fraction ?? forged / result.duration_s, 1), "of the video flagged"],
    [`${result.duration_s.toFixed(1)} s`, "video length"],
    [formatPercent(result.face_found_fraction), "frames with a face"],
    [formatPercent(kf?.peak_frame_score ?? Math.max(...result.frame_scores)), "highest frame score"],
    [`${(result.timings_s.total ?? 0).toFixed(1)} s`, "analysis time"],
  ];
  return (
    <dl className="grid grid-cols-2 overflow-hidden rounded-xl border border-border bg-surface sm:grid-cols-4">
      {tiles.map(([value, label]) => (
        <div key={label} className="border-b border-r border-grid p-3 sm:p-4" data-testid="key-figure">
          <dt className="text-xs text-ink-2">{label}</dt>
          <dd className="mt-1 text-xl font-semibold text-ink">{value}</dd>
        </div>
      ))}
    </dl>
  );
}
