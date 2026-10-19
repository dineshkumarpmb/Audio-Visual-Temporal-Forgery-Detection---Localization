import type { AnalysisResult } from "../api/types";
import { formatPercent } from "../lib/format";
import { ScoreChart } from "./ScoreChart";

interface Props {
  result: AnalysisResult;
  currentTime: number;
  onSeek: (time: number) => void;
}

/**
 * "Voice or face?" -- the same model rerun with one stream removed (src/inference/explain.py).
 * The caveat is always on screen, not behind a click: on LAV-DF the model leans on the
 * soundtrack for almost everything (PF-21), so a face-only edit can still read "voice".
 * The two traces are small multiples on the main chart's 0-100% scale, never a second axis.
 */
export function EvidencePanel({ result, currentTime, onSeek }: Props) {
  const m = result.evidence?.modality;
  if (!m?.available || !result.audio_only_scores || !result.visual_only_scores) return null;
  const rows: [string, number | undefined][] = [
    ["Both streams (the verdict)", m.both],
    ["Voice only — video removed", m.audio_only],
    ["Face only — audio removed", m.visual_only],
  ];
  return (
    <section aria-labelledby="evidence-title" className="space-y-4">
      <div>
        <h3 id="evidence-title" className="font-medium text-ink">
          Voice or face?
        </h3>
        <p className="mt-1 text-ink" data-testid="modality-reading">
          {m.reading}
        </p>
      </div>
      <table className="w-full max-w-md text-sm">
        <caption className="sr-only">Whole-video score with each stream removed</caption>
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label} className="border-b border-grid">
              <th scope="row" className="py-1.5 pr-3 text-left font-normal text-ink-2">
                {label}
              </th>
              <td className="py-1.5 text-right font-medium tabular-nums text-ink">
                {value === undefined ? "—" : formatPercent(value, 1)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="rounded-md bg-track px-3 py-2 text-sm text-ink-2" data-testid="modality-caveat">
        <span className="font-medium text-ink">Read with care: </span>
        {m.caveat}
      </p>
      <div className="grid gap-4 md:grid-cols-2">
        <ScoreChart
          scores={result.audio_only_scores}
          fps={result.fps}
          segments={result.segments}
          currentTime={currentTime}
          onSeek={onSeek}
          title="Voice only"
          subtitle="video removed"
          height={120}
          lineClass="stroke-series-audio"
          testId="chart-audio-only"
        />
        <ScoreChart
          scores={result.visual_only_scores}
          fps={result.fps}
          segments={result.segments}
          currentTime={currentTime}
          onSeek={onSeek}
          title="Face only"
          subtitle="audio removed"
          height={120}
          lineClass="stroke-series-visual"
          testId="chart-visual-only"
        />
      </div>
    </section>
  );
}
