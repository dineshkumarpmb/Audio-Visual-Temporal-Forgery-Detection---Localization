import type { Driver, Segment, SegmentEvidence } from "../api/types";
import { formatDuration, formatPercent, formatTime } from "../lib/format";

interface Props {
  segments: Segment[];
  evidence?: SegmentEvidence[];
  activeIndex: number | null;
  onPlay: (time: number, index: number) => void;
}

export const DRIVER_WORDS: Record<Driver, string> = {
  voice: "Voice",
  face: "Face",
  both: "Both",
  neither: "Unclear",
  unknown: "—",
};

const pct = (v: number | null | undefined) => (v == null ? "—" : formatPercent(v));

/** The timeline as a table: exact times for every span, and the accessible view of it.
 *  With evidence it also says how strongly each stream alone flags the span. */
export function SegmentList({ segments, evidence, activeIndex, onPlay }: Props) {
  if (segments.length === 0) return null;
  const rich = evidence !== undefined && evidence.length === segments.length;
  const th = "py-2 pr-3 font-normal";
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[34rem] text-left text-sm">
        <caption className="mb-2 text-left font-medium text-ink">Manipulated segments</caption>
        <thead className="text-ink-2">
          <tr className="border-b border-grid">
            <th scope="col" className={th}>#</th>
            <th scope="col" className={th}>From</th>
            <th scope="col" className={th}>To</th>
            <th scope="col" className={th}>Length</th>
            <th scope="col" className={th} title="Average manipulation score across the segment">Score</th>
            {rich && (
              <>
                <th scope="col" className={th}>Peak</th>
                <th scope="col" className={th} title="The span's score with the video removed">Voice only</th>
                <th scope="col" className={th} title="The span's score with the audio removed">Face only</th>
                <th scope="col" className={th}>Driven by</th>
              </>
            )}
            <th scope="col" className="py-2 font-normal"><span className="sr-only">Play</span></th>
          </tr>
        </thead>
        <tbody className="tabular-nums text-ink">
          {segments.map((s, i) => {
            const ev = rich ? evidence![i] : undefined;
            return (
              <tr
                key={`${s.start_s}-${i}`}
                className={["border-b border-grid", activeIndex === i ? "bg-fake-wash" : ""].join(" ")}
              >
                <td className="py-2 pr-3">{i + 1}</td>
                <td className="py-2 pr-3">{formatTime(s.start_s)}</td>
                <td className="py-2 pr-3">{formatTime(s.end_s)}</td>
                <td className="py-2 pr-3">{formatDuration(s.end_s - s.start_s)}</td>
                <td className="py-2 pr-3">{formatPercent(s.confidence)}</td>
                {ev && (
                  <>
                    <td className="py-2 pr-3">{pct(ev.peak_score)}</td>
                    <td className="py-2 pr-3">{pct(ev.audio_only_mean)}</td>
                    <td className="py-2 pr-3">{pct(ev.visual_only_mean)}</td>
                    <td className="py-2 pr-3" data-testid="segment-driver">{DRIVER_WORDS[ev.driver]}</td>
                  </>
                )}
                <td className="py-2 text-right">
                  <button
                    type="button"
                    onClick={() => onPlay(s.start_s, i)}
                    className="rounded-md border border-border px-2.5 py-1 text-ink hover:border-accent focus-visible:ring-2 focus-visible:ring-accent outline-none"
                  >
                    Play
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
