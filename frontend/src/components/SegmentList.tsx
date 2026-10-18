import type { Segment } from "../api/types";
import { formatDuration, formatPercent, formatTime } from "../lib/format";

interface Props {
  segments: Segment[];
  activeIndex: number | null;
  onPlay: (time: number, index: number) => void;
}

/** The timeline as a table: exact times for every span, and the accessible view of it. */
export function SegmentList({ segments, activeIndex, onPlay }: Props) {
  if (segments.length === 0) return null;
  return (
    <table className="w-full text-left text-sm">
      <caption className="mb-2 text-left font-medium text-ink">Manipulated segments</caption>
      <thead className="text-ink-2">
        <tr className="border-b border-grid">
          <th scope="col" className="py-2 pr-3 font-normal">#</th>
          <th scope="col" className="py-2 pr-3 font-normal">From</th>
          <th scope="col" className="py-2 pr-3 font-normal">To</th>
          <th scope="col" className="py-2 pr-3 font-normal">Length</th>
          <th scope="col" className="py-2 pr-3 font-normal" title="Average manipulation score across the segment">Score</th>
          <th scope="col" className="py-2 font-normal"><span className="sr-only">Play</span></th>
        </tr>
      </thead>
      <tbody className="tabular-nums text-ink">
        {segments.map((s, i) => (
          <tr
            key={`${s.start_s}-${i}`}
            className={["border-b border-grid", activeIndex === i ? "bg-fake-wash" : ""].join(" ")}
          >
            <td className="py-2 pr-3">{i + 1}</td>
            <td className="py-2 pr-3">{formatTime(s.start_s)}</td>
            <td className="py-2 pr-3">{formatTime(s.end_s)}</td>
            <td className="py-2 pr-3">{formatDuration(s.end_s - s.start_s)}</td>
            <td className="py-2 pr-3">{formatPercent(s.confidence)}</td>
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
        ))}
      </tbody>
    </table>
  );
}
