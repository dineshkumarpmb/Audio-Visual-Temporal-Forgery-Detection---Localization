import type { MouseEvent } from "react";

import type { Segment } from "../api/types";
import { formatPercent, formatTime } from "../lib/format";

interface Props {
  duration: number;
  segments: Segment[];
  currentTime?: number;
  activeIndex?: number | null;
  onSeek: (time: number, segmentIndex: number | null) => void;
}

/** Where a segment sits on the strip, in percent. Exported so tests check the geometry. */
export function spanGeometry(seg: Segment, duration: number): { left: number; width: number } {
  const clamp = (v: number) => Math.min(Math.max(v, 0), duration);
  const start = clamp(seg.start_s);
  const end = clamp(seg.end_s);
  return { left: (start / duration) * 100, width: ((end - start) / duration) * 100 };
}

function ticks(duration: number): number[] {
  const steps = [0.5, 1, 2, 5, 10, 15, 30, 60, 120];
  const step = steps.find((s) => duration / s <= 8) ?? 300;
  const out: number[] = [];
  for (let t = 0; t <= duration + 1e-9; t += step) out.push(t);
  return out;
}

/**
 * ⭐ The centrepiece (P14-4): the whole clip as one strip, forged spans marked on it.
 *
 * Forged spans are the reserved "critical" status colour *plus* a 45° hatch, a legend and an
 * accessible label -- never colour alone (red vs grey/green is the classic colour-blind
 * failure). Untouched time is a neutral track, not green: the model says "nothing found
 * here", which is weaker than "verified authentic". Each span is a button that seeks the
 * player to its start (P14-5); clicking the bare track seeks to that moment.
 */
export function Timeline({ duration, segments, currentTime = 0, activeIndex = null, onSeek }: Props) {
  const seekFromTrack = (e: MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width <= 0) return;
    const t = ((e.clientX - rect.left) / rect.width) * duration;
    onSeek(Math.min(Math.max(t, 0), duration), null);
  };
  const playhead = duration > 0 ? Math.min(currentTime / duration, 1) * 100 : 0;

  return (
    <figure className="m-0">
      <figcaption className="mb-2 flex flex-wrap items-baseline justify-between gap-x-3">
        <span className="font-medium text-ink">Timeline</span>
        <span className="text-sm text-ink-2">
          {segments.length === 0
            ? "No manipulated segments found"
            : `${segments.length} manipulated segment${segments.length > 1 ? "s" : ""} — click one to play it`}
        </span>
      </figcaption>

      <div
        className="relative h-12 cursor-pointer rounded-md bg-track ring-1 ring-inset ring-border"
        onClick={seekFromTrack}
        data-testid="timeline-track"
      >
        {segments.map((seg, i) => {
          const { left, width } = spanGeometry(seg, duration);
          const label =
            `Manipulated segment ${i + 1}: ${formatTime(seg.start_s)} to ${formatTime(seg.end_s)}, ` +
            `score ${formatPercent(seg.confidence)}. Play from here.`;
          return (
            <button
              key={`${seg.start_s}-${i}`}
              type="button"
              aria-label={label}
              title={label}
              data-testid="timeline-span"
              data-start={seg.start_s}
              data-end={seg.end_s}
              onClick={(e) => {
                e.stopPropagation();
                onSeek(seg.start_s, i);
              }}
              className={[
                "hatch absolute inset-y-0 rounded-[3px] bg-fake outline-none",
                "focus-visible:ring-2 focus-visible:ring-ink focus-visible:ring-offset-2",
                activeIndex === i ? "ring-2 ring-ink ring-offset-1" : "",
              ].join(" ")}
              // min-width keeps a 40 ms span clickable on a 10-minute clip
              style={{ left: `${left}%`, width: `max(${width}%, 4px)` }}
            />
          );
        })}
        <div
          aria-hidden="true"
          data-testid="timeline-playhead"
          className="pointer-events-none absolute inset-y-[-4px] w-0.5 rounded bg-ink"
          style={{ left: `${playhead}%` }}
        />
      </div>

      <div className="relative mt-1 h-5 text-xs text-muted tabular-nums" aria-hidden="true">
        {ticks(duration).map((t) => (
          <span
            key={t}
            className="absolute -translate-x-1/2 first:translate-x-0"
            style={{ left: `${(t / duration) * 100}%` }}
          >
            {formatTime(t)}
          </span>
        ))}
      </div>

      <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-sm text-ink-2">
        <span className="inline-flex items-center gap-2">
          <span aria-hidden="true" className="hatch inline-block h-3 w-5 rounded-sm bg-fake" />
          Manipulated (forged)
        </span>
        <span className="inline-flex items-center gap-2">
          <span aria-hidden="true" className="inline-block h-3 w-5 rounded-sm bg-track" />
          No manipulation detected
        </span>
      </div>
    </figure>
  );
}
