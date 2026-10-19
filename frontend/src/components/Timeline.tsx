import type { MouseEvent } from "react";

import type { Segment } from "../api/types";
import { formatPercent, formatTime } from "../lib/format";

interface Props {
  duration: number;
  segments: Segment[];
  currentTime?: number;
  activeIndex?: number | null;
  onSeek: (time: number, segmentIndex: number | null) => void;
  /** Per-frame 1/0: where a face was tracked. Frames without one were scored from audio. */
  faceFound?: number[] | null;
}

/** Runs of tracked-face frames as [startFrac, endFrac] of the strip. Exported for tests. */
export function faceRuns(found: number[]): [number, number][] {
  const out: [number, number][] = [];
  let start: number | null = null;
  for (let i = 0; i <= found.length; i++) {
    const on = i < found.length && found[i] === 1;
    if (on && start === null) start = i;
    if (!on && start !== null) {
      out.push([start / found.length, i / found.length]);
      start = null;
    }
  }
  return out;
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
export function Timeline({
  duration,
  segments,
  currentTime = 0,
  activeIndex = null,
  onSeek,
  faceFound,
}: Props) {
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

      {faceFound && faceFound.length > 0 && (
        <div className="mt-1.5" data-testid="face-band">
          <div className="relative h-1.5 rounded-full bg-track" aria-hidden="true">
            {faceRuns(faceFound).map(([a, b]) => (
              <span
                key={a}
                className="absolute inset-y-0 rounded-full bg-ink-2"
                style={{ left: `${a * 100}%`, width: `${(b - a) * 100}%` }}
              />
            ))}
          </div>
        </div>
      )}

      <div className="relative mt-1 h-5 text-xs text-muted tabular-nums" aria-hidden="true">
        {ticks(duration).map((t, i) => (
          <span
            key={t}
            // Every other label on phones: eight m:ss.s labels do not fit in 360 px.
            className={`absolute -translate-x-1/2 first:translate-x-0 ${i % 2 ? "hidden sm:inline" : ""}`}
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
        {faceFound && faceFound.length > 0 && (
          <span className="inline-flex items-center gap-2">
            <span aria-hidden="true" className="inline-block h-1.5 w-5 rounded-full bg-ink-2" />
            Face tracked (thin band)
          </span>
        )}
      </div>
    </figure>
  );
}
