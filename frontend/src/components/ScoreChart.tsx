import { useEffect, useRef, useState, type MouseEvent } from "react";

import type { Segment } from "../api/types";
import { formatPercent, formatTime } from "../lib/format";

interface Props {
  scores: number[];
  fps: number;
  segments: Segment[];
  /** The frame threshold segments were cut at (post-processing τ), if known. */
  threshold?: number | null;
  currentTime?: number;
  onSeek: (time: number) => void;
}

const HEIGHT = 160;
const PAD = { top: 10, right: 12, bottom: 22, left: 40 };

/** SVG path for the score line: x = frame centre time, y = P(fake). Exported for tests. */
export function linePath(scores: number[], width: number, height: number): string {
  const n = scores.length;
  if (n === 0) return "";
  const x = (i: number) => (n === 1 ? width / 2 : (i / (n - 1)) * width);
  const y = (v: number) => (1 - Math.min(Math.max(v, 0), 1)) * height;
  return scores.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(2)},${y(v).toFixed(2)}`).join("");
}

/**
 * Per-frame "how likely is this moment manipulated" (P14-7). One series, so no legend box:
 * the title names it. 2px line, hairline solid grid, forged spans washed behind the line so
 * the chart and the timeline above it read as the same story. Hover shows a crosshair and
 * the exact value; click seeks the player.
 */
export function ScoreChart({ scores, fps, segments, threshold, currentTime = 0, onSeek }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(640);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const el = wrap.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([entry]) => {
      if (entry.contentRect.width > 0) setWidth(entry.contentRect.width);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const n = scores.length;
  const duration = n / fps;
  const plotW = Math.max(width - PAD.left - PAD.right, 10);
  const plotH = HEIGHT - PAD.top - PAD.bottom;
  const xOfTime = (t: number) => (duration > 0 ? (t / duration) * plotW : 0);
  const yOf = (v: number) => (1 - v) * plotH;

  const frameAt = (e: MouseEvent<SVGRectElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width <= 0 || n === 0) return null;
    const frac = (e.clientX - rect.left) / rect.width;
    return Math.min(Math.max(Math.round(frac * (n - 1)), 0), n - 1);
  };

  const hx = hover === null ? 0 : n === 1 ? plotW / 2 : (hover / (n - 1)) * plotW;

  return (
    <figure className="m-0">
      <figcaption className="mb-2">
        <span className="font-medium text-ink">Manipulation score, frame by frame</span>
        <span className="ml-2 text-sm text-ink-2">higher = more likely manipulated</span>
      </figcaption>
      {/* overflow-hidden: width comes from the parent, never from the SVG, or the observer
          would measure its own content and a phone-width page would scroll sideways */}
      <div ref={wrap} className="relative w-full min-w-0 overflow-hidden">
        <svg
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={`Line chart of the manipulation score for ${n} frames`}
          className="block overflow-visible"
        >
          <g transform={`translate(${PAD.left},${PAD.top})`}>
            {segments.map((s, i) => (
              <rect
                key={i}
                data-testid="chart-span"
                x={xOfTime(s.start_s)}
                width={Math.max(xOfTime(s.end_s) - xOfTime(s.start_s), 2)}
                y={0}
                height={plotH}
                className="fill-fake-wash"
              />
            ))}
            {[0, 0.5, 1].map((v) => (
              <g key={v}>
                <line x1={0} x2={plotW} y1={yOf(v)} y2={yOf(v)}
                  className={v === 0 ? "stroke-axis" : "stroke-grid"} strokeWidth={1} />
                <text x={-8} y={yOf(v)} dy="0.32em" textAnchor="end"
                  className="fill-muted text-[11px] tabular-nums">
                  {v * 100}%
                </text>
              </g>
            ))}
            {threshold != null && (
              <g data-testid="chart-threshold">
                <line x1={0} x2={plotW} y1={yOf(threshold)} y2={yOf(threshold)}
                  className="stroke-ink-2" strokeWidth={1} strokeDasharray="4 3" />
                <text x={plotW} y={yOf(threshold) - 4} textAnchor="end"
                  className="fill-ink-2 text-[11px]">
                  segment threshold {formatPercent(threshold)}
                </text>
              </g>
            )}
            <path
              d={linePath(scores, plotW, plotH)}
              fill="none"
              className="stroke-accent"
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
              data-testid="chart-line"
            />
            <line x1={xOfTime(currentTime)} x2={xOfTime(currentTime)} y1={-4} y2={plotH}
              className="stroke-ink" strokeWidth={1.5} aria-hidden="true" />
            {[0, duration / 2, duration].map((t, i) => (
              <text key={i} x={xOfTime(t)} y={plotH + 16}
                textAnchor={i === 0 ? "start" : i === 2 ? "end" : "middle"}
                className="fill-muted text-[11px] tabular-nums">
                {formatTime(t)}
              </text>
            ))}
            {hover !== null && (
              <g aria-hidden="true" className="pointer-events-none">
                <line x1={hx} x2={hx} y1={0} y2={plotH} className="stroke-muted" strokeWidth={1} />
                <circle cx={hx} cy={yOf(scores[hover])} r={4.5}
                  className="fill-accent stroke-surface" strokeWidth={2} />
              </g>
            )}
            <rect
              x={0}
              y={0}
              width={plotW}
              height={plotH}
              fill="transparent"
              className="cursor-crosshair"
              onMouseMove={(e) => setHover(frameAt(e))}
              onMouseLeave={() => setHover(null)}
              onClick={(e) => {
                const f = frameAt(e);
                if (f !== null) onSeek(f / fps);
              }}
            />
          </g>
        </svg>
        {hover !== null && (
          <div
            role="tooltip"
            className="pointer-events-none absolute top-0 z-10 rounded-md border border-border bg-surface px-2 py-1 text-xs text-ink shadow-sm"
            style={{
              left: Math.min(PAD.left + hx + 10, width - 150),
            }}
          >
            <div className="tabular-nums text-ink-2">{formatTime(hover / fps)}</div>
            <div className="font-medium">{formatPercent(scores[hover])} likely manipulated</div>
          </div>
        )}
      </div>
    </figure>
  );
}
