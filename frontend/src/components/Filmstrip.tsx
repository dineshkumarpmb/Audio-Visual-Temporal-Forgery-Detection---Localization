import { useState } from "react";

import type { KeyframeMeta } from "../api/types";
import { formatTime } from "../lib/format";

interface Props {
  keyframes: KeyframeMeta[];
  onSeek: (time: number, segment: number | null) => void;
}

const ROLE_WORDS: Record<KeyframeMeta["role"], string> = {
  before: "before",
  start: "starts",
  middle: "middle",
  end: "ends",
  after: "after",
  sample: "sample",
};

/**
 * What the face looked like around each flagged span: half a second before, its first,
 * middle and last frame, and half a second after. Frames inside a span carry a red rule
 * (and the word, for colour-blind readers); clicking any frame plays from it.
 */
export function Filmstrip({ keyframes, onSeek }: Props) {
  const [broken, setBroken] = useState<Set<string>>(new Set());
  if (keyframes.length === 0) return null;

  const groups = new Map<number | null, KeyframeMeta[]>();
  for (const k of keyframes) {
    groups.set(k.segment, [...(groups.get(k.segment) ?? []), k]);
  }
  const visible = keyframes.filter((k) => !broken.has(k.url));
  if (visible.length === 0) return null; // thumbnails expired with the job

  return (
    <section aria-labelledby="filmstrip-title">
      <h3 id="filmstrip-title" className="mb-2 font-medium text-ink">
        Keyframes
      </h3>
      <div className="space-y-4">
        {[...groups.entries()].map(([segment, frames]) => (
          <div key={String(segment)} data-testid="filmstrip-group">
            <p className="mb-1.5 text-sm text-ink-2">
              {segment === null ? "Frames sampled across the video" : `Segment ${segment + 1}`}
            </p>
            <ol className="grid grid-cols-3 gap-2 sm:grid-cols-5">
              {frames.map((k) => {
                const inside = k.segment !== null && ["start", "middle", "end"].includes(k.role);
                return (
                  <li key={k.url}>
                    <button
                      type="button"
                      onClick={() => onSeek(k.time_s, k.segment)}
                      aria-label={`Play from ${formatTime(k.time_s)} (${ROLE_WORDS[k.role]})`}
                      className="group block w-full text-left outline-none focus-visible:ring-2 focus-visible:ring-accent rounded-md"
                    >
                      <img
                        src={k.url}
                        alt=""
                        loading="lazy"
                        onError={() => setBroken((b) => new Set(b).add(k.url))}
                        className={[
                          "aspect-square w-full rounded-md bg-track object-cover",
                          inside ? "ring-2 ring-fake" : "ring-1 ring-border",
                          "group-hover:opacity-90",
                        ].join(" ")}
                      />
                      <span className="mt-1 block text-xs tabular-nums text-ink-2">
                        {inside && <span className="text-fake">● </span>}
                        {ROLE_WORDS[k.role]} · {formatTime(k.time_s)}
                        {!k.face_found && " · no face"}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ol>
          </div>
        ))}
      </div>
    </section>
  );
}
