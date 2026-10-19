import { useCallback, useRef, useState, type ReactNode } from "react";

import type { Job, ModelInfo } from "../api/types";
import { DetailsPanel } from "./DetailsPanel";
import { Downloads } from "./Downloads";
import { EvidencePanel } from "./EvidencePanel";
import { Filmstrip } from "./Filmstrip";
import { KeyFigures } from "./KeyFigures";
import { ScoreChart } from "./ScoreChart";
import { SegmentList } from "./SegmentList";
import { Timeline } from "./Timeline";
import { Verdict } from "./Verdict";
import { VideoPlayer, type PlayerHandle } from "./VideoPlayer";

export { downloadJson } from "./Downloads";

interface Props {
  job: Job;
  videoUrl: string | null;
  info?: ModelInfo | null;
  /** Fallback for results without `evidence.frame_threshold` (older JSON). */
  segmentThreshold?: number | null;
  onReset: () => void;
}

function Card({ title, children, id }: { title?: string; children: ReactNode; id?: string }) {
  return (
    <section
      aria-labelledby={title ? id : undefined}
      className="space-y-5 rounded-xl border border-border bg-surface p-4 sm:p-5"
    >
      {title && (
        <h2 id={id} className="text-lg font-semibold text-ink">
          {title}
        </h2>
      )}
      {children}
    </section>
  );
}

/**
 * The full result, in the order a reader needs it: what the verdict is, how big the finding
 * is, where it is (video, timeline, score, keyframes), each span in detail, what drove it,
 * what was analysed and how -- then the same thing as a PDF to keep.
 */
export function ResultView({ job, videoUrl, info = null, segmentThreshold, onReset }: Props) {
  const result = job.result!;
  const player = useRef<PlayerHandle>(null);
  const [time, setTime] = useState(0);
  const [active, setActive] = useState<number | null>(null);

  const seek = useCallback((t: number, index: number | null = null) => {
    setActive(index);
    setTime(t);
    player.current?.seek(t);
  }, []);

  const threshold = result.evidence?.frame_threshold ?? segmentThreshold ?? null;

  return (
    <div className="space-y-6">
      <Verdict result={result} />
      <KeyFigures result={result} />

      <Card>
        {videoUrl && <VideoPlayer ref={player} src={videoUrl} onTime={setTime} />}
        <Timeline
          duration={result.duration_s}
          segments={result.segments}
          currentTime={time}
          activeIndex={active}
          onSeek={seek}
          faceFound={result.face_found}
        />
        <ScoreChart
          scores={result.frame_scores}
          fps={result.fps}
          segments={result.segments}
          threshold={threshold}
          currentTime={time}
          onSeek={(t) => seek(t)}
        />
        <Filmstrip keyframes={result.keyframes ?? []} onSeek={(t, seg) => seek(t, seg)} />
      </Card>

      {result.segments.length > 0 && (
        <Card>
          <SegmentList
            segments={result.segments}
            evidence={result.evidence?.segments}
            activeIndex={active}
            onPlay={seek}
          />
        </Card>
      )}

      {result.evidence?.modality.available && (
        <Card>
          <EvidencePanel result={result} currentTime={time} onSeek={(t) => seek(t)} />
        </Card>
      )}

      <Card>
        <DetailsPanel job={job} info={info} />
      </Card>

      <details className="rounded-xl border border-border bg-surface p-4 text-sm text-ink-2">
        <summary className="cursor-pointer font-medium text-ink">How to read this</summary>
        <ul className="mt-2 list-disc space-y-1 pl-5">
          <li>
            The <strong className="text-ink">timeline</strong> is the whole video. Hatched red blocks
            are moments the detector believes were altered — the face, the voice, or both.
          </li>
          <li>
            The <strong className="text-ink">score</strong> line shows, frame by frame, how likely
            each moment is to be manipulated. Blocks are drawn where it stays high.
          </li>
          <li>
            <strong className="text-ink">Voice only</strong> and{" "}
            <strong className="text-ink">face only</strong> rerun the model with one stream removed.
            They show what the model relied on, not how the video was edited.
          </li>
          <li>
            This is an automated estimate from a model trained on one research dataset (LAV-DF).
            It can be wrong, especially on videos unlike that data. Treat it as a lead, not proof.
          </li>
        </ul>
      </details>

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={onReset}
          className="rounded-lg bg-accent px-4 py-2 font-medium text-white outline-none hover:opacity-90 focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-accent"
        >
          Analyse another video
        </button>
        <Downloads job={job} />
      </div>
      <p className="text-sm text-muted">
        {job.filename} · analysed in {result.timings_s.total?.toFixed(1) ?? "?"} s
      </p>
    </div>
  );
}
