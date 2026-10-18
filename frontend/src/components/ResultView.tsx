import { useCallback, useRef, useState } from "react";

import type { Job } from "../api/types";
import { ScoreChart } from "./ScoreChart";
import { SegmentList } from "./SegmentList";
import { Timeline } from "./Timeline";
import { Verdict } from "./Verdict";
import { VideoPlayer, type PlayerHandle } from "./VideoPlayer";

interface Props {
  job: Job;
  videoUrl: string | null;
  segmentThreshold?: number | null;
  onReset: () => void;
}

export function downloadJson(job: Job): void {
  const blob = new Blob([JSON.stringify(job, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${job.filename.replace(/\.[^.]+$/, "")}.analysis.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

export function ResultView({ job, videoUrl, segmentThreshold, onReset }: Props) {
  const result = job.result!;
  const player = useRef<PlayerHandle>(null);
  const [time, setTime] = useState(0);
  const [active, setActive] = useState<number | null>(null);

  const seek = useCallback((t: number, index: number | null = null) => {
    setActive(index);
    setTime(t);
    player.current?.seek(t);
  }, []);

  return (
    <div className="space-y-6">
      <Verdict result={result} />

      <div className="rounded-xl border border-border bg-surface p-4 sm:p-5 space-y-6">
        {videoUrl && <VideoPlayer ref={player} src={videoUrl} onTime={setTime} />}
        <Timeline
          duration={result.duration_s}
          segments={result.segments}
          currentTime={time}
          activeIndex={active}
          onSeek={seek}
        />
        <ScoreChart
          scores={result.frame_scores}
          fps={result.fps}
          segments={result.segments}
          threshold={segmentThreshold}
          currentTime={time}
          onSeek={(t) => seek(t)}
        />
        <SegmentList segments={result.segments} activeIndex={active} onPlay={seek} />
      </div>

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
            This is an automated estimate from a model trained on one research dataset (LAV-DF).
            It can be wrong, especially on videos unlike that data. Treat it as a lead, not proof.
          </li>
        </ul>
      </details>

      <div className="flex flex-wrap gap-3">
        <button
          type="button"
          onClick={onReset}
          className="rounded-lg bg-accent px-4 py-2 font-medium text-white outline-none hover:opacity-90 focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-accent"
        >
          Analyse another video
        </button>
        <button
          type="button"
          onClick={() => downloadJson(job)}
          className="rounded-lg border border-border px-4 py-2 text-ink outline-none hover:border-accent focus-visible:ring-2 focus-visible:ring-accent"
        >
          Download full result (JSON)
        </button>
        <span className="self-center text-sm text-muted">
          {job.filename} · analysed in {result.timings_s.total?.toFixed(1) ?? "?"} s
        </span>
      </div>
    </div>
  );
}
