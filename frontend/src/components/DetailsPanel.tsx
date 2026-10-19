import type { Job, ModelInfo } from "../api/types";
import { formatBytes, formatPercent } from "../lib/format";

/** ffmpeg reports a demuxer alias list ("mov,mp4,m4a,3gp,3g2,mj2"); a reader wants "MP4". */
export function containerName(demuxer: string, filename: string): string {
  const ext = filename.includes(".") ? filename.split(".").pop()!.toLowerCase() : "";
  const aliases = demuxer ? demuxer.split(",") : [];
  if (ext && aliases.includes(ext)) return ext.toUpperCase();
  const friendly: Record<string, string> = { matroska: "MKV", avi: "AVI", mov: "MOV" };
  const hit = aliases.find((a) => a in friendly);
  return hit ? friendly[hit] : demuxer || "?";
}

function Rows({ rows }: { rows: [string, string][] }) {
  return (
    <dl className="divide-y divide-grid text-sm">
      {rows.map(([k, v]) => (
        <div key={k} className="grid grid-cols-[minmax(8rem,12rem)_1fr] gap-3 py-1.5">
          <dt className="text-ink-2">{k}</dt>
          <dd className="break-words text-ink">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

/** The input (always shown) and how it was analysed (collapsed: for the curious). */
export function DetailsPanel({ job, info }: { job: Job; info: ModelInfo | null }) {
  const r = job.result!;
  const m = job.media;
  const post = info?.postprocess ?? {};
  const timings = Object.entries(r.timings_s).filter(([k]) => k !== "total");
  return (
    <div className="grid gap-6 md:grid-cols-2">
      <section aria-labelledby="input-title">
        <h3 id="input-title" className="mb-2 font-medium text-ink">
          The input
        </h3>
        <Rows
          rows={[
            ["Format / codecs", `${containerName(m.container, job.filename)} · video ${m.video_codec} · audio ${m.audio_codec}`],
            ["Resolution", `${m.width} × ${m.height}`],
            ["Frame rate", m.fps ? `${m.fps} fps` : "unknown"],
            ["Length", `${m.duration_s.toFixed(2)} s`],
            ["Size", formatBytes(m.size_bytes)],
            ["Face tracked", `${formatPercent(r.face_found_fraction)} of frames`],
            ["Audio", r.audio_silent ? "silent" : "present"],
          ]}
        />
        {r.warnings.length > 0 && (
          <ul className="mt-3 space-y-1 text-sm text-ink-2">
            {r.warnings.map((w) => (
              <li key={w} className="flex gap-2">
                <span aria-hidden="true" className="text-warn">▲</span>
                <span>
                  <span className="sr-only">Warning: </span>
                  {w.charAt(0).toUpperCase() + w.slice(1)}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>
      <details className="group" data-testid="technical-details">
        <summary className="mb-2 cursor-pointer font-medium text-ink">How it was analysed</summary>
        <Rows
          rows={[
            ["Model", `${info?.name ?? "?"}${info?.sha256 ? ` (sha256 ${info.sha256.slice(0, 12)}…)` : ""}`],
            ["Architecture", `${info?.backbone ?? "?"} face features + ${info?.audio_feature ?? "?"} audio, cross-attention + Transformer`],
            ["Device", info?.device ?? "?"],
            ["Verdict threshold", formatPercent(r.threshold)],
            ["Segment threshold", formatPercent(r.evidence?.frame_threshold ?? 0.5)],
            ["Post-processing", Object.entries(post).filter(([k]) => k !== "fps").map(([k, v]) => `${k} ${String(v)}`).join(", ") || "—"],
            ["Time per stage", timings.map(([k, v]) => `${k} ${v.toFixed(2)} s`).join(" · ")],
            ["Job", `${job.job_id} · ${new Date(job.created_at).toLocaleString()}`],
          ]}
        />
      </details>
    </div>
  );
}
