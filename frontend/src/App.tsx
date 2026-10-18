import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, getModelInfo, uploadVideo } from "./api/client";
import type { ModelInfo } from "./api/types";
import { ResultView } from "./components/ResultView";
import { ErrorPanel, JobProgress } from "./components/StatusPanels";
import { Upload, UploadProgress } from "./components/Upload";
import { useJobPolling } from "./hooks/useJobPolling";

type Stage =
  | { kind: "idle" }
  | { kind: "uploading"; file: File; fraction: number }
  | { kind: "polling"; jobId: string }
  | { kind: "error"; code: string };

/** idle → uploading → polling → result | error. One video at a time, like the server. */
export default function App() {
  const [stage, setStage] = useState<Stage>({ kind: "idle" });
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    // Limits and the segment threshold; the UI works without them (defaults apply).
    getModelInfo().then(setInfo, () => setInfo(null));
  }, []);

  useEffect(() => () => {
    if (videoUrl) URL.revokeObjectURL(videoUrl);
  }, [videoUrl]);

  const jobId = stage.kind === "polling" ? stage.jobId : null;
  const { job, error: pollError, reconnecting } = useJobPolling(jobId);

  const reset = useCallback(() => {
    abort.current?.abort();
    setVideoUrl(null);
    setStage({ kind: "idle" });
  }, []);

  const start = useCallback(async (file: File) => {
    setVideoUrl(URL.createObjectURL(file));
    setStage({ kind: "uploading", file, fraction: 0 });
    abort.current = new AbortController();
    try {
      const created = await uploadVideo(
        file,
        (fraction) => setStage((s) => (s.kind === "uploading" ? { ...s, fraction } : s)),
        abort.current.signal,
      );
      setStage({ kind: "polling", jobId: created.job_id });
    } catch (e) {
      const code = e instanceof ApiError ? e.code : "unexpected_response";
      if (code !== "aborted") setStage({ kind: "error", code });
    }
  }, []);

  const threshold =
    typeof info?.postprocess?.threshold === "number" ? (info.postprocess.threshold as number) : null;

  let body;
  if (stage.kind === "idle") {
    body = <Upload limits={info?.limits} onFile={start} />;
  } else if (stage.kind === "uploading") {
    body = <UploadProgress fileName={stage.file.name} fraction={stage.fraction} />;
  } else if (stage.kind === "error") {
    body = <ErrorPanel code={stage.code} onReset={reset} />;
  } else if (pollError) {
    body = <ErrorPanel code={pollError.code} onReset={reset} />;
  } else if (job?.status === "failed") {
    body = <ErrorPanel code={job.error?.code ?? "internal_error"} onReset={reset} />;
  } else if (job?.status === "succeeded" && job.result) {
    body = <ResultView job={job} videoUrl={videoUrl} segmentThreshold={threshold} onReset={reset} />;
  } else {
    body = <JobProgress job={job} reconnecting={reconnecting} />;
  }

  return (
    <div className="min-h-screen bg-page font-sans text-ink">
      <main className="mx-auto max-w-4xl px-4 py-8 sm:py-12">
        <header className="mb-8">
          <h1 className="text-3xl font-semibold tracking-tight">Forgery Localizer</h1>
          <p className="mt-2 max-w-2xl text-ink-2">
            Upload a video of someone speaking. The detector checks whether the face or the voice
            was altered — and shows <em>where</em> in the video it happened.
          </p>
        </header>
        {body}
        <footer className="mt-12 text-xs text-muted">
          Audio-visual temporal forgery detection · model {info?.name ?? "—"} on {info?.device ?? "—"}
          {" · "}runs locally; your video never leaves this computer
        </footer>
      </main>
    </div>
  );
}
