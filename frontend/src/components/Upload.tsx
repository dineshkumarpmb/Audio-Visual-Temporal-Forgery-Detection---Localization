import { useRef, useState, type DragEvent } from "react";

import type { Limits } from "../api/types";
import { extensionOf, formatBytes } from "../lib/format";

export const DEFAULT_LIMITS: Pick<Limits, "max_upload_mb" | "allowed_extensions"> = {
  max_upload_mb: 200,
  allowed_extensions: [".mp4", ".avi", ".mov", ".mkv"],
};

interface Props {
  limits?: Pick<Limits, "max_upload_mb" | "allowed_extensions">;
  onFile: (file: File) => void;
  disabled?: boolean;
}

/**
 * Drag-drop or click to choose (P14-2). Checks the obvious things before uploading -- type
 * and size -- so a 300 MB mistake fails in a millisecond, not after a minute of upload. The
 * server re-checks everything; this is a courtesy, not the validation.
 */
export function Upload({ limits = DEFAULT_LIMITS, onFile, disabled = false }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const exts = limits.allowed_extensions;

  const accept = (file: File | undefined) => {
    if (!file || disabled) return;
    const ext = extensionOf(file.name);
    if (!exts.includes(ext)) {
      setProblem(
        `"${file.name}" isn't a supported video. Use ${exts.map((e) => e.slice(1).toUpperCase()).join(", ")}.`,
      );
      return;
    }
    if (file.size > limits.max_upload_mb * 1024 ** 2) {
      setProblem(
        `"${file.name}" is ${formatBytes(file.size)}; the limit is ${limits.max_upload_mb} MB.`,
      );
      return;
    }
    if (file.size === 0) {
      setProblem(`"${file.name}" is empty.`);
      return;
    }
    setProblem(null);
    onFile(file);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    accept(e.dataTransfer.files[0]);
  };

  return (
    <div>
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-disabled={disabled}
        aria-label="Choose a video to analyse"
        onClick={() => !disabled && input.current?.click()}
        onKeyDown={(e) => {
          if ((e.key === "Enter" || e.key === " ") && !disabled) {
            e.preventDefault();
            input.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={[
          "flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed",
          "px-6 py-14 text-center transition-colors outline-none",
          "focus-visible:ring-2 focus-visible:ring-accent",
          disabled ? "cursor-not-allowed opacity-60" : "cursor-pointer hover:border-accent",
          dragging ? "border-accent bg-surface" : "border-axis",
        ].join(" ")}
      >
        <svg aria-hidden="true" width="40" height="40" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="1.5" className="text-muted">
          <path d="M12 16V4m0 0l-4 4m4-4l4 4" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" strokeLinecap="round" />
        </svg>
        <p className="text-lg font-medium text-ink">
          Drop a video here, or <span className="text-accent underline">choose a file</span>
        </p>
        <p className="text-sm text-ink-2">
          A clip of someone speaking, with sound.{" "}
          {exts.map((e) => e.slice(1).toUpperCase()).join(", ")} up to {limits.max_upload_mb} MB.
        </p>
      </div>
      <input
        ref={input}
        type="file"
        accept={[...exts, "video/*"].join(",")}
        className="hidden"
        data-testid="file-input"
        onChange={(e) => {
          accept(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      {problem && (
        <p role="alert" className="mt-3 text-sm text-fake">
          {problem}
        </p>
      )}
    </div>
  );
}

export function UploadProgress({ fileName, fraction }: { fileName: string; fraction: number }) {
  const pct = Math.round(fraction * 100);
  return (
    <div className="rounded-xl border border-border bg-surface p-6">
      <p className="mb-3 text-ink">
        Uploading <span className="font-medium">{fileName}</span>
      </p>
      <div
        role="progressbar"
        aria-label="Upload progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct}
        className="h-2 overflow-hidden rounded-full bg-track"
      >
        <div className="h-full rounded-full bg-accent transition-[width]" style={{ width: `${pct}%` }} />
      </div>
      <p className="mt-2 text-sm text-ink-2 tabular-nums">{pct}%</p>
    </div>
  );
}
