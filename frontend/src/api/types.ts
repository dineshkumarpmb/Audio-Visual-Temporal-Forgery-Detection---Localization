// Mirrors api/schemas.py. If the API changes shape, change it there first, then here.

export type JobStatus = "queued" | "running" | "succeeded" | "failed";

export interface ErrorInfo {
  code: string;
  message: string;
  detail: Record<string, unknown> | null;
}

export interface Segment {
  start_s: number;
  end_s: number;
  confidence: number;
}

export interface MediaInfo {
  container: string;
  video_codec: string;
  audio_codec: string;
  width: number;
  height: number;
  fps: number | null;
  duration_s: number;
  size_bytes: number;
}

export interface KeyframeMeta {
  frame: number;
  time_s: number;
  role: "before" | "start" | "middle" | "end" | "after" | "sample";
  segment: number | null;
  face_found: boolean;
  url: string;
}

export type Driver = "voice" | "face" | "both" | "neither" | "unknown";

export interface SegmentEvidence {
  index: number;
  start_s: number;
  end_s: number;
  duration_s: number;
  frames: number[];
  mean_score: number;
  peak_score: number | null;
  audio_only_mean: number | null;
  visual_only_mean: number | null;
  face_coverage: number | null;
  driver: Driver;
  start_confidence?: number | null;
  end_confidence?: number | null;
}

export interface KeyFigures {
  clip_score: number;
  n_segments: number;
  forged_s: number;
  forged_fraction: number;
  duration_s: number;
  peak_frame_score: number;
  mean_frame_score: number;
  face_coverage: number;
}

export interface ModalitySummary {
  available: boolean;
  both?: number;
  audio_only?: number;
  visual_only?: number;
  reading?: string;
  caveat?: string;
}

export interface Evidence {
  summary: string;
  frame_threshold: number;
  key_figures: KeyFigures;
  segments: SegmentEvidence[];
  modality: ModalitySummary;
}

export interface AnalysisResult {
  verdict: "fake" | "real";
  is_fake: boolean;
  clip_score: number;
  threshold: number;
  n_frames: number;
  duration_s: number;
  fps: number;
  segments: Segment[];
  frame_scores: number[];
  face_found_fraction: number;
  audio_silent: boolean;
  warnings: string[];
  timings_s: Record<string, number>;
  // Report detail -- present on every result from the real analyzer, optional for old JSON.
  boundary_start_scores?: number[] | null;
  boundary_end_scores?: number[] | null;
  attention?: number[] | null;
  face_found?: number[] | null;
  audio_only_scores?: number[] | null;
  visual_only_scores?: number[] | null;
  audio_only_clip?: number | null;
  visual_only_clip?: number | null;
  keyframes?: KeyframeMeta[];
  evidence?: Evidence | null;
}

export interface JobCreated {
  job_id: string;
  status: JobStatus;
  status_url: string;
  queue_position: number;
}

export interface Job {
  job_id: string;
  status: JobStatus;
  filename: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  queue_position: number | null;
  media: MediaInfo;
  result: AnalysisResult | null;
  error: ErrorInfo | null;
}

export interface Limits {
  max_upload_mb: number;
  allowed_extensions: string[];
  min_duration_s: number;
  max_duration_s: number;
  job_timeout_s: number;
  max_concurrent_jobs: number;
  max_queued_jobs: number;
}

export interface ModelInfo {
  name: string;
  file?: string;
  sha256?: string;
  backbone?: string;
  audio_feature?: string;
  warmup_s?: number | null;
  device: string;
  threshold: number;
  fps: number;
  postprocess: Record<string, unknown>;
  limits: Limits;
}
