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
  device: string;
  threshold: number;
  fps: number;
  postprocess: Record<string, unknown>;
  limits: Limits;
}
