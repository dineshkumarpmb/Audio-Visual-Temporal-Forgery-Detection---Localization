import type { ErrorInfo, Job, JobCreated, ModelInfo } from "./types";

/** Every failure the UI can show: an API error body, or one we synthesise for transport. */
export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public detail: Record<string, unknown> | null = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function fromBody(status: number, text: string): ApiError {
  try {
    const body = JSON.parse(text) as { error?: ErrorInfo };
    if (body.error?.code) {
      return new ApiError(status, body.error.code, body.error.message, body.error.detail);
    }
  } catch {
    // not JSON: a proxy page, or the connection was cut mid-upload
  }
  if (status === 413) return new ApiError(413, "file_too_large", "the upload is too large");
  if (status === 502 || status === 503 || status === 504 || status === 0) {
    return new ApiError(status, "server_unreachable", "the analysis server is not responding");
  }
  return new ApiError(status, "unexpected_response", `unexpected response (HTTP ${status})`);
}

async function getJson<T>(url: string): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(url, { headers: { Accept: "application/json" } });
  } catch {
    throw new ApiError(0, "server_unreachable", "the analysis server is not responding");
  }
  const text = await resp.text();
  if (!resp.ok) throw fromBody(resp.status, text);
  return JSON.parse(text) as T;
}

export const getJob = (jobId: string) => getJson<Job>(`/api/v1/jobs/${encodeURIComponent(jobId)}`);
export const getModelInfo = () => getJson<ModelInfo>("/api/v1/model-info");

export const reportUrl = (jobId: string) => `/api/v1/jobs/${encodeURIComponent(jobId)}/report.pdf`;

/** `attachment; filename="x.pdf"` -> `x.pdf`; null when absent. */
export function filenameFrom(disposition: string | null): string | null {
  const m = disposition?.match(/filename\*?=(?:UTF-8'')?"?([^";]+)"?/i);
  return m ? decodeURIComponent(m[1]) : null;
}

/**
 * Fetch the PDF report and prove it is one before saving it.
 *
 * A bare `<a download>` saves whatever the server answers -- a 66-byte JSON error becomes a
 * "PDF" that will not open. So: a route-level 404 (`not_found`) means the API process
 * predates the report endpoint (`report_unavailable`), a job-level 404 means the job expired,
 * 409 means it has not finished, and a 200 that is not `application/pdf` is refused too.
 */
export async function fetchReport(jobId: string): Promise<{ blob: Blob; filename: string | null }> {
  let resp: Response;
  try {
    resp = await fetch(reportUrl(jobId), { headers: { Accept: "application/pdf" } });
  } catch {
    throw new ApiError(0, "server_unreachable", "the analysis server is not responding");
  }
  if (!resp.ok) {
    const err = fromBody(resp.status, await resp.text());
    if (err.status === 404 && err.code === "not_found") {
      throw new ApiError(404, "report_unavailable", "this server has no report endpoint");
    }
    throw err;
  }
  const type = resp.headers.get("content-type") ?? "";
  const blob = await resp.blob();
  if (!type.includes("application/pdf") || blob.size === 0) {
    throw new ApiError(resp.status, "unexpected_response", `expected a PDF, got ${type || "nothing"}`);
  }
  return { blob, filename: filenameFrom(resp.headers.get("content-disposition")) };
}

/**
 * Upload with progress. XMLHttpRequest, not fetch: fetch has no upload-progress events, and
 * a 200 MB upload with no progress bar looks frozen.
 */
export function uploadVideo(
  file: File,
  onProgress: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<JobCreated> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/v1/analyze");
    xhr.responseType = "text";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      if (xhr.status === 202) {
        onProgress(1);
        resolve(JSON.parse(xhr.responseText) as JobCreated);
      } else {
        reject(fromBody(xhr.status, xhr.responseText));
      }
    };
    xhr.onerror = () => reject(fromBody(0, ""));
    xhr.onabort = () => reject(new ApiError(0, "aborted", "the upload was cancelled"));
    signal?.addEventListener("abort", () => xhr.abort());
    const form = new FormData();
    form.append("file", file, file.name);
    xhr.send(form);
  });
}
