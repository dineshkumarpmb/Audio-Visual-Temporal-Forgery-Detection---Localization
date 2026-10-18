import { act, render, renderHook, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import * as client from "./api/client";
import { ApiError } from "./api/client";

vi.mock("./api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api/client")>();
  return { ...actual, uploadVideo: vi.fn(), getJob: vi.fn(), getModelInfo: vi.fn() };
});
import type { Job } from "./api/types";
import { MAX_POLL_FAILURES, POLL_INTERVAL_MS, useJobPolling } from "./hooks/useJobPolling";
import { failedJob, TWO_SPANS } from "./test/fixtures";

const created = { job_id: "j1", status: "queued" as const, status_url: "/api/v1/jobs/j1", queue_position: 0 };
const queued: Job = { ...TWO_SPANS, status: "queued", result: null, queue_position: 1 };
const running: Job = { ...TWO_SPANS, status: "running", result: null, queue_position: null };

beforeEach(() => {
  vi.mocked(client.getModelInfo).mockResolvedValue({
    name: "avtfd_g_seed0",
    device: "cuda",
    threshold: 0.5,
    fps: 25,
    postprocess: { threshold: 0.2 },
    limits: {
      max_upload_mb: 200,
      allowed_extensions: [".mp4", ".avi", ".mov", ".mkv"],
      min_duration_s: 0.5,
      max_duration_s: 300,
      job_timeout_s: 300,
      max_concurrent_jobs: 1,
      max_queued_jobs: 4,
    },
  });
});
afterEach(() => vi.clearAllMocks());

async function chooseFile() {
  const f = new File([new Uint8Array(100)], "clip.mp4", { type: "video/mp4" });
  await userEvent.upload(screen.getByTestId("file-input"), f);
}

describe("App flow", () => {
  it("upload → queued → running → result with timeline, chart and table", async () => {
    vi.mocked(client.uploadVideo).mockImplementation(async (_f, onProgress) => {
      onProgress(0.5);
      return created;
    });
    const getJob = vi
      .mocked(client.getJob)
      .mockResolvedValueOnce(queued)
      .mockResolvedValueOnce(running)
      .mockResolvedValue(TWO_SPANS);
    render(<App />);
    await chooseFile();

    expect(await screen.findByText(/1 video ahead/)).toBeInTheDocument();
    expect(await screen.findByText("Analysing the video…", {}, { timeout: 3000 })).toBeInTheDocument();
    expect(
      await screen.findByRole("heading", { name: "Likely manipulated" }, { timeout: 3000 }),
    ).toBeInTheDocument();
    expect(screen.getAllByTestId("timeline-span")).toHaveLength(2);
    expect(screen.getByTestId("chart-threshold")).toHaveTextContent("20%"); // from model-info
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByTestId("video")).toHaveAttribute("src", "blob:test");
    const polls = getJob.mock.calls.length;
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS + 200));
    expect(getJob.mock.calls.length).toBe(polls); // polling stopped on success
  });

  it("a failed job shows the plain-language error and can start over", async () => {
    vi.mocked(client.uploadVideo).mockResolvedValue(created);
    vi.mocked(client.getJob).mockResolvedValue(failedJob("no_face"));
    render(<App />);
    await chooseFile();
    expect(await screen.findByRole("heading", { name: "No face was found" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Try another video" }));
    expect(screen.getByRole("button", { name: "Choose a video to analyse" })).toBeInTheDocument();
  });

  it("an upload refused by the server shows its error", async () => {
    vi.mocked(client.uploadVideo).mockRejectedValue(
      new ApiError(422, "no_audio", "the file has no audio track"),
    );
    render(<App />);
    await chooseFile();
    expect(await screen.findByRole("heading", { name: "This video has no sound" })).toBeInTheDocument();
  });

  it("the server being down is explained, not a blank screen", async () => {
    vi.mocked(client.uploadVideo).mockRejectedValue(
      new ApiError(0, "server_unreachable", "down"),
    );
    render(<App />);
    await chooseFile();
    expect(await screen.findByText(/make api/)).toBeInTheDocument();
  });

  it("the JSON download carries the full job", async () => {
    vi.mocked(client.uploadVideo).mockResolvedValue(created);
    vi.mocked(client.getJob).mockResolvedValue(TWO_SPANS);
    const blobs: Blob[] = [];
    const create = vi.spyOn(URL, "createObjectURL").mockImplementation((b) => {
      if (b instanceof Blob) blobs.push(b);
      return "blob:test";
    });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    render(<App />);
    await chooseFile();
    await userEvent.click(await screen.findByRole("button", { name: /Download full result/ }));
    expect(click).toHaveBeenCalledTimes(1);
    const json = blobs.find((b) => b.type === "application/json");
    expect(json).toBeDefined();
    const text = await new Promise<string>((resolve) => {
      const reader = new FileReader(); // jsdom's Blob has no .text()
      reader.onload = () => resolve(reader.result as string);
      reader.readAsText(json!);
    });
    const body = JSON.parse(text) as Job;
    expect(body.result!.segments).toEqual(TWO_SPANS.result!.segments);
    expect(body.result!.frame_scores).toHaveLength(TWO_SPANS.result!.n_frames);
    create.mockRestore();
    click.mockRestore();
  });
});

describe("useJobPolling", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("polls every second until the job finishes, then stops", async () => {
    const fetchJob = vi
      .fn()
      .mockResolvedValueOnce(queued)
      .mockResolvedValueOnce(running)
      .mockResolvedValue(TWO_SPANS);
    const { result } = renderHook(() => useJobPolling("j1", fetchJob));
    await act(async () => {});
    expect(result.current.job?.status).toBe("queued");
    await act(async () => vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS));
    expect(result.current.job?.status).toBe("running");
    await act(async () => vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS));
    expect(result.current.job?.status).toBe("succeeded");
    await act(async () => vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS * 5));
    expect(fetchJob).toHaveBeenCalledTimes(3);
  });

  it("rides out a network blip, then recovers", async () => {
    const fetchJob = vi
      .fn()
      .mockRejectedValueOnce(new ApiError(0, "server_unreachable", "x"))
      .mockResolvedValue(TWO_SPANS);
    const { result } = renderHook(() => useJobPolling("j1", fetchJob));
    await act(async () => {});
    expect(result.current.reconnecting).toBe(true);
    expect(result.current.error).toBeNull();
    await act(async () => vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS));
    expect(result.current.job?.status).toBe("succeeded");
    expect(result.current.reconnecting).toBe(false);
  });

  it("gives up after repeated failures", async () => {
    const fetchJob = vi.fn().mockRejectedValue(new ApiError(0, "server_unreachable", "x"));
    const { result } = renderHook(() => useJobPolling("j1", fetchJob));
    await act(async () => vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS * (MAX_POLL_FAILURES + 2)));
    expect(result.current.error?.code).toBe("server_unreachable");
    expect(fetchJob).toHaveBeenCalledTimes(MAX_POLL_FAILURES);
  });

  it("a 404 (expired job) is final at once", async () => {
    const fetchJob = vi.fn().mockRejectedValue(new ApiError(404, "job_not_found", "x"));
    const { result } = renderHook(() => useJobPolling("j1", fetchJob));
    await act(async () => vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS * 3));
    expect(result.current.error?.code).toBe("job_not_found");
    expect(fetchJob).toHaveBeenCalledTimes(1);
  });

  it("does nothing without a job id", async () => {
    const fetchJob = vi.fn();
    renderHook(() => useJobPolling(null, fetchJob));
    await act(async () => vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS * 3));
    expect(fetchJob).not.toHaveBeenCalled();
  });
});

describe("format", () => {
  it("never shows a model as certain", async () => {
    const { formatPercent, formatTime } = await import("./lib/format");
    expect(formatPercent(0.99993)).toBe(">99.9%");
    expect(formatPercent(0.00004)).toBe("<0.1%");
    expect(formatPercent(1)).toBe("100%");
    expect(formatTime(6.96)).toBe("0:07.0");
    expect(formatTime(75.25)).toBe("1:15.3");
  });
});
