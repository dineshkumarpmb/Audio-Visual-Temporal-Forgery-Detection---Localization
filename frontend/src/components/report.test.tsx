import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { Job } from "../api/types";
import { FIXTURES, PARTIAL, REAL, TWO_SPANS } from "../test/fixtures";
import { containerName, DetailsPanel } from "./DetailsPanel";
import { Downloads, framesCsv } from "./Downloads";
import { EvidencePanel } from "./EvidencePanel";
import { Filmstrip } from "./Filmstrip";
import { KeyFigures } from "./KeyFigures";
import { ResultView } from "./ResultView";
import { SegmentList } from "./SegmentList";
import { faceRuns, Timeline } from "./Timeline";

describe("fixtures carry the report detail", () => {
  it("every recorded response has evidence, both single-stream traces and keyframes", () => {
    for (const [vid, job] of Object.entries(FIXTURES)) {
      const r = job.result!;
      expect(r.evidence, vid).toBeTruthy();
      expect(r.audio_only_scores, vid).toHaveLength(r.n_frames);
      expect(r.visual_only_scores, vid).toHaveLength(r.n_frames);
      expect(r.face_found, vid).toHaveLength(r.n_frames);
      expect(r.evidence!.segments, vid).toHaveLength(r.segments.length);
      expect(r.keyframes!.length, vid).toBeGreaterThan(0);
    }
  });
});

describe("KeyFigures", () => {
  it("shows the eight figures with the API's numbers", () => {
    render(<KeyFigures result={TWO_SPANS.result!} />);
    const tiles = screen.getAllByTestId("key-figure");
    expect(tiles).toHaveLength(8);
    const kf = TWO_SPANS.result!.evidence!.key_figures;
    expect(screen.getByText(`${kf.forged_s.toFixed(2)} s`)).toBeInTheDocument();
    expect(within(tiles[1]).getByText("2")).toBeInTheDocument();
  });
});

describe("Timeline face band", () => {
  it("faceRuns turns a 1/0 trace into strip fractions", () => {
    expect(faceRuns([1, 1, 0, 0, 1])).toEqual([
      [0, 0.4],
      [0.8, 1],
    ]);
    expect(faceRuns([0, 0])).toEqual([]);
  });

  it("draws the band and names it in the legend when the trace is present", () => {
    const r = TWO_SPANS.result!;
    render(<Timeline duration={r.duration_s} segments={r.segments} onSeek={() => {}} faceFound={r.face_found} />);
    expect(screen.getByTestId("face-band")).toBeInTheDocument();
    expect(screen.getByText("Face tracked (thin band)")).toBeInTheDocument();
  });
});

describe("Filmstrip", () => {
  it("groups thumbnails by segment, marks in-span frames in words, and seeks on click", () => {
    const r = TWO_SPANS.result!;
    const onSeek = vi.fn();
    render(<Filmstrip keyframes={r.keyframes!} onSeek={onSeek} />);
    expect(screen.getAllByTestId("filmstrip-group")).toHaveLength(2);
    const start = r.keyframes!.find((k) => k.role === "start")!;
    fireEvent.click(screen.getAllByRole("button", { name: /starts/ })[0]);
    expect(onSeek).toHaveBeenCalledWith(start.time_s, start.segment);
  });

  it("a real video shows sampled frames, not segments", () => {
    render(<Filmstrip keyframes={REAL.result!.keyframes!} onSeek={() => {}} />);
    expect(screen.getByText("Frames sampled across the video")).toBeInTheDocument();
  });

  it("hides itself when every thumbnail has expired on the server", () => {
    const one = [TWO_SPANS.result!.keyframes![0]];
    const { container } = render(<Filmstrip keyframes={one} onSeek={() => {}} />);
    fireEvent.error(container.querySelector("img")!);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("EvidencePanel", () => {
  it("states the reading, keeps the caveat on screen, and draws both small multiples", () => {
    const r = TWO_SPANS.result!;
    render(<EvidencePanel result={r} currentTime={0} onSeek={() => {}} />);
    expect(screen.getByTestId("modality-reading")).toHaveTextContent(r.evidence!.modality.reading!);
    expect(screen.getByTestId("modality-caveat")).toHaveTextContent(/not how the video was edited/);
    expect(screen.getByTestId("chart-audio-only")).toBeInTheDocument();
    expect(screen.getByTestId("chart-visual-only")).toBeInTheDocument();
  });

  it("the face-only fake is attributed to the voice -- and the caveat says why that can happen", () => {
    // dev_005675 is a visual_only fake (LAV-DF label); the model leans on audio (PF-21).
    const r = FIXTURES.dev_005675.result!;
    expect(r.evidence!.segments.every((s) => s.driver === "voice")).toBe(true);
    render(<EvidencePanel result={r} currentTime={0} onSeek={() => {}} />);
    expect(screen.getByTestId("modality-caveat")).toHaveTextContent(/leans on the soundtrack/);
  });

  it("renders nothing for a result without the single-stream runs", () => {
    const old = { ...PARTIAL.result!, audio_only_scores: null, evidence: null };
    const { container } = render(<EvidencePanel result={old} currentTime={0} onSeek={() => {}} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("SegmentList with evidence", () => {
  it("adds peak, voice-only, face-only and the driving stream", () => {
    const r = TWO_SPANS.result!;
    render(<SegmentList segments={r.segments} evidence={r.evidence!.segments} activeIndex={null} onPlay={() => {}} />);
    expect(screen.getByRole("columnheader", { name: "Voice only" })).toBeInTheDocument();
    expect(screen.getAllByTestId("segment-driver").map((c) => c.textContent)).toEqual(["Voice", "Voice"]);
  });
});

describe("DetailsPanel", () => {
  it("names the container like a person would", () => {
    expect(containerName("mov,mp4,m4a,3gp,3g2,mj2", "clip.mp4")).toBe("MP4");
    expect(containerName("matroska,webm", "clip.mkv")).toBe("MKV");
    expect(containerName("avi", "x.avi")).toBe("AVI");
  });

  it("shows the input, and keeps the technical details collapsed", () => {
    render(<DetailsPanel job={TWO_SPANS} info={null} />);
    expect(screen.getByText(/MP4 · video h264/)).toBeInTheDocument();
    expect(screen.getByTestId("technical-details")).not.toHaveAttribute("open");
  });
});

describe("Downloads", () => {
  it("targets the server's PDF for this job", () => {
    render(<Downloads job={TWO_SPANS} />);
    expect(screen.getByTestId("download-pdf")).toHaveAttribute(
      "data-href",
      `/api/v1/jobs/${TWO_SPANS.job_id}/report.pdf`,
    );
  });

  it("the CSV has one row per frame and every per-frame trace", () => {
    const csv = framesCsv(TWO_SPANS).trim().split("\n");
    const r = TWO_SPANS.result!;
    expect(csv).toHaveLength(r.n_frames + 1);
    expect(csv[0]).toBe(
      "frame,time_s,score,voice_only,face_only,boundary_start,boundary_end,face_found,attention,in_segment",
    );
    const inSpan = csv.slice(1).filter((line) => line.endsWith(",1")).length;
    const expected = r.segments.reduce((a, s) => a + Math.round((s.end_s - s.start_s) * r.fps), 0);
    expect(inSpan).toBe(expected);
  });
});

describe("ResultView", () => {
  it("assembles every section for a partially fake video", () => {
    render(<ResultView job={TWO_SPANS} videoUrl={null} onReset={() => {}} />);
    expect(screen.getByRole("heading", { name: "Likely manipulated" })).toBeInTheDocument();
    expect(screen.getAllByTestId("key-figure")).toHaveLength(8);
    expect(screen.getAllByTestId("timeline-span")).toHaveLength(2);
    expect(screen.getByRole("heading", { name: "Keyframes" })).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Manipulated segments" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Voice or face?" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "The input" })).toBeInTheDocument();
    expect(screen.getByTestId("download-pdf")).toBeInTheDocument();
  });

  it("a real video has no segment table but still the evidence and downloads", () => {
    const job: Job = REAL;
    render(<ResultView job={job} videoUrl={null} onReset={() => {}} />);
    expect(screen.queryByRole("table", { name: "Manipulated segments" })).toBeNull();
    expect(screen.getByRole("heading", { name: "No manipulation detected" })).toBeInTheDocument();
    expect(screen.getByTestId("download-pdf")).toBeInTheDocument();
  });
});
