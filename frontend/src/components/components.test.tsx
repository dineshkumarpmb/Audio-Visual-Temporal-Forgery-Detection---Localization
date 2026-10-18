import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { AnalysisResult } from "../api/types";
import { PARTIAL, REAL, TWO_SPANS } from "../test/fixtures";
import { ScoreChart, linePath } from "./ScoreChart";
import { SegmentList } from "./SegmentList";
import { ErrorPanel } from "./StatusPanels";
import { Upload } from "./Upload";
import { Verdict, summarise } from "./Verdict";

const file = (name: string, bytes = 10, type = "video/mp4") =>
  new File([new Uint8Array(bytes)], name, { type });

describe("Verdict", () => {
  it("fake: headline, segment count, share of the video, score", () => {
    render(<Verdict result={TWO_SPANS.result!} />);
    expect(screen.getByRole("heading", { name: "Likely manipulated" })).toBeInTheDocument();
    expect(screen.getByText(/2 manipulated segments found/)).toBeInTheDocument();
    expect(screen.getByTestId("confidence")).toHaveTextContent(/>99\.9%/);
  });

  it("real: says nothing was flagged", () => {
    render(<Verdict result={REAL.result!} />);
    expect(screen.getByRole("heading", { name: "No manipulation detected" })).toBeInTheDocument();
    expect(screen.getByText("No part of this video was flagged as manipulated.")).toBeInTheDocument();
  });

  it("covers the awkward combinations in words", () => {
    const base = PARTIAL.result!;
    expect(summarise({ ...base, segments: [] })).toMatch(/no specific moment stood out/);
    expect(summarise({ ...base, is_fake: false })).toMatch(/looks genuine, though 1 short moment was flagged/);
  });

  it("shows warnings as text with a warning label", () => {
    const r: AnalysisResult = { ...REAL.result!, warnings: ["the audio track is silent; the score is unreliable"] };
    render(<Verdict result={r} />);
    expect(screen.getByText(/The audio track is silent/)).toBeInTheDocument();
    expect(screen.getByText("Warning:")).toBeInTheDocument();
  });
});

describe("ScoreChart", () => {
  it("linePath has one vertex per frame, scores mapped top-down", () => {
    const d = linePath([0, 1, 0.5], 100, 50);
    expect(d).toBe("M0.00,50.00L50.00,0.00L100.00,25.00");
  });

  it("renders the line, the forged spans behind it, and the threshold", () => {
    const r = TWO_SPANS.result!;
    render(
      <ScoreChart scores={r.frame_scores} fps={r.fps} segments={r.segments} threshold={0.2} onSeek={() => {}} />,
    );
    const d = screen.getByTestId("chart-line").getAttribute("d")!;
    expect(d.split(/[ML]/).filter(Boolean)).toHaveLength(r.frame_scores.length);
    expect(screen.getAllByTestId("chart-span")).toHaveLength(2);
    expect(screen.getByTestId("chart-threshold")).toHaveTextContent("segment threshold 20%");
  });

  it("omits the threshold when the API did not report one", () => {
    render(<ScoreChart scores={[0.1, 0.2]} fps={25} segments={[]} onSeek={() => {}} />);
    expect(screen.queryByTestId("chart-threshold")).toBeNull();
  });
});

describe("SegmentList", () => {
  it("one row per segment with exact times; Play seeks", () => {
    const r = TWO_SPANS.result!;
    const onPlay = vi.fn();
    render(<SegmentList segments={r.segments} activeIndex={null} onPlay={onPlay} />);
    expect(screen.getAllByRole("row")).toHaveLength(3); // header + 2
    expect(screen.getByText("0:07.0")).toBeInTheDocument(); // 6.96 s
    fireEvent.click(screen.getAllByRole("button", { name: "Play" })[1]);
    expect(onPlay).toHaveBeenCalledWith(r.segments[1].start_s, 1);
  });

  it("renders nothing for a real clip", () => {
    const { container } = render(<SegmentList segments={[]} activeIndex={null} onPlay={() => {}} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("Upload", () => {
  it("accepts a supported video", async () => {
    const onFile = vi.fn();
    render(<Upload onFile={onFile} />);
    await userEvent.upload(screen.getByTestId("file-input"), file("clip.MP4"));
    expect(onFile).toHaveBeenCalledTimes(1);
  });

  it("refuses an unsupported type before uploading", async () => {
    const onFile = vi.fn();
    render(<Upload onFile={onFile} />);
    await userEvent.upload(screen.getByTestId("file-input"), file("clip.webm", 10, "video/webm"), {
      applyAccept: false,
    });
    expect(onFile).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent(/isn't a supported video/);
  });

  it("refuses a file over the server's limit", async () => {
    const onFile = vi.fn();
    render(<Upload onFile={onFile} limits={{ max_upload_mb: 0.001, allowed_extensions: [".mp4"] }} />);
    await userEvent.upload(screen.getByTestId("file-input"), file("big.mp4", 4096));
    expect(onFile).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent(/the limit is 0.001 MB/);
  });

  it("accepts a dropped file", () => {
    const onFile = vi.fn();
    render(<Upload onFile={onFile} />);
    fireEvent.drop(screen.getByRole("button", { name: "Choose a video to analyse" }), {
      dataTransfer: { files: [file("drop.mov", 10, "video/quicktime")] },
    });
    expect(onFile).toHaveBeenCalledTimes(1);
  });
});

describe("ErrorPanel", () => {
  it.each([
    ["no_face", "No face was found"],
    ["no_audio", "This video has no sound"],
    ["corrupt_video", "The video couldn't be read"],
    ["queue_full", "The server is busy"],
    ["something_new", "Something went wrong"],
  ])("%s → plain-language title", (code, title) => {
    render(<ErrorPanel code={code} onReset={() => {}} />);
    expect(screen.getByRole("heading", { name: title })).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent(`error code: ${code}`);
  });
});
