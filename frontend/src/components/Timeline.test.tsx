import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { FIXTURES, GROUND_TRUTH, PARTIAL, REAL, TWO_SPANS } from "../test/fixtures";
import { Timeline, spanGeometry } from "./Timeline";

describe("spanGeometry", () => {
  it("places a span by percent of the clip", () => {
    expect(spanGeometry({ start_s: 2, end_s: 3, confidence: 1 }, 10)).toEqual({ left: 20, width: 10 });
  });

  it("clamps spans that spill past either end", () => {
    const g = spanGeometry({ start_s: -1, end_s: 12, confidence: 1 }, 10);
    expect(g.left).toBe(0);
    expect(g.width).toBe(100);
  });
});

describe("Timeline", () => {
  it("draws one span per returned segment, at the returned times — every real fixture", () => {
    for (const [vid, job] of Object.entries(FIXTURES)) {
      const r = job.result!;
      const { unmount } = render(
        <Timeline duration={r.duration_s} segments={r.segments} onSeek={() => {}} />,
      );
      const spans = screen.queryAllByTestId("timeline-span");
      expect(spans, vid).toHaveLength(r.segments.length);
      spans.forEach((el, i) => {
        const seg = r.segments[i];
        const { left, width } = spanGeometry(seg, r.duration_s);
        expect(Number(el.dataset.start), vid).toBe(seg.start_s);
        expect(Number(el.dataset.end), vid).toBe(seg.end_s);
        expect(el.style.left).toBe(`${left}%`);
        expect(el.style.width).toBe(`max(${width}%, 4px)`);
      });
      unmount();
    }
  });

  it("puts the partially fake clip's span over its labelled forgery (P14-10)", () => {
    // dev_034103: the API's segment must overlap LAV-DF's fake_periods for this clip.
    const r = PARTIAL.result!;
    const [[gs, ge]] = GROUND_TRUTH.dev_034103.fake_periods;
    render(<Timeline duration={r.duration_s} segments={r.segments} onSeek={() => {}} />);
    const [span] = screen.getAllByTestId("timeline-span");
    const start = Number(span.dataset.start);
    const end = Number(span.dataset.end);
    const overlap = Math.min(end, ge) - Math.max(start, gs);
    expect(overlap / (Math.max(end, ge) - Math.min(start, gs))).toBeGreaterThan(0.8);
  });

  it("a real clip shows no spans and says so", () => {
    const r = REAL.result!;
    render(<Timeline duration={r.duration_s} segments={r.segments} onSeek={() => {}} />);
    expect(screen.queryAllByTestId("timeline-span")).toHaveLength(0);
    expect(screen.getByText("No manipulated segments found")).toBeInTheDocument();
  });

  it("clicking a span seeks to its start and reports which span", () => {
    const r = TWO_SPANS.result!;
    const onSeek = vi.fn();
    render(<Timeline duration={r.duration_s} segments={r.segments} onSeek={onSeek} />);
    fireEvent.click(screen.getAllByTestId("timeline-span")[1]);
    expect(onSeek).toHaveBeenCalledTimes(1); // the track's own click must not also fire
    expect(onSeek).toHaveBeenCalledWith(r.segments[1].start_s, 1);
  });

  it("spans are keyboard-reachable buttons with a spoken description", () => {
    const r = TWO_SPANS.result!;
    render(<Timeline duration={r.duration_s} segments={r.segments} onSeek={() => {}} />);
    const buttons = screen.getAllByRole("button", { name: /Manipulated segment \d/ });
    expect(buttons).toHaveLength(2);
    expect(buttons[0]).toHaveAccessibleName(/0:07\.0 to 0:08\.5.*Play from here/);
  });

  it("clicking the bare track seeks to that moment", () => {
    const onSeek = vi.fn();
    render(<Timeline duration={10} segments={[]} onSeek={onSeek} />);
    const track = screen.getByTestId("timeline-track");
    track.getBoundingClientRect = () => ({ left: 0, width: 200, top: 0, height: 48 }) as DOMRect;
    fireEvent.click(track, { clientX: 50 });
    expect(onSeek).toHaveBeenCalledWith(2.5, null);
  });

  it("moves the playhead with the current time", () => {
    render(<Timeline duration={10} segments={[]} currentTime={4} onSeek={() => {}} />);
    expect(screen.getByTestId("timeline-playhead").style.left).toBe("40%");
  });

  it("has a legend that names both states, not colour alone", () => {
    render(<Timeline duration={10} segments={[]} onSeek={() => {}} />);
    expect(screen.getByText("Manipulated (forged)")).toBeInTheDocument();
    expect(screen.getByText("No manipulation detected")).toBeInTheDocument();
  });
});
