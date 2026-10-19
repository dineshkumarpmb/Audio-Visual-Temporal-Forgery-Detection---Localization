import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { filenameFrom } from "../api/client";
import { TWO_SPANS } from "../test/fixtures";
import { Downloads } from "./Downloads";

/**
 * The PDF button must never save something that is not a PDF. The bug this guards: with the
 * API process older than the page (Vite hot-reloads, uvicorn does not), the old `<a download>`
 * saved the server's 66-byte JSON 404 as a "PDF" that would not open.
 */

const PDF_BYTES = new Uint8Array([0x25, 0x50, 0x44, 0x46, 0x2d, 0x31, 0x2e, 0x34]); // %PDF-1.4

function respond(body: BodyInit, status: number, headers: Record<string, string>) {
  return vi.fn().mockResolvedValue(new Response(body, { status, headers }));
}

function errorBody(code: string) {
  return JSON.stringify({ error: { code, message: code, detail: null } });
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

async function clickPdf() {
  const saved: { href: string; download: string }[] = [];
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
    saved.push({ href: this.href, download: this.download });
  });
  render(<Downloads job={TWO_SPANS} />);
  await userEvent.click(screen.getByTestId("download-pdf"));
  return saved;
}

describe("PDF download", () => {
  it("saves the PDF under the server's filename", async () => {
    const fetchMock = respond(PDF_BYTES, 200, {
      "content-type": "application/pdf",
      "content-disposition": 'attachment; filename="dev_004623-forgery-report.pdf"',
    });
    vi.stubGlobal("fetch", fetchMock);
    const saved = await clickPdf();
    await waitFor(() => expect(saved).toHaveLength(1));
    expect(fetchMock).toHaveBeenCalledWith(`/api/v1/jobs/${TWO_SPANS.job_id}/report.pdf`, expect.anything());
    expect(saved[0].download).toBe("dev_004623-forgery-report.pdf");
    expect(screen.queryByTestId("pdf-error")).toBeNull();
    expect(screen.getByTestId("download-pdf")).toHaveTextContent("Download PDF report");
  });

  it("an API process older than the page is explained, and nothing is saved", async () => {
    vi.stubGlobal("fetch", respond(errorBody("not_found"), 404, { "content-type": "application/json" }));
    const saved = await clickPdf();
    expect(await screen.findByTestId("pdf-error")).toHaveTextContent("can't make PDF reports yet");
    expect(screen.getByTestId("pdf-error")).toHaveTextContent("make api");
    expect(saved).toHaveLength(0);
  });

  it.each([
    ["job_not_found", 404, "This result has expired"],
    ["job_not_ready", 409, "hasn't finished"],
  ])("%s is explained", async (code, status, text) => {
    vi.stubGlobal("fetch", respond(errorBody(code), status, { "content-type": "application/json" }));
    const saved = await clickPdf();
    expect(await screen.findByTestId("pdf-error")).toHaveTextContent(text);
    expect(saved).toHaveLength(0);
  });

  it("a 200 that is not a PDF is refused rather than saved", async () => {
    vi.stubGlobal("fetch", respond("<html>proxy page</html>", 200, { "content-type": "text/html" }));
    const saved = await clickPdf();
    expect(await screen.findByTestId("pdf-error")).toHaveTextContent("Something went wrong");
    expect(saved).toHaveLength(0);
  });

  it("an unreachable server is explained", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await clickPdf();
    expect(await screen.findByTestId("pdf-error")).toHaveTextContent("Can't reach the analysis server");
  });

  it("filenameFrom reads both plain and RFC 5987 forms", () => {
    expect(filenameFrom('attachment; filename="a-forgery-report.pdf"')).toBe("a-forgery-report.pdf");
    expect(filenameFrom("attachment; filename*=UTF-8''r%C3%A9sum%C3%A9.pdf")).toBe("résumé.pdf");
    expect(filenameFrom(null)).toBeNull();
  });
});
