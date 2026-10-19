"""The PDF report: one finished job -> a self-contained, printable document.

    GET /api/v1/jobs/{id}/report.pdf

Everything in it comes from the job the API already returned -- the verdict, the arrays and the
derived evidence (`src/inference/explain.py`) -- so the PDF and the result page cannot disagree.
Charts are drawn as vector graphics with reportlab's own primitives (no raster screenshots), in
the page's colours: forged spans are the reserved critical red *and* a 45° hatch, never colour
alone, so the report also survives a greyscale printer.

Pages: (1) verdict, key figures, timeline, score chart; (2) evidence -- which stream drove
each span, the single-stream traces, the segment table and keyframes; (3) the input and how
it was analysed; (4) how to read the report and what the model cannot do.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# -- palette: the frontend's light-theme roles (frontend/src/index.css) ----------------------
INK = colors.HexColor("#0b0b0b")
INK_2 = colors.HexColor("#52514e")
MUTED = colors.HexColor("#898781")
GRID = colors.HexColor("#e1e0d9")
AXIS = colors.HexColor("#c3c2b7")
TRACK = colors.HexColor("#e8e7e1")
SURFACE = colors.HexColor("#fcfcfb")
ACCENT = colors.HexColor("#2a78d6")
FAKE = colors.HexColor("#d03b3b")
FAKE_WASH = colors.HexColor("#f8e2e2")
REAL = colors.HexColor("#0ca30c")
WARN = colors.HexColor("#8a5a00")
SERIES_AUDIO = colors.HexColor("#eb6834")  # categorical slot 2 (orange) - distinct from red
SERIES_VISUAL = colors.HexColor("#6250d6")  # categorical slot 7 (violet)

PAGE_W, PAGE_H = A4
MARGIN = 16 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

# Measured once on LAV-DF's held-out test split for this model (reports/test_once.md, P11-8).
TEST_FIGURES = {
    "clip ROC-AUC": "0.984",
    "segment AP@0.5": "0.80",
    "segment AP@0.5, face-only edits": "0.71",
    "false segments on real videos": "2.6%",
}

DRIVER_WORDS = {
    "voice": "Voice",
    "face": "Face",
    "both": "Both",
    "neither": "Unclear",
    "unknown": "—",
}


# ---------------------------------------------------------------- text


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5,
                          leading=13, textColor=INK, alignment=TA_LEFT)  # fmt: skip
    return {
        "title": ParagraphStyle(
            "title", parent=body, fontName="Helvetica-Bold", fontSize=20, leading=24, spaceAfter=2
        ),  # fmt: skip
        "subtitle": ParagraphStyle("subtitle", parent=body, textColor=INK_2, fontSize=10),
        "h2": ParagraphStyle(
            "h2",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            spaceBefore=10,
            spaceAfter=4,
        ),  # fmt: skip
        "h3": ParagraphStyle(
            "h3",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=14,
            spaceBefore=6,
            spaceAfter=2,
        ),  # fmt: skip
        "verdict": ParagraphStyle(
            "verdict", parent=body, fontName="Helvetica-Bold", fontSize=17, leading=21
        ),  # fmt: skip
        "body": body,
        "small": ParagraphStyle("small", parent=body, fontSize=8, leading=10.5, textColor=INK_2),
        "tile_value": ParagraphStyle(
            "tv", parent=body, fontName="Helvetica-Bold", fontSize=14, leading=17
        ),  # fmt: skip
        "tile_label": ParagraphStyle(
            "tl", parent=body, fontSize=7.5, leading=9.5, textColor=INK_2
        ),  # fmt: skip
        "cell": ParagraphStyle("cell", parent=body, fontSize=8, leading=10),
    }


def _bullets(items: list[str], st: dict) -> list[Paragraph]:
    """One paragraph per point, hanging indent -- markup in items is kept, so escape first."""
    style = ParagraphStyle("bullet", parent=st["body"], leftIndent=10, bulletIndent=0,
                           spaceAfter=2)  # fmt: skip
    return [Paragraph(item, style, bulletText="•") for item in items]


def container_name(demuxer: str, filename: str) -> str:
    """ffmpeg reports a demuxer alias list ("mov,mp4,m4a,3gp,3g2,mj2"); a reader wants "MP4"."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    aliases = demuxer.split(",") if demuxer else []
    if ext and ext in aliases:
        return ext.upper()
    friendly = {"matroska": "MKV", "avi": "AVI", "mov": "MOV"}
    return next((friendly[a] for a in aliases if a in friendly), demuxer or "?")


def _esc(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt_time(s: float) -> str:
    m = int(abs(s) // 60)
    return f"{m}:{abs(s) - m * 60:04.1f}"


def fmt_pct(p: float, digits: int = 0) -> str:
    v = p * 100
    if 99.9 < v < 100:
        return ">99.9%"
    if 0 < v < 0.1:
        return "<0.1%"
    return f"{v:.{digits}f}%"


# ---------------------------------------------------------------- drawings


def _hatch(d: Drawing, x0: float, y0: float, w: float, h: float, step: float = 4.0) -> None:
    """45° hatch lines clipped to a rectangle -- the colour-independent mark of a forged span."""
    c = x0 - h
    while c < x0 + w:
        sx, sy = c, y0
        ex, ey = c + h, y0 + h
        if sx < x0:
            sy += x0 - sx
            sx = x0
        if ex > x0 + w:
            ey -= ex - (x0 + w)
            ex = x0 + w
        if ex > sx:
            d.add(Line(sx, sy, ex, ey, strokeColor=colors.white, strokeWidth=0.8))
        c += step


def _ticks(duration: float) -> list[float]:
    step = next((s for s in (0.5, 1, 2, 5, 10, 15, 30, 60, 120) if duration / s <= 8), 300)
    out, t = [], 0.0
    while t <= duration + 1e-9:
        out.append(t)
        t += step
    return out


def timeline_drawing(result: dict, width: float = CONTENT_W) -> Drawing:
    """The clip as a strip: forged spans red + hatch, face-tracked frames as a band below."""
    duration = float(result["duration_s"]) or 1.0
    h = 74
    d = Drawing(width, h)
    track_y, track_h = 30, 26
    d.add(Rect(0, track_y, width, track_h, fillColor=TRACK, strokeColor=AXIS, strokeWidth=0.5))
    for seg in result["segments"]:
        x0 = width * max(seg["start_s"], 0) / duration
        x1 = width * min(seg["end_s"], duration) / duration
        w = max(x1 - x0, 2.0)
        d.add(Rect(x0, track_y, w, track_h, fillColor=FAKE, strokeColor=None))
        _hatch(d, x0, track_y, w, track_h)
    face = result.get("face_found")
    if face:
        n = len(face)
        d.add(String(0, 22, "face tracked", fontName="Helvetica", fontSize=6.5, fillColor=MUTED))
        band_y, band_h = 14, 5
        d.add(Rect(0, band_y, width, band_h, fillColor=TRACK, strokeColor=None))
        start = None
        for i in range(n + 1):
            on = i < n and face[i]
            if on and start is None:
                start = i
            elif not on and start is not None:
                d.add(Rect(width * start / n, band_y, width * (i - start) / n, band_h,
                           fillColor=INK_2, strokeColor=None))  # fmt: skip
                start = None
    for t in _ticks(duration):
        x = width * t / duration
        anchor = "start" if t == 0 else ("end" if x > width - 12 else "middle")
        d.add(String(x, 2, fmt_time(t), fontName="Helvetica", fontSize=6.5, fillColor=MUTED,
                     textAnchor=anchor))  # fmt: skip
    return d


def line_chart(
    series: list[tuple[list[float], Any, float]],
    duration: float,
    segments: list[dict],
    *,
    width: float = CONTENT_W,
    height: float = 120,
    threshold: float | None = None,
    label: str | None = None,
) -> Drawing:
    """Score traces on one 0-100% axis (never two y-scales), forged spans washed behind."""
    left, bottom, top = 30, 16, 8
    pw, ph = width - left - 6, height - bottom - top
    d = Drawing(width, height)
    duration = duration or 1.0
    for seg in segments:
        x0 = left + pw * max(seg["start_s"], 0) / duration
        x1 = left + pw * min(seg["end_s"], duration) / duration
        d.add(Rect(x0, bottom, max(x1 - x0, 1.5), ph, fillColor=FAKE_WASH, strokeColor=None))
    for v in (0.0, 0.5, 1.0):
        y = bottom + ph * v
        d.add(Line(left, y, left + pw, y, strokeColor=AXIS if v == 0 else GRID, strokeWidth=0.5))
        d.add(String(left - 4, y - 2.5, f"{int(v * 100)}%", fontName="Helvetica", fontSize=6.5,
                     fillColor=MUTED, textAnchor="end"))  # fmt: skip
    if threshold is not None:
        y = bottom + ph * threshold
        d.add(Line(left, y, left + pw, y, strokeColor=INK_2, strokeWidth=0.6,
                   strokeDashArray=[3, 2]))  # fmt: skip
        d.add(String(left + pw, y + 2.5, f"segment threshold {fmt_pct(threshold)}",
                     fontName="Helvetica", fontSize=6.5, fillColor=INK_2,
                     textAnchor="end"))  # fmt: skip
    for values, colour, stroke in series:
        n = len(values)
        if n == 0:
            continue
        pts: list[float] = []
        for i, v in enumerate(values):
            x = left + (pw / 2 if n == 1 else pw * i / (n - 1))
            pts += [x, bottom + ph * min(max(float(v), 0.0), 1.0)]
        d.add(PolyLine(pts, strokeColor=colour, strokeWidth=stroke, strokeLineJoin=1))
    for t, anchor in ((0.0, "start"), (duration / 2, "middle"), (duration, "end")):
        d.add(String(left + pw * t / duration, 3, fmt_time(t), fontName="Helvetica",
                     fontSize=6.5, fillColor=MUTED, textAnchor=anchor))  # fmt: skip
    if label:
        d.add(String(left + 2, bottom + ph - 9, label, fontName="Helvetica-Bold", fontSize=7.5,
                     fillColor=INK_2))  # fmt: skip
    return d


# ---------------------------------------------------------------- blocks


def _verdict_block(result: dict, st: dict) -> Table:
    fake = result["is_fake"]
    ev = result.get("evidence") or {}
    title = "Likely manipulated" if fake else "No manipulation detected"
    summary = ev.get("summary", "")
    score = (
        f"Manipulation score for the whole video: <b>{fmt_pct(result['clip_score'], 1)}</b> "
        f"(flagged above {fmt_pct(result['threshold'])})"
    )
    badge = Drawing(28, 28)
    badge.add(Rect(0, 0, 28, 28, rx=14, ry=14, fillColor=FAKE if fake else REAL, strokeColor=None))
    badge.add(String(14, 8.5, "!" if fake else "OK", fontName="Helvetica-Bold",
                     fontSize=13 if fake else 9.5, fillColor=colors.white,
                     textAnchor="middle"))  # fmt: skip
    body = [
        Paragraph(title, st["verdict"]),
        Paragraph(_esc(summary), st["body"]),
        Paragraph(score, st["small"]),
    ]
    t = Table([[badge, body]], colWidths=[36, CONTENT_W - 36])
    t.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 1.5, FAKE if fake else REAL),
                ("BACKGROUND", (0, 0), (-1, -1), FAKE_WASH if fake else SURFACE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return t


def _tiles(result: dict, st: dict) -> Table:
    kf = (result.get("evidence") or {}).get("key_figures") or {}
    timings = result.get("timings_s") or {}
    items = [
        (fmt_pct(result["clip_score"], 1), "manipulation score"),
        (str(len(result["segments"])), "manipulated segments"),
        (f"{kf.get('forged_s', 0):.2f} s", "flagged time"),
        (fmt_pct(kf.get("forged_fraction", 0), 1), "of the video flagged"),
        (f"{result['duration_s']:.1f} s", "video length"),
        (fmt_pct(result["face_found_fraction"]), "frames with a face"),
        (fmt_pct(kf.get("peak_frame_score", 0)), "highest frame score"),
        (f"{timings.get('total', 0):.1f} s", "analysis time"),
    ]
    cells = [[Paragraph(v, st["tile_value"]), Paragraph(lbl, st["tile_label"])] for v, lbl in items]
    rows = [cells[:4], cells[4:]]
    t = Table(rows, colWidths=[CONTENT_W / 4] * 4)
    t.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, GRID),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


def _kv_table(rows: list[tuple[str, str]], st: dict, col1: float = 48 * mm) -> Table:
    data = [[Paragraph(_esc(k), st["small"]), Paragraph(_esc(v), st["cell"])] for k, v in rows]
    t = Table(data, colWidths=[col1, CONTENT_W - col1])
    t.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def _segment_table(result: dict, st: dict) -> Table | Paragraph:
    segs = (result.get("evidence") or {}).get("segments") or []
    if not segs:
        return Paragraph("No manipulated segments were found.", st["body"])
    head = ["#", "From", "To", "Length", "Score", "Peak", "Voice only", "Face only",
            "Driven by", "Face"]  # fmt: skip

    def pct(v: float | None) -> str:
        return "—" if v is None else fmt_pct(v)

    rows = [head] + [
        [
            str(s["index"] + 1),
            fmt_time(s["start_s"]),
            fmt_time(s["end_s"]),
            f"{s['duration_s']:.2f} s",
            pct(s["mean_score"]),
            pct(s["peak_score"]),
            pct(s["audio_only_mean"]),
            pct(s["visual_only_mean"]),
            DRIVER_WORDS.get(s["driver"], s["driver"]),
            pct(s["face_coverage"]),
        ]
        for s in segs
    ]
    widths = [8, 15, 15, 16, 14, 14, 18, 18, 20, 14]
    t = Table(rows, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.5),
                ("FONT", (0, 1), (-1, -1), "Helvetica", 8),
                ("TEXTCOLOR", (0, 0), (-1, 0), INK_2),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, GRID),
                ("ALIGN", (3, 0), (-1, -1), "RIGHT"),
                ("ALIGN", (8, 0), (8, -1), "LEFT"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def _keyframe_rows(result: dict, thumbs: list[bytes], st: dict) -> list:
    metas = result.get("keyframes") or []
    if not metas or not thumbs:
        return []
    groups: dict[Any, list[int]] = {}
    for i, k in enumerate(metas):
        groups.setdefault(k["segment"], []).append(i)
    out: list = []
    per_row = 5
    cell_w = CONTENT_W / per_row - 4
    for seg, idxs in groups.items():
        title = "Sampled frames" if seg is None else f"Segment {seg + 1}"
        cells = []
        for i in idxs[:per_row]:
            k = metas[i]
            if i >= len(thumbs):
                continue
            iw, ih = ImageReader(io.BytesIO(thumbs[i])).getSize()
            img = Image(io.BytesIO(thumbs[i]), width=cell_w, height=cell_w * ih / iw)
            cap = f"{k['role']} · {fmt_time(k['time_s'])}" + (
                "" if k["face_found"] else " · no face"
            )
            cells.append([img, Paragraph(_esc(cap), st["small"])])
        if not cells:
            continue
        while len(cells) < per_row:
            cells.append("")
        t = Table([cells], colWidths=[CONTENT_W / per_row] * per_row)
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 2),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))  # fmt: skip
        out.append(KeepTogether([Paragraph(title, st["h3"]), t]))
    return out


# ---------------------------------------------------------------- page furniture


def _numbered_canvas(header: str, footer: str):
    class NumberedCanvas(rl_canvas.Canvas):
        """Two-pass page numbers ("Page 2 of 4") plus a header and footer on every page."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._states: list[dict] = []

        def showPage(self):  # noqa: N802 - reportlab API
            self._states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._states)
            for state in self._states:
                self.__dict__.update(state)
                self._furniture(total)
                super().showPage()
            super().save()

        def _furniture(self, total: int) -> None:
            self.setFont("Helvetica", 7.5)
            self.setFillColor(MUTED)
            self.drawString(MARGIN, PAGE_H - 10 * mm, header)
            self.drawRightString(PAGE_W - MARGIN, PAGE_H - 10 * mm, "AVTFD forgery report")
            self.setStrokeColor(GRID)
            self.setLineWidth(0.5)
            self.line(MARGIN, PAGE_H - 11.5 * mm, PAGE_W - MARGIN, PAGE_H - 11.5 * mm)
            self.line(MARGIN, 11.5 * mm, PAGE_W - MARGIN, 11.5 * mm)
            self.drawString(MARGIN, 8 * mm, footer)
            self.drawRightString(PAGE_W - MARGIN, 8 * mm, f"Page {self._pageNumber} of {total}")

    return NumberedCanvas


# ---------------------------------------------------------------- the document


def build_report(job: dict, model_info: dict, thumbs: list[bytes]) -> bytes:
    """`job` is the `GET /api/v1/jobs/{id}` body of a succeeded job."""
    st = _styles()
    result = job["result"]
    media = job.get("media") or {}
    ev = result.get("evidence") or {}
    duration = float(result["duration_s"])
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    filename = job.get("filename", "video")

    story: list = [
        Paragraph("Forgery analysis report", st["title"]),
        Paragraph(
            f"{_esc(filename)} · analysed {generated} · job {_esc(job['job_id'][:8])}",
            st["subtitle"],
        ),
        Spacer(1, 8),
        _verdict_block(result, st),
        Spacer(1, 8),
        _tiles(result, st),
        Paragraph("Timeline", st["h2"]),
        Paragraph(
            "The whole video, left to right. Red hatched blocks are the moments the detector "
            "believes were altered; the thin band below shows where a face was tracked.",
            st["small"],
        ),
        Spacer(1, 4),
        timeline_drawing(result),
        Paragraph("Manipulation score, frame by frame", st["h2"]),
        Paragraph(
            "How likely each moment is to be manipulated. Segments are drawn where the score "
            "stays above the segment threshold.",
            st["small"],
        ),
        Spacer(1, 4),
        line_chart(
            [(result["frame_scores"], ACCENT, 1.4)],
            duration,
            result["segments"],
            threshold=ev.get("frame_threshold"),
        ),  # fmt: skip
        PageBreak(),
        Paragraph("Evidence", st["h2"]),
    ]

    modality = ev.get("modality") or {}
    if modality.get("available"):
        story += [
            Paragraph("<b>Voice or face?</b> " + _esc(modality["reading"]), st["body"]),
            Spacer(1, 3),
            _kv_table(
                [
                    ("Both streams (the verdict)", fmt_pct(modality["both"], 1)),
                    ("Voice only (video removed)", fmt_pct(modality["audio_only"], 1)),
                    ("Face only (audio removed)", fmt_pct(modality["visual_only"], 1)),
                ],
                st,
                col1=62 * mm,
            ),
            Paragraph(_esc(modality["caveat"]), st["small"]),
            Spacer(1, 6),
        ]
    if result.get("audio_only_scores") and result.get("visual_only_scores"):
        half = (CONTENT_W - 8) / 2
        charts = Table(
            [[
                line_chart([(result["audio_only_scores"], SERIES_AUDIO, 1.2)], duration,
                           result["segments"], width=half, height=96, label="Voice only"),
                line_chart([(result["visual_only_scores"], SERIES_VISUAL, 1.2)], duration,
                           result["segments"], width=half, height=96, label="Face only"),
            ]],
            colWidths=[half + 4, half + 4],
        )  # fmt: skip
        story += [
            Paragraph("The same video scored with one stream removed", st["h3"]),
            charts,
        ]
    story += [Paragraph("Manipulated segments", st["h3"]), _segment_table(result, st)]
    story += [Spacer(1, 4)] + _keyframe_rows(result, thumbs, st)

    timings = result.get("timings_s") or {}
    post = model_info.get("postprocess") or {}
    story += [
        PageBreak(),
        Paragraph("The input", st["h2"]),
        _kv_table(
            [
                ("File", filename),
                ("Format / codecs", f"{container_name(media.get('container', ''), filename)} · "
                 f"video {media.get('video_codec', '?')} · audio {media.get('audio_codec', '?')}"),
                ("Resolution", f"{media.get('width', '?')} × {media.get('height', '?')}"),
                ("Frame rate (source)", f"{media.get('fps') or '?'} fps"),
                ("Length", f"{media.get('duration_s', duration)} s"),
                ("Size", f"{media.get('size_bytes', 0) / 1024**2:.1f} MB"),
                ("Frames analysed", f"{result['n_frames']} at {result['fps']:g} fps"),
                ("Face tracked", fmt_pct(result["face_found_fraction"]) + " of frames"),
                ("Audio", "silent" if result.get("audio_silent") else "present"),
            ],
            st,
        ),
    ]  # fmt: skip
    if result.get("warnings"):
        story.append(Paragraph("Warnings", st["h3"]))
        for w in result["warnings"]:
            story.append(Paragraph("• " + _esc(w[0].upper() + w[1:]), st["body"]))
    story += [
        Paragraph("How it was analysed", st["h2"]),
        _kv_table(
            [
                ("Model", f"{model_info.get('name', '?')} (sha256 {str(model_info.get('sha256', '?'))[:12]}…)"),
                ("Architecture", f"{model_info.get('backbone', '?')} face features + "
                 f"{model_info.get('audio_feature', '?')} audio, cross-attention + Transformer"),
                ("Device", str(model_info.get("device", "?"))),
                ("Verdict threshold", fmt_pct(result["threshold"])),
                ("Segment threshold", fmt_pct(ev.get("frame_threshold", post.get("threshold", 0.5)))),
                ("Post-processing", ", ".join(f"{k} {v}" for k, v in post.items() if k != "fps")),
                ("Time per stage", " · ".join(f"{k} {v:.2f} s" for k, v in timings.items()
                                              if k != "total")),
                ("Total analysis time", f"{timings.get('total', 0):.2f} s"),
            ],
            st,
        ),
    ]  # fmt: skip
    story += [
        Paragraph("How to read this report", st["h2"]),
        *_bullets(
            [
                "The <b>verdict</b> is the model's estimate that the video was manipulated "
                "anywhere in it.",
                "The <b>timeline</b> and <b>segments</b> say where: each is a stretch whose "
                "frame-by-frame score stayed above the segment threshold.",
                "<b>Voice only</b> and <b>face only</b> rerun the same model with one stream "
                "removed; they show what the model relied on, not how the video was made.",
            ],
            st,
        ),
        Paragraph("What this model can and cannot do", st["h2"]),
        Paragraph(
            "It was trained and evaluated on one research dataset, LAV-DF, in which a word or "
            "two of a real speech video has its face, its voice or both replaced. Measured "
            "once on that dataset's held-out test split:",
            st["body"],
        ),
        Spacer(1, 3),
        _kv_table(list(TEST_FIGURES.items()), st, col1=62 * mm),
        Spacer(1, 6),
        *_bullets(
            [
                "It leans on the soundtrack for nearly every decision, so edits to the face "
                "alone are found less reliably — and may be attributed to the voice.",
                "Videos unlike LAV-DF — other languages, other generators, heavy compression, "
                "no clear speaking face — are outside what was measured; scores there may be "
                "wrong.",
                "A high score is a reason to look closer, not proof. A low score is not proof "
                "that a video is genuine.",
            ],
            st,
        ),
        Spacer(1, 10),
        Paragraph(
            "Generated locally by the AVTFD API; the video was not sent anywhere. The full "
            "machine-readable result is available as JSON from the same page.",
            st["small"],
        ),
    ]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN + 2 * mm,
        bottomMargin=MARGIN,
        title=f"Forgery analysis report — {filename}",
        author="AVTFD",
        subject="Audio-visual temporal forgery detection and localization",
    )
    footer = f"Generated {generated} · model {model_info.get('name', '?')} · job {job['job_id']}"
    doc.build(story, canvasmaker=_numbered_canvas(f"Forgery analysis report — {filename}", footer))
    return buf.getvalue()
