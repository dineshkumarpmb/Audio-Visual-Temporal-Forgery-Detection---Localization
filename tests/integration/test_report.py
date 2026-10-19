"""The PDF report and the keyframe endpoint (`api/services/report.py`, `api/routes/analyze.py`).

The document is rendered from the *recorded real API responses* the frontend's tests use
(`frontend/src/test/fixtures/`), then read back with pypdf: right page count, the verdict
and every section heading present, each segment's times printed, and the caveat about voice
attribution never dropped. The endpoints are exercised on the real app with a fake analyzer.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from api.main import create_app
from api.services.report import build_report, container_name, fmt_time
from api.settings import ApiSettings
from src.inference.keyframes import Keyframe, render_thumbnail

FIXTURES = Path("frontend/src/test/fixtures")
MODEL_INFO = {
    "name": "avtfd_g_seed0",
    "sha256": "b88955423e8a" + "0" * 52,
    "device": "cuda",
    "backbone": "mobilenet_v2",
    "audio_feature": "logmel",
    "postprocess": {"threshold": 0.2, "median_kernel": 5, "closing": 3, "fps": 25.0},
}


def fixture(vid: str) -> dict:
    return json.loads((FIXTURES / f"{vid}.json").read_text(encoding="utf-8"))


def thumbs(n: int) -> list[bytes]:
    frame = np.full((224, 224, 3), 120, dtype=np.uint8)
    return [render_thumbnail(frame, None, highlight=False) for _ in range(n)]


def pdf_text(pdf: bytes) -> tuple[int, str]:
    reader = PdfReader(io.BytesIO(pdf))
    return len(reader.pages), "\n".join(p.extract_text() for p in reader.pages)


# ---------------------------------------------------------------- the document


class TestDocument:
    def test_partially_fake_report(self):
        job = fixture("dev_004623")
        pdf = build_report(job, MODEL_INFO, thumbs(len(job["result"]["keyframes"])))
        assert pdf[:5] == b"%PDF-"
        pages, text = pdf_text(pdf)
        assert pages == 3
        for heading in (
            "Forgery analysis report",
            "Likely manipulated",
            "Timeline",
            "Manipulation score, frame by frame",
            "Evidence",
            "Voice or face?",
            "Manipulated segments",
            "The input",
            "How it was analysed",
            "How to read this report",
            "What this model can and cannot do",
        ):
            assert heading in text, heading
        assert job["result"]["evidence"]["summary"] in text.replace("\n", " ")
        for seg in job["result"]["segments"]:
            assert fmt_time(seg["start_s"]) in text and fmt_time(seg["end_s"]) in text
        assert "not how the video was edited" in text  # the attribution caveat
        assert "Page 3 of 3" in text
        assert "dev_004623.mp4" in text and "MP4" in text

    def test_real_video_report(self):
        job = fixture("dev_004337")
        pages, text = pdf_text(build_report(job, MODEL_INFO, thumbs(4)))
        assert pages >= 2
        assert "No manipulation detected" in text
        assert "No manipulated segments were found." in text
        assert "Sampled frames" in text

    def test_report_survives_missing_detail_and_thumbnails(self):
        """An older result (no evidence, no traces) still renders -- just with less in it."""
        job = fixture("dev_034103")
        for key in (
            "evidence",
            "audio_only_scores",
            "visual_only_scores",
            "face_found",
            "keyframes",
        ):
            job["result"].pop(key, None)
        pages, text = pdf_text(build_report(job, {}, []))
        assert pages >= 2 and "Likely manipulated" in text

    def test_metadata(self):
        job = fixture("dev_004623")
        meta = PdfReader(io.BytesIO(build_report(job, MODEL_INFO, []))).metadata
        assert meta.title == "Forgery analysis report — dev_004623.mp4"

    @pytest.mark.parametrize(
        ("demuxer", "name", "want"),
        [
            ("mov,mp4,m4a,3gp,3g2,mj2", "a.mp4", "MP4"),
            ("mov,mp4,m4a,3gp,3g2,mj2", "a.mov", "MOV"),
            ("matroska,webm", "a.mkv", "MKV"),
            ("avi", "a.avi", "AVI"),
        ],
    )
    def test_container_names(self, demuxer, name, want):
        assert container_name(demuxer, name) == want


# ---------------------------------------------------------------- the endpoints


@dataclass
class ReportResult:
    face_found_fraction: float = 1.0
    audio_silent: bool = False
    is_fake: bool = True
    clip_score: float = 0.95
    n_frames: int = 25
    duration_s: float = 1.0
    fps: float = 25.0
    segments: list = field(default_factory=lambda: [(0.2, 0.6, 0.8)])
    frame_scores: np.ndarray = field(
        default_factory=lambda: np.linspace(0, 1, 25).astype(np.float32)
    )
    audio_only_scores: np.ndarray = field(default_factory=lambda: np.full(25, 0.7, np.float32))
    visual_only_scores: np.ndarray = field(default_factory=lambda: np.full(25, 0.1, np.float32))
    audio_only_clip: float = 0.94
    visual_only_clip: float = 0.3
    boundary_scores: np.ndarray | None = None
    face_found: np.ndarray = field(default_factory=lambda: np.ones(25, bool))
    keyframes: list = field(default_factory=list)

    def as_dict(self, *, include_frames: bool = True) -> dict:
        return {
            "n_frames": self.n_frames,
            "duration_s": self.duration_s,
            "fps": self.fps,
            "clip_score": self.clip_score,
            "is_fake": self.is_fake,
            "segments": [{"start_s": s, "end_s": e, "confidence": c} for s, e, c in self.segments],
            "face_found_fraction": self.face_found_fraction,
            "audio_silent": self.audio_silent,
            "timings_s": {"total": 0.5},
            "frame_scores": [float(x) for x in self.frame_scores],
            "audio_only_scores": [float(x) for x in self.audio_only_scores],
            "visual_only_scores": [float(x) for x in self.visual_only_scores],
            "audio_only_clip": self.audio_only_clip,
            "visual_only_clip": self.visual_only_clip,
            "face_found": [int(x) for x in self.face_found],
            "keyframes": [k.meta() for k in self.keyframes],
        }


class ReportAnalyzer:
    threshold = 0.5

    def __init__(self, gate=None):
        self.gate = gate
        self.seen: dict = {}

    def analyze(self, path, *, deadline=None, details=False, keyframes=False):
        self.seen = {"details": details, "keyframes": keyframes}
        if self.gate is not None:
            self.gate.wait(10)
        jpeg = thumbs(1)[0]
        kfs = [Keyframe(frame=f, time_s=f / 25, role=r, segment=0, face_found=True, jpeg=jpeg)
               for f, r in ((5, "start"), (10, "middle"), (14, "end"))]  # fmt: skip
        return ReportResult(keyframes=kfs)

    def close(self):
        pass


INFO = {**MODEL_INFO, "file": "-", "size_mb": 0.0, "threshold": 0.5, "fps": 25.0,
        "model_kwargs": {}, "parameters": {}, "source": {}, "warmup_s": None}  # fmt: skip


@pytest.fixture
def client_for(tmp_path):
    made = []

    def make(analyzer):
        settings = ApiSettings(_env_file=None, upload_dir=tmp_path / "up", warmup=False,
                               max_duration_s=3.0)  # fmt: skip
        c = TestClient(create_app(settings, analyzer_factory=lambda s: (analyzer, dict(INFO))))
        c.__enter__()
        made.append(c)
        return c

    yield make
    for c in made:
        c.__exit__(None, None, None)


def run_job(client, media: Path) -> str:
    with media.open("rb") as f:
        job_id = client.post(
            "/api/v1/analyze", files={"file": (media.name, f, "video/x-matroska")}
        ).json()["job_id"]
    end = time.monotonic() + 30
    while client.get(f"/api/v1/jobs/{job_id}").json()["status"] not in ("succeeded", "failed"):
        assert time.monotonic() < end
        time.sleep(0.05)
    return job_id


TINY = Path("tests/fixtures/tiny_fake.mkv")


class TestEndpoints:
    def test_service_asks_for_details_and_keyframes(self, client_for):
        analyzer = ReportAnalyzer()
        client = client_for(analyzer)
        run_job(client, TINY)
        assert analyzer.seen == {"details": True, "keyframes": True}

    def test_result_carries_evidence_and_keyframe_urls(self, client_for):
        client = client_for(ReportAnalyzer())
        job_id = run_job(client, TINY)
        r = client.get(f"/api/v1/jobs/{job_id}").json()["result"]
        assert r["evidence"]["segments"][0]["driver"] == "voice"
        assert r["evidence"]["modality"]["available"] is True
        assert [k["url"] for k in r["keyframes"]] == [
            f"/api/v1/jobs/{job_id}/keyframes/{i}.jpg" for i in range(3)
        ]

    def test_keyframe_endpoint(self, client_for):
        client = client_for(ReportAnalyzer())
        job_id = run_job(client, TINY)
        ok = client.get(f"/api/v1/jobs/{job_id}/keyframes/1.jpg")
        assert ok.status_code == 200 and ok.headers["content-type"] == "image/jpeg"
        assert ok.content[:2] == b"\xff\xd8"
        missing = client.get(f"/api/v1/jobs/{job_id}/keyframes/9.jpg")
        assert (
            missing.status_code == 404 and missing.json()["error"]["code"] == "keyframe_not_found"
        )

    def test_report_endpoint_returns_a_named_pdf_and_caches_it(self, client_for):
        client = client_for(ReportAnalyzer())
        job_id = run_job(client, TINY)
        first = client.get(f"/api/v1/jobs/{job_id}/report.pdf")
        assert first.status_code == 200 and first.headers["content-type"] == "application/pdf"
        assert (
            first.headers["content-disposition"]
            == 'attachment; filename="tiny_fake-forgery-report.pdf"'
        )
        pages, text = pdf_text(first.content)
        assert pages >= 2 and "Likely manipulated" in text and "tiny_fake.mkv" in text
        assert client.get(f"/api/v1/jobs/{job_id}/report.pdf").content == first.content

    def test_unknown_job(self, client_for):
        client = client_for(ReportAnalyzer())
        for url in (
            "/api/v1/jobs/" + "f" * 32 + "/report.pdf",
            "/api/v1/jobs/" + "f" * 32 + "/keyframes/0.jpg",
        ):
            r = client.get(url)
            assert r.status_code == 404 and r.json()["error"]["code"] == "job_not_found"

    def test_unfinished_job_is_409(self, client_for):
        import threading

        gate = threading.Event()
        client = client_for(ReportAnalyzer(gate=gate))
        with TINY.open("rb") as f:
            job_id = client.post(
                "/api/v1/analyze", files={"file": (TINY.name, f, "video/x-matroska")}
            ).json()["job_id"]
        r = client.get(f"/api/v1/jobs/{job_id}/report.pdf")
        assert r.status_code == 409 and r.json()["error"]["code"] == "job_not_ready"
        gate.set()
