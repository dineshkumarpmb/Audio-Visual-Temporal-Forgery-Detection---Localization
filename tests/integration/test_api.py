"""Phase 13: the API under every error case (P13-11, ⛔ P13-12).

Two layers:

* **Contract tests** run the real app -- real streaming upload, real PyAV vetting, real job
  store and worker -- around a *fake* analyzer, so every error path is exercised in seconds.
  Media are generated with ffmpeg; nothing here needs the dataset.
* **`TestRealModel`** boots the app on the shipped `models/avtfd_g_seed0.pt` once and runs
  real uploads through it: a LAV-DF clip end to end, a faceless clip (`no_face`), and the
  load-once check (P13-8).

Every error response must have the `{"error": {"code", "message", "detail"}}` shape and no
traceback text (P13-10).
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import ApiSettings

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg on PATH")

MODEL = Path("models/avtfd_g_seed0.pt")
LAVDF_DEV = Path("data/raw/LAV-DF/dev-2k")


# ---------------------------------------------------------------- media


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, capture_output=True)


def make_video(
    path: Path,
    seconds: float = 1.0,
    *,
    audio: bool = True,
    vcodec: str = "libx264",
    acodec: str = "aac",
) -> Path:
    src = ["-f", "lavfi", "-i", f"testsrc=size=160x120:rate=25:duration={seconds}"]
    if audio:
        src += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=16000:duration={seconds}"]
    # MJPEG is full-range: ffmpeg < 7 (Ubuntu's, in CI) refuses limited-range yuv420p for it.
    pix_fmt = "yuvj420p" if vcodec == "mjpeg" else "yuv420p"
    codecs = ["-pix_fmt", pix_fmt, "-c:v", vcodec]
    codecs += ["-c:a", acodec, "-shortest"] if audio else ["-an"]
    _ffmpeg(*src, *codecs, str(path))
    return path


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> dict[str, Path]:
    if not HAVE_FFMPEG:
        pytest.skip("needs ffmpeg on PATH")
    d = tmp_path_factory.mktemp("api_media")
    out = {
        "mp4": make_video(d / "ok.mp4"),
        "mov": make_video(d / "ok.mov"),
        "mkv": make_video(d / "ok.mkv"),
        "avi": make_video(d / "ok.avi", vcodec="mpeg4", acodec="mp3"),
        "no_audio": make_video(d / "mute.mp4", audio=False),
        "too_short": make_video(d / "short.mp4", 0.2),
        "too_long": make_video(d / "long.mp4", 4.0),
    }
    rng = np.random.default_rng(0)
    out["garbage"] = d / "garbage.mp4"
    out["garbage"].write_bytes(rng.bytes(64 * 1024))
    good = out["mp4"].read_bytes()
    out["truncated"] = d / "truncated.mp4"
    out["truncated"].write_bytes(good[: len(good) // 3])
    out["avi_as_mp4"] = d / "avi_inside.mp4"
    out["avi_as_mp4"].write_bytes(out["avi"].read_bytes())
    # Valid headers, garbage stream: an MJPEG AVI with its frame payloads overwritten.
    mjpeg = make_video(d / "mj.avi", vcodec="mjpeg", acodec="pcm_s16le")
    raw = bytearray(mjpeg.read_bytes())
    for i in range(len(raw) // 2, len(raw) - 16, 3):
        raw[i] = 0
    out["broken_stream"] = d / "broken.avi"
    out["broken_stream"].write_bytes(bytes(raw))
    return out


# ---------------------------------------------------------------- fake analyzer


@dataclass
class FakeResult:
    face_found_fraction: float = 1.0
    audio_silent: bool = False
    is_fake: bool = True
    clip_score: float = 0.9

    def as_dict(self, *, include_frames: bool = True) -> dict:
        return {
            "n_frames": 25,
            "duration_s": 1.0,
            "fps": 25.0,
            "clip_score": self.clip_score,
            "is_fake": self.is_fake,
            "segments": [{"start_s": 0.2, "end_s": 0.6, "confidence": 0.8}],
            "face_found_fraction": self.face_found_fraction,
            "audio_silent": self.audio_silent,
            "timings_s": {"total": 0.01},
            "frame_scores": [0.1] * 25,
        }


@dataclass
class FakeAnalyzer:
    """Stands in for VideoAnalyzer. `gate` (if set) blocks every analysis until released."""

    result: FakeResult = field(default_factory=FakeResult)
    raises: BaseException | None = None
    gate: threading.Event | None = None
    threshold: float = 0.5
    calls: int = 0
    closed: bool = False
    seen_deadlines: list = field(default_factory=list)

    def analyze(self, path, *, deadline=None):
        self.calls += 1
        self.seen_deadlines.append(deadline)
        assert Path(path).is_file(), "the upload must still exist while it is analysed"
        if self.gate is not None:
            assert self.gate.wait(10), "test gate never released"
        if self.raises is not None:
            raise self.raises
        return self.result

    def close(self) -> None:
        self.closed = True


FAKE_INFO = {
    "name": "fake",
    "file": "fake.pt",
    "size_mb": 0.0,
    "sha256": "0" * 64,
    "device": "cpu",
    "threshold": 0.5,
    "fps": 25.0,
    "backbone": "mobilenet_v2",
    "audio_feature": "logmel",
    "model_kwargs": {},
    "postprocess": {},
    "parameters": {"model": 1, "backbone": 1},
    "source": {},
    "warmup_s": None,
}


def _settings(tmp_path: Path, **kw) -> ApiSettings:
    base = {"upload_dir": tmp_path / "uploads", "max_duration_s": 3.0, "warmup": False}
    return ApiSettings(_env_file=None, **{**base, **kw})


@pytest.fixture
def make_client(tmp_path):
    clients = []

    def _make(analyzer: FakeAnalyzer | None = None, **settings_kw):
        analyzer = analyzer or FakeAnalyzer()
        settings = _settings(tmp_path, **settings_kw)
        app = create_app(settings, analyzer_factory=lambda s: (analyzer, dict(FAKE_INFO)))
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        clients.append(client)
        return client, analyzer, settings

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def upload(client, path: Path, *, name: str | None = None, mime: str = "video/mp4"):
    with path.open("rb") as f:
        return client.post("/api/v1/analyze", files={"file": (name or path.name, f, mime)})


def wait_done(client, job_id: str, timeout: float = 30.0) -> dict:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        body = client.get(f"/api/v1/jobs/{job_id}").json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout} s")


def assert_error(resp, status: int, code: str) -> dict:
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "detail"}
    assert body["error"]["code"] == code, body
    assert "Traceback" not in resp.text and 'File "' not in resp.text
    return body["error"]


def uploads_left(settings: ApiSettings) -> list[Path]:
    return [p for p in settings.upload_dir.iterdir() if p.is_file()]


# ---------------------------------------------------------------- service endpoints


class TestServiceEndpoints:
    def test_health(self, make_client):
        client, _, _ = make_client()
        body = client.get("/health").json()
        assert body["status"] == "ok" and body["model_loaded"] is True
        assert body["busy"] is False and body["queued"] == 0

    def test_model_info_includes_limits(self, make_client):
        client, _, _ = make_client(max_upload_mb=50)
        body = client.get("/api/v1/model-info").json()
        assert body["name"] == "fake"
        assert body["limits"]["max_upload_mb"] == 50
        assert body["limits"]["max_concurrent_jobs"] == 1
        assert body["limits"]["allowed_extensions"] == [".mp4", ".avi", ".mov", ".mkv"]

    def test_openapi_and_docs_render(self, make_client):
        client, _, _ = make_client()
        spec = client.get("/openapi.json").json()
        assert {"/health", "/api/v1/model-info", "/api/v1/analyze", "/api/v1/jobs/{job_id}"} <= set(
            spec["paths"]
        )
        body = spec["paths"]["/api/v1/analyze"]["post"]["requestBody"]
        assert "multipart/form-data" in body["content"]
        docs = client.get("/docs")
        assert docs.status_code == 200 and "swagger" in docs.text.lower()

    def test_unknown_route_is_structured(self, make_client):
        client, _, _ = make_client()
        assert_error(client.get("/nope"), 404, "not_found")

    def test_wrong_method_is_structured(self, make_client):
        client, _, _ = make_client()
        assert_error(client.get("/api/v1/analyze"), 405, "method_not_allowed")

    def test_unknown_job(self, make_client):
        client, _, _ = make_client()
        assert_error(client.get("/api/v1/jobs/" + "f" * 32), 404, "job_not_found")

    def test_concurrency_cannot_be_configured_above_one(self, tmp_path):
        with pytest.raises(ValueError):
            _settings(tmp_path, max_concurrent_jobs=2)


# ---------------------------------------------------------------- happy path


@needs_ffmpeg
class TestHappyPath:
    @pytest.mark.parametrize("kind", ["mp4", "mov", "mkv", "avi"])
    def test_each_container_runs_to_a_result(self, make_client, media, kind):
        client, analyzer, settings = make_client()
        resp = upload(client, media[kind], mime="application/octet-stream")
        assert resp.status_code == 202, resp.text
        created = resp.json()
        assert created["status_url"] == f"/api/v1/jobs/{created['job_id']}"
        body = wait_done(client, created["job_id"])
        assert body["status"] == "succeeded", body
        res = body["result"]
        assert res["verdict"] == "fake" and res["threshold"] == 0.5
        assert res["segments"] == [{"start_s": 0.2, "end_s": 0.6, "confidence": 0.8}]
        assert len(res["frame_scores"]) == res["n_frames"]
        assert body["media"]["duration_s"] == pytest.approx(1.0, abs=0.1)
        assert body["error"] is None and body["started_at"] and body["finished_at"]
        assert analyzer.calls == 1
        assert uploads_left(settings) == [], "uploads must be deleted after analysis"

    def test_job_timeout_is_passed_as_a_deadline(self, make_client, media):
        client, analyzer, _ = make_client(job_timeout_s=123)
        t0 = time.monotonic()
        wait_done(client, upload(client, media["mp4"]).json()["job_id"])
        (deadline,) = analyzer.seen_deadlines
        assert t0 + 120 < deadline < time.monotonic() + 124

    def test_low_face_and_silence_produce_warnings(self, make_client, media):
        client, _, _ = make_client(
            FakeAnalyzer(FakeResult(face_found_fraction=0.2, audio_silent=True))
        )
        body = wait_done(client, upload(client, media["mp4"]).json()["job_id"])
        assert body["status"] == "succeeded"
        assert len(body["result"]["warnings"]) == 2

    def test_real_verdict(self, make_client, media):
        client, _, _ = make_client(FakeAnalyzer(FakeResult(is_fake=False, clip_score=0.1)))
        body = wait_done(client, upload(client, media["mp4"]).json()["job_id"])
        assert body["result"]["verdict"] == "real"


# ---------------------------------------------------------------- upload rejections


@needs_ffmpeg
class TestUploadRejections:
    """Refused at upload time: 4xx with a code, never a job, never a file left behind."""

    def _rejected(self, client, settings, resp, status, code, analyzer):
        err = assert_error(resp, status, code)
        assert analyzer.calls == 0
        assert uploads_left(settings) == []
        return err

    def test_oversized_by_content_length(self, make_client, media):
        client, analyzer, settings = make_client(max_upload_mb=0.01)
        err = self._rejected(
            client, settings, upload(client, media["mp4"]), 413, "file_too_large", analyzer
        )
        assert err["detail"]["max_bytes"] == int(0.01 * 1024**2)

    def test_oversized_streamed_without_content_length(self, make_client, media):
        """Chunked body, no Content-Length: the cap is enforced while streaming."""
        client, analyzer, settings = make_client(max_upload_mb=0.01)
        boundary = "testboundary"
        head = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="big.mp4"\r\nContent-Type: video/mp4\r\n\r\n'
        ).encode()

        def body():
            yield head
            for _ in range(64):
                yield b"\0" * 4096
            yield f"\r\n--{boundary}--\r\n".encode()

        resp = client.post(
            "/api/v1/analyze",
            content=body(),
            headers={"content-type": f"multipart/form-data; boundary={boundary}"},
        )
        self._rejected(client, settings, resp, 413, "file_too_large", analyzer)

    def test_disallowed_extension(self, make_client, media):
        client, analyzer, settings = make_client()
        resp = upload(client, media["mp4"], name="clip.webm")
        self._rejected(client, settings, resp, 415, "unsupported_format", analyzer)

    def test_no_extension(self, make_client, media):
        client, analyzer, settings = make_client()
        resp = upload(client, media["mp4"], name="clip")
        self._rejected(client, settings, resp, 415, "unsupported_format", analyzer)

    def test_wrong_mime(self, make_client, media):
        client, analyzer, settings = make_client()
        resp = upload(client, media["mp4"], mime="text/plain")
        self._rejected(client, settings, resp, 415, "unsupported_media_type", analyzer)

    def test_not_multipart(self, make_client):
        client, analyzer, settings = make_client()
        resp = client.post(
            "/api/v1/analyze", content=b"{}", headers={"content-type": "application/json"}
        )
        self._rejected(client, settings, resp, 415, "unsupported_media_type", analyzer)

    def test_missing_file_field(self, make_client):
        client, analyzer, settings = make_client()
        resp = client.post("/api/v1/analyze", data={"other": "x"}, files={"x": ("a.txt", b"x")})
        self._rejected(client, settings, resp, 422, "missing_file", analyzer)

    def test_empty_file(self, make_client):
        client, analyzer, settings = make_client()
        resp = client.post("/api/v1/analyze", files={"file": ("empty.mp4", b"", "video/mp4")})
        self._rejected(client, settings, resp, 422, "empty_file", analyzer)

    def test_malformed_multipart(self, make_client):
        client, analyzer, settings = make_client()
        resp = client.post(
            "/api/v1/analyze",
            content=b"--b\r\nthis is not a header block",
            headers={"content-type": "multipart/form-data; boundary=b"},
        )
        assert resp.status_code in (400, 422)
        assert resp.json()["error"]["code"] in ("malformed_upload", "missing_file")
        assert analyzer.calls == 0 and uploads_left(settings) == []

    @pytest.mark.parametrize("kind", ["garbage", "truncated"])
    def test_corrupt_container(self, make_client, media, kind):
        client, analyzer, settings = make_client()
        self._rejected(
            client, settings, upload(client, media[kind]), 422, "corrupt_video", analyzer
        )

    def test_broken_stream_with_valid_headers(self, make_client, media):
        """Either refused at vetting, or fails as a job -- never a 500."""
        client, _, settings = make_client()
        resp = upload(client, media["broken_stream"], mime="video/x-msvideo")
        if resp.status_code == 202:
            body = wait_done(client, resp.json()["job_id"])
            assert body["status"] in ("succeeded", "failed")
        else:
            assert_error(resp, 422, "corrupt_video")
        assert uploads_left(settings) == []

    def test_contents_do_not_match_extension(self, make_client, media):
        client, analyzer, settings = make_client()
        resp = upload(client, media["avi_as_mp4"])
        err = self._rejected(client, settings, resp, 415, "format_mismatch", analyzer)
        assert err["detail"]["container"] == "avi"

    def test_no_audio(self, make_client, media):
        client, analyzer, settings = make_client()
        self._rejected(
            client, settings, upload(client, media["no_audio"]), 422, "no_audio", analyzer
        )

    @pytest.mark.parametrize("kind", ["too_short", "too_long"])
    def test_duration_bounds(self, make_client, media, kind):
        client, analyzer, settings = make_client()
        err = self._rejected(
            client, settings, upload(client, media[kind]), 422, "duration_out_of_range", analyzer
        )
        assert "duration_s" in err["detail"]


# ---------------------------------------------------------------- job failures


@needs_ffmpeg
class TestJobFailures:
    """Accepted, then failed during analysis: status `failed` with a code, never a 500."""

    @pytest.mark.parametrize(
        ("exc", "code"),
        [
            ("timeout", "timeout"),
            ("audio", "no_audio"),
            ("decode", "corrupt_video"),
            ("other", "internal_error"),
        ],
    )
    def test_analysis_exceptions_map_to_codes(self, make_client, media, exc, code):
        from src.inference.pipeline import AnalysisTimeoutError
        from src.preprocessing.audio import AudioError
        from src.preprocessing.video import DecodeError

        raises = {
            "timeout": AnalysisTimeoutError("late"),
            "audio": AudioError("ffmpeg exit 1 for C:/secret/path.mp4"),
            "decode": DecodeError("decode failed"),
            "other": ZeroDivisionError("boom"),
        }[exc]
        client, _, settings = make_client(FakeAnalyzer(raises=raises))
        body = wait_done(client, upload(client, media["mp4"]).json()["job_id"])
        assert body["status"] == "failed" and body["result"] is None
        assert body["error"]["code"] == code
        # Internal messages (paths, exception text) never reach the client.
        assert "secret" not in str(body) and "boom" not in str(body)
        assert uploads_left(settings) == []

    def test_no_face(self, make_client, media):
        client, _, _ = make_client(FakeAnalyzer(FakeResult(face_found_fraction=0.0)))
        body = wait_done(client, upload(client, media["mp4"]).json()["job_id"])
        assert body["status"] == "failed"
        assert body["error"]["code"] == "no_face"
        assert body["error"]["detail"] == {"face_found_fraction": 0.0}

    def test_no_face_can_be_scored_from_audio_when_allowed(self, make_client, media):
        client, _, _ = make_client(
            FakeAnalyzer(FakeResult(face_found_fraction=0.0)), require_face=False
        )
        body = wait_done(client, upload(client, media["mp4"]).json()["job_id"])
        assert body["status"] == "succeeded"
        assert any("face" in w for w in body["result"]["warnings"])

    def test_server_keeps_serving_after_a_failure(self, make_client, media):
        analyzer = FakeAnalyzer(raises=RuntimeError("first one fails"))
        client, _, _ = make_client(analyzer)
        assert (
            wait_done(client, upload(client, media["mp4"]).json()["job_id"])["status"] == "failed"
        )
        analyzer.raises = None
        assert (
            wait_done(client, upload(client, media["mp4"]).json()["job_id"])["status"]
            == "succeeded"
        )


# ---------------------------------------------------------------- concurrency


@needs_ffmpeg
class TestConcurrency:
    def test_one_analysis_at_a_time_then_queue_then_503(self, make_client, media):
        """⛔ P13-9: with the worker blocked, one job runs, `max_queued_jobs` wait, the next
        upload gets 503 + Retry-After; released, all finish and none ever overlapped."""
        gate = threading.Event()
        analyzer = FakeAnalyzer(gate=gate)
        client, _, settings = make_client(analyzer, max_queued_jobs=2)
        ids = []
        for _ in range(3):
            resp = upload(client, media["mp4"])
            assert resp.status_code == 202, resp.text
            ids.append(resp.json()["job_id"])

        end = time.monotonic() + 5
        while client.get("/health").json()["busy"] is False and time.monotonic() < end:
            time.sleep(0.02)
        health = client.get("/health").json()
        assert health["busy"] is True and health["queued"] == 2
        statuses = [client.get(f"/api/v1/jobs/{i}").json() for i in ids]
        assert [s["status"] for s in statuses] == ["running", "queued", "queued"]
        assert [s["queue_position"] for s in statuses] == [None, 1, 2]

        resp = upload(client, media["mp4"])
        assert_error(resp, 503, "queue_full")
        assert resp.headers["retry-after"] == "10"

        gate.set()
        assert all(wait_done(client, i)["status"] == "succeeded" for i in ids)
        service = client.app.state.service
        assert service.peak_concurrency == 1
        assert analyzer.calls == 3
        assert uploads_left(settings) == []

    def test_concurrent_http_uploads_never_overlap(self, make_client, media):
        """Uploads arriving on parallel connections still serialise in the worker."""
        active = 0
        peak = 0
        lock = threading.Lock()

        class Overlap(FakeAnalyzer):
            def analyze(self, path, *, deadline=None):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.05)
                with lock:
                    active -= 1
                return super().analyze(path, deadline=deadline)

        client, _, _ = make_client(Overlap(), max_queued_jobs=8)
        results: list = []

        def go():
            results.append(upload(client, media["mp4"]))

        threads = [threading.Thread(target=go) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r.status_code == 202 for r in results), [r.text for r in results]
        for r in results:
            assert wait_done(client, r.json()["job_id"])["status"] == "succeeded"
        assert peak == 1

    def test_shutdown_closes_the_analyzer(self, tmp_path):
        analyzer = FakeAnalyzer()
        app = create_app(
            _settings(tmp_path), analyzer_factory=lambda s: (analyzer, dict(FAKE_INFO))
        )
        with TestClient(app):
            pass
        assert analyzer.closed

    def test_stale_uploads_are_cleared_at_startup(self, tmp_path):
        settings = _settings(tmp_path)
        settings.upload_dir.mkdir(parents=True)
        (settings.upload_dir / "leftover.mp4").write_bytes(b"x")
        app = create_app(settings, analyzer_factory=lambda s: (FakeAnalyzer(), dict(FAKE_INFO)))
        with TestClient(app):
            assert uploads_left(settings) == []


# ---------------------------------------------------------------- the real model


def _lavdf_clip() -> Path | None:
    return next(iter(sorted(LAVDF_DEV.glob("*.mp4"))), None) if LAVDF_DEV.is_dir() else None


@pytest.mark.slow
@pytest.mark.skipif(
    not (HAVE_FFMPEG and MODEL.is_file()), reason="needs ffmpeg and models/avtfd_g_seed0.pt"
)
class TestRealModel:
    """The shipped model behind the real app, loaded once for the whole class."""

    @pytest.fixture(scope="class")
    def real(self, tmp_path_factory):
        from api.main import load_analyzer

        calls = []

        def factory(s):
            calls.append(1)
            return load_analyzer(s)

        settings = ApiSettings(
            _env_file=None, upload_dir=tmp_path_factory.mktemp("real_uploads"), max_duration_s=30
        )
        with TestClient(create_app(settings, analyzer_factory=factory)) as client:
            yield client, calls

    def test_model_info_describes_the_shipped_bundle(self, real):
        client, _ = real
        info = client.get("/api/v1/model-info").json()
        assert info["name"] == "avtfd_g_seed0"
        assert info["backbone"] == "mobilenet_v2" and info["audio_feature"] == "logmel"
        assert info["size_mb"] < 50 and len(info["sha256"]) == 64
        assert info["warmup_s"] is not None

    @pytest.mark.dataset
    def test_lavdf_clip_end_to_end_and_model_loads_once(self, real):
        """P13-8: two requests, one model load, and the second is not slower than a warm run."""
        clip = _lavdf_clip()
        if clip is None:
            pytest.skip("no LAV-DF dev clip on disk")
        client, calls = real
        bodies = []
        for _ in range(2):
            bodies.append(wait_done(client, upload(client, clip).json()["job_id"], timeout=120))
        a, b = (x["result"] for x in bodies)
        assert all(x["status"] == "succeeded" for x in bodies), bodies
        assert a["face_found_fraction"] > 0
        # Parity with training: the same face mask the cached features were built with.
        cached = Path("data/features/visual/0d8eaa32") / f"{clip.stem}.npz"
        if cached.is_file():
            found = np.load(cached)["found"]
            assert a["face_found_fraction"] == pytest.approx(found.mean(), abs=1e-4)
        assert a["frame_scores"] == b["frame_scores"]  # deterministic across requests
        assert len(a["frame_scores"]) == a["n_frames"]
        assert calls == [1]
        # Load + warm-up is ~40 s; a reload per request would show up here.
        assert b["timings_s"]["total"] < 30

    def test_faceless_video_is_no_face(self, real, tmp_path):
        client, _ = real
        clip = make_video(tmp_path / "pattern.mp4", 1.0)
        body = wait_done(client, upload(client, clip).json()["job_id"], timeout=120)
        assert body["status"] == "failed"
        assert body["error"]["code"] == "no_face"
