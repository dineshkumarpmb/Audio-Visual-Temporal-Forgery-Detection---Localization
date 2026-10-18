"""⛔ P13-12: the Phase 13 gate -- the API tested and stable under every error case.

    python scripts/43_phase13_gate.py

Two halves:

1. **P13-11, the test suite** -- `tests/integration/test_api.py` (contract tests around a fake
   analyzer, plus the shipped model behind the real app) must pass in full.
2. **The real server.** Boots `uvicorn api.main:app --workers 1` exactly as `make api` does,
   on the shipped model, and drives it over real HTTP:

   * P13-3/4: `/health`, `/api/v1/model-info`, `/openapi.json` and `/docs` render;
   * P13-6/10: every error case -- oversized (with and without `Content-Length`), wrong
     extension, wrong MIME, not multipart, empty, garbage, truncated, contents/extension
     mismatch, no audio, too short, no face -- answers a structured 4xx (or a structured
     `failed` job), never a 5xx, never traceback text;
   * P13-8: LAV-DF dev clips run end to end; the second request on the same clip is not
     slower than the first beyond noise (no per-request load) and returns identical scores;
   * ⛔ P13-9: a burst of uploads is serialised -- no two jobs' `[started, finished]`
     intervals overlap, the queue fills to `max_queued_jobs`, and the next upload gets 503;
   * after all of it the server is still up, `/health` answers, no upload is left on disk,
     and the server log holds no traceback.

Server RSS is sampled throughout and reported against PF-29's ~1.1 GB budget; it is
recorded, not gated, because the 6 GB host's free memory swings with whatever else runs.
Dev clips only -- the test split is spent (`reports/test_once.lock`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.manifest import read_manifest  # noqa: E402
from src.training.reporting import git_sha  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"
ANALYZE = "/api/v1/analyze"
STARTUP_TIMEOUT_S = 300
JOB_TIMEOUT_S = 180
REPEAT_SLACK = 1.5  # second request may be at most this much slower than the first (noise)
RSS_BUDGET_GB = 1.1  # PF-29, reported only
FACE_CACHE = Path("data/features/visual/0d8eaa32")  # MobileNetV2 Stage-A cache (D-1)


# ---------------------------------------------------------------- media


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, capture_output=True)


def make_video(path: Path, seconds: float = 1.0, *, audio: bool = True, codec="libx264") -> Path:
    src = ["-f", "lavfi", "-i", f"testsrc=size=160x120:rate=25:duration={seconds}"]
    if audio:
        src += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=16000:duration={seconds}"]
    tail = ["-c:a", "mp3" if path.suffix == ".avi" else "aac", "-shortest"] if audio else ["-an"]
    ffmpeg(*src, "-pix_fmt", "yuv420p", "-c:v", codec, *tail, str(path))
    return path


def build_media(d: Path) -> dict[str, Path]:
    m = {
        "faceless": make_video(d / "pattern.mp4"),
        "no_audio": make_video(d / "mute.mp4", audio=False),
        "too_short": make_video(d / "short.mp4", 0.2),
        "avi": make_video(d / "clip.avi", codec="mpeg4"),
    }
    m["garbage"] = d / "garbage.mp4"
    m["garbage"].write_bytes(np.random.default_rng(0).bytes(64 * 1024))
    good = m["faceless"].read_bytes()
    m["truncated"] = d / "truncated.mp4"
    m["truncated"].write_bytes(good[: len(good) // 3])
    m["avi_as_mp4"] = d / "avi_inside.mp4"
    m["avi_as_mp4"].write_bytes(m["avi"].read_bytes())
    return m


# ---------------------------------------------------------------- server


class Server:
    def __init__(self, port: int, log_path: Path) -> None:
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.rss_peak = 0
        self._stop = threading.Event()

    def start(self) -> float:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", encoding="utf-8")
        cmd = [sys.executable, "-m", "uvicorn", "api.main:app"]
        cmd += ["--host", "127.0.0.1", "--port", str(self.port), "--workers", "1"]
        t0 = time.monotonic()
        self.proc = subprocess.Popen(
            cmd, cwd=REPO_ROOT, stdout=self._log, stderr=subprocess.STDOUT, env=os.environ.copy()
        )
        threading.Thread(target=self._sample_rss, daemon=True).start()
        while time.monotonic() - t0 < STARTUP_TIMEOUT_S:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited during startup; see {self.log_path}")
            try:
                if httpx.get(self.base + "/health", timeout=2).status_code == 200:
                    return time.monotonic() - t0
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"server not up after {STARTUP_TIMEOUT_S} s")

    def rss(self) -> int:
        """The process tree: a venv `python.exe` on Windows is a launcher with a child."""
        try:
            p = psutil.Process(self.proc.pid)
            return sum(q.memory_info().rss for q in [p, *p.children(recursive=True)])
        except psutil.Error:
            return 0

    def _sample_rss(self) -> None:
        while not self._stop.is_set():
            self.rss_peak = max(self.rss_peak, self.rss())
            time.sleep(0.25)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        self._stop.set()
        if self.proc is None:
            return
        try:
            parent = psutil.Process(self.proc.pid)
            for p in [*parent.children(recursive=True), parent]:
                p.terminate()
            psutil.wait_procs([parent], timeout=20)
        except psutil.Error:
            pass
        self._log.close()


# ---------------------------------------------------------------- checks


def iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))  # 3.10 does not parse "Z"


def structured(resp: httpx.Response) -> bool:
    try:
        body = resp.json()
    except ValueError:
        return False
    return (
        set(body) == {"error"}
        and set(body["error"]) == {"code", "message", "detail"}
        and "Traceback" not in resp.text
        and 'File "' not in resp.text
    )


def post_file(client: httpx.Client, path: Path, name=None, mime="video/mp4") -> httpx.Response:
    with path.open("rb") as f:
        return client.post(ANALYZE, files={"file": (name or path.name, f, mime)})


def wait_job(client: httpx.Client, job_id: str) -> dict:
    end = time.monotonic() + JOB_TIMEOUT_S
    while time.monotonic() < end:
        body = client.get(f"/api/v1/jobs/{job_id}").json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.2)
    raise TimeoutError(job_id)


def oversized_stream(size: int, *, with_length: bool):
    boundary = "gateboundary"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="big.mp4"\r\n'
        "Content-Type: video/mp4\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    chunk = b"\0" * (1 << 20)

    def body():
        yield head
        for _ in range(size // len(chunk)):
            yield chunk
        yield tail

    headers = {"content-type": f"multipart/form-data; boundary={boundary}"}
    if with_length:
        headers["content-length"] = str(len(head) + (size // len(chunk)) * len(chunk) + len(tail))
    return body(), headers


def error_matrix(client: httpx.Client, media: dict[str, Path], max_bytes: int) -> list[dict]:
    """Each case: (name, request thunk, expected status, expected code, job-level?)."""
    over = max_bytes + (8 << 20)

    def oversized(with_length: bool):
        def go():
            content, headers = oversized_stream(over, with_length=with_length)
            return client.post(ANALYZE, content=content, headers=headers)

        return go

    cases = [
        ("oversized, Content-Length", oversized(True), 413, "file_too_large", False),
        ("oversized, chunked (no length)", oversized(False), 413, "file_too_large", False),
        (
            "wrong extension (.webm)",
            lambda: post_file(client, media["faceless"], name="x.webm"),
            415,
            "unsupported_format",
            False,
        ),
        (
            "wrong MIME (text/plain)",
            lambda: post_file(client, media["faceless"], mime="text/plain"),
            415,
            "unsupported_media_type",
            False,
        ),
        (
            "not multipart (JSON body)",
            lambda: client.post(ANALYZE, json={"file": "x"}),
            415,
            "unsupported_media_type",
            False,
        ),
        (
            "empty file",
            lambda: client.post(ANALYZE, files={"file": ("e.mp4", b"", "video/mp4")}),
            422,
            "empty_file",
            False,
        ),
        (
            "random bytes as .mp4",
            lambda: post_file(client, media["garbage"]),
            422,
            "corrupt_video",
            False,
        ),
        (
            "truncated .mp4",
            lambda: post_file(client, media["truncated"]),
            422,
            "corrupt_video",
            False,
        ),
        (
            "AVI contents named .mp4",
            lambda: post_file(client, media["avi_as_mp4"]),
            415,
            "format_mismatch",
            False,
        ),
        ("no audio track", lambda: post_file(client, media["no_audio"]), 422, "no_audio", False),
        (
            "0.2 s clip",
            lambda: post_file(client, media["too_short"]),
            422,
            "duration_out_of_range",
            False,
        ),
        (
            "no face (test pattern)",
            lambda: post_file(client, media["faceless"]),
            202,
            "no_face",
            True,
        ),
        (
            "unknown job id",
            lambda: client.get("/api/v1/jobs/" + "0" * 32),
            404,
            "job_not_found",
            False,
        ),
    ]
    rows = []
    for name, go, status, code, job_level in cases:
        try:
            resp = go()
        except httpx.HTTPError as exc:
            rows.append({"case": name, "ok": False, "got": f"{type(exc).__name__}: {exc}"})
            continue
        if job_level:
            ok_submit = resp.status_code == status
            body = wait_job(client, resp.json()["job_id"]) if ok_submit else {}
            got_code = (body.get("error") or {}).get("code")
            ok = ok_submit and body.get("status") == "failed" and got_code == code
            got = f"{resp.status_code} -> job {body.get('status')} / {got_code}"
        else:
            got_code = resp.json().get("error", {}).get("code") if structured(resp) else None
            ok = resp.status_code == status and structured(resp) and got_code == code
            got = f"{resp.status_code} {got_code}"
        rows.append(
            {
                "case": name,
                "expected": f"{status} {code}",
                "got": got,
                "ok": ok,
                "5xx": resp.status_code >= 500,
            }
        )
    return rows


# ---------------------------------------------------------------- main


def has_face(clip: Path) -> bool:
    """Cached mask from Stage A, when there is one. MediaPipe finds no face on 4 of 1,159
    cached clips; those correctly fail `no_face`, which the error matrix already covers."""
    cached = REPO_ROOT / FACE_CACHE / f"{clip.stem}.npz"
    return not cached.is_file() or float(np.load(cached)["found"].mean()) >= 0.5


DIRECT_PROBE = r"""
import gc, sys, psutil
sys.path.insert(0, ".")
from src.inference.pipeline import VideoAnalyzer
a = VideoAnalyzer(sys.argv[1]); a.warmup(); gc.collect()
print(psutil.Process().memory_info().rss)
"""


def direct_rss(model: Path) -> int:
    """Same load + warm-up with no server around it, now, under today's free memory.

    Windows RSS is the working set, which the OS lets grow when RAM is free and trims under
    pressure -- PF-29's 1,057 MB was measured with 1.86 GB free. The fair comparison for the
    server is this probe in the same conditions, not a figure from another day.
    """
    out = subprocess.run(
        [sys.executable, "-c", DIRECT_PROBE, str(model)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip().splitlines()[-1])


def run_pytest() -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/integration/test_api.py",
            "-q",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", tail)) else 0
    return {"returncode": proc.returncode, "summary": tail, "passed": passed}


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="P13-12 gate: the API under every error case")
    ap.add_argument("--port", type=int, default=8013)
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--clips", default="data/raw/LAV-DF/dev-2k")
    ap.add_argument("--n-clips", type=int, default=5)
    ap.add_argument("--log", default="logs/phase13_server.log")
    ap.add_argument("--out", default="reports/phase13_gate.json")
    ap.add_argument("--md", default="reports/phase13_gate.md")
    ap.add_argument("--skip-pytest", action="store_true")
    args = ap.parse_args()

    from api.settings import ApiSettings

    settings = ApiSettings()
    free_at_start = psutil.virtual_memory().available
    print(f"{DIM}{'=' * 72}{RESET}\n  ⛔ P13-12  Phase 13 gate -- API\n{DIM}{'=' * 72}{RESET}")

    print("  P13-11  tests/integration/test_api.py ...", flush=True)
    tests = {"returncode": None, "summary": "skipped", "passed": 0}
    if not args.skip_pytest:
        tests = run_pytest()
        print(f"    {tests['summary']}")

    manifest = read_manifest(args.manifest).set_index("video_id")
    clips = [
        p
        for p in sorted(Path(args.clips).glob("*.mp4"))
        if p.stem in manifest.index and manifest.loc[p.stem, "split"] == "dev"
    ]
    # Both classes in the sample, deterministically: alternate fake / real from the sorted list.
    fakes = [p for p in clips if manifest.loc[p.stem, "label"] == 1]
    reals = [p for p in clips if manifest.loc[p.stem, "label"] == 0]
    sample = [c for pair in zip(fakes, reals, strict=False) for c in pair][: args.n_clips]
    skipped_faceless = [p.stem for p in sample if not has_face(p)]
    sample = [p for p in sample if has_face(p)]
    assert len(sample) >= 3, "need at least three local dev clips"

    print("  direct VideoAnalyzer RSS baseline (fresh process, no server) ...", flush=True)
    rss_direct = direct_rss(settings.model_path)
    print(f"    {rss_direct / 1e9:.2f} GB")
    tmp = Path(tempfile.mkdtemp(prefix="avtfd_p13_"))
    server = Server(args.port, Path(args.log))
    result: dict = {}
    try:
        media = build_media(tmp)
        startup_s = server.start()
        rss_idle = server.rss()
        print(f"  server up in {startup_s:.1f} s, RSS {rss_idle / 1e9:.2f} GB (idle, model loaded)")
        client = httpx.Client(base_url=server.base, timeout=120)

        # -- P13-3 / P13-4
        info = client.get("/api/v1/model-info").json()
        spec = client.get("/openapi.json").json()
        docs = client.get("/docs")
        endpoints = {
            "/health 200": client.get("/health").status_code == 200,
            "/api/v1/model-info names the shipped model": info.get("name") == "avtfd_g_seed0",
            "/openapi.json lists all four endpoints": {
                "/health",
                "/api/v1/model-info",
                "/api/v1/analyze",
                "/api/v1/jobs/{job_id}",
            }
            <= set(spec.get("paths", {})),
            "/docs renders (Swagger UI)": docs.status_code == 200
            and "swagger" in docs.text.lower(),
        }

        # -- P13-6 / P13-10
        print("  error matrix ...", flush=True)
        errors = error_matrix(client, media, settings.max_upload_bytes)
        for r in errors:
            mark = GREEN + "ok" if r["ok"] else RED + "NO"
            print(f"    [{mark}{RESET}] {r['case']:<32} {DIM}{r['got']}{RESET}")

        # -- P13-8: same clip twice, then the rest of the sample
        print("  real clips ...", flush=True)
        runs = []
        for i, clip in enumerate([sample[0], *sample]):
            t0 = time.monotonic()
            resp = post_file(client, clip)
            body = wait_job(client, resp.json()["job_id"]) if resp.status_code == 202 else {}
            res = body.get("result") or {}
            runs.append(
                {
                    "clip": clip.stem,
                    "label": int(manifest.loc[clip.stem, "label"]),
                    "status": body.get("status", f"http {resp.status_code}"),
                    "verdict": res.get("verdict"),
                    "clip_score": res.get("clip_score"),
                    "n_segments": len(res.get("segments", [])),
                    "face_found_fraction": res.get("face_found_fraction"),
                    "analysis_s": (res.get("timings_s") or {}).get("total"),
                    "wall_s": round(time.monotonic() - t0, 2),
                    "frame_scores": res.get("frame_scores"),
                }
            )
            r = runs[-1]
            print(
                f"    {i}  {r['clip']}  label {r['label']}  {r['status']}  {r['verdict']}  "
                f"score {r['clip_score']}  {r['analysis_s']} s"
            )
        first, second = runs[0], runs[1]
        repeat_ok = (
            first["analysis_s"] is not None
            and second["analysis_s"] is not None
            and second["analysis_s"] <= REPEAT_SLACK * first["analysis_s"]
        )
        load_once = {
            "every real clip succeeded": all(r["status"] == "succeeded" for r in runs),
            f"2nd request on the same clip not slower than {REPEAT_SLACK}x the 1st "
            f"({second['analysis_s']} vs {first['analysis_s']} s)": repeat_ok,
            "repeat request returns identical frame scores": first["frame_scores"]
            == second["frame_scores"],
            "model warmed at startup (warmup_s recorded)": info.get("warmup_s") is not None,
        }

        # -- P13-9: a burst past the queue
        print("  burst ...", flush=True)
        capacity = settings.max_concurrent_jobs + settings.max_queued_jobs
        burst = [sample[i % len(sample)] for i in range(capacity + 1)]
        responses = [post_file(client, c) for c in burst]
        codes = [r.status_code for r in responses]
        mid = client.get("/health").json()
        accepted = [r.json()["job_id"] for r in responses if r.status_code == 202]
        jobs = [wait_job(client, j) for j in accepted]
        spans = sorted((iso(j["started_at"]), iso(j["finished_at"])) for j in jobs)
        overlaps = sum(1 for a, b in zip(spans, spans[1:], strict=False) if b[0] < a[1])
        rejected = [r for r in responses if r.status_code != 202]
        concurrency = {
            f"{capacity} accepted, upload #{capacity + 1} refused ({codes})": codes
            == [202] * capacity + [503],
            "refusal is structured 503 queue_full with Retry-After": len(rejected) == 1
            and structured(rejected[0])
            and rejected[0].json()["error"]["code"] == "queue_full"
            and "retry-after" in rejected[0].headers,
            f"/health mid-burst: busy={mid['busy']}, queued={mid['queued']}": mid["busy"]
            and mid["queued"] >= 1,
            f"no two jobs' run intervals overlap ({overlaps} overlaps in {len(spans)} jobs)": overlaps
            == 0,
            "every burst job succeeded": all(j["status"] == "succeeded" for j in jobs),
        }

        # -- stability
        time.sleep(1.0)
        log_text = Path(args.log).read_text(encoding="utf-8", errors="replace")
        leftovers = [p.name for p in settings.upload_dir.iterdir() if p.is_file()]
        stability = {
            "server still running": server.alive(),
            "/health ok after everything": client.get("/health").json().get("status") == "ok",
            "no 5xx on any error case": not any(r.get("5xx") for r in errors),
            "no upload left on disk": not leftovers,
            "no traceback in the server log": "Traceback" not in log_text,
        }
        rss_peak = server.rss_peak
        client.close()
    finally:
        server.stop()
        shutil.rmtree(tmp, ignore_errors=True)

    gate = {
        f"P13-11 test suite green ({tests['summary']})": tests["returncode"] == 0,
        "P13-3/4 endpoints and /docs": all(endpoints.values()),
        f"P13-6/10 error matrix {sum(r['ok'] for r in errors)}/{len(errors)}": all(
            r["ok"] for r in errors
        ),
        "P13-8 model loaded once, warm per request": all(load_once.values()),
        "⛔ P13-9 concurrency of one under a burst": all(concurrency.values()),
        "stable after all of the above": all(stability.values()),
    }
    passed = all(gate.values())
    sections = {
        "Endpoints": endpoints,
        "P13-8 load once": load_once,
        "⛔ P13-9 concurrency": concurrency,
        "Stability": stability,
    }
    for title, checks in sections.items():
        print(f"\n  {title}")
        for k, v in checks.items():
            print(f"    [{GREEN + 'ok' if v else RED + 'NO'}{RESET}] {k}")
    print(
        f"\n  RSS: server idle {rss_idle / 1e9:.2f} GB, peak {rss_peak / 1e9:.2f} GB; "
        f"direct VideoAnalyzer now {rss_direct / 1e9:.2f} GB (reported only)"
    )
    print("\n  ⛔ P13-12 gate")
    for k, v in gate.items():
        print(f"    [{GREEN + 'ok' if v else RED + 'NO'}{RESET}] {k}")

    for r in runs:
        r.pop("frame_scores")
    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": git_sha(),
        "model": {k: info.get(k) for k in ("name", "file", "size_mb", "sha256", "device")},
        "warmup_s": info.get("warmup_s"),
        "startup_s": round(startup_s, 2),
        "rss_gb": {
            "server_idle": round(rss_idle / 1e9, 3),
            "server_peak": round(rss_peak / 1e9, 3),
            "direct_analyzer_same_conditions": round(rss_direct / 1e9, 3),
            "free_at_start": round(free_at_start / 1e9, 3),
            "pf29_fresh_process_reference": RSS_BUDGET_GB,
        },
        "skipped_faceless_clips": skipped_faceless,
        "limits": info.get("limits"),
        "tests": tests,
        "endpoints": endpoints,
        "error_matrix": errors,
        "real_clips": runs,
        "load_once": load_once,
        "burst": {"codes": codes, "overlaps": overlaps, "n_jobs": len(spans)},
        "concurrency": concurrency,
        "stability": stability,
        "gate": gate,
        "gate_passed": passed,
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Phase 13 gate -- the API under every error case (P13-12)",
        "",
        f"Generated by `scripts/43_phase13_gate.py` at `{result['git_sha']}` on "
        f"{result['generated_at']}. Real `uvicorn api.main:app --workers 1` on "
        f"`{info.get('file')}` ({info.get('device')}), driven over HTTP. Dev clips only.",
        "",
        f"Startup {startup_s:.1f} s (warm-up {info.get('warmup_s')} s). Server RSS "
        f"{rss_idle / 1e9:.2f} GB idle, **{rss_peak / 1e9:.2f} GB peak**; the same load + "
        f"warm-up with no server, in a fresh process just before, **{rss_direct / 1e9:.2f} GB** "
        f"({free_at_start / 1e9:.2f} GB free at start). Windows RSS is the working set, which "
        "grows when RAM is free and is trimmed under pressure, so PF-29's 1,057 MB (taken with "
        "1.86 GB free) is not today's comparison -- the direct probe is. Reported, not gated.",
        "",
        "## Error matrix (P13-6, P13-10)",
        "",
        "| case | expected | got | |",
        "|---|---|---|---|",
        *[
            f"| {r['case']} | {r.get('expected', '')} | {r['got']} | {'✅' if r['ok'] else '❌'} |"
            for r in errors
        ],
        "",
        "## Real clips (P13-8)",
        "",
        "The first clip is submitted twice; the repeat must be no slower than "
        f"{REPEAT_SLACK}x and score identically. Not an accuracy measurement (P12-5 is).",
        "",
        "| # | clip | label | status | verdict | clip score | segments | face found | analysis s |",
        "|---|---|---|---|---|---|---|---|---|",
        *[
            f"| {i} | {r['clip']} | {'fake' if r['label'] else 'real'} | {r['status']} | "
            f"{r['verdict']} | {r['clip_score']} | {r['n_segments']} | "
            f"{r['face_found_fraction']} | {r['analysis_s']} |"
            for i, r in enumerate(runs)
        ],
        "",
    ]
    for title, checks in sections.items():
        md += [f"## {title}", "", *[f"- [{'x' if v else ' '}] {k}" for k, v in checks.items()], ""]
    md += [
        "## ⛔ P13-12 gate -- " + ("PASSED" if passed else "FAILED"),
        "",
        *[f"- [{'x' if v else ' '}] {k}" for k, v in gate.items()],
    ]
    Path(args.md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n  wrote {args.out}, {args.md}, server log {args.log}")
    print(f"\n{GREEN if passed else RED}{'GATE PASSED' if passed else 'GATE FAILED'}{RESET}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
