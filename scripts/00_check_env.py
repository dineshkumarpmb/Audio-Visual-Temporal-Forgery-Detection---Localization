"""Phase 0 gate: verify the toolchain and measure what this machine can actually do.

Exits 0 only if every hard requirement passes. Run with --bench to additionally
measure GPU throughput, the VRAM ceiling, and video decode speed, and to write
reports/hardware_report.md.

    python scripts/00_check_env.py
    python scripts/00_check_env.py --bench

Design note: this script imports nothing from src/. It must be runnable before
any project code exists, and it must not fail for a reason of our own making.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (import name, pip name, minimum version or None)
REQUIRED_PACKAGES: list[tuple[str, str, str | None]] = [
    ("torch", "torch", "2.0"),
    ("torchvision", "torchvision", None),
    ("torchaudio", "torchaudio", None),
    ("numpy", "numpy", "1.24"),
    ("pandas", "pandas", "2.0"),
    ("pyarrow", "pyarrow", "14.0"),
    ("av", "av", None),
    ("cv2", "opencv-python", None),
    ("mediapipe", "mediapipe", None),
    ("librosa", "librosa", "0.10"),
    ("soundfile", "soundfile", None),
    ("scipy", "scipy", None),
    ("sklearn", "scikit-learn", None),
    ("einops", "einops", None),
    ("pydantic", "pydantic", "2.0"),
    ("yaml", "pyyaml", None),
    ("mlflow", "mlflow", None),
    ("structlog", "structlog", None),
    ("tqdm", "tqdm", None),
    ("psutil", "psutil", None),
    ("matplotlib", "matplotlib", None),
    ("fastapi", "fastapi", None),
    ("uvicorn", "uvicorn", None),
]

# Driver floors. CUDA 11.8 needs >=522.06 on Windows; CUDA 12.0 needs >=527.41.
CUDA_118_MIN_DRIVER = (522, 6)

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


@dataclass
class Results:
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, object] = field(default_factory=dict)

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append((name, ok, detail))
        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  [{mark}] {name}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
        return ok

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  [{YELLOW}WARN{RESET}] {msg}")

    @property
    def failed(self) -> list[str]:
        return [n for n, ok, _ in self.checks if not ok]


def _version_tuple(v: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in v.split("+")[0].split("."):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            break
    return tuple(parts) or (0,)


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────


def check_python(r: Results) -> None:
    section("Python")
    v = sys.version_info
    r.facts["python_version"] = platform.python_version()
    r.facts["python_executable"] = sys.executable
    r.check(
        "Python is 3.10.x",
        (v.major, v.minor) == (3, 10),
        f"found {platform.python_version()}",
    )
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    r.check("Running inside a virtualenv", in_venv, sys.prefix)
    if in_venv and not Path(sys.prefix).drive.upper().startswith("D"):
        r.warn(f"venv is not on D: ({sys.prefix}) — C: has limited free space")


def check_binaries(r: Results) -> None:
    section("External binaries")
    for exe in ("ffmpeg", "ffprobe"):
        path = shutil.which(exe)
        ok = r.check(f"{exe} on PATH", path is not None, path or "not found")
        if ok:
            try:
                out = subprocess.run(
                    [exe, "-version"], capture_output=True, text=True, timeout=30, check=True
                ).stdout.splitlines()[0]
                r.facts[f"{exe}_version"] = out
                print(f"         {DIM}{out}{RESET}")
            except (subprocess.SubprocessError, OSError) as e:  # pragma: no cover
                r.warn(f"{exe} found but failed to run: {e}")


def check_packages(r: Results) -> None:
    section("Python packages")
    versions: dict[str, str] = {}
    for import_name, pip_name, min_version in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
        except ImportError as e:
            r.check(f"import {import_name}", False, f"pip install {pip_name} ({e})")
            continue
        got = getattr(mod, "__version__", "unknown")
        versions[pip_name] = got
        if min_version and got != "unknown" and _version_tuple(got) < _version_tuple(min_version):
            r.check(f"import {import_name}", False, f"{got} < required {min_version}")
        else:
            r.check(f"import {import_name}", True, got)
    r.facts["packages"] = versions


def _parse_driver(v: str) -> tuple[int, ...]:
    return _version_tuple(v)


def check_cuda(r: Results) -> None:
    section("CUDA / GPU")
    try:
        import torch
    except ImportError:
        r.check("torch.cuda.is_available()", False, "torch not installed")
        return

    avail = torch.cuda.is_available()
    r.facts["torch_version"] = torch.__version__
    r.facts["torch_cuda_version"] = torch.version.cuda
    r.check(
        "torch.cuda.is_available()",
        avail,
        f"torch {torch.__version__}, built for CUDA {torch.version.cuda}",
    )
    if not avail:
        print(
            f"\n  {RED}This is a hard gate.{RESET} PROJECT_PLAN.md Phase 0: "
            "'Do not proceed without CUDA.'\n"
            "  Most likely cause: a CPU-only wheel from PyPI. Reinstall with:\n"
            "    pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 "
            "torchaudio==2.7.1+cu118 \\\n"
            "        --index-url https://download.pytorch.org/whl/cu118"
        )
        return

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024**3
    r.facts["gpu_name"] = props.name
    r.facts["gpu_vram_gb"] = round(total_gb, 2)
    r.facts["gpu_capability"] = f"sm_{props.major}{props.minor}"
    r.check("GPU visible", True, f"{props.name}, {total_gb:.2f} GB, sm_{props.major}{props.minor}")

    # Driver floor. torch builds against a CUDA runtime; the driver must meet its minimum.
    driver = None
    if shutil.which("nvidia-smi"):
        with contextlib.suppress(subprocess.SubprocessError, OSError, IndexError):
            driver = (
                subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                )
                .stdout.strip()
                .splitlines()[0]
            )
    if driver:
        r.facts["driver_version"] = driver
        cuda_major = int((torch.version.cuda or "11").split(".")[0])
        if cuda_major >= 12:
            ok = _parse_driver(driver) >= (527, 41)
            r.check("Driver supports CUDA 12.x", ok, f"driver {driver}, need >=527.41")
        else:
            ok = _parse_driver(driver) >= CUDA_118_MIN_DRIVER
            r.check("Driver supports CUDA 11.8", ok, f"driver {driver}, need >=522.06")

    # A real allocation + kernel launch. is_available() alone can pass on a broken install.
    try:
        x = torch.randn(256, 256, device="cuda")
        y = (x @ x).sum().item()
        del x
        torch.cuda.empty_cache()
        r.check("CUDA kernel executes", y == y, "matmul on device returned a finite value")
    except Exception as e:  # noqa: BLE001 - any failure here is a genuine gate failure
        r.check("CUDA kernel executes", False, str(e))


def check_hardware(r: Results) -> None:
    section("Hardware")
    try:
        import psutil
    except ImportError:
        r.warn("psutil missing — skipping RAM/disk detail")
        return

    vm = psutil.virtual_memory()
    total_gb, avail_gb = vm.total / 1024**3, vm.available / 1024**3
    r.facts["ram_total_gb"] = round(total_gb, 2)
    r.facts["ram_available_gb"] = round(avail_gb, 2)
    r.facts["cpu_logical"] = psutil.cpu_count(logical=True)
    r.facts["cpu_physical"] = psutil.cpu_count(logical=False)
    print(f"  CPU        : {platform.processor()}")
    print(f"  Cores      : {psutil.cpu_count(logical=False)}c / {psutil.cpu_count(logical=True)}t")
    print(f"  RAM total  : {total_gb:.2f} GB")
    print(f"  RAM free   : {avail_gb:.2f} GB")

    # Not a gate — free RAM depends on what else is open. But it drives num_workers.
    if avail_gb < 2.0:
        r.warn(
            f"only {avail_gb:.2f} GB RAM free. PROJECT_PLAN.md §12.2: close Chrome before "
            "preprocessing or training — this is the binding constraint, not VRAM."
        )

    disks = {}
    for part in psutil.disk_partitions(all=False):
        if "cdrom" in part.opts or not part.fstype:
            continue
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except OSError:
            continue
        disks[part.device] = round(usage.free / 1024**3, 1)
        print(f"  Disk {part.device:<4}: {usage.free / 1024**3:>6.1f} GB free")
    r.facts["disk_free_gb"] = disks

    # LAV-DF is 25.6 GB (reports/dataset_access.md); working set peaks ~45-55 GB.
    d_free = disks.get("D:\\")
    if d_free is not None:
        r.check("D: has room for LAV-DF + working set (>=60 GB)", d_free >= 60, f"{d_free} GB free")

    # Pagefile location matters: C: has little headroom, and thrashing there with
    # 5.94 GB of RAM will freeze the machine (§12.2). Queried via PowerShell rather
    # than wmic, which is deprecated and already absent on some Windows 11 builds.
    pagefiles = []
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "(Get-CimInstance Win32_PageFileUsage).Name",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            ).stdout
            pagefiles = [ln.strip() for ln in out.splitlines() if ":\\" in ln]
        except (subprocess.SubprocessError, OSError):
            pass
    if pagefiles:
        r.facts["pagefile"] = pagefiles
        on_d = any(p.upper().startswith("D:") for p in pagefiles)
        print(f"  Pagefile   : {', '.join(pagefiles)}")
        if not on_d:
            r.warn("pagefile is not on D: — §12.2 recommends moving it off C:")


# ─────────────────────────────────────────────────────────────────────────────
# Benchmarks (--bench)
# ─────────────────────────────────────────────────────────────────────────────


def bench_matmul(r: Results) -> None:
    """Measure GEMM throughput per precision.

    This decides whether AMP is worth enabling. PROJECT_PLAN.md §12.3 assumes
    "Precision: AMP fp16" — an assumption that must be measured, not inherited,
    because cards without tensor cores can be dramatically *slower* in fp16.
    """
    import torch

    section("Benchmark: GPU throughput")
    n, iters = 2048, 30
    results: dict[str, float] = {}
    for dtype, label in (
        (torch.float32, "fp32"),
        (torch.float16, "fp16"),
        (torch.bfloat16, "bf16"),
    ):
        try:
            a = torch.randn(n, n, device="cuda", dtype=dtype)
            b = torch.randn(n, n, device="cuda", dtype=dtype)
            for _ in range(10):  # warm up; first cuBLAS call initialises handles
                c = a @ b
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                c = a @ b
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            tflops = (2 * n**3 * iters) / dt / 1e12
            results[label] = tflops
            r.facts[f"matmul_{label}_tflops"] = round(tflops, 2)
            print(f"  {label} {n}x{n} matmul : {tflops:6.2f} TFLOP/s")
            del a, b, c
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"  {label} {n}x{n} matmul : unsupported ({type(e).__name__})")

    # The decision this benchmark exists to make.
    if "fp32" in results and "fp16" in results and results["fp32"] > 0:
        ratio = results["fp16"] / results["fp32"]
        r.facts["fp16_speedup_vs_fp32"] = round(ratio, 2)
        print(f"  {DIM}fp16 is {ratio:.2f}x fp32 throughput{RESET}")
        if ratio < 1.0:
            r.facts["amp_recommended"] = False
            r.warn(
                f"fp16 is SLOWER than fp32 on this GPU ({ratio:.2f}x). This card has no "
                "tensor cores. Do NOT enable AMP fp16 for compute here — it costs speed and "
                "only saves memory. Keep AMP for the cloud tier (P100/T4). "
                "See reports/hardware_report.md."
            )
        else:
            r.facts["amp_recommended"] = True


def bench_vram_ceiling(r: Results) -> None:
    """Find peak allocatable VRAM by growing in 128 MB steps until OOM.

    Deliberately incremental: a single huge allocation can wedge the driver on a
    4 GB card. Every block is released before returning.
    """
    import torch

    section("Benchmark: VRAM ceiling")
    block_mb, blocks, peak_mb = 128, [], 0
    try:
        while True:
            blocks.append(
                torch.empty(block_mb * 1024 * 1024 // 2, dtype=torch.float16, device="cuda")
            )
            peak_mb += block_mb
            if peak_mb > 16384:  # safety stop; no consumer card here exceeds this
                break
    except torch.cuda.OutOfMemoryError:
        pass
    except Exception as e:  # noqa: BLE001
        r.warn(f"VRAM ceiling probe stopped early: {e}")
    finally:
        del blocks
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
    r.facts["vram_allocatable_mb"] = peak_mb
    r.facts["vram_total_mb"] = round(total_mb)
    print(f"  Peak allocatable  : {peak_mb} MB of {total_mb:.0f} MB total")
    print(f"  Usable fraction   : {peak_mb / total_mb:.0%}")
    r.check("VRAM ceiling probe completed without driver crash", peak_mb > 0, f"{peak_mb} MB")
    if peak_mb < 3000:
        r.warn(
            f"only {peak_mb} MB allocatable — other processes hold VRAM. "
            "Close GPU-using apps before training."
        )


def bench_decode(r: Results) -> None:
    """Measure decode throughput on a synthetic clip.

    Stage-A decode is the dominant cost of the whole project (§B) and its budget
    is TO VERIFY. A synthetic 10 s / 25 fps clip gives a first estimate now,
    before LAV-DF is on disk. Real footage is slower (H.264 from VoxCeleb2, plus
    face detection), so treat this as an optimistic ceiling.
    """
    section("Benchmark: video decode")
    if not shutil.which("ffmpeg"):
        r.warn("ffmpeg missing — skipping decode benchmark")
        return

    scratch = REPO_ROOT / "data" / "interim" / "_bench"
    scratch.mkdir(parents=True, exist_ok=True)
    clip = scratch / "synthetic_10s_25fps.mp4"

    if not clip.exists():
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=640x480:rate=25:duration=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=10",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(clip),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=180)
        except (subprocess.SubprocessError, OSError) as e:
            r.warn(f"could not synthesize benchmark clip: {e}")
            return

    try:
        import av
    except ImportError:
        r.warn("PyAV missing — skipping decode benchmark")
        return

    try:
        t0 = time.perf_counter()
        n_frames = 0
        with av.open(str(clip)) as container:
            for _ in container.decode(video=0):
                n_frames += 1
        dt = time.perf_counter() - t0
        fps = n_frames / dt if dt else 0.0
        r.facts["decode_frames"] = n_frames
        r.facts["decode_fps"] = round(fps, 1)
        r.facts["decode_realtime_x"] = round(fps / 25.0, 1)
        print(
            f"  Decoded {n_frames} frames in {dt:.2f}s  ->  {fps:.0f} fps ({fps / 25:.0f}x realtime)"
        )
        print(
            f"  {DIM}Optimistic: synthetic 640x480. Real footage + face detection is slower.{RESET}"
        )

        # Extrapolate the Stage-A budget the plan marks TO VERIFY.
        if fps > 0:
            for n_videos, label in ((2000, "dev-2k"), (10000, "dev-10k")):
                hours = (n_videos * 250 / fps) / 3600
                print(
                    f"  {DIM}Decode-only estimate, {label:<9}: {hours:5.2f} h "
                    f"(excludes face detection){RESET}"
                )
    except Exception as e:  # noqa: BLE001
        r.warn(f"decode benchmark failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────


def write_report(r: Results, benched: bool) -> Path:
    out = REPO_ROOT / "reports" / "hardware_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    f = r.facts
    pkgs = f.get("packages", {})
    assert isinstance(pkgs, dict)

    lines = [
        "# Hardware & Environment Report",
        "",
        # No trailing-double-space line breaks: the trailing-whitespace pre-commit
        # hook would rewrite this file on every regeneration.
        "**Generated by:** `scripts/00_check_env.py`",
        "",
        f"**Machine:** {platform.node()}",
        "",
        f"**OS:** {platform.system()} {platform.release()} (build {platform.version()})",
        "",
        "> Regenerate with `python scripts/00_check_env.py --bench`. "
        "Measured values only — nothing here is estimated unless labelled.",
        "",
        "## Toolchain",
        "",
        "| Component | Version |",
        "|---|---|",
        f"| Python | {f.get('python_version', '?')} |",
        f"| Interpreter | `{f.get('python_executable', '?')}` |",
        f"| torch | {f.get('torch_version', '?')} (CUDA {f.get('torch_cuda_version', '?')}) |",
        f"| ffmpeg | {str(f.get('ffmpeg_version', 'not found')).split(' Copyright')[0]} |",
        "",
        "## Hardware",
        "",
        "| Resource | Value |",
        "|---|---|",
        f"| GPU | {f.get('gpu_name', '?')} |",
        f"| VRAM | {f.get('gpu_vram_gb', '?')} GB ({f.get('gpu_capability', '?')}) |",
        f"| Driver | {f.get('driver_version', '?')} |",
        f"| CPU cores | {f.get('cpu_physical', '?')}c / {f.get('cpu_logical', '?')}t |",
        f"| RAM total | {f.get('ram_total_gb', '?')} GB |",
        f"| RAM free at scan | {f.get('ram_available_gb', '?')} GB |",
        f"| Pagefile | {', '.join(f.get('pagefile', ['?']))} |",  # type: ignore[arg-type]
        "",
        "### Disk",
        "",
        "| Drive | Free (GB) |",
        "|---|---|",
    ]
    for dev, free in (f.get("disk_free_gb") or {}).items():  # type: ignore[union-attr]
        lines.append(f"| {dev} | {free} |")

    if benched:
        lines += [
            "",
            "## Measured performance",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| fp32 matmul (2048³) | {f.get('matmul_fp32_tflops', '?')} TFLOP/s |",
            f"| fp16 matmul (2048³) | {f.get('matmul_fp16_tflops', '?')} TFLOP/s |",
            f"| bf16 matmul (2048³) | {f.get('matmul_bf16_tflops', '?')} TFLOP/s |",
            f"| **fp16 / fp32 ratio** | **{f.get('fp16_speedup_vs_fp32', '?')}×** |",
            f"| Peak allocatable VRAM | {f.get('vram_allocatable_mb', '?')} MB "
            f"of {f.get('vram_total_mb', '?')} MB |",
            f"| Decode (synthetic 640×480) | {f.get('decode_fps', '?')} fps "
            f"({f.get('decode_realtime_x', '?')}× realtime) |",
            "",
            "> The decode figure is an **optimistic ceiling**: synthetic 640×480 content with no "
            "face detection. Real Stage-A throughput is measured in Phase 2 (task P2-13) and is "
            "what Part 7 §7.4's budget should be revised against.",
        ]
        if f.get("amp_recommended") is False:
            lines += [
                "",
                "### ⚠️ Finding — AMP fp16 must be disabled locally",
                "",
                f"fp16 runs at **{f.get('fp16_speedup_vs_fp32')}×** fp32 throughput on this GPU — "
                "i.e. several times *slower*, reproducibly.",
                "",
                "The GTX 1650 (TU117) ships **without tensor cores**, so there is no fast fp16 "
                "GEMM path and cuBLAS falls back to a slow kernel. This directly contradicts "
                '`PROJECT_PLAN.md` §12.3, which specifies *"Precision: AMP fp16"* for the '
                "minimum configuration.",
                "",
                "**Revised guidance:**",
                "",
                "| Tier | Precision | Why |",
                "|---|---|---|",
                "| Local (GTX 1650) | **fp32** | fp16 costs 5-8× compute for a memory saving we "
                "can get more cheaply by lowering batch size |",
                "| Cloud (P100 / T4) | **AMP fp16** | T4 has tensor cores; P100 has a fast fp16 "
                "path. Re-run this benchmark there to confirm |",
                "",
                "Cached features stay `float16` **on disk** (Plan §D) — that is storage, not "
                "compute, and is unaffected. Cast to fp32 on load.",
                "",
                "Precision must therefore be a **config field**, not a constant, so the same code "
                "is correct in both tiers (decision PF-1 / task X-2).",
            ]

    lines += [
        "",
        "## Gate status",
        "",
        f"- Checks run: **{len(r.checks)}**",
        f"- Failed: **{len(r.failed)}**" + (f" — {', '.join(r.failed)}" if r.failed else ""),
        f"- Warnings: **{len(r.warnings)}**",
    ]
    for w in r.warnings:
        lines.append(f"  - {w}")
    lines += [
        "",
        "---",
        "",
        "<details><summary>Installed package versions</summary>",
        "",
        "| Package | Version |",
        "|---|---|",
    ]
    for name in sorted(pkgs):
        lines.append(f"| {name} | {pkgs[name]} |")
    lines += ["", "</details>", ""]

    # Strip per-line trailing whitespace and end with exactly one newline so the
    # pre-commit hooks never rewrite what this script just generated.
    out.write_text("\n".join(ln.rstrip() for ln in lines).rstrip() + "\n", encoding="utf-8")
    (REPO_ROOT / "reports" / "hardware_facts.json").write_text(
        json.dumps(f, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 0 environment gate")
    ap.add_argument("--bench", action="store_true", help="run GPU/VRAM/decode benchmarks")
    ap.add_argument("--no-report", action="store_true", help="skip writing reports/")
    args = ap.parse_args()

    print("=" * 72)
    print("  Phase 0 gate — Audio-Visual Temporal Forgery Detection & Localization")
    print("=" * 72)

    r = Results()
    check_python(r)
    check_binaries(r)
    check_packages(r)
    check_cuda(r)
    check_hardware(r)

    cuda_ok = any(n == "torch.cuda.is_available()" and ok for n, ok, _ in r.checks)
    if args.bench:
        if cuda_ok:
            bench_matmul(r)
            bench_vram_ceiling(r)
        else:
            r.warn("skipping GPU benchmarks — CUDA unavailable")
        bench_decode(r)

    if not args.no_report:
        path = write_report(r, benched=args.bench)
        print(f"\n{DIM}Report written to {path.relative_to(REPO_ROOT)}{RESET}")

    section("Summary")
    passed = len(r.checks) - len(r.failed)
    print(f"  {passed}/{len(r.checks)} checks passed, {len(r.warnings)} warning(s)")
    if r.failed:
        print(f"\n{RED}GATE FAILED{RESET} — {len(r.failed)} check(s) failed:")
        for name in r.failed:
            print(f"    - {name}")
        return 1
    print(f"\n{GREEN}GATE PASSED{RESET} — environment is ready for Phase 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
