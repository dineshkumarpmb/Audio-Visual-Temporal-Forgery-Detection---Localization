"""P12-1 / P12-3 / P12-4: end-to-end latency, memory and the optimisation matrix -> Table 8.2.3.

    python scripts/41_benchmark.py

The plan names this `scripts/09_benchmark.py`; 09 was taken by Phase 2's face extraction, so
it lands at 41 (as 37 did in Phase 11).

**What is timed is the real path**, `VideoAnalyzer.analyze` on raw dev `.mp4` files -- decode,
landmarks, alignment, MobileNetV2, log-Mel, the section 5.1 model, post-processing -- never
cached features. Clips are dev, never test (the test split is spent, `reports/test_once.lock`).

**P12-3's optimisations are measured, not assumed** -- each is a variant run on the same clips:

| variant          | what changes                                                         |
|---|---|
| `two_pass`       | decode the file twice, as Phase 2-4's extraction did (the naive port)  |
| `optimized`      | decode once, keep frames within a RAM budget; models cached + warmed   |
| `backbone_fp16`  | `optimized` + MobileNetV2 in fp16                                     |
| `model_fp16`     | `optimized` + the Stage-B model in fp16                               |

PF-4 measured fp16 arithmetic at 0.13x fp32 on this GTX 1650; the fp16 variants re-measure
that on the real pipeline, and record their numerical drift against fp32 so a "faster"
variant that changes the answer cannot be adopted by accident.

**Normalisation.** LAV-DF dev clips run 3-20 s, so each clip's time is scaled to a 10 s clip
(`total * 250 / T`) for the gate's "< 10 s per 10 s clip"; clips of >= 9 s are also reported
unscaled. Medians, because this machine's background load (Chrome alone holds ~1.6 cores
idle) makes means noisy -- ambient CPU and free RAM are recorded with every run.

Peak RAM is process RSS sampled every 10 ms; peak VRAM is `torch.cuda.max_memory_allocated`
over the variant, plus an isolated probe per GPU component for Table 8.2.3's per-row column.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.manifest import read_manifest  # noqa: E402
from src.data.subset import load_subset  # noqa: E402
from src.evaluation.runs import load_predictions  # noqa: E402
from src.inference.pipeline import STAGES, VideoAnalyzer  # noqa: E402
from src.training.reporting import git_sha  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
VARIANTS = {
    "two_pass": {"frame_budget_mb": 0.0},
    "optimized": {},
    "backbone_fp16": {"backbone_fp16": True},
    "model_fp16": {"model_fp16": True},
}
TARGET_S_PER_10S = 10.0
FRAMES_10S = 250


class RssSampler:
    """Background thread recording (t, rss) every `period` seconds."""

    def __init__(self, period: float = 0.01) -> None:
        self.period = period
        self.samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._proc = psutil.Process()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append((time.perf_counter(), self._proc.memory_info().rss))
            time.sleep(self.period)

    def __enter__(self) -> RssSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def peak(self, t0: float | None = None, t1: float | None = None) -> int:
        vals = [r for t, r in self.samples if (t0 is None or t >= t0) and (t1 is None or t <= t1)]
        return max(vals) if vals else 0


def pick_clips(manifest, ids: list[str], roots: list[Path], n: int) -> list[tuple[str, Path, int]]:
    """`n` dev clips evenly spaced by length (shortest to longest), deterministic."""
    rows = []
    for vid in ids:
        if vid not in manifest.index or manifest.loc[vid, "split"] != "dev":
            continue
        path = next((r / f"{vid}.mp4" for r in roots if (r / f"{vid}.mp4").exists()), None)
        if path is not None:
            rows.append((vid, path, int(manifest.loc[vid, "n_frames"])))
    rows.sort(key=lambda r: (r[2], r[0]))
    if len(rows) <= n:
        return rows
    idx = np.unique(np.linspace(0, len(rows) - 1, n).round().astype(int))
    return [rows[i] for i in idx]


def ambient() -> dict:
    vm = psutil.virtual_memory()
    chrome = sum(1 for p in psutil.process_iter(["name"]) if (p.info["name"] or "") == "chrome.exe")
    return {
        "cpu_percent_2s": psutil.cpu_percent(interval=2.0),
        "ram_available_gb": round(vm.available / 1e9, 2),
        "ram_total_gb": round(vm.total / 1e9, 2),
        "chrome_processes": chrome,
    }


@torch.no_grad()
def component_vram(analyzer: VideoAnalyzer, device: str) -> dict:
    """Isolated peak VRAM per GPU component, for Table 8.2.3's per-row column (MB)."""
    if device != "cuda":
        return {}
    out = {}
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    x = torch.zeros(analyzer.visual_batch, 3, 112, 112, dtype=torch.uint8, device=device)
    analyzer.backbone(x.half() / 255.0 if analyzer.backbone_fp16 else x)
    torch.cuda.synchronize()
    out["visual"] = (torch.cuda.max_memory_allocated() - base) / 1024**2
    del x

    t = 497  # the longest LAV-DF clip (P1 statistics)
    torch.cuda.reset_peak_memory_stats()
    analyzer._forward(
        np.zeros((t, analyzer.backbone.feature_dim), np.float32),
        np.zeros((t, analyzer.model.audio_dim), np.float32),
        np.ones(t, dtype=bool),
    )
    torch.cuda.synchronize()
    out["model"] = (torch.cuda.max_memory_allocated() - base) / 1024**2
    out["weights_resident"] = base / 1024**2
    return {k: round(v, 1) for k, v in out.items()}


FRESH_PROBE = r"""
import json, sys, psutil
sys.path.insert(0, ".")
p = psutil.Process()
r0 = p.memory_info().rss
from src.inference.pipeline import VideoAnalyzer
a = VideoAnalyzer(sys.argv[1], device=sys.argv[3])
a.warmup()
r1 = p.memory_info().rss
a.analyze(sys.argv[2])
r2 = p.memory_info().rss
print(json.dumps({"start_mb": r0 / 2**20, "loaded_mb": r1 / 2**20, "after_longest_clip_mb": r2 / 2**20,
                  "peak_working_set_mb": getattr(p.memory_info(), "peak_wset", 0) / 2**20}))
"""


def fresh_process_rss(bundle: Path, clip: Path, device: str) -> dict:
    """RSS of a *new* process that loads, warms and analyses the longest clip once.

    The in-benchmark RSS figures are inflated by the burn-in and earlier variants living in
    the same process; this is the number Phase 13's server will actually hold on a 6 GB box.
    """
    import subprocess

    out = subprocess.run(
        [sys.executable, "-c", FRESH_PROBE, str(bundle), str(clip), device],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
        cwd=REPO_ROOT,
    )
    vals = json.loads(out.stdout.strip().splitlines()[-1])
    return {k: round(v, 1) for k, v in vals.items()}


def run_variant(name: str, kwargs: dict, bundle: Path, clips, reference: dict, device: str) -> dict:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    rss0 = psutil.Process().memory_info().rss

    with RssSampler() as sampler:
        t0 = time.perf_counter()
        analyzer = VideoAnalyzer(bundle, device=device, **kwargs)
        construct_s = time.perf_counter() - t0
        warmup_s = analyzer.warmup()
        rss_loaded = psutil.Process().memory_info().rss

        per_clip = []
        stage_ram = {s: 0 for s in ("detect", "align+visual", "audio", "model", "postprocess")}
        for vid, path, _ in clips:
            start = time.perf_counter()
            res = analyzer.analyze(path)
            tm = res.timings
            # Stages run in order; rebuild their windows from the cumulative durations.
            edges = np.cumsum(
                [
                    0,
                    tm["detect"],
                    tm["align"] + tm["visual"],
                    tm["audio"],
                    tm["model"],
                    tm["postprocess"],
                ]
            )
            for k, name_ in enumerate(stage_ram):
                stage_ram[name_] = max(
                    stage_ram[name_], sampler.peak(start + edges[k], start + edges[k + 1])
                )
            ref = reference["frames"].get(vid)
            n = min(len(ref), len(res.frame_scores)) if ref is not None else 0
            per_clip.append(
                {
                    "video_id": vid,
                    "n_frames": res.n_frames,
                    "duration_s": res.duration_s,
                    **{f"{k}_s": tm[k] for k in (*STAGES, "total")},
                    "s_per_10s": tm["total"] * FRAMES_10S / res.n_frames,
                    "clip_score": res.clip_score,
                    "n_segments": len(res.segments),
                    "max_abs_frame_diff_vs_dev": (
                        float(np.abs(res.frame_scores[:n] - ref[:n]).max()) if n else None
                    ),
                    "abs_clip_diff_vs_dev": (
                        abs(res.clip_score - reference["clip_scores"][vid])
                        if ref is not None
                        else None
                    ),
                    "frame_scores": res.frame_scores,
                }
            )
            print(
                f"    {name:<14} {vid}  T={res.n_frames:>3}  {tm['total']:5.2f}s"
                f"  ({tm['total'] * FRAMES_10S / res.n_frames:5.2f}s/10s)  {DIM}"
                + " ".join(f"{s}={tm[s]:.2f}" for s in STAGES)
                + RESET,
                flush=True,
            )
        # Read the run's peak *before* the probe, which resets the counter.
        peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0.0
        vram = component_vram(analyzer, device)
        analyzer.close()
        del analyzer

    def med(key: str) -> float:
        return float(statistics.median(c[key] for c in per_clip))

    norm = [c["s_per_10s"] for c in per_clip]
    long_clips = [c["total_s"] for c in per_clip if c["n_frames"] >= 225]
    diffs = [
        c["max_abs_frame_diff_vs_dev"]
        for c in per_clip
        if c["max_abs_frame_diff_vs_dev"] is not None
    ]
    return {
        "variant": name,
        "kwargs": kwargs,
        "construct_s": round(construct_s, 2),
        "warmup_s": round(warmup_s, 2),
        "n_clips": len(per_clip),
        "median_stage_s": {s: round(med(f"{s}_s"), 4) for s in (*STAGES, "total")},
        "median_stage_s_per_10s": {
            s: round(
                float(
                    statistics.median(c[f"{s}_s"] * FRAMES_10S / c["n_frames"] for c in per_clip)
                ),
                4,
            )
            for s in (*STAGES, "total")
        },
        "s_per_10s": {
            "median": round(float(np.median(norm)), 3),
            "p90": round(float(np.percentile(norm, 90)), 3),
            "max": round(float(np.max(norm)), 3),
        },
        "long_clips_total_s": {
            "n": len(long_clips),
            "median": round(float(np.median(long_clips)), 3) if long_clips else None,
            "max": round(float(np.max(long_clips)), 3) if long_clips else None,
        },
        "real_time_factor_median": round(
            float(np.median([c["total_s"] / c["duration_s"] for c in per_clip])), 3
        ),
        "peak_vram_mb": round(peak_vram, 1),
        "component_vram_mb": vram,
        "rss_before_mb": round(rss0 / 1024**2, 1),
        "rss_loaded_mb": round(rss_loaded / 1024**2, 1),
        "peak_rss_mb": round(sampler.peak() / 1024**2, 1),
        "peak_rss_by_stage_mb": {k: round(v / 1024**2, 1) for k, v in stage_ram.items()},
        "max_abs_frame_diff_vs_dev": max(diffs) if diffs else None,
        "clips": per_clip,
    }


def drift_vs(base: dict, other: dict) -> dict:
    """Numerical change a variant introduces relative to fp32 `optimized`, clip by clip."""
    a = {c["video_id"]: c for c in base["clips"]}
    frame = [
        float(np.abs(c["frame_scores"] - a[c["video_id"]]["frame_scores"]).max())
        for c in other["clips"]
    ]
    clip = [abs(c["clip_score"] - a[c["video_id"]]["clip_score"]) for c in other["clips"]]
    seg = sum(c["n_segments"] != a[c["video_id"]]["n_segments"] for c in other["clips"])
    return {
        "max_abs_frame_diff": max(frame),
        "max_abs_clip_diff": max(clip),
        "clips_with_different_segment_count": int(seg),
    }


def table_8_2_3(res: dict, chosen: str) -> str:
    """Per-stage RAM is the peak RSS *above the loaded, warmed analyzer* -- the process's
    resident baseline (CUDA context, torch, MediaPipe, weights) is reported once, separately,
    because charging it to every row would make each row read ~2 GB and say nothing."""
    v = res["variants"][chosen]
    per10 = v["median_stage_s_per_10s"]
    base = v["rss_loaded_mb"]
    vram = v["component_vram_mb"]

    def ram(stage: str) -> str:
        peak = v["peak_rss_by_stage_mb"][stage]
        return f"+{max(peak - base, 0.0):.0f}" if peak else "< sampling period"

    rows = [
        ("Decode + face landmarks (MediaPipe, CPU)", per10["detect"], "--", ram("detect")),
        ("Alignment (warp, CPU)", per10["align"], "--", ram("align+visual")),
        (
            "Visual features (MobileNetV2, GPU, batch 64)",
            per10["visual"],
            vram.get("visual", "--"),
            ram("align+visual"),
        ),
        ("Audio features (ffmpeg + log-Mel, CPU)", per10["audio"], "--", ram("audio")),
        (
            "Stage-B model (section 5.1, GPU, T=497)",
            per10["model"],
            vram.get("model", "--"),
            ram("model"),
        ),
        ("Post-processing (section 6.4)", per10["postprocess"], "--", ram("postprocess")),
    ]
    lines = [
        "| Component | Median latency per 10 s clip | Peak VRAM, isolated (MB) | Peak RAM above loaded baseline (MB) |",
        "|---|---|---|---|",
    ]
    for name, s, vr, r in rows:
        lines.append(f"| {name} | {s * 1000:.0f} ms | {vr} | {r} |")
    lines.append(
        f"| **End to end** | **{v['s_per_10s']['median']:.2f} s** (p90 {v['s_per_10s']['p90']:.2f}, "
        f"max {v['s_per_10s']['max']:.2f}) | **{v['peak_vram_mb']}** (whole run) | "
        f"**{v['peak_rss_mb']}** total RSS |"
    )
    lines += [
        "",
        f"Loaded baseline: process RSS **{base} MB** after construction + warm-up (before: "
        f"{v['rss_before_mb']} MB); weights resident on the GPU "
        f"{vram.get('weights_resident', '--')} MB.",
    ]
    return "\n".join(lines)


def write_markdown(res: dict, path: Path) -> None:
    chosen = res["chosen_variant"]
    v = res["variants"]
    lines = [
        "# Phase 12 benchmark -- Table 8.2.3",
        "",
        f"Generated by `scripts/41_benchmark.py` at `{res['git_sha']}` on {res['generated_at']}.",
        f"{res['n_clips']} dev clips ({res['clip_frames_min']}-{res['clip_frames_max']} frames), "
        f"device `{res['device']}` ({res['gpu']}). Test split not used (spent, P11-8).",
        "",
        f"Ambient at start: CPU {res['ambient']['cpu_percent_2s']}% busy, "
        f"{res['ambient']['ram_available_gb']} GB of {res['ambient']['ram_total_gb']} GB RAM free, "
        f"{res['ambient']['chrome_processes']} Chrome processes running.",
        "",
        f"## Table 8.2.3 -- performance (GTX 1650, 4 GB), variant `{chosen}`",
        "",
        table_8_2_3(res, chosen),
        "",
        f"Model file: `{res['model']['path']}`, **{res['model']['size_mb']} MB** "
        f"({res['model']['params_stage_b'] / 1e6:.2f}M Stage-B + {res['model']['params_backbone'] / 1e6:.2f}M "
        f"backbone params, fp32) + `face_landmarker.task` {res['model']['face_landmarker_mb']} MB.",
        f"Cold start: construct {v[chosen]['construct_s']} s + warm-up {v[chosen]['warmup_s']} s, "
        "paid once at startup.",
        "",
        f"**Fresh-process memory** (what a Phase 13 server holds; in-benchmark RSS above is inflated "
        f"by the burn-in and earlier variants sharing the process): "
        f"{res['fresh_process_rss_mb']['loaded_mb']:.0f} MB loaded + warm, "
        f"{res['fresh_process_rss_mb']['after_longest_clip_mb']:.0f} MB after analysing the longest "
        f"clip, peak working set {res['fresh_process_rss_mb']['peak_working_set_mb']:.0f} MB.",
        "",
        "## P12-3 -- the optimisation matrix (measured, not assumed)",
        "",
        "| variant | s per 10 s (median) | p90 | detect | align | visual | audio | model | peak VRAM MB | peak RSS MB | max drift vs fp32 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in v.items():
        m = r["median_stage_s_per_10s"]
        drift = res["drift_vs_optimized"].get(name)
        lines.append(
            f"| `{name}` | {r['s_per_10s']['median']:.2f} | {r['s_per_10s']['p90']:.2f} | "
            f"{m['detect']:.2f} | {m['align']:.2f} | {m['visual']:.3f} | {m['audio']:.3f} | "
            f"{m['model']:.3f} | {r['peak_vram_mb']} | {r['peak_rss_mb']} | "
            + (f"{drift['max_abs_frame_diff']:.1e}" if drift else "--")
            + " |"
        )
    lines += ["", "## Notes", ""] + [f"- {n}" for n in res["notes"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="P12-1/3/4: end-to-end benchmark -> Table 8.2.3")
    ap.add_argument("--export-facts", default="reports/model_export.json")
    ap.add_argument("--subsets", nargs="+", default=["smoke-100", "dev-2k"])
    ap.add_argument("--subset-dir", default="data/manifests/subsets")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--video-root", default="data/raw/LAV-DF")
    ap.add_argument("--n-clips", type=int, default=20)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--no-burn-in", dest="burn_in", action="store_false")
    ap.add_argument("--out", default="reports/benchmark.json")
    ap.add_argument("--md", default="reports/benchmark.md")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    export = json.loads(Path(args.export_facts).read_text(encoding="utf-8"))
    bundle = Path(export["path"])
    reference = load_predictions(Path(export["source_run"]) / "dev_predictions.npz")

    manifest = read_manifest(args.manifest).set_index("video_id")
    ids = sorted({v for s in args.subsets for v in load_subset(s, args.subset_dir)})
    roots = [Path(args.video_root) / s for s in args.subsets]
    clips = pick_clips(manifest, ids, roots, args.n_clips)
    # Pre-read every clip so no variant pays for a cold disk that the next one gets warm --
    # the dry run showed run order alone moving the median by 2x.
    for _, path, _ in clips:
        path.read_bytes()

    print(f"{DIM}{'=' * 72}{RESET}\n  P12-1/3/4  End-to-end benchmark\n{DIM}{'=' * 72}{RESET}")
    amb = ambient()
    print(
        f"  ambient: CPU {amb['cpu_percent_2s']}%, {amb['ram_available_gb']} GB RAM free, "
        f"{amb['chrome_processes']} chrome.exe"
    )
    if amb["chrome_processes"]:
        print(f"  {YELLOW}Chrome is running -- timings will be pessimistic (X-5){RESET}")
    print(f"  {len(clips)} dev clips, {clips[0][2]}-{clips[-1][2]} frames; model {bundle}")

    fresh = fresh_process_rss(bundle, clips[-1][1], args.device)
    print(
        f"  fresh process RSS: {fresh['start_mb']:.0f} MB at start, {fresh['loaded_mb']:.0f} MB "
        f"loaded+warm, {fresh['after_longest_clip_mb']:.0f} MB after the longest clip "
        f"(peak working set {fresh['peak_working_set_mb']:.0f} MB)",
        flush=True,
    )

    # Burn-in, unrecorded: the dry run showed whichever variant ran first paying ~2x on
    # *identical* work (process-level first-use costs), which would credit the variants that
    # happen to run later. One full pass with the default analyzer absorbs it.
    if args.burn_in:
        print(f"  burn-in pass over all clips {DIM}(not recorded){RESET}", flush=True)
        with VideoAnalyzer(bundle, device=args.device) as warm:
            warm.warmup()
            for _, path, _ in clips:
                warm.analyze(path)
        gc.collect()

    results = {}
    for name in args.variants:
        print(f"\n  variant {name} {DIM}{VARIANTS[name]}{RESET}")
        results[name] = run_variant(name, VARIANTS[name], bundle, clips, reference, args.device)
        r = results[name]
        print(
            f"    -> median {r['s_per_10s']['median']:.2f} s per 10 s (p90 {r['s_per_10s']['p90']:.2f}),"
            f" construct {r['construct_s']} s, warm-up {r['warmup_s']} s,"
            f" peak VRAM {r['peak_vram_mb']} MB, peak RSS {r['peak_rss_mb']} MB"
        )

    drift = {}
    if "optimized" in results:
        for name, r in results.items():
            if name != "optimized":
                drift[name] = drift_vs(results["optimized"], r)

    # The shipped configuration: the fastest variant that changes no answer. fp16 variants
    # only qualify if they introduce no drift beyond fp32's own run-to-run noise (1e-5) --
    # P12-5's "accuracy unchanged" is then true by construction, not by luck.
    def admissible(name: str) -> bool:
        return (
            name in ("optimized", "two_pass")
            or drift.get(name, {}).get("max_abs_frame_diff", 1.0) <= 1e-5
        )

    chosen = min(
        (n for n in results if admissible(n)), key=lambda n: results[n]["s_per_10s"]["median"]
    )

    notes = []
    if "two_pass" in results and "optimized" in results:
        a, b = results["two_pass"], results["optimized"]
        notes.append(
            f"Single decode saves {a['s_per_10s']['median'] - b['s_per_10s']['median']:.2f} s per 10 s "
            f"({a['s_per_10s']['median']:.2f} -> {b['s_per_10s']['median']:.2f}); peak RSS "
            f"{a['peak_rss_mb']} -> {b['peak_rss_mb']} MB."
        )
    for name in ("backbone_fp16", "model_fp16"):
        if name in results and "optimized" in results:
            r, b = results[name], results["optimized"]
            stage = "visual" if name == "backbone_fp16" else "model"
            notes.append(
                f"`{name}`: {stage} stage {b['median_stage_s_per_10s'][stage] * 1000:.0f} -> "
                f"{r['median_stage_s_per_10s'][stage] * 1000:.0f} ms per 10 s; max frame-score drift "
                f"{drift[name]['max_abs_frame_diff']:.1e}, segment count changed on "
                f"{drift[name]['clips_with_different_segment_count']} clip(s) -> "
                + ("adopted." if name == chosen else "not adopted.")
            )
    if "optimized" in results:
        m = results["optimized"]["median_stage_s_per_10s"]
        share = m["detect"] / m["total"] if m["total"] else 0.0
        notes.append(
            f"Decode + MediaPipe landmarks is {share:.0%} of the optimized pipeline; the GPU stages "
            f"together are {(m['visual'] + m['model']) / m['total']:.1%}. The lever left is the "
            "detection cadence (`detect_every_n`), which is a preprocessing change and would need "
            "the model re-validated -- out of Phase 12's scope."
        )

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": git_sha(),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "ambient": amb,
        "n_clips": len(clips),
        "clip_frames_min": clips[0][2],
        "clip_frames_max": clips[-1][2],
        "clip_ids": [c[0] for c in clips],
        "model": export,
        "target_s_per_10s": TARGET_S_PER_10S,
        "chosen_variant": chosen,
        "fresh_process_rss_mb": fresh,
        "drift_vs_optimized": drift,
        "notes": notes,
        "variants": {
            n: {
                **r,
                "clips": [{k: v for k, v in c.items() if k != "frame_scores"} for c in r["clips"]],
            }
            for n, r in results.items()
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    write_markdown(out, Path(args.md))
    print(f"\n  chosen variant: {chosen}")
    for n in notes:
        print(f"  - {n}")
    print(f"  wrote {args.out}, {args.md}")

    # Pre-committed: the p90, not the median, must clear the target -- most clips, not a typical one.
    ok = out["variants"][chosen]["s_per_10s"]["p90"] < TARGET_S_PER_10S
    print(
        f"\n{GREEN if ok else RED}{'BENCHMARK WITHIN TARGET' if ok else 'BENCHMARK OVER TARGET'}{RESET}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
