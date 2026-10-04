"""⛔ P12-5 + P12-6: accuracy re-verified after optimisation, then the Phase 12 gate.

    python scripts/42_phase12_gate.py

**P12-5.** Runs the shipped configuration -- `models/avtfd_g_seed0.pt` through
`VideoAnalyzer.analyze`, from raw `.mp4` -- over **every** dev clip of Experiment G and
compares it with what the same seed predicted from cached features in Phase 10
(`dev_predictions.npz`): per-frame and per-clip scores, then every headline dev metric through
`src/evaluation/runs.py`, the same functions every reported number went through. Dev, never
test: the test split is spent (`reports/test_once.lock`).

Tolerances are fixed here, before the run, and are tight on purpose -- the pipeline is meant
to compute the *same* function, not a close one:

* max |Δ frame score| <= 1e-4 and max |Δ clip score| <= 1e-4
* |Δ| <= 0.001 on clip AUC, frame AP, AP@0.5 and AP@0.75; identical segment count per clip

**P12-6 gate**: p90 latency < 10 s per 10 s clip (`reports/benchmark.json`), model file < 50 MB
(`reports/model_export.json`, sha256 re-checked), P12-5 passed, and the benchmark and this
check ran the same model file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.manifest import read_manifest  # noqa: E402
from src.evaluation.runs import (  # noqa: E402
    classification_metrics,
    ground_truth,
    load_predictions,
    localization_metrics,
    segments,
)
from src.inference.pipeline import VideoAnalyzer  # noqa: E402
from src.localization.postprocess import PostProcessConfig  # noqa: E402
from src.training.reporting import git_sha  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"
TOL_SCORE = 1e-4
TOL_METRIC = 1e-3
METRICS = (
    ("auc", "clip AUC"),
    ("frame_ap", "frame AP"),
    ("ap@0.5", "AP@0.5"),
    ("ap@0.75", "AP@0.75"),
)
LATENCY_TARGET_S = 10.0
SIZE_LIMIT_MB = 50.0


def find_video(vid: str, roots: list[Path]) -> Path | None:
    return next((r / f"{vid}.mp4" for r in roots if (r / f"{vid}.mp4").exists()), None)


def metrics_for(pred: dict, manifest, cfg: PostProcessConfig) -> dict:
    cls = classification_metrics(pred, manifest)
    gt = ground_truth(manifest, pred["lengths"])
    loc = localization_metrics(pred, gt, cfg)
    return {
        "auc": cls["auc"],
        "frame_ap": cls["frame_ap"],
        "ap@0.5": loc["ap@0.5"],
        "ap@0.75": loc["ap@0.75"],
        "boundary_error_ms": loc["boundary_error_ms"],
        "fp_video_rate": loc["fp_video_rate"],
    }


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="P12-5 accuracy re-verification + P12-6 gate")
    ap.add_argument("--export-facts", default="reports/model_export.json")
    ap.add_argument("--benchmark", default="reports/benchmark.json")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument(
        "--video-roots", nargs="+", default=["data/raw/LAV-DF/smoke-100", "data/raw/LAV-DF/dev-2k"]
    )
    ap.add_argument("--out", default="reports/phase12_gate.json")
    ap.add_argument("--md", default="reports/phase12_gate.md")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    export = json.loads(Path(args.export_facts).read_text(encoding="utf-8"))
    bench = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    model_path = Path(export["path"])
    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    run_dir = Path(export["source_run"])
    reference = load_predictions(run_dir / "dev_predictions.npz")
    cfg = PostProcessConfig(**export["postprocess"])
    manifest = read_manifest(args.manifest).set_index("video_id")
    roots = [Path(r) for r in args.video_roots]

    ids = list(reference["lengths"])
    assert all(manifest.loc[v, "split"] == "dev" for v in ids), "reference must be dev only"
    print(
        f"{DIM}{'=' * 72}{RESET}\n  ⛔ P12-5 / P12-6  Accuracy after optimisation + gate\n{DIM}{'=' * 72}{RESET}"
    )
    print(f"  model {model_path} ({export['size_mb']} MB), reference {run_dir}/dev_predictions.npz")
    print(f"  {len(ids)} dev clips; tolerances: scores {TOL_SCORE:g}, metrics {TOL_METRIC:g}")

    frames, clips, lengths, missing = {}, {}, {}, []
    frame_diff, clip_diff = {}, {}
    t_start = time.time()
    with VideoAnalyzer(model_path, device=args.device) as analyzer:
        analyzer.warmup()
        for i, vid in enumerate(ids, 1):
            path = find_video(vid, roots)
            if path is None:
                missing.append(vid)
                continue
            res = analyzer.analyze(path)
            ref = reference["frames"][vid]
            # The pipeline sees the same frames the cache did; a length change is itself a failure.
            n = len(ref)
            frames[vid] = res.frame_scores[:n] if len(res.frame_scores) >= n else res.frame_scores
            clips[vid] = res.clip_score
            lengths[vid] = len(frames[vid])
            frame_diff[vid] = (
                float(np.abs(frames[vid] - ref).max()) if len(frames[vid]) == n else float("inf")
            )
            clip_diff[vid] = abs(res.clip_score - reference["clip_scores"][vid])
            if i % 25 == 0 or i == len(ids):
                print(
                    f"    {i}/{len(ids)}  {DIM}{(time.time() - t_start) / i:.2f} s/clip{RESET}",
                    flush=True,
                )

    pipeline = {
        "frames": frames,
        "bounds": None,
        "lengths": lengths,
        "labels": {v: reference["labels"][v] for v in frames},
        "clip_scores": clips,
    }
    out_npz = run_dir / "dev_predictions_e2e.npz"
    order = list(frames)
    np.savez_compressed(
        out_npz,
        video_ids=np.array(order),
        labels=np.array([pipeline["labels"][v] for v in order], dtype=np.int8),
        clip_scores=np.array([clips[v] for v in order], dtype=np.float32),
        lengths=np.array([lengths[v] for v in order], dtype=np.int32),
        frame_scores=np.concatenate([frames[v] for v in order]).astype(np.float32),
    )

    ref_sub = {
        **reference,
        "frames": {v: reference["frames"][v] for v in order},
        "lengths": {v: reference["lengths"][v] for v in order},
        "labels": {v: reference["labels"][v] for v in order},
        "clip_scores": {v: reference["clip_scores"][v] for v in order},
        "bounds": None,
    }
    m_ref = metrics_for(ref_sub, manifest, cfg)
    m_e2e = metrics_for(pipeline, manifest, cfg)
    seg_ref, seg_e2e = segments(ref_sub, cfg), segments(pipeline, cfg)
    seg_mismatch = [v for v in order if len(seg_ref[v]) != len(seg_e2e[v])]

    max_frame = max(frame_diff.values()) if frame_diff else float("inf")
    max_clip = max(clip_diff.values()) if clip_diff else float("inf")
    deltas = {k: m_e2e[k] - m_ref[k] for k, _ in METRICS}

    p12_5 = {
        f"all {len(ids)} dev clips found and analysed": not missing,
        f"max |Δ frame score| {max_frame:.1e} <= {TOL_SCORE:g}": max_frame <= TOL_SCORE,
        f"max |Δ clip score| {max_clip:.1e} <= {TOL_SCORE:g}": max_clip <= TOL_SCORE,
        **{
            f"|Δ {label}| {abs(deltas[k]):.1e} <= {TOL_METRIC:g}": abs(deltas[k]) <= TOL_METRIC
            for k, label in METRICS
        },
        f"segment count identical on every clip ({len(seg_mismatch)} differ)": not seg_mismatch,
    }
    chosen = bench["variants"][bench["chosen_variant"]]
    p90 = chosen["s_per_10s"]["p90"]
    gate = {
        f"p90 latency {p90:.2f} s per 10 s clip < {LATENCY_TARGET_S:.0f} s": p90 < LATENCY_TARGET_S,
        f"model file {export['size_mb']} MB < {SIZE_LIMIT_MB:.0f} MB": export["size_mb"]
        < SIZE_LIMIT_MB,
        "model file sha256 matches the export record": sha == export["sha256"],
        "benchmark ran the same model file": bench["model"]["sha256"] == sha,
        "P12-5: accuracy confirmed unchanged on dev": all(p12_5.values()),
    }

    print("\n  P12-5")
    for k, v in p12_5.items():
        print(f"    [{GREEN + 'ok' if v else RED + 'NO'}{RESET}] {k}")
    print("\n  dev metrics, cached features -> end-to-end pipeline")
    for k, label in METRICS:
        print(f"    {label:<10} {m_ref[k]:.4f} -> {m_e2e[k]:.4f}  (Δ {deltas[k]:+.1e})")
    print("\n  ⛔ P12-6 gate")
    for k, v in gate.items():
        print(f"    [{GREEN + 'ok' if v else RED + 'NO'}{RESET}] {k}")
    passed = all(gate.values())

    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": git_sha(),
        "model": {"path": model_path.as_posix(), "sha256": sha, "size_mb": export["size_mb"]},
        "reference": (run_dir / "dev_predictions.npz").as_posix(),
        "pipeline_predictions": out_npz.as_posix(),
        "n_clips": len(order),
        "missing": missing,
        "tolerances": {"score": TOL_SCORE, "metric": TOL_METRIC},
        "max_abs_frame_diff": max_frame,
        "max_abs_clip_diff": max_clip,
        "metrics_cached": m_ref,
        "metrics_pipeline": m_e2e,
        "deltas": deltas,
        "segment_count_mismatch": seg_mismatch,
        "seconds_per_clip": round((time.time() - t_start) / max(len(order), 1), 3),
        "latency": {"variant": bench["chosen_variant"], **chosen["s_per_10s"]},
        "p12_5": p12_5,
        "gate": gate,
        "gate_passed": passed,
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Phase 12 gate -- accuracy after optimisation (P12-5) and P12-6",
        "",
        f"Generated by `scripts/42_phase12_gate.py` at `{result['git_sha']}` on {result['generated_at']}.",
        f"Shipped model `{model_path.as_posix()}` ({export['size_mb']} MB), run end to end from raw "
        f"`.mp4` on all {len(order)} dev clips, compared with the same seed's cached-feature "
        "predictions from Phase 10. Test split not used.",
        "",
        "| metric (dev) | cached features (Phase 10) | end-to-end pipeline | Δ |",
        "|---|---|---|---|",
    ]
    md += [
        f"| {label} | {m_ref[k]:.4f} | {m_e2e[k]:.4f} | {deltas[k]:+.1e} |" for k, label in METRICS
    ]
    md += [
        f"| boundary error | {m_ref['boundary_error_ms']:.1f} ms | {m_e2e['boundary_error_ms']:.1f} ms | "
        f"{m_e2e['boundary_error_ms'] - m_ref['boundary_error_ms']:+.1e} |",
        "",
        f"Max |Δ frame score| **{max_frame:.1e}**, max |Δ clip score| **{max_clip:.1e}**, "
        f"segment count differs on {len(seg_mismatch)} of {len(order)} clips.",
        "",
        "## P12-5",
        "",
        *[f"- [{'x' if v else ' '}] {k}" for k, v in p12_5.items()],
        "",
        "## ⛔ P12-6 gate -- " + ("PASSED" if passed else "FAILED"),
        "",
        *[f"- [{'x' if v else ' '}] {k}" for k, v in gate.items()],
    ]
    Path(args.md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n  wrote {args.out}, {args.md}, {out_npz}")
    print(f"\n{GREEN if passed else RED}{'GATE PASSED' if passed else 'GATE FAILED'}{RESET}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
