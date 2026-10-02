"""⭐ P10-6 .. P10-14: Experiment G -- post-processing tuned on dev, frozen, then scored.

    python scripts/35_experiment_g.py

Reads only what `scripts/34_train_localization.py` saved (`dev_predictions*.npz`) plus the
manifest's labels. No model, no GPU: section L's "evaluation is a pure function".

1. **Ground truth** is the manifest's `fake_periods` in seconds, clipped to each clip's
   feature length (the model sees `T` frames; a span past `T / fps` cannot be predicted).
   Real videos are listed with `[]`, so every segment predicted on one is a false positive.
2. **⛔ P10-10 grid search on dev, per arm**, maximising mean AP@0.5 over the arm's seeds:
   threshold 0.1-0.9, median kernel {1, 3, 5, 9}, min duration {0 ... 1.0 s}, merge gap
   {0 ... 1.0 s}, both step orders, and boundary refinement radius {0, 2, 4} for arms with
   the boundary head. The winner is written to `configs/postprocess_frozen.json` **before**
   any number is reported, and the test split is never scored here (P11-8 does that once).
3. **Every section 6.6 number** under the frozen config: AP@{0.5, 0.75, 0.95}, AR@{100, 50,
   20, 10}, boundary error in ms, segment-count error and the FP-segment rate on real videos
   (section 6.8) -- plus AP under the LAV-DF authors' convention for comparability with the
   literature, and the same numbers under section 6.4's *default* config to show what tuning
   bought.
4. **Arms compared paired by seed** under section 7.3 criteria 1-2, on AP@0.5 and AP@0.75.
5. **PF-21 for localization**: AP@0.5 with the visual stream zeroed.
6. **P10-13**: timelines for 20 dev videos, for a human to compare against ground truth.
7. **⛔ P10-14 gate**.

⚠️ Everything here is tuned *and* scored on dev (175 clips): model selection (frame AP),
λ, and post-processing all read it. These are development numbers, stated as such; the one
honest held-out number is Phase 11's single test run.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.manifest import read_manifest  # noqa: E402
from src.evaluation.localization import (  # noqa: E402
    AP_IOU_THRESHOLDS,
    AR_N_PROPOSALS,
    average_precision_at_iou,
    interval_iou,
    localization_report,
)
from src.localization.postprocess import (  # noqa: E402
    PostProcessConfig,
    assert_segment_contract,
    scores_to_segments,
    smooth,
)
from src.localization.targets import normalise_periods  # noqa: E402
from src.training.reporting import git_sha  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
FPS = 25.0
MAIN_ARMS = ("bce", "focal", "full")
TUNING_ARMS = ("full_l1", "full_l4", "full_b1")
LABELS = {
    "bce": "BCE control (F0 objective)",
    "focal": "focal, no boundary head",
    "full": "focal + boundary head (5.4)",
    "full_l1": "full, λ_loc = 1",
    "full_l4": "full, λ_loc = 4",
    "full_b1": "full, λ_bnd = 1",
}

GRID = {
    "threshold": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "median_kernel": [1, 3, 5, 9],
    "min_duration_s": [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0],
    "merge_gap_s": [0.0, 0.1, 0.2, 0.3, 0.5, 1.0],
    "merge_first": [False, True],
}
REFINE = [0, 2, 4]


# ---------------------------------------------------------------- loading


def load_predictions(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    offsets = np.concatenate(([0], np.cumsum(d["lengths"])))
    ids = [str(v) for v in d["video_ids"]]
    frames = {v: d["frame_scores"][offsets[i] : offsets[i + 1]] for i, v in enumerate(ids)}
    bnd = d["boundary_scores"] if "boundary_scores" in d.files else np.zeros((0, 2))
    bounds = (
        {v: bnd[offsets[i] : offsets[i + 1]] for i, v in enumerate(ids)}
        if len(bnd) == len(d["frame_scores"])
        else None
    )
    return {
        "frames": frames,
        "bounds": bounds,
        "lengths": dict(zip(ids, d["lengths"].tolist(), strict=True)),
        "labels": dict(zip(ids, d["labels"].tolist(), strict=True)),
    }


def load_arm(root: Path, arm: str) -> list[dict]:
    runs = []
    for seed_dir in sorted((root / f"phase10_{arm}").glob("seed*")):
        res, pred = seed_dir / "result.json", seed_dir / "dev_predictions.npz"
        if not (res.exists() and pred.exists()):
            continue
        run = {
            "seed_dir": seed_dir,
            "result": json.loads(res.read_text(encoding="utf-8")),
            "pred": load_predictions(pred),
        }
        nv = seed_dir / "dev_predictions_drop_visual.npz"
        run["pred_nv"] = load_predictions(nv) if nv.exists() else None
        runs.append(run)
    return runs


def ground_truth(manifest, lengths: dict[str, int]) -> dict[str, list[tuple[float, float]]]:
    """Label seconds, clipped to what the model saw (`T / fps`)."""
    gt = {}
    for vid, t in lengths.items():
        end_of_clip = t / FPS
        spans = []
        for s, e in normalise_periods(manifest.loc[vid, "fake_periods"]):
            if s < end_of_clip:
                spans.append((s, min(e, end_of_clip)))
        gt[vid] = spans
    return gt


# ---------------------------------------------------------------- post-processing


def segments(pred: dict, cfg: PostProcessConfig, smoothed: dict | None = None) -> dict:
    bounds = pred["bounds"] if cfg.refine_radius else None
    return {
        vid: scores_to_segments(
            s,
            cfg,
            boundary_scores=None if bounds is None else bounds[vid],
            smoothed=None if smoothed is None else smoothed[vid],
        )
        for vid, s in pred["frames"].items()
    }


def _grid_for_kernel(job: tuple) -> list[dict]:
    """One median kernel's slice of the grid. Top-level so worker processes can import it."""
    kernel, preds, gt, refine = job
    smoothed = [{vid: smooth(s, kernel) for vid, s in p["frames"].items()} for p in preds]
    rows = []
    for tau, mind, gap, mf, rad in itertools.product(
        GRID["threshold"],
        GRID["min_duration_s"],
        GRID["merge_gap_s"],
        GRID["merge_first"],
        refine,
    ):
        # With no min-duration filter or no merging, the two step orders are identical --
        # score the configuration once rather than twice.
        if mf and (mind == 0.0 or gap == 0.0):
            continue
        cfg = PostProcessConfig(
            threshold=tau,
            median_kernel=kernel,
            min_duration_s=mind,
            merge_gap_s=gap,
            merge_first=mf,
            refine_radius=rad,
        )
        aps = [
            average_precision_at_iou(segments(p, cfg, sm), gt, 0.5)
            for p, sm in zip(preds, smoothed, strict=True)
        ]
        rows.append({"cfg": cfg.as_dict(), "ap50": float(np.mean(aps))})
    return rows


def grid_search(
    runs: list[dict], gt: dict, has_boundary: bool, workers: int = 4
) -> tuple[dict, list[dict]]:
    """⛔ P10-10: exhaustive search on dev maximising the mean AP@0.5 over seeds.

    Split by median kernel across processes; the result is identical to a serial sweep
    because the rows are re-assembled in grid order before the (stable) sort.
    """
    refine = REFINE if has_boundary else [0]
    preds = [r["pred"] for r in runs]
    jobs = [(k, preds, gt, refine) for k in GRID["median_kernel"]]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            parts = list(pool.map(_grid_for_kernel, jobs))
    else:
        parts = [_grid_for_kernel(j) for j in jobs]
    table = [row for part in parts for row in part]
    table.sort(key=lambda row: -row["ap50"])  # stable: grid order breaks exact ties
    return table[0]["cfg"], table


# ---------------------------------------------------------------- scoring


def score_run(run: dict, gt: dict, cfg: PostProcessConfig) -> dict:
    segs = segments(run["pred"], cfg)
    for vid, s in segs.items():  # ⛔ P10-9 on real predictions, not just synthetic ones
        assert_segment_contract(s, run["pred"]["lengths"][vid] / FPS)
    rep = localization_report(segs, gt)
    rep["frame_ap"] = run["result"]["dev"]["frame_ap"]
    rep["frame_auc"] = run["result"]["dev"]["frame_auc"]
    rep["clip_auc"] = run["result"]["dev"]["auc"]
    # PF-21 for localization. Refinement is off on *both* sides of this pair: the saved
    # drop-visual predictions carry no boundary channel, and a like-for-like pair matters
    # more than the absolute number.
    if run["pred_nv"] is not None:
        plain = PostProcessConfig(**{**cfg.as_dict(), "refine_radius": 0})
        rep["ap@0.5_no_refine"] = average_precision_at_iou(segments(run["pred"], plain), gt, 0.5)
        rep["ap@0.5_drop_visual"] = average_precision_at_iou(
            segments(run["pred_nv"], plain), gt, 0.5
        )
    return rep


def summarise(reports: list[dict]) -> dict:
    keys = [k for k, v in reports[0].items() if isinstance(v, int | float)]
    return {
        k: {
            "mean": float(np.nanmean([r[k] for r in reports])),
            "std": float(np.nanstd([r[k] for r in reports])),
        }
        for k in keys
    }


def compare(a: list[dict], b: list[dict], seeds_a, seeds_b, key: str) -> dict:
    """`b - a`, paired by seed, section 7.3 criteria 1 and 2 (as in Experiment F)."""
    pa = dict(zip(seeds_a, [r[key] for r in a], strict=True))
    pb = dict(zip(seeds_b, [r[key] for r in b], strict=True))
    shared = sorted(set(pa) & set(pb))
    diffs = np.array([pb[s] - pa[s] for s in shared])
    delta = float(np.mean([pb[s] for s in shared]) - np.mean([pa[s] for s in shared]))
    spread = max(float(np.std(list(pa.values()))), float(np.std(list(pb.values()))))
    consistent = bool(diffs.size and ((diffs > 0).all() or (diffs < 0).all()))
    return {
        "delta": delta,
        "seed_spread": spread,
        "per_seed": {int(s): float(pb[s] - pa[s]) for s in shared},
        "consistent": consistent,
        "significant": bool(abs(delta) > spread and consistent and len(shared) >= 3),
        "n_paired": len(shared),
    }


# ---------------------------------------------------------------- figures (P10-13)


def timelines(run: dict, gt: dict, cfg: PostProcessConfig, manifest, out: Path, n: int = 20):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    vids = sorted(run["pred"]["frames"])
    by_class: dict[str, list[str]] = {}
    for v in vids:
        by_class.setdefault(str(manifest.loc[v, "class_name"]), []).append(v)
    quota = {"visual_only": 5, "audio_only": 5, "both": 5, "real": 5}
    chosen = []
    for cls, k in quota.items():
        pool = by_class.get(cls, [])
        chosen += list(rng.choice(pool, size=min(k, len(pool)), replace=False))
    chosen = chosen[:n]
    segs = segments(run["pred"], cfg)

    fig, axes = plt.subplots(len(chosen), 1, figsize=(11, 1.25 * len(chosen)), sharex=False)
    rows = []
    for ax, vid in zip(np.atleast_1d(axes), chosen, strict=True):
        scores = run["pred"]["frames"][vid]
        t = np.arange(len(scores)) / FPS
        for s, e in gt[vid]:
            ax.axvspan(s, e, ymin=0.55, ymax=1.0, color="#d62728", alpha=0.35, lw=0)
        for s, e, _ in segs[vid]:
            ax.axvspan(s, e, ymin=0.0, ymax=0.45, color="#2ca02c", alpha=0.45, lw=0)
        ax.plot(t, scores, color="black", lw=0.8)
        ax.axhline(cfg.threshold, color="grey", lw=0.6, ls="--")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, len(scores) / FPS)
        best = max(
            (interval_iou(p[:2], g) for g in gt[vid] for p in segs[vid]), default=float("nan")
        )
        tail = f"best IoU {best:.2f}" if gt[vid] else f"real -- {len(segs[vid])} false segment(s)"
        label = f"{vid}  {manifest.loc[vid, 'class_name']}  {tail}"
        ax.set_title(label, fontsize=8, loc="left", pad=2)
        ax.tick_params(labelsize=7)
        rows.append(
            {
                "video_id": vid,
                "class": str(manifest.loc[vid, "class_name"]),
                "ground_truth": gt[vid],
                "predicted": [list(p) for p in segs[vid]],
                "best_iou": best,
            }
        )
    fig.suptitle(
        "P10-13: red (top) = ground truth, green (bottom) = predicted segments, "
        "line = per-frame score, dashed = frozen threshold",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return rows


# ---------------------------------------------------------------- main


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Experiment G and the Phase 10 gate")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--exp-root", default="experiments")
    ap.add_argument("--frozen", default="configs/postprocess_frozen.json")
    ap.add_argument("--out", default="reports/experiment_g.md")
    ap.add_argument("--figure", default="reports/figures/localization/timelines_dev20.png")
    ap.add_argument("--min-seeds", type=int, default=3)
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="processes for the grid search")
    args = ap.parse_args()

    t_start = time.time()
    manifest = read_manifest(args.manifest).set_index("video_id")
    root = Path(args.exp_root)
    arms = {a: load_arm(root, a) for a in (*MAIN_ARMS, *TUNING_ARMS)}
    arms = {a: r for a, r in arms.items() if r}
    missing = [a for a in MAIN_ARMS if a not in arms]
    if missing:
        print(f"{RED}missing arm(s): {missing} -- run scripts/34_train_localization.py{RESET}")
        return 1

    lengths = arms["bce"][0]["pred"]["lengths"]
    gt = ground_truth(manifest, lengths)
    n_real = sum(1 for g in gt.values() if not g)
    print(
        f"{DIM}{'=' * 78}{RESET}\n  ⭐ Experiment G -- temporal localization on dev "
        f"({len(gt)} clips, {sum(len(g) for g in gt.values())} spans, {n_real} real)\n"
        f"{DIM}{'=' * 78}{RESET}"
    )
    for a, runs in arms.items():
        for r in runs:  # every arm must be scored on exactly the same clips
            if r["pred"]["lengths"] != lengths:
                print(f"{RED}{a} {r['seed_dir']}: dev clips differ from the control{RESET}")
                return 1

    # ---- ⛔ P10-10: tune on dev, per arm ------------------------------------------------
    tuned, tables = {}, {}
    for a in MAIN_ARMS:
        t0 = time.time()
        has_bnd = arms[a][0]["pred"]["bounds"] is not None
        tuned[a], tables[a] = grid_search(arms[a], gt, has_bnd, args.workers)
        print(
            f"  grid {a:<6} {len(tables[a]):>5} configs  best mean AP@0.5 "
            f"{tables[a][0]['ap50']:.4f}  {DIM}({time.time() - t0:.0f}s){RESET}"
        )
    # λ tuning (P10-5): single-seed arms, compared on seed 0 with seed-0-only tuning, so every
    # candidate -- including `full` -- gets the same treatment.
    lam = {}
    for a in ("full", *[x for x in TUNING_ARMS if x in arms]):
        seed0 = [r for r in arms[a] if r["seed_dir"].name == "seed0"]
        if not seed0:
            continue
        has_bnd = seed0[0]["pred"]["bounds"] is not None
        cfg0, tab0 = grid_search(seed0, gt, has_bnd, args.workers)
        lam[a] = {
            "ap50_seed0": tab0[0]["ap50"],
            "cfg": cfg0,
            **{k: seed0[0]["result"][k] for k in ("lambda_loc", "lambda_bnd")},
        }

    # Headline = the best 3-seed arm by mean AP@0.5 under its own tuned config.
    eligible = [a for a in MAIN_ARMS if len(arms[a]) >= args.min_seeds]
    headline = max(eligible, key=lambda a: tables[a][0]["ap50"]) if eligible else "full"
    frozen = PostProcessConfig(**tuned[headline])

    # ⛔ Freeze BEFORE reporting anything, and record that no test prediction exists.
    test_preds = sorted(str(p) for p in root.glob("phase10_*/seed*/test_predictions*.npz"))
    frozen_doc = {
        "arm": headline,
        "config": frozen.as_dict(),
        "tuned_on": "dev",
        "objective": "mean AP@0.5 (standard convention) over the arm's seeds",
        "grid": {**GRID, "refine_radius": REFINE},
        "n_configs_searched": len(tables[headline]),
        "git_sha": git_sha(),
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "test_split_scored_before_freeze": bool(test_preds),
        "per_arm": {a: tuned[a] for a in MAIN_ARMS},
    }
    Path(args.frozen).parent.mkdir(parents=True, exist_ok=True)
    Path(args.frozen).write_text(json.dumps(frozen_doc, indent=2) + "\n", encoding="utf-8")
    print(f"\n  ⛔ frozen -> {args.frozen}  (arm '{headline}'): {frozen.as_dict()}")

    # ---- score every arm under its own tuned config, and under section 6.4's defaults ------
    results, defaults = {}, {}
    for a in MAIN_ARMS:
        cfg = PostProcessConfig(**tuned[a])
        results[a] = [score_run(r, gt, cfg) for r in arms[a]]
        defaults[a] = [score_run(r, gt, PostProcessConfig()) for r in arms[a]]
    seeds = {a: [int(r["result"]["seed"]) for r in arms[a]] for a in MAIN_ARMS}
    summ = {a: summarise(results[a]) for a in MAIN_ARMS}
    summ_default = {a: summarise(defaults[a]) for a in MAIN_ARMS}

    hdr = (
        f"\n  {'arm':<30}{'AP@.5':>8}{'AP@.75':>8}{'AP@.95':>8}{'AR@10':>8}{'AR@100':>8}"
        f"{'bnd ms':>8}{'FP real':>8}{'frameAP':>9}"
    )
    print(hdr)
    for a in MAIN_ARMS:
        s = summ[a]
        print(
            f"  {LABELS[a]:<30}{s['ap@0.5']['mean']:>8.4f}{s['ap@0.75']['mean']:>8.4f}"
            f"{s['ap@0.95']['mean']:>8.4f}{s['ar@10']['mean']:>8.4f}{s['ar@100']['mean']:>8.4f}"
            f"{s['boundary_error_ms']['mean']:>8.1f}{s['fp_video_rate']['mean']:>8.3f}"
            f"{s['frame_ap']['mean']:>9.4f}"
        )
        print(
            f"  {DIM}{'  sd':<30}{s['ap@0.5']['std']:>8.4f}{s['ap@0.75']['std']:>8.4f}"
            f"{s['ap@0.95']['std']:>8.4f}{s['ar@10']['std']:>8.4f}{s['ar@100']['std']:>8.4f}"
            f"{s['boundary_error_ms']['std']:>8.1f}{s['fp_video_rate']['std']:>8.3f}"
            f"{s['frame_ap']['std']:>9.4f}{RESET}"
        )

    comps = {}
    for name, (x, y) in {
        "focal - bce": ("bce", "focal"),
        "full - focal": ("focal", "full"),
        "full - bce": ("bce", "full"),
    }.items():
        comps[name] = {
            k: compare(results[x], results[y], seeds[x], seeds[y], k)
            for k in ("ap@0.5", "ap@0.75", "boundary_error_ms", "frame_ap")
        }
    print("\n  paired by seed (section 7.3 criteria 1-2)")
    for name, c in comps.items():
        for k in ("ap@0.5", "ap@0.75"):
            v = c[k]
            tag = "significant" if v["significant"] else "not significant"
            print(
                f"    {name:<14}{k:<9}{v['delta']:+.4f}  {DIM}(spread {v['seed_spread']:.4f}, "
                f"seeds {', '.join(f'{d:+.3f}' for d in v['per_seed'].values())}){RESET}  {tag}"
            )

    tuning_gain = {
        a: summ[a]["ap@0.5"]["mean"] - summ_default[a]["ap@0.5"]["mean"] for a in MAIN_ARMS
    }
    reliance = {
        a: float(np.mean([r["ap@0.5_no_refine"] - r["ap@0.5_drop_visual"] for r in results[a]]))
        for a in MAIN_ARMS
        if "ap@0.5_drop_visual" in results[a][0]
    }
    print(
        "\n  section 6.4 defaults -> tuned, AP@0.5: "
        + ", ".join(f"{a} {tuning_gain[a]:+.4f}" for a in MAIN_ARMS)
    )
    print(
        "  PF-21, AP@0.5 lost without video: "
        + ", ".join(f"{a} {v:+.4f}" for a, v in reliance.items())
    )

    # ---- P10-13 figure ---------------------------------------------------------------------
    head_run = next(r for r in arms[headline] if r["seed_dir"].name == "seed0")
    rows = timelines(head_run, gt, frozen, manifest, Path(args.figure))
    print(f"\n  P10-13 timelines -> {args.figure}")

    # ---- ⛔ P10-14 gate ---------------------------------------------------------------------
    if args.skip_tests:
        tests_ok = None
    else:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/unit/test_localization_metrics.py",
                "tests/unit/test_postprocess.py",
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        )
        tests_ok = proc.returncode == 0
        print(f"  AP + post-processing unit tests: {proc.stdout.strip().splitlines()[-1]}")
    xc = (
        json.loads(Path("reports/ap_crosscheck.json").read_text(encoding="utf-8"))
        if Path("reports/ap_crosscheck.json").exists()
        else {"passed": False}
    )
    gta = (
        json.loads(Path("reports/gt_alignment.json").read_text(encoding="utf-8"))
        if Path("reports/gt_alignment.json").exists()
        else {"passed": False}
    )
    h = summ[headline]
    metric_keys = [f"ap@{t:g}" for t in AP_IOU_THRESHOLDS] + [f"ar@{n}" for n in AR_N_PROPOSALS]
    metrics_ok = all(np.isfinite(h[k]["mean"]) for k in metric_keys)
    checks = {
        "AP/AR + post-processing unit tests pass (P10-7, P10-8, P10-9)": tests_ok,
        "AP/AR cross-checked against the LAV-DF authors' evaluator": bool(xc.get("passed")),
        "ground-truth round trip on the full manifest (P10-1, P10-2)": bool(gta.get("passed")),
        "AP@{0.5,0.75,0.95} + AR@{100,50,20,10} computed": metrics_ok,
        "post-processing frozen before any test-split scoring": not test_preds,
        f">= {args.min_seeds} seeds for the headline arm": len(arms[headline]) >= args.min_seeds,
    }
    gate = all(v is not False and v is not None for v in checks.values())
    print("\n  Gate P10-14: localization metrics computed with a verified AP implementation")
    for k, v in checks.items():
        print(f"    {k:<66}: {v}")
    print(f"\n  {(GREEN if gate else RED)}GATE P10-14 {'PASSED' if gate else 'NOT PASSED'}{RESET}")

    facts = {
        "headline_arm": headline,
        "frozen": frozen_doc,
        "n_dev_clips": len(gt),
        "n_dev_spans": int(sum(len(g) for g in gt.values())),
        "n_dev_real": n_real,
        "arms": {
            a: {
                "label": LABELS[a],
                "seeds": seeds[a],
                "tuned_config": tuned[a],
                "summary": summ[a],
                "summary_default_postprocess": summ_default[a],
                "runs": results[a],
                "top5_configs": tables[a][:5],
                "git_sha": sorted({r["result"]["git_sha"] for r in arms[a]}),
                "epochs_mean": float(np.mean([r["result"]["epochs_run"] for r in arms[a]])),
                "wall_clock_s_mean": float(np.mean([r["result"]["wall_clock_s"] for r in arms[a]])),
                "n_params": arms[a][0]["result"]["n_params"],
            }
            for a in MAIN_ARMS
        },
        "lambda_tuning_seed0": lam,
        "comparisons": comps,
        "tuning_gain_ap50": tuning_gain,
        "visual_reliance_ap50": reliance,
        "timelines": rows,
        "checks": checks,
        "gate_passed": gate,
        "seconds": round(time.time() - t_start, 1),
    }
    out = Path(args.out)
    out.with_suffix(".json").write_text(
        json.dumps(facts, indent=2, default=float) + "\n", encoding="utf-8"
    )
    write_markdown(out, facts)
    print(f"  wrote {out} and {out.with_suffix('.json')}")
    return 0 if gate else 1


def _pm(s: dict, k: str, fmt: str = ".4f") -> str:
    return f"{s[k]['mean']:{fmt}} ± {s[k]['std']:{fmt}}"


def write_markdown(path: Path, f: dict) -> None:
    h = f["headline_arm"]
    hs = f["arms"][h]["summary"]
    fz = f["frozen"]["config"]
    lines = [
        "# Experiment G — temporal localization",
        "",
        f"**Headline ({f['arms'][h]['label']}, dev, 3 seeds): AP@0.5 = {_pm(hs, 'ap@0.5')}, "
        f"AP@0.75 = {_pm(hs, 'ap@0.75')}, AP@0.95 = {_pm(hs, 'ap@0.95')}; boundaries within "
        f"{hs['boundary_error_ms']['mean']:.0f} ms on average; a span is flagged on "
        f"{100 * hs['fp_video_rate']['mean']:.1f}% of real videos.**",
        "",
        "Generated by `scripts/35_experiment_g.py` from the saved dev predictions of "
        "`scripts/34_train_localization.py`. Dev: "
        f"{f['n_dev_clips']} clips, {f['n_dev_spans']} forged spans, {f['n_dev_real']} real videos.",
        "",
        "> ⚠️ **Development numbers.** Model selection (frame AP), λ and post-processing were all",
        "> chosen on this same dev split, so every figure here is optimistic by construction. The",
        "> held-out number is Phase 11's single test run (P11-8), made with the frozen config below.",
        "",
        "## Arms",
        "",
        "Every arm is Phase 9's F0 (the section 5.1 model, no augmentation — PF-25) with one change",
        "to the objective; all select `best.pt` on dev frame AP; each gets its own post-processing,",
        "tuned on dev by the same grid.",
        "",
        "| arm | AP@0.5 | AP@0.75 | AP@0.95 | AR@10 | AR@100 | boundary error (ms) | FP rate on real |"
        " frame AP |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for d in f["arms"].values():
        s = d["summary"]
        lines.append(
            f"| {d['label']} | {_pm(s, 'ap@0.5')} | {_pm(s, 'ap@0.75')} | {_pm(s, 'ap@0.95')} | "
            f"{_pm(s, 'ar@10')} | {_pm(s, 'ar@100')} | {_pm(s, 'boundary_error_ms', '.0f')} | "
            f"{_pm(s, 'fp_video_rate', '.3f')} | {_pm(s, 'frame_ap')} |"
        )
    lines += [
        "",
        "AR@N counts a span as recalled if any of its video's top-N segments overlaps it; averaged",
        "over IoU 0.5:0.05:0.95. Section 6.4's chain emits a handful of segments per clip, so",
        "AR@100, @50 and @20 coincide — a property of the post-processing, not of the metric.",
        "",
        "### Under the LAV-DF authors' AP convention",
        "",
        'For comparison with published LAV-DF numbers only (`convention="lavdf"`, cross-checked',
        "against their evaluator in `reports/ap_crosscheck.json`): it drops the first recall step",
        "and matches by list order.",
        "",
        "| arm | AP@0.5 | AP@0.75 | AP@0.95 |",
        "|---|---|---|---|",
    ]
    for d in f["arms"].values():
        s = d["summary"]
        lines.append(
            f"| {d['label']} | {_pm(s, 'ap@0.5_lavdf')} | {_pm(s, 'ap@0.75_lavdf')} | "
            f"{_pm(s, 'ap@0.95_lavdf')} |"
        )

    lines += ["", "## ⛔ Comparisons, paired by seed (section 7.3 criteria 1–2)", ""]
    lines += [
        "| comparison | metric | Δ | seed spread | per seed | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for name, c in f["comparisons"].items():
        for k in ("ap@0.5", "ap@0.75", "boundary_error_ms"):
            v = c[k]
            fmt = ".1f" if k == "boundary_error_ms" else ".4f"
            lines.append(
                f"| {name} | {k} | {v['delta']:+{fmt}} | {v['seed_spread']:{fmt}} | "
                + ", ".join(f"{d:+{fmt}}" for d in v["per_seed"].values())
                + f" | {'**significant**' if v['significant'] else 'not significant'} |"
            )
    lines += [
        "",
        "Lower is better for boundary error, so a negative Δ there is an improvement.",
        "",
        "## ⛔ P10-10 — post-processing, tuned on dev and frozen",
        "",
        f"Grid: {len(f['frozen']['grid']['threshold'])} thresholds × "
        f"{len(f['frozen']['grid']['median_kernel'])} kernels × "
        f"{len(f['frozen']['grid']['min_duration_s'])} min durations × "
        f"{len(f['frozen']['grid']['merge_gap_s'])} merge gaps × 2 step orders"
        " (× 3 refinement radii for arms with a boundary head), maximising mean AP@0.5 over seeds.",
        "",
        f"**Frozen** (`configs/postprocess_frozen.json`, arm `{h}`): threshold "
        f"{fz['threshold']}, median kernel {fz['median_kernel']}, closing {fz['closing']}, "
        f"min duration {fz['min_duration_s']} s, merge gap {fz['merge_gap_s']} s, "
        f"{'merge before' if fz['merge_first'] else 'min-duration before'} the other step, "
        f"boundary refinement radius {fz['refine_radius']}.",
        "",
        "What tuning bought over section 6.4's defaults (threshold 0.5, kernel 5, min duration",
        "0.4 s, merge gap 0.3 s), AP@0.5:",
        "",
        "| arm | defaults | tuned | gain |",
        "|---|---|---|---|",
    ]
    for a, d in f["arms"].items():
        lines.append(
            f"| {d['label']} | {d['summary_default_postprocess']['ap@0.5']['mean']:.4f} | "
            f"{d['summary']['ap@0.5']['mean']:.4f} | {f['tuning_gain_ap50'][a]:+.4f} |"
        )
    lines += [
        "",
        "Top configurations for the headline arm (a flat top means the choice is not fragile):",
        "",
    ]
    lines += [
        "| threshold | kernel | min dur | merge gap | merge first | refine | mean AP@0.5 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in f["arms"][h]["top5_configs"]:
        c = row["cfg"]
        lines.append(
            f"| {c['threshold']} | {c['median_kernel']} | {c['min_duration_s']} | "
            f"{c['merge_gap_s']} | {c['merge_first']} | {c['refine_radius']} | {row['ap50']:.4f} |"
        )

    if f["lambda_tuning_seed0"]:
        lines += [
            "",
            "## P10-5 — λ tuning (seed 0, each candidate with its own seed-0-tuned post-processing)",
            "",
            "| arm | λ_loc | λ_bnd | AP@0.5 |",
            "|---|---|---|---|",
        ]
        for a, d in f["lambda_tuning_seed0"].items():
            lines.append(f"| {a} | {d['lambda_loc']} | {d['lambda_bnd']} | {d['ap50_seed0']:.4f} |")
        lines += [
            "",
            "Single-seed, so a difference smaller than the 3-seed spread of `full` is not a result.",
        ]

    lines += [
        "",
        "## ⛔ PF-21 — does localization use the video?",
        "",
        "AP@0.5 lost when the visual stream is zeroed (boundary refinement off on both sides of the",
        "pair). The same 0.01 threshold Phases 6–9 used for clip AUC.",
        "",
        "| arm | AP@0.5 lost without video |",
        "|---|---|",
    ]
    for a, v in f["visual_reliance_ap50"].items():
        lines.append(f"| {f['arms'][a]['label']} | {v:+.4f} |")

    lines += [
        "",
        "## P10-13 — timelines for a human check",
        "",
        "`reports/figures/localization/timelines_dev20.png`: 20 dev clips (5 per class, seeded",
        "choice), headline arm seed 0, frozen config. ✋ Still needs a human to look at it.",
        "",
        "## ⛔ Gate P10-14",
        "",
        "> localization metrics computed with a **verified** AP implementation",
        "",
        "| check | result |",
        "|---|---|",
    ]
    for k, v in f["checks"].items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        f"**{'PASSED' if f['gate_passed'] else 'NOT PASSED'}.**",
        "",
        "## ⚠️ Read alongside PF-17, PF-19, PF-21 and PF-22",
        "",
        "The model localizes almost entirely from audio (PF-21), and the audio pathway reads a",
        "global processing fingerprint (PF-19). Localization is harder to game with a clip-level",
        "fingerprint than classification is — the span still has to be found — but the absolute",
        "numbers remain upper bounds on this data, and the deltas between arms are the trustworthy part.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
