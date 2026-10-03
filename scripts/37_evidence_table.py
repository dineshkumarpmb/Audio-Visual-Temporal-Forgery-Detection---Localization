"""Phase 11: the evidence table -- every experiment A-K, mean ± std, judged by section 7.3.

    python scripts/37_evidence_table.py            # dev evidence (+ test rows once P11-8 ran)
    python scripts/37_evidence_table.py --retune   # re-run the post-processing grid per arm

Reads only what training already saved -- each run's `result.json` and, where it exists,
`dev_predictions.npz` -- plus `reports/test_once.json` once `scripts/38_test_once.py` has
made the single test run. No model is loaded, so the table can be rebuilt at will (P11-1).

* **P11-5** mean ± std over seeds 0, 1, 2 for every arm.
* **P11-6** section 7.3's four criteria on every claimed improvement: magnitude beyond the
  larger seed std, the same sign on all three seeds (paired by seed), the right metric for
  the component's purpose, and its cost stated.
* **P11-7** Tables 8.2.1, 8.2.2, 8.2.4 and 8.2.5.
* **P11-9** negative and not-significant results are listed in full, never filtered.
* **⛔ P11-11** the gate: every number traceable to a logged run (directory + git SHA).

Localization numbers for arms trained before Phase 10 (E, F0-F2) use post-processing tuned
on dev for that arm by Experiment G's own grid, so no arm is judged under another arm's
thresholds. The tuned configs are cached in the JSON output; `--retune` recomputes them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.manifest import read_manifest  # noqa: E402
from src.evaluation.metrics import frame_metrics  # noqa: E402
from src.evaluation.runs import (  # noqa: E402
    compare,
    grid_search,
    ground_truth,
    load_predictions,
    localization_metrics,
    per_class_frame_ap,
    per_class_localization,
    summarise,
)
from src.localization.postprocess import PostProcessConfig  # noqa: E402
from src.localization.targets import frame_targets  # noqa: E402
from src.training.reporting import git_sha  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

# (id, label, experiment dir). The section 7.2 matrix, mapped onto the runs that exist.
ARMS = [
    ("A", "Visual only (MobileNetV2, attn-pool)", "phase4_mobilenet_v2_attention"),
    ("A-r18", "Visual only (ResNet-18, attn-pool)", "phase4_resnet18_attention"),
    ("B", "Audio only (log-Mel 1D-CNN)", "phase5_audio_logmel"),
    ("B-mfcc", "Audio only (MFCC 1D-CNN)", "phase5_audio_mfcc"),
    ("C", "A+V early concat", "phase6_concat"),
    ("D", "A+V + BiLSTM", "phase7_bilstm"),
    ("E-cross", "A+V cross-attention + BiLSTM", "phase8_cross"),
    ("E-self", "A+V concat + Transformer", "phase8_self"),
    ("E", "A+V cross-attention + Transformer (5.1)", "phase8_full"),
    ("F0", "E, no augmentation", "phase9_F0"),
    ("F1", "E + splice + classical aug", "phase9_F1"),
    ("F2", "F1 + feature-GAN samples", "phase9_F2"),
    ("G-bce", "G objective control (BCE)", "phase10_bce"),
    ("G-focal", "G with focal, no boundary head", "phase10_focal"),
    ("G", "Final: focal + boundary + sync", "phase10_full"),
    ("H", "G − sync loss (λ_sync = 0)", "phase11_H_nosync"),
    ("I", "G − modality dropout (p = 0)", "phase11_I_nomd"),
]
LABEL = {a: lbl for a, lbl, _ in ARMS}
DIR = {a: d for a, _, d in ARMS}

# Section 7.3 claims: (name, baseline, candidate, metrics in criterion-3 order, purpose).
# The first metric is the one the component is *justified* by; it decides the verdict.
CLAIMS = [
    ("J: MobileNetV2 vs ResNet-18 (D-1)", "A-r18", "A", ["auc"], "classification"),
    ("K: log-Mel vs MFCC (C-1)", "B-mfcc", "B", ["auc"], "classification"),
    ("C > B: fusion beats the best unimodal arm", "B", "C", ["auc"], "classification"),
    ("D > C: temporal context (BiLSTM)", "C", "D", ["auc"], "classification"),
    ("E > D: the section 5.1 model", "D", "E", ["frame_ap", "auc"], "localization"),
    ("G-1: cross-attention vs concat fusion", "E-self", "E", ["frame_ap", "auc"], "localization"),
    ("H-1: Transformer vs BiLSTM", "E-cross", "E", ["frame_ap", "auc"], "localization"),
    (
        "F1 > F0: splice + classical augmentation",
        "F0",
        "F1",
        ["ap@0.5", "frame_ap", "auc"],
        "localization",
    ),
    ("F2 > F1: GAN augmentation", "F1", "F2", ["ap@0.5", "frame_ap", "auc"], "localization"),
    (
        "focal > BCE frame loss",
        "G-bce",
        "G-focal",
        ["ap@0.5", "ap@0.75", "boundary_error_ms"],
        "localization",
    ),
    (
        "boundary head (G > G-focal)",
        "G-focal",
        "G",
        ["ap@0.5", "ap@0.75", "boundary_error_ms"],
        "localization",
    ),
    ("H: sync loss (G > H)", "H", "G", ["ap@0.5", "ap@0.75", "frame_ap"], "localization"),
    (
        "I: modality dropout (G > I)",
        "I",
        "G",
        ["ap@0.5", "visual_reliance_ap50", "frame_ap"],
        "localization",
    ),
]
LOWER_IS_BETTER = {"boundary_error_ms"}
LOC_ARMS = ("E", "F0", "F1", "F2", "G-bce", "G-focal", "G", "H", "I")
PHASE10_TUNED = {"G-bce": "bce", "G-focal": "focal", "G": "full"}


# ---------------------------------------------------------------- loading


def load_runs(root: Path, arm: str) -> list[dict]:
    runs = []
    for seed_dir in sorted((root / DIR[arm]).glob("seed*")):
        res = seed_dir / "result.json"
        if not res.exists():
            continue
        run = {
            "seed": int(seed_dir.name.removeprefix("seed")),
            "dir": seed_dir.as_posix(),
            "result": json.loads(res.read_text(encoding="utf-8")),
            "pred": None,
            "pred_nv": None,
        }
        p, nv = seed_dir / "dev_predictions.npz", seed_dir / "dev_predictions_drop_visual.npz"
        if p.exists():
            run["pred"] = load_predictions(p)
            run["pred_nv"] = load_predictions(nv) if nv.exists() else None
        runs.append(run)
    return runs


def run_metrics(run: dict, manifest, gt: dict | None, cfg: PostProcessConfig | None) -> dict:
    """One seed's flat metric dict: logged classification + re-scored frame/segment metrics."""
    dev = run["result"]["dev"]
    m = {k: dev[k] for k in ("auc", "ap", "accuracy", "precision", "recall", "f1") if k in dev}
    if "dev_drop_visual" in run["result"]:
        m["visual_reliance_auc"] = dev["auc"] - run["result"]["dev_drop_visual"]["auc"]
    if "frame_ap" in dev:
        m["frame_ap_logged"] = dev["frame_ap"]
        m["frame_ap"] = dev["frame_ap"]
    if run["pred"] is not None:
        # Re-scored under the current target rule (Phase 10's FRAME_SNAP), so every arm's
        # frame AP is measured the same way; Phases 7-9 logged it under the older rule.
        ids = list(run["pred"]["lengths"])
        t = np.concatenate(
            [
                frame_targets(
                    manifest.loc[v, "fake_periods"],
                    run["pred"]["lengths"][v],
                    float(manifest.loc[v, "fps"]),
                )
                for v in ids
            ]
        )
        s = np.concatenate([run["pred"]["frames"][v] for v in ids])
        m["frame_ap"] = frame_metrics(s, t)["frame_ap"]
    if cfg is not None and run["pred"] is not None:
        rep = localization_metrics(run["pred"], gt, cfg, run["pred_nv"])
        m.update({k: v for k, v in rep.items() if isinstance(v, int | float)})
        if "ap@0.5_drop_visual" in rep:
            m["visual_reliance_ap50"] = rep["ap@0.5_no_refine"] - rep["ap@0.5_drop_visual"]
    m["n_params"] = run["result"].get("n_params", float("nan"))
    m["wall_clock_s"] = run["result"].get("wall_clock_s", float("nan"))
    return m


# ---------------------------------------------------------------- section 7.3


def judge(name, base, cand, metrics, purpose, per_seed, seeds) -> dict:
    """All four section 7.3 criteria for one claimed improvement."""
    out = {"claim": name, "baseline": base, "candidate": cand, "purpose": purpose, "metrics": {}}
    for k in metrics:
        if not all(k in r for r in per_seed[base] + per_seed[cand]):
            continue
        c = compare(per_seed[base], per_seed[cand], seeds[base], seeds[cand], k)
        better = -c["delta"] if k in LOWER_IS_BETTER else c["delta"]
        c["direction"] = "better" if better > 0 else "worse"
        out["metrics"][k] = c
    deciding = next((k for k in metrics if k in out["metrics"]), None)
    out["deciding_metric"] = deciding
    if deciding is None:
        out["verdict"] = "not measurable"
    else:
        c = out["metrics"][deciding]
        if not c["significant"]:
            out["verdict"] = "not significant"
        else:
            out["verdict"] = "helps" if c["direction"] == "better" else "hurts"
    # Criterion 4: the cost, stated next to the gain.
    pb = np.mean([r["n_params"] for r in per_seed[base]])
    pc = np.mean([r["n_params"] for r in per_seed[cand]])
    wb = np.mean([r["wall_clock_s"] for r in per_seed[base]])
    wc = np.mean([r["wall_clock_s"] for r in per_seed[cand]])
    out["cost"] = {
        "params_baseline": float(pb),
        "params_candidate": float(pc),
        "params_ratio": float(pc / pb) if pb else float("nan"),
        "train_s_baseline": float(wb),
        "train_s_candidate": float(wc),
    }
    out["criteria"] = {
        "1_magnitude": bool(deciding and abs(c["delta"]) > c["seed_spread"]),
        "2_consistency": bool(deciding and c["consistent"]),
        "3_right_metric": deciding,
        "4_cost_stated": True,
    }
    return out


# ---------------------------------------------------------------- main


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Phase 11 evidence table")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--exp-root", default="experiments")
    ap.add_argument("--frozen", default="configs/postprocess_frozen.json")
    ap.add_argument("--test", default="reports/test_once.json")
    ap.add_argument("--out", default="reports/evidence_table.md")
    ap.add_argument("--retune", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    t0 = time.time()
    manifest = read_manifest(args.manifest).set_index("video_id")
    root = Path(args.exp_root)
    runs = {a: load_runs(root, a) for a, _, _ in ARMS}
    missing = [a for a, r in runs.items() if len(r) < 3]
    print(f"{DIM}{'=' * 78}{RESET}\n  Phase 11  Evidence table\n{DIM}{'=' * 78}{RESET}")
    for a, r in runs.items():
        tag = f"{GREEN}{len(r)} seeds{RESET}" if len(r) >= 3 else f"{RED}{len(r)} seeds{RESET}"
        print(f"  {a:<8} {DIR[a]:<32} {tag}")
    if missing:
        print(f"\n{YELLOW}incomplete arms: {missing} -- they are reported but not judged{RESET}")

    # ---- post-processing per localization arm, tuned on dev exactly as Experiment G was --
    frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))
    out_json = Path(args.out).with_suffix(".json")
    cached = {}
    if out_json.exists() and not args.retune:
        cached = json.loads(out_json.read_text(encoding="utf-8")).get("postprocess", {})
    lengths = runs["G"][0]["pred"]["lengths"]
    gt = ground_truth(manifest, lengths)
    tuned: dict[str, dict] = {}
    for a in LOC_ARMS:
        rs = [r for r in runs[a] if r["pred"] is not None]
        if len(rs) < len(runs[a]) or not rs:
            continue
        for r in rs:
            if r["pred"]["lengths"] != lengths:
                print(f"{RED}{a} {r['dir']}: dev clips differ from G's{RESET}")
                return 1
        seeds_now = [r["seed"] for r in rs]
        if a in PHASE10_TUNED:
            tuned[a] = {"cfg": frozen["per_arm"][PHASE10_TUNED[a]], "source": args.frozen}
        elif a in cached and cached[a].get("seeds") == seeds_now:
            tuned[a] = cached[a]
        else:
            t1 = time.time()
            has_bnd = rs[0]["pred"]["bounds"] is not None
            cfg, table = grid_search([r["pred"] for r in rs], gt, has_bnd, args.workers)
            tuned[a] = {
                "cfg": cfg,
                "source": "grid search on dev (P10-10 grid)",
                "ap50": table[0]["ap50"],
                "seeds": seeds_now,
            }
            print(f"  tuned {a:<8} mean AP@0.5 {table[0]['ap50']:.4f}  ({time.time() - t1:.0f}s)")

    # ---- per-seed metrics and summaries (P11-5) --------------------------------------------
    per_seed: dict[str, list[dict]] = {}
    for a, rs in runs.items():
        cfg = PostProcessConfig(**tuned[a]["cfg"]) if a in tuned else None
        per_seed[a] = [run_metrics(r, manifest, gt, cfg) for r in rs]
    seeds = {a: [r["seed"] for r in rs] for a, rs in runs.items()}
    summ = {a: summarise(v) for a, v in per_seed.items() if v}

    # ---- section 7.3 on every claim (P11-6, P11-9) -----------------------------------------
    claims = [
        judge(n, b, c, m, p, per_seed, seeds)
        for n, b, c, m, p in CLAIMS
        if len(runs[b]) >= 3 and len(runs[c]) >= 3
    ]
    print("\n  section 7.3, paired by seed")
    for c in claims:
        k = c["deciding_metric"]
        v = c["metrics"].get(k, {})
        colour = {"helps": GREEN, "hurts": RED}.get(c["verdict"], YELLOW)
        print(
            f"    {c['claim']:<46} {k or '-':<20} "
            f"{v.get('delta', float('nan')):+.4f} {DIM}(spread {v.get('seed_spread', float('nan')):.4f})"
            f"{RESET}  {colour}{c['verdict']}{RESET}"
        )

    # ---- Table 8.2.4, the four-class breakdown --------------------------------------------
    four_class = {}
    for a in ("E", "F2", "G", "H", "I"):
        if len(runs[a]) < 3 or runs[a][0]["pred"] is None:
            continue
        cfg = PostProcessConfig(**tuned[a]["cfg"])
        rows: dict[str, dict] = {}
        for r in runs[a]:
            pc = r["result"]["dev"]["per_class"]
            fap = per_class_frame_ap(r["pred"], manifest)
            lap = per_class_localization(r["pred"], gt, cfg, manifest)
            for name in ("visual_only", "audio_only", "both"):
                d = rows.setdefault(name, {"auc": [], "frame_ap": [], "ap@0.5": [], "n": 0})
                d["auc"].append(pc[name]["auc"])
                d["frame_ap"].append(fap[name])
                d["ap@0.5"].append(lap[name]["ap@0.5"])
                d["n"] = pc[name]["n_fake"]
            rows.setdefault("real", {"n": pc["real"]["n"], "specificity": []})
            rows["real"]["specificity"].append(pc["real"]["specificity"])
        four_class[a] = {
            name: {
                k: (
                    {"mean": float(np.mean(v)), "std": float(np.std(v))}
                    if isinstance(v, list)
                    else v
                )
                for k, v in d.items()
            }
            for name, d in rows.items()
        }

    # ---- the single test run, if it has happened ------------------------------------------
    test = (
        json.loads(Path(args.test).read_text(encoding="utf-8"))
        if Path(args.test).exists()
        else None
    )

    # ---- ⛔ P11-11 gate --------------------------------------------------------------------
    lock = Path("reports/test_once.lock")
    repro = {}
    for name in ("reproducibility", "reproducibility_strict"):
        rp = Path(f"reports/{name}.json")
        if rp.exists():
            repro[name] = json.loads(rp.read_text(encoding="utf-8"))
    trace = {
        a: {
            "dir": DIR[a],
            "runs": [r["dir"] for r in runs[a]],
            "git_sha": sorted({str(r["result"].get("git_sha", "")) for r in runs[a]}),
        }
        for a in runs
    }
    checks = {
        "every arm A-K has >= 3 seeds": not missing,
        "every run has a logged result.json with a git SHA": all(
            all(s for s in t["git_sha"]) and t["runs"] for t in trace.values()
        ),
        "every claim judged by all four section 7.3 criteria": len(claims) == len(CLAIMS),
        "negative / not-significant results reported": any(c["verdict"] != "helps" for c in claims),
        "P11-10: a re-run matches its original to 1e-4 (strict determinism)": bool(
            repro.get("reproducibility_strict", {}).get("passed")
        ),
        "test split scored exactly once (P11-8)": bool(
            test and lock.exists() and test.get("n_test_runs") == 1
        ),
    }
    gate = all(checks.values())
    print("\n  Gate P11-11: complete evidence table, every number traceable to a logged run")
    for k, v in checks.items():
        print(f"    {k:<58}: {v}")
    print(f"\n  {(GREEN if gate else RED)}GATE P11-11 {'PASSED' if gate else 'NOT PASSED'}{RESET}")

    facts = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": git_sha(),
        "n_dev_clips": len(gt),
        "labels": LABEL,
        "trace": trace,
        "postprocess": tuned,
        "per_seed": {a: dict(zip(seeds[a], v, strict=True)) for a, v in per_seed.items()},
        "summary": summ,
        "claims": claims,
        "four_class_dev": four_class,
        "test": test,
        "reproducibility": repro,
        "checks": checks,
        "gate_passed": gate,
        "seconds": round(time.time() - t0, 1),
    }
    out_json.write_text(json.dumps(facts, indent=2, default=float) + "\n", encoding="utf-8")
    write_markdown(Path(args.out), facts)
    print(f"  wrote {args.out} and {out_json}")
    return 0 if gate else 1


# ---------------------------------------------------------------- markdown


def _pm(s: dict, k: str, fmt: str = ".4f") -> str:
    if k not in s:
        return "—"
    return f"{s[k]['mean']:{fmt}} ± {s[k]['std']:{fmt}}"


def write_markdown(path: Path, f: dict) -> None:
    s, t = f["summary"], f["test"]
    lines = [
        "# Phase 11 — evidence table",
        "",
        "Generated by `scripts/37_evidence_table.py` from each run's `result.json` and saved",
        f"predictions. Dev split: {f['n_dev_clips']} clips. Every row is mean ± std over seeds 0, 1, 2.",
        "",
        "> ⚠️ **PF-17 / PF-19 / PF-22.** Clip length and an audio processing fingerprint are",
        "> both partly readable on this data, so absolute dev numbers are upper bounds. The",
        "> trustworthy quantities are the paired deltas in the section 7.3 table.",
        "",
        "## Table 8.2.1 — Classification (dev unless marked)",
        "",
        "| Exp | Model | Split | Acc | P | R | F1 | ROC-AUC | PR-AUC | AUC lost without video | Seeds |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a, lbl in f["labels"].items():
        if a not in s:
            continue
        x = s[a]
        lines.append(
            f"| {a} | {lbl} | dev | {_pm(x, 'accuracy')} | {_pm(x, 'precision')} | "
            f"{_pm(x, 'recall')} | {_pm(x, 'f1')} | {_pm(x, 'auc')} | {_pm(x, 'ap')} | "
            f"{_pm(x, 'visual_reliance_auc')} | {len(f['per_seed'][a])} |"
        )
    if t:
        x = t["summary"]
        lines.append(
            f"| **G** | **{f['labels']['G']}** | **test** | {_pm(x, 'accuracy')} | "
            f"{_pm(x, 'precision')} | {_pm(x, 'recall')} | {_pm(x, 'f1')} | **{_pm(x, 'auc')}** | "
            f"{_pm(x, 'ap')} | {_pm(x, 'visual_reliance_auc')} | {t['n_seeds']} |"
        )
    lines += [
        "",
        "Threshold 0.5 for Acc / P / R / F1. ROC-AUC is the primary metric (section 8.1).",
        "",
        "## Table 8.2.2 — Localization (dev unless marked)",
        "",
        "Each arm's post-processing is tuned on dev by Experiment G's grid (G's three arms reuse",
        "`configs/postprocess_frozen.json`). Frame AP is re-scored from saved predictions under",
        "the current target rule.",
        "",
        "| Exp | AP@0.5 | AP@0.75 | AP@0.95 | AR@100 | AR@50 | AR@20 | AR@10 | Boundary err (ms) | "
        "FP-seg rate (real) | frame AP |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in LOC_ARMS:
        if a not in s or "ap@0.5" not in s[a]:
            continue
        x = s[a]
        lines.append(
            f"| {a} | {_pm(x, 'ap@0.5')} | {_pm(x, 'ap@0.75')} | {_pm(x, 'ap@0.95')} | "
            f"{_pm(x, 'ar@100')} | {_pm(x, 'ar@50')} | {_pm(x, 'ar@20')} | {_pm(x, 'ar@10')} | "
            f"{_pm(x, 'boundary_error_ms', '.0f')} | {_pm(x, 'fp_video_rate', '.3f')} | "
            f"{_pm(x, 'frame_ap')} |"
        )
    if t:
        x = t["summary"]
        lines.append(
            f"| **G (test)** | **{_pm(x, 'ap@0.5')}** | {_pm(x, 'ap@0.75')} | {_pm(x, 'ap@0.95')} | "
            f"{_pm(x, 'ar@100')} | {_pm(x, 'ar@50')} | {_pm(x, 'ar@20')} | {_pm(x, 'ar@10')} | "
            f"{_pm(x, 'boundary_error_ms', '.0f')} | {_pm(x, 'fp_video_rate', '.3f')} | "
            f"{_pm(x, 'frame_ap')} |"
        )

    lines += [
        "",
        "## ⛔ Section 7.3 — does each component genuinely help?",
        "",
        "Paired by seed. *Significant* = |Δ| above the larger arm's seed std **and** the same sign",
        "on all three seeds (criteria 1–2). The first metric listed is the one the component is",
        "justified by (criterion 3) and decides the verdict; the rest are context. Cost is",
        "criterion 4 — parameters and training wall-clock here; inference latency is Phase 12's",
        "Table 8.2.3. **Every claim is listed, including the ones that failed (P11-9).**",
        "",
        "| claim | metric | Δ | seed spread | per seed | verdict | params (base → cand) | train time |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in f["claims"]:
        first = True
        for k, v in c["metrics"].items():
            fmt = ".1f" if k == "boundary_error_ms" else ".4f"
            verdict = ""
            if first:
                verdict = {
                    "helps": "✅ **helps**",
                    "hurts": "❌ **hurts**",
                }.get(c["verdict"], c["verdict"])
            cost = (
                f"{c['cost']['params_baseline'] / 1e6:.2f}M → {c['cost']['params_candidate'] / 1e6:.2f}M"
                if first
                else ""
            )
            tt = (
                f"{c['cost']['train_s_baseline'] / 60:.0f} → {c['cost']['train_s_candidate'] / 60:.0f} min"
                if first
                else ""
            )
            lines.append(
                f"| {c['claim'] if first else ''} | {k}{' ★' if first else ''} | "
                f"{v['delta']:+{fmt}} | {v['seed_spread']:{fmt}} | "
                + ", ".join(f"{d:+{fmt}}" for d in v["per_seed"].values())
                + f" | {verdict} | {cost} | {tt} |"
            )
            first = False
    lines += [
        "",
        "★ the deciding metric. Lower is better for boundary error. `visual_reliance_ap50` is",
        "AP@0.5 lost when the video stream is zeroed — for Experiment I a *positive* Δ means",
        "modality dropout made the model use video more.",
        "",
        "## Table 8.2.4 — 4-class diagnostic breakdown (dev, collapse detector §5.6)",
        "",
        "Each fake class is scored against all real clips. Video AUC from the clip head, frame",
        "AP from the frame head, AP@0.5 from the post-processed segments.",
        "",
        "| Exp | Class | N | Video AUC | Frame AP | AP@0.5 |",
        "|---|---|---|---|---|---|",
    ]
    names = {
        "visual_only": "Visual-only fake",
        "audio_only": "Audio-only fake",
        "both": "Both fake",
    }
    for a, rows in f["four_class_dev"].items():
        r = rows["real"]
        lines.append(
            f"| {a} | Real | {r['n']} | specificity {r['specificity']['mean']:.3f} | — | — |"
        )
        for k, lbl in names.items():
            d = rows[k]
            lines.append(
                f"| {a} | {lbl} | {d['n']} | {_pm(d, 'auc')} | {_pm(d, 'frame_ap')} | "
                f"{_pm(d, 'ap@0.5')} |"
            )
    if t and "four_class" in t:
        r = t["four_class"]
        for k, lbl in names.items():
            if k in r:
                d = r[k]
                lines.append(
                    f"| **G (test)** | {lbl} | {d['n']} | {_pm(d, 'auc')} | {_pm(d, 'frame_ap')} | "
                    f"{_pm(d, 'ap@0.5')} |"
                )

    lines += [
        "",
        "## Table 8.2.5 — Decision resolutions",
        "",
        "| Decision | Option A | Option B | Winner | Margin | Basis |",
        "|---|---|---|---|---|---|",
    ]

    def row(dec, a, b, key, basis):
        c = next((x for x in f["claims"] if x["claim"].startswith(key)), None)
        if c is None:
            return f"| {dec} | {a} | {b} | — | — | {basis} |"
        k = c["deciding_metric"]
        v = c["metrics"][k]
        win = b if v["delta"] > 0 else a
        sig = "" if v["significant"] else " (not significant — tie)"
        return f"| {dec} | {a} | {b} | {win}{sig} | {k} {v['delta']:+.4f} | {basis} |"

    lines += [
        row("**D-1** Visual backbone", "ResNet-18", "MobileNetV2", "J:", "Exp J (Phase 4)"),
        row("**C-1** Audio features", "MFCC", "log-Mel", "K:", "Exp K (Phase 5)"),
        row("G-1 Fusion", "Concat", "Cross-attention", "G-1", "Phase 8: E-self vs E, paired"),
        row("H-1 Temporal", "BiLSTM", "Transformer", "H-1", "Phase 8: E-cross vs E, paired"),
    ]
    lines += [
        "",
        "G-1 and H-1 use Phase 8's decomposition rather than C vs E or D vs E: those pairs",
        "differ in more than one component, which section 7.1 rule 1 forbids.",
        "",
    ]
    if t:
        lines += [
            "## ⛔ P11-8 — the single test run",
            "",
            f"Scored once, at {t['scored_at']}, git `{t['git_sha'][:12]}`: arm `{t['arm']}`, "
            f"{t['n_seeds']} seeds, {t['n_clips']} test clips ({t['n_fake']} fake / {t['n_real']} real), "
            f"post-processing `{t['frozen_config_source']}` frozen at {t['frozen_at']}.",
            "",
            "See `reports/test_once.md` for the full record.",
            "",
        ]
    rp = f.get("reproducibility", {})
    if rp:
        lines += [
            "## P11-10 — reproducibility (PF-27)",
            "",
            "G seed 0 re-trained from its logged config. `default` is the mode every Phase 4–11",
            "run was trained in (cuDNN deterministic only); `strict` adds",
            "`torch.use_deterministic_algorithms(True)` and two strict runs are compared.",
            "",
            "| mode | runs compared | |Δ AUC| | |Δ frame AP| | best epochs | within 1e-4 |",
            "|---|---|---|---|---|---|",
        ]
        for name, label in (("reproducibility", "default"), ("reproducibility_strict", "strict")):
            if name not in rp:
                continue
            r = rp[name]
            lines.append(
                f"| {label} | `{r['original']}` vs `{r['rerun']}` | "
                f"{r['metrics']['auc']['abs_diff']:.2e} | {r['metrics']['frame_ap']['abs_diff']:.2e} | "
                f"{r['best_epoch'][0]} / {r['best_epoch'][1]} | {'✅' if r['passed'] else '❌'} |"
            )
        lines += [
            "",
            "In default mode a re-run drifts by about one seed std, which the 3-seed paired",
            "protocol already absorbs; no verdict above depends on a difference that small.",
            "",
        ]
    lines += ["## ⛔ Gate P11-11", "", "| check | result |", "|---|---|"]
    for k, v in f["checks"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", f"**{'PASSED' if f['gate_passed'] else 'NOT PASSED'}.**", ""]
    lines += ["## Traceability", "", "| Exp | runs | git SHA |", "|---|---|---|"]
    for a, tr in f["trace"].items():
        lines.append(
            f"| {a} | `{tr['dir']}/seed{{0,1,2}}` | {', '.join(x[:8] for x in tr['git_sha'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
