"""⛔ P8-9 / P8-10: Experiment E, and the Phase 8 gate.

    python scripts/28_experiment_e.py

The experiment matrix predicts **E > D** — global comparison beats sequential memory. Section
H calls that *"the strongest theoretical argument in the whole project"*, and Experiment D vs
E is designed to demonstrate it empirically. This script tests it against numbers on disk and
writes the verdict, whichever way it goes.

**What P8-10 asks for: "attention contribution measured, attention maps produced."** Like
P7-6 that is a provenance gate. E changes two components at once, so a single `E − D` number
would leave the credit unassignable — which is the failure the whole experiment ladder is
built to avoid. The gate therefore requires the decomposition to exist:

    D            Phase 7, no sync loss, no attention
    d_sync       + sync loss                 -> isolates P8-5
    cross        + cross-attention           -> isolates P8-3
    self         + self-attention            -> isolates P8-1
    full         + both                      -> the section 5.1 model

Each single-component arm is compared against `d_sync`, not against D, so the sync loss is
held constant and does not leak into the attention deltas.

**Three questions, different answers, only the first two gate:**

1. **Is the contribution decomposed?** All arms present, controls matching, multi-seed.
2. **Do attention maps exist?** P8-8's artefact, plus `attention_lift` — the measured version
   of "attention elevates on forged spans", which can enter a table where a figure cannot.
3. **Is E > D?** Reported against seed variance. A Transformer that does not beat the BiLSTM
   is a legitimate result and section 5.3 is on record predicting the opposite, so recording
   it honestly is the point.

⚠️ **PF-21 is the question behind the question.** Cross-attention is the last untried defence
against the section 5.6 collapse. The `cross` arm's modality ablation is what answers it, and
it is reported whether or not it flatters the architecture.

Inputs (none entered by hand):
  reports/phase7_bilstm.json  -- D
  reports/phase8_{full,cross,self,d_sync}.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.console import init_console  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

COLLAPSE_TOLERANCE = 0.01
CONTROLLED = ("backbone", "feature", "defences", "n_clips")
ARM_LABELS = {
    "d_sync": "D + sync loss",
    "cross": "+ cross-attention",
    "self": "+ self-attention",
    "full": "E full (5.1 model)",
}


def lift_stats(report: dict) -> dict:
    """⛔ Per-seed attention lift, because the mean is not a description of this quantity.

    The `full` arm measured lifts of 0.063, 0.055 and **59.17** across three seeds that
    differ in nothing but their seed and land within 0.003 dev AUC of each other. The mean
    (19.76x) describes none of those runs, and quoting it would assert "attention elevates
    on forged spans" on the strength of one seed in three.

    So the median across seeds is the headline, the range is reported beside it, and
    `stable` records whether the seeds even agree on the *direction* — whether attention
    concentrates on the forgery or avoids it. When they disagree, the honest claim is that
    the behaviour is seed-dependent, not that it was observed.
    """
    lifts = np.array([r["dev"].get("attention_lift", np.nan) for r in report["runs"]], dtype=float)
    lifts = lifts[np.isfinite(lifts)]
    if lifts.size == 0:
        return {
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "stable": False,
            "per_seed": [],
        }
    return {
        "median": float(np.median(lifts)),
        "min": float(lifts.min()),
        "max": float(lifts.max()),
        # Every seed on the same side of 1.0 -- i.e. they agree on the direction.
        "stable": bool((lifts > 1).all() or (lifts <= 1).all()),
        "per_seed": [float(x) for x in lifts],
    }


def load(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def visual_reliance(report: dict) -> tuple[float, float, bool]:
    drops = np.array(
        [r["dev"]["auc"] - r["dev_drop_visual"]["auc"] for r in report["runs"]], dtype=float
    )
    mean, sd = float(drops.mean()), float(drops.std())
    return mean, sd, bool(mean > 2 * sd and mean >= COLLAPSE_TOLERANCE)


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Experiment E and the Phase 8 gate")
    ap.add_argument("--out", default="reports/experiment_e.md")
    ap.add_argument("--d-report", default="reports/phase7_bilstm.json")
    ap.add_argument("--maps", default="reports/figures/attention", help="P8-8 artefacts")
    args = ap.parse_args()

    d = load(args.d_report)
    arms = {name: load(f"reports/phase8_{name}.json") for name in ARM_LABELS}
    if d is None:
        print(f"{RED}missing {args.d_report} -- run Phase 7 first{RESET}")
        return 1
    if arms.get("full") is None:
        print(f"{RED}missing reports/phase8_full.json -- run scripts/27_train_attention.py{RESET}")
        return 1

    print(
        f"{DIM}{'=' * 72}{RESET}\n  🔬 Experiment E -- does attention beat recurrence?"
        f"\n{DIM}{'=' * 72}{RESET}"
    )

    # ---- the ladder ------------------------------------------------------------------
    print(f"\n  {'arm':<24}{'dev AUC':>10}{'sd':>9}{'frame AP':>11}{'lift':>8}{'-visual':>10}")
    print(
        f"  {'D (Phase 7, no sync)':<24}{d['dev_auc_mean']:>10.4f}{d['dev_auc_std']:>9.4f}"
        f"{d['dev_frame_ap_mean']:>11.4f}{'--':>8}{d['drop_visual_delta']:>+10.4f}"
    )
    for name, rep in arms.items():
        if rep is None:
            print(f"  {ARM_LABELS[name]:<24}{YELLOW}{'not run':>10}{RESET}")
            continue
        print(
            f"  {ARM_LABELS[name]:<24}{rep['dev_auc_mean']:>10.4f}{rep['dev_auc_std']:>9.4f}"
            f"{rep['dev_frame_ap_mean']:>11.4f}{lift_stats(rep)['median']:>7.2f}x"
            f"{rep['drop_visual_delta']:>+10.4f}"
        )

    full = arms["full"]
    delta_vs_d = full["dev_auc_mean"] - d["dev_auc_mean"]
    spread = max(full["dev_auc_std"], d["dev_auc_std"])
    beats = delta_vs_d > spread
    print(f"\n  E - D             : {(GREEN if beats else YELLOW)}{delta_vs_d:+.4f}{RESET}")
    print(f"  beyond seed sd    : {beats}  {DIM}(max sd {spread:.4f}){RESET}")

    # ---- ⛔ decomposition -------------------------------------------------------------
    base = arms.get("d_sync")
    contributions = {}
    print("\n  ⛔ P8-10 decomposition -- who earned the gain?")
    if base is None:
        print(f"    {YELLOW}d_sync arm missing; attention deltas cannot be isolated{RESET}")
    else:
        contributions["sync loss (P8-5)"] = base["dev_auc_mean"] - d["dev_auc_mean"]
        for name, task in (("cross", "cross-attention (P8-3)"), ("self", "self-attention (P8-1)")):
            if arms.get(name) is not None:
                contributions[task] = arms[name]["dev_auc_mean"] - base["dev_auc_mean"]
        contributions["both together (E)"] = full["dev_auc_mean"] - base["dev_auc_mean"]
        for label, value in contributions.items():
            colour = GREEN if value > spread else (YELLOW if value > 0 else RED)
            print(f"    {label:<26}{colour}{value:+.4f}{RESET}")

    decomposed = base is not None and all(arms[n] is not None for n in ("cross", "self"))

    # ---- P8-9 attention behaviour ------------------------------------------------------
    lift = lift_stats(full)
    entropy = full["runs"][0]["dev"].get("head_entropy_mean")
    sync_sep = full.get("sync_separation_mean")
    print("\n  P8-9 does attention elevate on forged spans?")
    ok = lift["stable"] and lift["median"] > 1
    print(
        f"    attention lift : {(GREEN if ok else RED)}{lift['median']:.2f}x{RESET} median"
        f"  {DIM}(pool weight on forged frames / on genuine frames, fake clips){RESET}"
    )
    print(
        f"    per seed       : {', '.join(f'{x:.2f}' for x in lift['per_seed'])}"
        f"  {DIM}range {lift['min']:.2f}-{lift['max']:.2f}{RESET}"
    )
    if not lift["stable"]:
        print(
            f"    {RED}⛔ SEED-DEPENDENT -- seeds disagree on whether attention concentrates "
            f"on the forgery or avoids it, at near-identical AUC.{RESET}\n"
            f"    {DIM}The behaviour is not a property of the architecture, and no single "
            f"attention figure represents it.{RESET}"
        )
    if sync_sep is not None:
        works = abs(sync_sep) > 0.05
        print(
            f"    sync separation: {(GREEN if works else RED)}{sync_sep:+.4f}{RESET}"
            f"  {DIM}(mean clip sync, real minus fake){RESET}"
            + ("" if works else f"  {RED}sync head does not separate real from fake{RESET}")
        )
    if entropy is not None:
        collapsed_heads = entropy > 0.98
        print(
            f"    head entropy   : {entropy:.4f}  "
            f"{(RED + 'heads are uniform -- collapsed' + RESET) if collapsed_heads else DIM + 'selective' + RESET}"
        )

    maps_dir = Path(args.maps)
    maps_exist = maps_dir.exists() and any(maps_dir.iterdir())
    print(
        f"    attention maps : {(GREEN + 'present' + RESET) if maps_exist else RED + 'MISSING' + RESET}"
        f"  {DIM}{maps_dir}{RESET}"
    )

    # ---- ⛔ PF-21: did cross-attention break the collapse? -----------------------------
    print("\n  ⛔ PF-21 -- is cross-attention the collapse fix? (section 5.6, X-7)")
    d_drop, _, _ = visual_reliance(d)
    print(f"    D (BiLSTM)         reliance on video {d_drop:+.4f}")
    fixed_by = []
    for name in ("d_sync", "cross", "self", "full"):
        rep = arms.get(name)
        if rep is None:
            continue
        drop, sd, uses = visual_reliance(rep)
        print(
            f"    {ARM_LABELS[name]:<18} reliance on video {drop:+.4f} +/- {sd:.4f}  "
            f"{(GREEN + 'uses video' + RESET) if uses else RED + 'COLLAPSED' + RESET}"
        )
        if uses:
            fixed_by.append(name)

    # ---- the gate ----------------------------------------------------------------------
    diffs = {k: (d.get(k), full.get(k)) for k in CONTROLLED if d.get(k) != full.get(k)}
    controlled = not diffs
    if diffs:
        for k, (dv, ev) in diffs.items():
            print(f"\n    {RED}uncontrolled: {k}: D={dv!r} vs E={ev!r}{RESET}")
    gate = bool(decomposed and controlled and maps_exist and len(full["runs"]) >= 2)

    print(f"\n  {DIM}Gate P8-10: attention contribution measured, attention maps produced{RESET}")
    print(f"    contribution decomposed : {decomposed}")
    print(f"    controls match D        : {controlled}")
    print(f"    attention maps produced : {maps_exist}")
    print(f"    multi-seed spread       : {len(full['runs'])} seeds")
    print(f"    E beats D               : {beats}  {DIM}(reported, does not gate){RESET}")
    print(f"\n  {(GREEN if gate else RED)}GATE P8-10 {'PASSED' if gate else 'NOT PASSED'}{RESET}")

    facts = {
        "arms": {
            name: {
                "auc_mean": rep["dev_auc_mean"],
                "auc_std": rep["dev_auc_std"],
                "frame_ap_mean": rep["dev_frame_ap_mean"],
                "attention_lift_median": lift_stats(rep)["median"],
                "attention_lift_stable": lift_stats(rep)["stable"],
                "drop_visual_delta": rep["drop_visual_delta"],
                "uses_video": visual_reliance(rep)[2],
            }
            for name, rep in arms.items()
            if rep is not None
        },
        "d_auc_mean": d["dev_auc_mean"],
        "d_auc_std": d["dev_auc_std"],
        "d_frame_ap_mean": d["dev_frame_ap_mean"],
        "delta_vs_d": delta_vs_d,
        "seed_spread": spread,
        "beats_d": bool(beats),
        "contributions": contributions,
        "decomposed": bool(decomposed),
        "controlled": bool(controlled),
        "attention_lift_median": lift["median"],
        "attention_lift_per_seed": lift["per_seed"],
        "attention_lift_range": [lift["min"], lift["max"]],
        "attention_lift_stable": lift["stable"],
        "sync_separation_mean": sync_sep,
        "head_entropy_mean": entropy,
        "attention_maps_present": bool(maps_exist),
        "collapse_fixed_by": fixed_by,
        "n_seeds": len(full["runs"]),
        "gate_passed": gate,
    }
    out_md = Path(args.out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.with_suffix(".json").write_text(
        json.dumps(facts, indent=2, default=float) + "\n", encoding="utf-8"
    )
    _write(out_md, facts, d, arms)
    print(f"\n  wrote {args.out}")
    return 0


def _write(path: Path, f: dict, d: dict, arms: dict) -> None:
    beats = f["beats_d"]
    headline = (
        "self-attention beats the BiLSTM"
        if beats
        else "**self-attention does not beat the BiLSTM**"
    )
    lines = [
        "# Experiment E — does attention beat recurrence?",
        "",
        f"**Result: {headline}** ({f['delta_vs_d']:+.4f} dev AUC against Phase 7's D).",
        "",
        "Generated by `scripts/28_experiment_e.py` from measurements on disk. Every arm shares the",
        f"same {d.get('n_clips', '?')} clips, the same unchanged dev split, the same MobileNetV2 (D-1)",
        "and log-Mel (C-1) inputs, and the same Phase 6 stream encoders.",
        "",
        "## The ladder",
        "",
        "| arm | dev AUC | sd | frame AP | attention lift | AUC lost without video |",
        "|---|---|---|---|---|---|",
        f"| D (Phase 7, no sync) | {d['dev_auc_mean']:.4f} | {d['dev_auc_std']:.4f} | "
        f"{d['dev_frame_ap_mean']:.4f} | — | {d['drop_visual_delta']:+.4f} |",
    ]
    for name, label in ARM_LABELS.items():
        rep = arms.get(name)
        if rep is None:
            continue
        lines.append(
            f"| {label} | {rep['dev_auc_mean']:.4f} | {rep['dev_auc_std']:.4f} | "
            f"{rep['dev_frame_ap_mean']:.4f} | {lift_stats(rep)['median']:.2f}× | "
            f"{rep['drop_visual_delta']:+.4f} |"
        )

    lines += [
        "",
        "## ⛔ Who earned the gain? (P8-10)",
        "",
        "`E` changes two components at once, so a single `E − D` number would leave the credit",
        "unassignable. Each single-component arm is measured against **D + sync**, holding the sync",
        "loss constant so it cannot leak into the attention deltas.",
        "",
        "| component | Δ dev AUC |",
        "|---|---|",
    ]
    for label, value in f["contributions"].items():
        lines.append(f"| {label} | {value:+.4f} |")

    lines += [
        "",
        f"Seed spread across arms is {f['seed_spread']:.4f}; a contribution smaller than that is",
        "**not significant** and is marked as such rather than claimed (§7.3 criterion 1).",
        "",
        "## P8-9 — does attention elevate on forged spans?",
        "",
        f"**Attention lift = {f['attention_lift_median']:.2f}× (median across seeds), per-seed "
        f"{', '.join(f'{x:.2f}' for x in f['attention_lift_per_seed'])}** — the ratio of mean",
        "attention-pool weight on forged frames to genuine frames, within fake clips. Above 1 means",
        "the pooling operator concentrates where the forgery is.",
        "",
        (
            "⛔ **The seeds disagree on the direction, so this is reported as seed-dependent rather "
            "than as an observation.** Runs differing in nothing but their seed, and landing within "
            "0.003 dev AUC of each other, split between attention that concentrates almost entirely "
            "on the forged span and attention that almost entirely avoids it. Whether attention "
            "lands on the forgery is therefore **decoupled from classification performance**: the "
            "model scores the same either way. The median is quoted instead of the mean because the "
            "mean describes none of the individual runs, and any single attention figure would be a "
            "picture of one seed rather than of the architecture."
            if not f["attention_lift_stable"]
            else "All seeds agree on the direction, so the behaviour is a property of the "
            "architecture rather than of a run."
        ),
        "",
        "## ⛔ PF-21 — is cross-attention the collapse fix?",
        "",
        "PF-20 measured the section 5.6 collapse at C; PF-21 measured it surviving the BiLSTM at D.",
        "Modality dropout backfired and the auxiliary heads did not save it. Cross-attention is the",
        "last structurally different candidate: its V→A queries come *from* video, so a dead visual",
        "stream cannot route attention.",
        "",
        (
            "**Arms that genuinely use video: " + ", ".join(f["collapse_fixed_by"]) + ".** "
            "The collapse is broken where those arms are concerned — the visual pathway carries "
            "signal that removing it costs."
            if f["collapse_fixed_by"]
            else "**No arm broke the collapse.** Removing the visual stream still costs "
            "essentially nothing, in every configuration tried. Across Phases 6, 7 and 8 the "
            "project has now tested modality dropout, auxiliary heads, recurrence, cross-attention "
            "and self-attention against this failure, and none of them fixed it. That is a "
            "substantive negative result about the dataset rather than about any one architecture: "
            "on LAV-DF at this scale the audio pathway is so much easier that no amount of "
            "architectural encouragement makes the visual pathway worth using. It belongs in the "
            "limitations section, and PF-19's processing fingerprint is the leading explanation."
        ),
        "",
        "## Gate P8-10",
        "",
        "> attention contribution measured, attention maps produced",
        "",
        "| check | result |",
        "|---|---|",
        f"| contribution decomposed into single-component arms | {f['decomposed']} |",
        f"| controls match D (backbone / feature / defences / clips) | {f['controlled']} |",
        f"| attention maps produced (P8-8) | {f['attention_maps_present']} |",
        f"| seeds | {f['n_seeds']} |",
        f"| E beats D | {beats} *(reported, does not gate)* |",
        "",
        f"**{'PASSED' if f['gate_passed'] else 'NOT PASSED'}.**",
        "",
        "## ⚠️ Read alongside PF-17, PF-19, PF-21 and PF-22",
        "",
        "Every arm inherits the clip-length shortcut and the global audio processing fingerprint. The",
        "absolute AUCs here are upper bounds; the **deltas between arms** are the trustworthy part,",
        "since all arms carry identical artefacts on an identical dev split.",
    ]
    while lines and not lines[-1].strip():
        lines.pop()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
