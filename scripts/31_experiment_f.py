"""⛔ P9-6 / P9-7: Experiment F, and the Phase 9 gate.

    python scripts/31_experiment_f.py

Three arms, because two would be uninterpretable:

    F0   no augmentation                      the floor -- the section 5.1 model unchanged
    F1   splicing + classical (P9-1, P9-2)    does augmentation help at all?
    F2   F1 + GAN samples (P9-3)              ⛔ F2 - F1 is the ONLY valid GAN claim

**P9-6 is a pre-commitment, not a summary.** Section I is explicit that most published GAN
ablations compare "GAN augmentation" against "no augmentation" and therefore credit the GAN
with everything ordinary augmentation did. `F2 - F1` holds ordinary augmentation constant, and
this script reports it whichever way it lands. **F2 ~= F1 -> "the GAN added nothing beyond
classical augmentation" is a legitimate result** and is written into the report as the
headline when the measurement says so.

The section 7.3 criteria are applied rather than quoted:

1. **magnitude** -- the difference must exceed the seed spread of the arms being compared;
2. **consistency** -- seeds are paired (arm F1 seed *k* against F2 seed *k*, same
   `1337 + k`, same data order), so the sign has to agree on all of them, not just on the mean;
3. **right metric** -- clip AUC *and* frame AP are both reported, because augmentation aimed
   at boundaries can improve localization while leaving classification flat;
4. **cost** -- wall-clock and training-set size per arm, so a gain that is really just more
   compute is visible as one.

⚠️ **PF-21 is the question behind the question, again.** Splicing can leave audio untouched
(`SpliceConfig.modality_weights`), so F1 and F2 train on fakes that are *undetectable from
audio by construction* -- the first thing in this project able to force the visual pathway to
matter, since nothing in LAV-DF's own training mix does. Whether that moved the section 5.6
collapse is measured here against the same 0.01 threshold Phases 6-8 used, and reported
whether or not it flatters the augmentation.

Inputs (none entered by hand):
  reports/phase9_{F0,F1,F2}.json  -- written by scripts/30_train_augmented.py
  reports/phase8_full.json        -- E full, the architecture F0 should reproduce
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
CONTROLLED = ("backbone", "feature", "defences", "n_clips", "n_params")
ARM_LABELS = {
    "F0": "F0 no augmentation",
    "F1": "F1 splice + classical",
    "F2": "F2 + GAN samples",
}
CONTENTS = {
    "F0": "no augmentation",
    "F1": "splice (P9-1) + classical (P9-2)",
    "F2": "F1 + GAN samples (P9-3)",
}


def load(path: str | Path) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def per_seed(report: dict, key: str = "auc") -> dict[int, float]:
    """Metric keyed by seed, so arms can be compared *paired* rather than only in the mean."""
    return {int(r["seed"]): float(r["dev"].get(key, np.nan)) for r in report["runs"]}


def compare(a: dict, b: dict, key: str = "auc") -> dict:
    """⛔ `b - a` under the section 7.3 criteria, paired by seed.

    The mean difference is not enough on its own: three seeds whose differences are
    +0.01, +0.01, -0.02 average to zero, and three that are +0.003 each average to something
    smaller than either arm's spread while agreeing perfectly. Both are reported, and
    `significant` requires magnitude *and* unanimity.
    """
    pa, pb = per_seed(a, key), per_seed(b, key)
    shared = sorted(set(pa) & set(pb))
    diffs = np.array([pb[s] - pa[s] for s in shared], dtype=float)
    finite = diffs[np.isfinite(diffs)]
    mean_a = float(np.nanmean(list(pa.values())))
    mean_b = float(np.nanmean(list(pb.values())))
    delta = mean_b - mean_a
    spread = max(float(np.nanstd(list(pa.values()))), float(np.nanstd(list(pb.values()))))
    consistent = bool(finite.size and ((finite > 0).all() or (finite < 0).all()))
    return {
        "delta": delta,
        "seed_spread": spread,
        "beyond_spread": bool(abs(delta) > spread),
        "per_seed": {int(s): float(pb[s] - pa[s]) for s in shared},
        "consistent": consistent,
        "n_paired": len(shared),
        # Criteria 1 and 2 together. Either alone has produced a published non-result.
        "significant": bool(abs(delta) > spread and consistent),
    }


def visual_reliance(report: dict) -> tuple[float, float, bool]:
    drops = np.array(
        [r["dev"]["auc"] - r["dev_drop_visual"]["auc"] for r in report["runs"]], dtype=float
    )
    mean, sd = float(drops.mean()), float(drops.std())
    return mean, sd, bool(mean > 2 * sd and mean >= COLLAPSE_TOLERANCE)


def gan_health(report: dict) -> dict:
    """P9-4's numbers as they were measured at the moment the samples were admitted."""
    rows = [r.get("gan_diversity") for r in report["runs"] if r.get("gan_diversity")]
    if not rows:
        return {"measured": False, "any_collapsed": True}
    return {
        "measured": True,
        "n_seeds": len(rows),
        "pairwise_ratio_mean": float(np.mean([r["pairwise_ratio"] for r in rows])),
        "std_ratio_mean": float(np.mean([r["std_ratio"] for r in rows])),
        "nn_ratio_mean": float(np.mean([r["nn_ratio"] for r in rows])),
        "any_collapsed": bool(any(r["collapsed"] for r in rows)),
        "per_seed": rows,
    }


def cost(report: dict) -> dict:
    """Criterion 4 -- a gain bought with more compute has to show its price."""
    return {
        "wall_clock_s_mean": float(np.mean([r["wall_clock_s"] for r in report["runs"]])),
        "epochs_mean": float(np.mean([r["epochs_run"] for r in report["runs"]])),
        "n_clips_train": int(report["runs"][0]["n_clips_train"]),
    }


def verdict_line(name: str, cmp: dict) -> str:
    if cmp["significant"]:
        colour = GREEN if cmp["delta"] > 0 else RED
        tail = "significant"
    elif cmp["beyond_spread"]:
        colour = YELLOW
        tail = "beyond spread but seeds disagree -- NOT significant"
    else:
        colour = YELLOW
        tail = "within seed spread -- NOT significant"
    return (
        f"    {name:<26}{colour}{cmp['delta']:+.4f}{RESET}  "
        f"{DIM}(spread {cmp['seed_spread']:.4f}, {cmp['n_paired']} seeds){RESET}  {tail}"
    )


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Experiment F and the Phase 9 gate")
    ap.add_argument("--out", default="reports/experiment_f.md")
    ap.add_argument("--report-dir", default="reports")
    ap.add_argument("--e-report", default="reports/phase8_full.json", help="E full, for control")
    ap.add_argument("--min-seeds", type=int, default=3)
    args = ap.parse_args()

    rdir = Path(args.report_dir)
    arms = {name: load(rdir / f"phase9_{name}.json") for name in ARM_LABELS}
    missing = [n for n, r in arms.items() if r is None]
    if missing:
        print(
            f"{RED}missing arm(s) {', '.join(missing)} -- run "
            f"scripts/30_train_augmented.py --arms {' '.join(missing)}{RESET}"
        )
        return 1
    e_full = load(args.e_report)

    print(
        f"{DIM}{'=' * 78}{RESET}\n  🔬 Experiment F -- does augmentation help, and did the GAN "
        f"add anything?\n{DIM}{'=' * 78}{RESET}"
    )

    # ---- the ladder --------------------------------------------------------------------
    print(
        f"\n  {'arm':<24}{'dev AUC':>10}{'sd':>9}{'frame AP':>11}{'-visual':>10}"
        f"{'train':>8}{'wall':>9}"
    )
    if e_full is not None:
        print(
            f"  {DIM}{'E full (Phase 8)':<24}{e_full['dev_auc_mean']:>10.4f}"
            f"{e_full['dev_auc_std']:>9.4f}{e_full['dev_frame_ap_mean']:>11.4f}"
            f"{e_full['drop_visual_delta']:>+10.4f}"
            f"{e_full['runs'][0]['n_clips_train']:>8}{'--':>9}{RESET}"
        )
    for name, rep in arms.items():
        c = cost(rep)
        print(
            f"  {ARM_LABELS[name]:<24}{rep['dev_auc_mean']:>10.4f}{rep['dev_auc_std']:>9.4f}"
            f"{rep['dev_frame_ap_mean']:>11.4f}{rep['drop_visual_delta']:>+10.4f}"
            f"{c['n_clips_train']:>8}{c['wall_clock_s_mean'] / 60:>8.0f}m"
        )

    # ---- the two questions, in order -----------------------------------------------------
    aug = compare(arms["F0"], arms["F1"])
    gan = compare(arms["F1"], arms["F2"])
    aug_frame = compare(arms["F0"], arms["F1"], "frame_ap")
    gan_frame = compare(arms["F1"], arms["F2"], "frame_ap")

    print("\n  the two questions (paired by seed, criteria 1 and 2 of section 7.3)")
    print(verdict_line("F1 - F0  augmentation", aug))
    print(verdict_line("  same, frame AP", aug_frame))
    print(verdict_line("⛔ F2 - F1  the GAN", gan))
    print(verdict_line("  same, frame AP", gan_frame))
    print(
        f"    {DIM}per-seed F2-F1: "
        f"{', '.join(f'{v:+.4f}' for v in gan['per_seed'].values())}{RESET}"
    )

    # ---- ⛔ P9-4, as admitted ------------------------------------------------------------
    health = gan_health(arms["F2"])
    print("\n  ⛔ P9-4 mode-collapse gate, at the moment samples were admitted")
    if not health["measured"]:
        print(f"    {RED}no diversity recorded on the F2 runs{RESET}")
    else:
        state = RED + "COLLAPSED" + RESET if health["any_collapsed"] else GREEN + "diverse" + RESET
        print(
            f"    pairwise ratio {health['pairwise_ratio_mean']:.2f}   "
            f"std ratio {health['std_ratio_mean']:.2f}   "
            f"nn ratio {health['nn_ratio_mean']:.2f}   {state}"
            f"  {DIM}(ratios against the real reference batch){RESET}"
        )

    # ---- ⛔ PF-21 again, now with audio-untouched fakes in training -----------------------
    print("\n  ⛔ PF-21 -- do audio-untouched spliced fakes force the visual pathway?")
    reliance = {}
    for name, rep in arms.items():
        drop, sd, uses = visual_reliance(rep)
        reliance[name] = {"delta": drop, "sd": sd, "uses_video": uses}
        print(
            f"    {ARM_LABELS[name]:<24}reliance on video {drop:+.4f} +/- {sd:.4f}  "
            f"{(GREEN + 'uses video' + RESET) if uses else RED + 'COLLAPSED' + RESET}"
        )
    collapse_broken = [n for n, r in reliance.items() if r["uses_video"]]

    # ---- F0 as a control on the whole ladder ---------------------------------------------
    reproduces_e = None
    if e_full is not None:
        gap = abs(arms["F0"]["dev_auc_mean"] - e_full["dev_auc_mean"])
        tol = max(arms["F0"]["dev_auc_std"], e_full["dev_auc_std"])
        reproduces_e = bool(gap <= 2 * tol)
        print(
            f"\n  control: F0 reproduces Phase 8's E full  "
            f"{(GREEN if reproduces_e else RED)}{reproduces_e}{RESET}  "
            f"{DIM}(|delta| {gap:.4f} vs 2x sd {2 * tol:.4f}){RESET}"
        )

    # ---- ⛔ the gate ----------------------------------------------------------------------
    diffs: dict[str, dict] = {}
    for key in CONTROLLED:
        values = {n: rep.get(key) for n, rep in arms.items()}
        if len({repr(v) for v in values.values()}) > 1:
            diffs[key] = values
    controlled = not diffs
    for key, values in diffs.items():
        print(f"\n    {RED}uncontrolled: {key}: {values}{RESET}")

    seeds_ok = all(len(rep["runs"]) >= args.min_seeds for rep in arms.values())
    three_arms = True  # enforced above -- a missing arm exits non-zero before this point
    gan_measured = bool(health["measured"] and not health["any_collapsed"])
    gate = bool(three_arms and seeds_ok and controlled and gan_measured)

    print(f"\n  {DIM}Gate P9-7: three arms run, conclusion recorded honestly{RESET}")
    print(f"    three arms present          : {three_arms}")
    print(f"    >= {args.min_seeds} seeds each            : {seeds_ok}")
    print(f"    controls match across arms  : {controlled}")
    print(f"    P9-4 measured, not collapsed: {gan_measured}")
    print(
        f"    F2 vs F1 reported           : True  "
        f"{DIM}({'significant' if gan['significant'] else 'not significant'}, "
        f"reported either way){RESET}"
    )
    print(f"\n  {(GREEN if gate else RED)}GATE P9-7 {'PASSED' if gate else 'NOT PASSED'}{RESET}")

    facts = {
        "arms": {
            name: {
                "auc_mean": rep["dev_auc_mean"],
                "auc_std": rep["dev_auc_std"],
                "frame_ap_mean": rep["dev_frame_ap_mean"],
                "frame_ap_std": rep["dev_frame_ap_std"],
                "drop_visual_delta": rep["drop_visual_delta"],
                "drop_audio_delta": rep["drop_audio_delta"],
                "uses_video": reliance[name]["uses_video"],
                "n_seeds": len(rep["runs"]),
                **cost(rep),
            }
            for name, rep in arms.items()
        },
        "augmentation_auc": aug,
        "augmentation_frame_ap": aug_frame,
        "gan_auc": gan,
        "gan_frame_ap": gan_frame,
        "gan_health": health,
        "visual_reliance": reliance,
        "collapse_broken_by": collapse_broken,
        "collapse_tolerance": COLLAPSE_TOLERANCE,
        "f0_reproduces_e_full": reproduces_e,
        "e_full_auc_mean": None if e_full is None else e_full["dev_auc_mean"],
        "controlled": controlled,
        "uncontrolled": diffs,
        "seeds_ok": seeds_ok,
        "git_sha": arms["F0"].get("git_sha"),
        "gate_passed": gate,
    }
    out_md = Path(args.out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.with_suffix(".json").write_text(
        json.dumps(facts, indent=2, default=float) + "\n", encoding="utf-8"
    )
    _write(out_md, facts)
    print(f"\n  wrote {args.out}")
    return 0


def _describe(cmp: dict) -> str:
    """One clause that is true whichever way the measurement went."""
    if cmp["significant"]:
        return (
            f"**{cmp['delta']:+.4f}**, beyond the {cmp['seed_spread']:.4f} seed spread and "
            f"consistent on all {cmp['n_paired']} seeds"
        )
    if cmp["beyond_spread"]:
        return (
            f"{cmp['delta']:+.4f}, larger than the {cmp['seed_spread']:.4f} seed spread but "
            "**the seeds disagree on the sign**, so not significant"
        )
    return (
        f"{cmp['delta']:+.4f}, **inside** the {cmp['seed_spread']:.4f} seed spread — "
        "not significant"
    )


def _write(path: Path, f: dict) -> None:
    gan, aug = f["gan_auc"], f["augmentation_auc"]
    if gan["significant"] and gan["delta"] > 0:
        headline = "the GAN added something beyond classical augmentation"
    elif gan["significant"]:
        headline = "**the GAN made things worse**"
    else:
        headline = "**the GAN added nothing beyond classical augmentation**"

    lines = [
        "# Experiment F — does augmentation help, and did the GAN add anything?",
        "",
        f"**Result: {headline}** (`F2 − F1` = {gan['delta']:+.4f} dev AUC, seed spread "
        f"{gan['seed_spread']:.4f}).",
        "",
        "Generated by `scripts/31_experiment_f.py` from measurements on disk. Every arm shares the",
        "same section 5.1 model (E full), the same MobileNetV2 (D-1) and log-Mel (C-1) inputs, the",
        "same unchanged dev split and the same three seeds (1337, 1338, 1339). The arms differ in",
        "exactly one object — the training dataset.",
        "",
        "## The three arms",
        "",
        "| arm | contents | dev AUC | sd | frame AP | AUC lost without video | train clips |"
        " wall clock |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, a in f["arms"].items():
        lines.append(
            f"| {name} | {CONTENTS[name]} | {a['auc_mean']:.4f} | {a['auc_std']:.4f} | "
            f"{a['frame_ap_mean']:.4f} | {a['drop_visual_delta']:+.4f} | "
            f"{a['n_clips_train']} | {a['wall_clock_s_mean'] / 60:.0f} min |"
        )
    if f["e_full_auc_mean"] is not None:
        lines += [
            "",
            "**Control:** F0 is the Phase 8 `E full` architecture with the augmentation removed, and",
            f"it reproduces that arm's {f['e_full_auc_mean']:.4f} — `{f['f0_reproduces_e_full']}` at a",
            "2×sd tolerance. Without that check a difference in the training harness would be",
            "indistinguishable from a difference in augmentation.",
        ]

    lines += [
        "",
        "## ⛔ P9-6 — the two questions, in order",
        "",
        "Section I warns that comparing *GAN augmentation* against *no augmentation* credits the GAN",
        "with everything ordinary augmentation did. So the ladder asks two separate questions and only",
        "the second one is about the GAN.",
        "",
        "| question | comparison | Δ dev AUC | Δ frame AP |",
        "|---|---|---|---|",
        f"| does augmentation help at all? | F1 − F0 | {aug['delta']:+.4f} | "
        f"{f['augmentation_frame_ap']['delta']:+.4f} |",
        f"| **did the GAN add anything?** | **F2 − F1** | **{gan['delta']:+.4f}** | "
        f"{f['gan_frame_ap']['delta']:+.4f} |",
        "",
        f"**Augmentation (F1 − F0):** {_describe(aug)}. Per seed: "
        + ", ".join(f"{v:+.4f}" for v in aug["per_seed"].values())
        + ".",
        "",
        f"**⛔ The GAN (F2 − F1):** {_describe(gan)}. Per seed: "
        + ", ".join(f"{v:+.4f}" for v in gan["per_seed"].values())
        + ".",
        "",
        (
            "F2 ≈ F1 is a **legitimate published result**, and P9-6 pre-committed to reporting it."
            " The conditional feature-space GAN is the most elaborate component in Phase 9;"
            " measuring that it buys nothing over splicing and SpecAugment is worth more than a"
            " number that quietly folds the two together. Two structural reasons were known in"
            " advance and are stated in `SequenceGenerator`'s docstring: the generator learns a"
            " **snapshot** of F1's fused space and F2 then trains from scratch, so the classifier"
            " moves away from the distribution the samples describe; and generated samples reach"
            " only the temporal encoder and the two output heads, since the auxiliary and sync"
            " terms are defined on the pre-fusion streams that a synthesised sequence does not have."
            if not (gan["significant"] and gan["delta"] > 0)
            else "The GAN clears both criteria against F1, which is the only comparison that can"
            " support a claim about it."
        ),
        "",
        "### Section 7.3 criteria, applied",
        "",
        "| criterion | F1 − F0 | F2 − F1 |",
        "|---|---|---|",
        f"| 1. magnitude > seed spread | {aug['beyond_spread']} | {gan['beyond_spread']} |",
        f"| 2. consistent on every seed | {aug['consistent']} | {gan['consistent']} |",
        "| 3. right metric | clip AUC **and** frame AP both reported | ← |",
        f"| 4. cost stated | {f['arms']['F1']['n_clips_train']} vs "
        f"{f['arms']['F0']['n_clips_train']} train clips, "
        f"{f['arms']['F1']['wall_clock_s_mean'] / 60:.0f} vs "
        f"{f['arms']['F0']['wall_clock_s_mean'] / 60:.0f} min | "
        f"+ GAN fit, {f['arms']['F2']['wall_clock_s_mean'] / 60:.0f} min |",
        "",
        "## ⛔ P9-4 — the mode-collapse gate",
        "",
    ]
    h = f["gan_health"]
    if not h["measured"]:
        lines.append("No diversity was recorded on the F2 runs.")
    else:
        lines += [
            "Measured on the samples at the moment they were admitted to training, as ratios against",
            "a real reference batch — never as absolute floors, because this project's own fused",
            "embeddings sit in a narrow cone (200 real clips: mean pairwise cosine distance 0.0264)",
            "and an absolute threshold would ask the generator to be more varied than the data it",
            "imitates.",
            "",
            "| measurement | value | floor | meaning if low |",
            "|---|---|---|---|",
            f"| pairwise ratio | {h['pairwise_ratio_mean']:.2f} | 0.50 | one sequence regardless"
            " of noise |",
            f"| std ratio | {h['std_ratio_mean']:.2f} | 0.10 | samples occupy a thin subspace |",
            f"| nn ratio | {h['nn_ratio_mean']:.2f} | 0.50 | memorising the training embeddings |",
            "",
            (
                "No F2 seed was refused."
                if not h["any_collapsed"]
                else "**A seed collapsed and its samples were refused.**"
            )
            + " `generate_samples` raises `ModeCollapseError` rather than warning: a collapsed"
            " generator injects a perfectly learnable constant marker, and F2 would then beat F1"
            " for a reason that has nothing to do with augmentation quality.",
        ]

    lines += [
        "",
        "## ⛔ PF-21 — do audio-untouched spliced fakes force the visual pathway?",
        "",
        "Splicing sets the modality of each generated fake, so two thirds of them have **untouched",
        "audio** and cannot be detected from the audio stream at all — the first training signal in",
        "this project able to force the visual pathway to matter. Nothing in LAV-DF's own training",
        "mix provides it. The measurement is the same one Phases 6–8 used: AUC lost when the visual",
        f"stream is zeroed, against a {f['collapse_tolerance']} threshold.",
        "",
        "| arm | AUC lost without video | verdict |",
        "|---|---|---|",
    ]
    for name, r in f["visual_reliance"].items():
        lines.append(
            f"| {name} | {r['delta']:+.4f} ± {r['sd']:.4f} | "
            f"{'uses video' if r['uses_video'] else '**collapsed**'} |"
        )
    lines += [
        "",
        (
            "**"
            + ", ".join(f["collapse_broken_by"])
            + " genuinely use the visual stream.** Removing it costs measurably more than the"
            " threshold and more than twice the seed spread, which is the first time in this project"
            " any arm has done so. Augmentation, not architecture, is what moved it — consistent"
            " with PF-21's conclusion that the collapse was a property of the training distribution."
            if f["collapse_broken_by"]
            else "**No arm broke the collapse, including the ones trained on fakes whose audio is"
            " untouched by construction.** This is the strongest form of the PF-21 result the"
            " project can produce: the model was given forgeries it *cannot* detect from audio, and"
            " still did not learn to use video. Modality dropout, auxiliary heads, recurrence,"
            " cross-attention, self-attention and now targeted augmentation have all failed against"
            " it. It belongs in the limitations section, and PF-19's global audio processing"
            " fingerprint remains the leading explanation — the audio pathway can score the spliced"
            " fakes too, because the fingerprint is a property of the file rather than of the"
            " forged span."
        ),
        "",
        "## Gate P9-7",
        "",
        "> three arms run, conclusion recorded honestly",
        "",
        "| check | result |",
        "|---|---|",
        "| three arms present (F0 / F1 / F2) | True |",
        f"| ≥ 3 seeds per arm | {f['seeds_ok']} |",
        "| controls match across arms (backbone / feature / defences / clips / params) |"
        f" {f['controlled']} |",
        f"| P9-4 measured and not collapsed |"
        f" {bool(h.get('measured') and not h.get('any_collapsed', True))} |",
        "| F2 vs F1 reported whichever way it landed | True |",
        "",
        f"**{'PASSED' if f['gate_passed'] else 'NOT PASSED'}.**",
        "",
        "## ⚠️ Read alongside PF-17, PF-19 and PF-22",
        "",
        "Every arm inherits the clip-length shortcut and the global audio processing fingerprint, and",
        "the spliced examples inherit them from their recipients. The absolute AUCs here are upper",
        "bounds; the **deltas between arms** are the trustworthy part, since all three arms carry",
        "identical artefacts on an identical dev split.",
    ]
    while lines and not lines[-1].strip():
        lines.pop()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
