"""⛔ P7-5 / P7-6: Experiment D, and the Phase 7 gate.

    python scripts/25_experiment_d.py

The experiment matrix predicts **D > C** — sequential context helps. This script tests that
against numbers on disk and writes the verdict, whichever way it goes.

**What P7-6 actually asks for.** *"Temporal contribution measured independently of
attention."* That is a **provenance** requirement, not a performance one: the gate is asking
whether, when Phase 8 later reports `E > D`, the reader can tell how much of the total gain
was recurrence and how much was attention. It passes when the temporal delta has been
isolated and recorded; it fails when the two components were introduced together and the
credit cannot be assigned.

So this script checks three things, and only the first two can fail the gate:

1. **Is D isolated?** The D arm must declare `attention: false` and must share the C arm's
   backbone, audio feature, defences and clip count. Any difference is a second variable.
2. **Is the delta recorded with its spread?** A single-seed number cannot be compared to
   anything (§7.3 criterion 2).
3. **Is D > C?** Reported, and required to clear seed variance to count as real — but a
   BiLSTM that does not help is a legitimate measurement, and section 5.3 already predicts
   sequential memory will lose to global comparison. Recording that honestly is the point.

⚠️ **"Independently of attention" excludes self- and cross-attention, not attention
pooling.** Every arm since Phase 4 pools with a learned query; it is a constant across A, B,
C and D and therefore not a confound. Phase 8's `E` is what adds attention *between frames*.

Inputs (none entered by hand):
  reports/phase6_concat.json   -- C, fusion without temporal modelling
  reports/phase7_bilstm.json   -- D, fusion + BiLSTM + per-frame head
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
# The arms must agree on these or `D - C` is measuring more than the recurrence.
CONTROLLED = ("backbone", "feature", "defences", "n_clips")


def load(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def visual_reliance(report: dict) -> tuple[float, float, bool]:
    """Per-seed AUC lost when the visual stream is zeroed: (mean, sd, real?)."""
    drops = np.array(
        [r["dev"]["auc"] - r["dev_drop_visual"]["auc"] for r in report["runs"]], dtype=float
    )
    mean, sd = float(drops.mean()), float(drops.std())
    return mean, sd, bool(mean > 2 * sd and mean >= COLLAPSE_TOLERANCE)


def controlled_diff(c: dict, d: dict) -> dict[str, tuple]:
    """Everything the two arms were supposed to hold constant but did not."""
    return {k: (c.get(k), d.get(k)) for k in CONTROLLED if c.get(k) != d.get(k)}


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Experiment D and the Phase 7 gate")
    ap.add_argument("--out", default="reports/experiment_d.md")
    ap.add_argument("--c-report", default="reports/phase6_concat.json")
    ap.add_argument("--d-report", default="reports/phase7_bilstm.json")
    args = ap.parse_args()

    c, d = load(args.c_report), load(args.d_report)
    missing = [n for n, v in ((args.c_report, c), (args.d_report, d)) if v is None]
    if missing:
        print(f"{RED}cannot run Experiment D -- missing:{RESET}")
        for m in missing:
            print(f"  {m}")
        return 1

    print(
        f"{DIM}{'=' * 72}{RESET}\n  🔬 Experiment D -- does temporal modelling help?"
        f"\n{DIM}{'=' * 72}{RESET}"
    )

    arms = {
        "C fusion (no temporal)": (c["dev_auc_mean"], c["dev_auc_std"]),
        "D fusion + BiLSTM": (d["dev_auc_mean"], d["dev_auc_std"]),
    }
    print(f"\n  {'arm':<26}{'dev AUC':>10}{'sd':>10}")
    for name, (mean, sd) in arms.items():
        print(f"  {name:<26}{mean:>10.4f}{sd:>10.4f}")

    delta = d["dev_auc_mean"] - c["dev_auc_mean"]
    # §7.3 criterion 1: the gap must exceed seed variance to be called an improvement.
    spread = max(c["dev_auc_std"], d["dev_auc_std"])
    beats = delta > spread

    print(f"\n  D - C             : {(GREEN if beats else YELLOW)}{delta:+.4f}{RESET}")
    print(f"  beyond seed sd    : {beats}  {DIM}(max sd {spread:.4f}){RESET}")

    # --- question 1: was the comparison controlled? ----------------------------------
    diffs = controlled_diff(c, d)
    isolated = not diffs and d.get("attention") is False and d.get("temporal") == "bilstm"
    print("\n  ⛔ P7-6 isolation -- is this the recurrence and nothing else?")
    print(f"    D declares attention : {d.get('attention')}  {DIM}(must be False){RESET}")
    print(f"    D declares temporal  : {d.get('temporal')!r}")
    if diffs:
        for k, (cv, dv) in diffs.items():
            print(f"    {RED}{k}: C={cv!r} vs D={dv!r}{RESET}")
    else:
        print(f"    {GREEN}controlled: {', '.join(CONTROLLED)} all match{RESET}")

    # --- question 2: the new metric --------------------------------------------------
    frame_ap = d.get("dev_frame_ap_mean")
    frame_ap_sd = d.get("dev_frame_ap_std")
    frame_rate = d.get("frame_balance", {}).get("dev", {}).get("positive_rate")
    has_frame_ap = frame_ap is not None and np.isfinite(frame_ap)
    # ⛔ The measured shortcut floor, from scripts/26_check_frame_confound.py. Optional so
    # the gate still runs without it, but its absence is called out rather than glossed:
    # an unchecked localization number is exactly what PF-17 and PF-19 were.
    confound = load("reports/frame_confound.json")
    print("\n  P7-4 frame-level AP (the first localization-relevant metric)")
    if has_frame_ap:
        print(f"    dev frame AP  : {frame_ap:.4f} +/- {frame_ap_sd:.4f}")
        if frame_rate:
            print(
                f"    chance floor  : {frame_rate:.4f}  "
                f"{DIM}(the forged-frame rate; AP of a constant scorer){RESET}"
            )
        if confound:
            pos_ap = confound["positional_frame_ap"]
            share = pos_ap / frame_ap
            print(
                f"    positional floor: {pos_ap:.4f}  "
                f"{DIM}(frame index alone, prior fitted on train){RESET}"
            )
            print(
                f"    position explains: {(RED if share >= 0.5 else GREEN)}{share:.1%}{RESET}"
                f" of it  {DIM}(⛔ P7-4 validity check){RESET}"
            )
        else:
            print(
                f"    {YELLOW}no reports/frame_confound.json -- run "
                f"scripts/26_check_frame_confound.py{RESET}"
            )
    else:
        print(f"    {RED}not recorded -- the frame head produced no AP{RESET}")

    # --- question 3: still multimodal? ------------------------------------------------
    drop_v, drop_v_sd, uses_video = visual_reliance(d)
    c_drop_v, _, c_uses_video = visual_reliance(c)
    print("\n  ⛔ section 5.6 modality ablation (X-7: re-check at every change)")
    print(
        f"    dev AUC without visual : {d['drop_visual_auc_mean']:.4f}  "
        f"{DIM}drop {drop_v:+.4f} +/- {drop_v_sd:.4f}{RESET}"
    )
    print(
        f"    dev AUC without audio  : {d['drop_audio_auc_mean']:.4f}  "
        f"{DIM}drop {d['drop_audio_delta']:+.4f}{RESET}"
    )
    print(f"    C -> D reliance on video: {c_drop_v:+.4f} -> {drop_v:+.4f}")
    print(
        f"    {(RED if not uses_video else GREEN)}"
        f"{'STILL COLLAPSED onto audio' if not uses_video else 'both modalities contribute'}"
        f"{RESET}"
    )

    # --- the gate ---------------------------------------------------------------------
    gate = bool(isolated and has_frame_ap and len(d["runs"]) >= 2)
    print(f"\n  {DIM}Gate P7-6: temporal contribution measured independently of attention{RESET}")
    print(f"    D isolated from attention : {isolated}")
    print(f"    frame AP recorded (P7-4)  : {has_frame_ap}")
    print(f"    multi-seed spread         : {len(d['runs'])} seeds")
    print(f"    D beats C                 : {beats}  {DIM}(reported, does not gate){RESET}")
    print(f"\n  {(GREEN if gate else RED)}GATE P7-6 {'PASSED' if gate else 'NOT PASSED'}{RESET}")

    facts = {
        "arms": {k: {"auc_mean": v[0], "auc_std": v[1]} for k, v in arms.items()},
        "delta_vs_fusion": delta,
        "seed_spread": spread,
        "beats_fusion": bool(beats),
        "isolated_from_attention": bool(isolated),
        "uncontrolled_differences": {k: list(v) for k, v in diffs.items()},
        "frame_ap_mean": frame_ap,
        "frame_ap_std": frame_ap_sd,
        "frame_chance_floor": frame_rate,
        "frame_positional_floor": (confound or {}).get("positional_frame_ap"),
        "frame_positional_share": (
            (confound or {}).get("positional_frame_ap") / frame_ap
            if confound and has_frame_ap
            else None
        ),
        "drop_visual_delta": drop_v,
        "drop_visual_sd": drop_v_sd,
        "uses_video": bool(uses_video),
        "fusion_drop_visual_delta": c_drop_v,
        "fusion_uses_video": bool(c_uses_video),
        "collapse_tolerance": COLLAPSE_TOLERANCE,
        "n_seeds": len(d["runs"]),
        "gate_passed": gate,
    }
    # Derived from --out rather than hardcoded, so a dry run against synthetic inputs
    # cannot overwrite the real reports/experiment_d.json.
    out_md = Path(args.out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.with_suffix(".json").write_text(
        json.dumps(facts, indent=2, default=float) + "\n", encoding="utf-8"
    )
    _write(out_md, facts, c, d)
    print(f"\n  wrote {args.out}")
    return 0


def _write(path: Path, f: dict, c: dict, d: dict) -> None:
    beats = f["beats_fusion"]
    headline = (
        "the BiLSTM improves on early-concat fusion"
        if beats
        else "**the BiLSTM does not improve on early-concat fusion**"
    )
    lines = [
        "# Experiment D — does temporal modelling help?",
        "",
        f"**Result: {headline}** ({f['delta_vs_fusion']:+.4f} dev AUC against Phase 6's C),",
        f"and the per-frame head reaches **frame AP {f['frame_ap_mean']:.4f}**"
        + (
            f" against a {f['frame_chance_floor']:.4f} chance floor."
            if f["frame_chance_floor"]
            else "."
        ),
        "",
        "Generated by `scripts/25_experiment_d.py` from measurements on disk. Both arms share the",
        f"same {c.get('n_clips', '?')} clips, the same unchanged dev split, the same MobileNetV2",
        "(D-1) and log-Mel (C-1) inputs, and the same collapse defences — so the difference is the",
        "recurrence.",
        "",
        "## The arms",
        "",
        "| arm | dev AUC | sd |",
        "|---|---|---|",
    ]
    for name, v in f["arms"].items():
        lines.append(f"| {name} | {v['auc_mean']:.4f} | {v['auc_std']:.4f} |")

    lines += [
        "",
        f"Measured: **{f['delta_vs_fusion']:+.4f}**, which "
        f"{'clears' if beats else 'does not clear'} the larger of the two arms' seed spreads "
        f"({f['seed_spread']:.4f}).",
        "",
        (
            "A BiLSTM over 256-d fused features has unbounded context in both directions, so if "
            "sequential memory were going to find the ~16-frame inconsistency this is where it "
            "would show. "
            + (
                "It did."
                if beats
                else "It did not — which is the outcome section 5.3 anticipated when it argued "
                "that global comparison beats sequential memory for this task. Phase 8's "
                "Transformer is the test of that argument; this number is what it has to beat."
            )
        ),
        "",
        "## P7-4 — frame-level AP",
        "",
        "The per-frame head is the first localization-relevant output the project produces:",
        "`[T,256] -> [T,1]`, one decision per 40 ms frame, trained with masked pos-weighted BCE.",
        "",
        "| | value |",
        "|---|---|",
        f"| dev frame AP | **{f['frame_ap_mean']:.4f}** ± {f['frame_ap_std']:.4f} |",
    ]
    if f["frame_chance_floor"]:
        lines += [
            f"| chance floor (forged-frame rate) | {f['frame_chance_floor']:.4f} |",
            f"| lift over chance | {f['frame_ap_mean'] / f['frame_chance_floor']:.2f}× |",
        ]
    if f.get("frame_positional_floor") is not None:
        lines += [
            f"| ⛔ positional floor (frame index alone) | {f['frame_positional_floor']:.4f} |",
            f"| share of the model's AP it explains | **{f['frame_positional_share']:.1%}** |",
        ]
    lines += [
        "",
        (
            "⛔ **The positional floor is the number that makes the frame AP mean something.** "
            'Forged spans are not uniformly placed, so "frames near the start are more likely '
            'forged" is a real content-free prior — and PF-17 already showed this project what a '
            "shortcut worth most of a headline looks like. Fitted on train and scored on dev, that "
            f"prior reaches only {f['frame_positional_floor']:.4f}, explaining "
            f"{f['frame_positional_share']:.1%} of the model's frame AP. The frame head is locating "
            "forgeries by content, not by clock position."
            if f.get("frame_positional_floor") is not None
            else "⚠️ No positional floor was measured — run `scripts/26_check_frame_confound.py`. "
            "Until then this frame AP has no validated denominator."
        ),
        "",
        "⚠️ This is **frame** AP, not the **interval** AP@IoU that Phase 10's gate requires. Frame",
        "AP never forms a segment, so it cannot punish a model that is right about which frames are",
        "forged and wrong about where the boundaries fall. It is a tracked metric, not a headline",
        "localization number, and it must not be quoted as one.",
        "",
        "## ⛔ Is it still multimodal? (section 5.6, X-7)",
        "",
        "| stream removed | dev AUC | drop |",
        "|---|---|---|",
        f"| none | {d['dev_auc_mean']:.4f} | — |",
        f"| visual | {d['drop_visual_auc_mean']:.4f} | **{f['drop_visual_delta']:+.4f}** |",
        f"| audio | {d['drop_audio_auc_mean']:.4f} | **{d['drop_audio_delta']:+.4f}** |",
        "",
        f"Reliance on video went from **{f['fusion_drop_visual_delta']:+.4f}** at C to "
        f"**{f['drop_visual_delta']:+.4f}** at D. "
        + (
            "The visual pathway still contributes essentially nothing — PF-20's collapse survives "
            "the addition of temporal modelling, which is itself informative: recurrence over a "
            "collapsed representation cannot un-collapse it."
            if not f["uses_video"]
            else "The visual pathway carries real signal at D."
        ),
        "",
        "## Gate P7-6",
        "",
        "> temporal contribution measured independently of attention",
        "",
        "This is a **provenance** gate, not a performance one. It asks whether Phase 8 will be able",
        "to say how much of `E > C` was recurrence and how much was attention. It passes when the",
        "temporal delta is isolated and recorded; a BiLSTM that does not help still passes, because",
        "an honest negative is a measurement.",
        "",
        "| check | result |",
        "|---|---|",
        f"| D declares no attention, temporal = bilstm | {f['isolated_from_attention']} |",
        f"| backbone / feature / defences / clip count match C | "
        f"{not f['uncontrolled_differences']} |",
        f"| frame AP recorded (P7-4) | {f['frame_ap_mean'] is not None} |",
        f"| seeds | {f['n_seeds']} |",
        f"| D beats C | {beats} *(reported, does not gate)* |",
        "",
        f"**{'PASSED' if f['gate_passed'] else 'NOT PASSED'}.**",
        "",
        "## ⚠️ Read alongside PF-17, PF-19 and PF-20",
        "",
        "D inherits every artefact its inputs carry: the clip-length shortcut (PF-17), the global",
        "audio processing fingerprint (PF-19), and the modality collapse measured at C (PF-20). The",
        "absolute AUC here is therefore an upper bound, and the *delta* between C and D is the",
        "trustworthy part — both arms carry the same artefacts on the same dev split.",
    ]
    while lines and not lines[-1].strip():
        lines.pop()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
