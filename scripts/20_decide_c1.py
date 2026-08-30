"""⛔ P5-7: resolve Decision C-1 from the measurements, and write the justification.

    python scripts/20_decide_c1.py

C-1 (MFCC vs log-Mel) is one of the two decisions PF-3 committed to settling **by
experiment rather than by preference**. The source report specifies MFCC; PROJECT_PLAN
section C recommends log-Mel, with a stated mechanism:

> MFCC applies a DCT and keeps ~13-40 coefficients, which decorrelates and *compresses
> away* fine spectral structure. That structure is precisely where TTS/vocoder artifacts
> live [...] MFCC was designed for *speech recognition*, where speaker and channel detail
> is nuisance; here it is the signal.

**The rule, pre-committed before the numbers were looked at** (and mirroring the one
section 5.2 fixed for D-1, so the two decisions are resolved on the same standard):

> Take the higher-scoring arm iff the dev-AUC gap exceeds the 0.01 noise threshold **and**
> the seed sd intervals do not overlap. Otherwise the experiment has not separated the
> arms, and section C's stated default (log-Mel) stands unrebutted.

Unlike D-1 there is no speed tie-break: both arms read a cache of the same shape through
the same encoder, so extraction cost is identical by construction and cannot decide
anything.

Inputs (produced by `scripts/19_train_audio.py`, none entered by hand):
  reports/phase5_audio_logmel.json
  reports/phase5_audio_mfcc.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.console import init_console  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

ARMS = ("mfcc", "logmel")
AUC_NOISE_THRESHOLD = 0.01
PRIOR = "logmel"  # section C's reasoned default; the report's own choice was MFCC


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Resolve Decision C-1 from measurements")
    ap.add_argument("--out", default="reports/decision_c1.md")
    args = ap.parse_args()

    training = {a: load(Path(f"reports/phase5_audio_{a}.json")) for a in ARMS}
    missing = [a for a in ARMS if training[a] is None]
    if missing:
        print(f"{RED}cannot decide C-1 -- missing measurements:{RESET}")
        for a in missing:
            print(f"  reports/phase5_audio_{a}.json")
        print("\n  Run: python scripts/19_train_audio.py --feature <arm> --seeds 3")
        return 1

    print(f"{DIM}{'=' * 72}{RESET}\n  ⛔ Decision C-1 -- MFCC vs log-Mel\n{DIM}{'=' * 72}{RESET}")

    rows = {}
    for a in ARMS:
        t = training[a]
        rows[a] = {
            "input_dim": 40 if a == "mfcc" else 80,
            "n_params": t["n_params"],
            "receptive_field": t["receptive_field"],
            "auc_mean": t["dev_auc_mean"],
            "auc_std": t["dev_auc_std"],
            "acc_mean": t["dev_acc_mean"],
            "n_seeds": len(t["runs"]),
            "n_clips": t["n_clips"],
            "baseline_auc": t["baseline_zero"]["majority_class"]["auc"],
            "audio_only_auc": t["audio_only_auc_mean"],
            "splits": {k: v["n"] for k, v in t.get("splits", {}).items()},
            "per_class": {
                c: sum(r["dev"]["per_class"][c]["auc"] for r in t["runs"]) / len(t["runs"])
                for c in ("visual_only", "audio_only", "both")
            },
        }

    mf, lm = rows["mfcc"], rows["logmel"]
    auc_gap = lm["auc_mean"] - mf["auc_mean"]  # positive => log-Mel ahead
    overlap = abs(auc_gap) <= (mf["auc_std"] + lm["auc_std"])
    within_noise = abs(auc_gap) < AUC_NOISE_THRESHOLD
    separated = not within_noise and not overlap

    print(f"\n  {'':<24}{'MFCC':>14}{'log-Mel':>14}")
    for label, key, fmt in (
        ("input dim", "input_dim", "{:d}"),
        ("encoder params", "n_params", "{:,d}"),
        ("receptive field (frames)", "receptive_field", "{:d}"),
        ("dev AUC (mean)", "auc_mean", "{:.4f}"),
        ("dev AUC (sd)", "auc_std", "{:.4f}"),
        ("dev acc (mean)", "acc_mean", "{:.4f}"),
        ("seeds", "n_seeds", "{:d}"),
        ("audio_only AUC", "audio_only_auc", "{:.4f}"),
    ):
        print(f"  {label:<24}{fmt.format(mf[key]):>14}{fmt.format(lm[key]):>14}")

    print(f"\n  AUC gap (log-Mel - MFCC)  : {auc_gap:+.4f}")
    print(f"  Majority-class floor      : {mf['baseline_auc']:.4f}")
    print(
        f"\n  {DIM}Pre-committed rule: higher arm wins iff |gap| >= {AUC_NOISE_THRESHOLD} "
        f"AND sd intervals disjoint{RESET}"
    )
    print(f"    |gap| >= {AUC_NOISE_THRESHOLD}   : {not within_noise}  ({abs(auc_gap):.4f})")
    print(f"    sd intervals disjoint: {not overlap}")

    if separated:
        winner = "logmel" if auc_gap > 0 else "mfcc"
        reason = (
            f"the dev-AUC gap ({auc_gap:+.4f} in log-Mel's favour) exceeds the "
            f"{AUC_NOISE_THRESHOLD} noise threshold and the seed sd intervals are disjoint, so "
            f"the experiment separates the arms and the higher-scoring one wins."
            if auc_gap > 0
            else f"the dev-AUC gap ({auc_gap:+.4f}) exceeds the {AUC_NOISE_THRESHOLD} noise "
            f"threshold and the seed sd intervals are disjoint, so MFCC wins on measurement -- "
            f"overturning section C's stated prior, which is what deciding by experiment means."
        )
    else:
        winner = PRIOR
        reason = (
            f"the dev-AUC gap ({auc_gap:+.4f}) does not clear the {AUC_NOISE_THRESHOLD} noise "
            f"threshold with disjoint seed intervals, so the experiment does not separate the "
            f"arms. Section C's stated default (log-Mel) therefore stands unrebutted -- on its "
            f"mechanism argument, not on a fractional AUC difference."
        )

    print(f"\n  {GREEN}C-1 RESOLVED: {winner}{RESET}")
    print(f"  {reason}")

    facts = {
        "decision": winner,
        "reason": reason,
        "auc_gap_logmel_minus_mfcc": auc_gap,
        "within_noise": within_noise,
        "sd_overlap": overlap,
        "separated": separated,
        "prior": PRIOR,
        "rule": {"auc_noise_threshold": AUC_NOISE_THRESHOLD},
        "arms": rows,
    }
    Path("reports/c1_facts.json").write_text(
        json.dumps(facts, indent=2, default=float) + "\n", encoding="utf-8"
    )
    _write(Path(args.out), facts)
    print(f"\n  wrote {args.out}")
    return 0


def _write(path: Path, f: dict) -> None:
    mf, lm = f["arms"]["mfcc"], f["arms"]["logmel"]
    win = f["arms"][f["decision"]]
    splits = " / ".join(
        f"{k} {mf['splits'][k]}" for k in ("train", "dev", "test") if k in mf.get("splits", {})
    )
    lines = [
        "# Decision C-1 — MFCC vs log-Mel",
        "",
        f"**Resolved: `{f['decision']}`.** {f['reason']}",
        "",
        "Generated by `scripts/20_decide_c1.py` from measurements on disk. The source report",
        "specifies MFCC; PROJECT_PLAN section C recommends log-Mel with a stated mechanism, and",
        "PF-3 committed to settling the disagreement by Experiment K rather than by preference.",
        "",
        "## Measurements",
        "",
        "| | MFCC (B2a) | log-Mel (B2b) |",
        "|---|---|---|",
        f"| Input dim | {mf['input_dim']} | {lm['input_dim']} |",
        f"| Encoder params | {mf['n_params']:,} | {lm['n_params']:,} |",
        f"| **dev AUC** | **{mf['auc_mean']:.4f} ± {mf['auc_std']:.4f}** | "
        f"**{lm['auc_mean']:.4f} ± {lm['auc_std']:.4f}** |",
        f"| dev accuracy | {mf['acc_mean']:.4f} | {lm['acc_mean']:.4f} |",
        f"| Seeds | {mf['n_seeds']} | {lm['n_seeds']} |",
        f"| Majority-class floor | {mf['baseline_auc']:.4f} | {lm['baseline_auc']:.4f} |",
        "",
        f"**{mf['n_clips']} clips** — {splits} — on the locally available corpus, the same clips",
        "and the same unchanged dev split Phase 4 used, so the audio and visual arms are directly",
        "comparable.",
        "",
        "## The rule, pre-committed",
        "",
        "> Take the higher-scoring arm iff the dev-AUC gap exceeds the 0.01 noise threshold **and**",
        "> the seed sd intervals do not overlap. Otherwise section C's stated default (log-Mel)",
        "> stands unrebutted.",
        "",
        "| Condition | Threshold | Measured | Met |",
        "|---|---|---|---|",
        f"| AUC gap exceeds noise | ≥ {f['rule']['auc_noise_threshold']} | "
        f"{abs(f['auc_gap_logmel_minus_mfcc']):.4f} | {not f['within_noise']} |",
        f"| Seed sd intervals disjoint | — | {not f['sd_overlap']} | {not f['sd_overlap']} |",
        "",
        "There is deliberately **no speed tie-break** as there is for D-1: both arms run the same",
        "encoder over a cache of the same shape, so extraction cost is identical by construction",
        "and cannot decide anything.",
        "",
        "## Per-class behaviour",
        "",
        "| class | MFCC | log-Mel |",
        "|---|---|---|",
        *[
            f"| {c} | {mf['per_class'][c]:.4f} | {lm['per_class'][c]:.4f} |"
            for c in ("visual_only", "audio_only", "both")
        ],
        "",
        "`audio_only` is the diagnostic that matters: those forgeries are **pixel-identical to",
        "their originals**, so they are the class audio should own outright and vision should be",
        "blind to. `visual_only` is the mirror image — audio has nothing to detect there, and a",
        "high score on it would indicate the same clip-length shortcut PF-17 found in Phase 4",
        "rather than any acoustic evidence.",
        "",
        "## ⛔ The PF-17 control",
        "",
        "Phase 4's visual baseline scored **0.7551** on `audio_only` — clips whose video track is",
        "byte-identical to the real original. It had nothing to see and scored well anyway, off a",
        "clip-length artefact. The audio arm is the control for that claim:",
        "",
        f"- Audio ({f['decision']}) on `audio_only`: **{win['per_class']['audio_only']:.4f}**",
        "- Visual on `audio_only`: **0.7551**",
        "",
        (
            "Audio is ahead on exactly the class it should own, which is the evidence that it is "
            "reading acoustic content rather than the shared shortcut."
            if win["per_class"]["audio_only"] > 0.7551
            else "⚠️ **Audio does not beat the visual model on fakes the visual model cannot see.** "
            "That is the outcome PF-17 warned about: it points at both arms riding the same "
            "clip-length shortcut rather than at either one detecting forgery. The length-matched "
            "evaluation PF-17 defers to Phase 6 should be settled before fusion, because fusing "
            "two arms that share a confound compounds it rather than cancelling it."
        ),
    ]
    while lines and not lines[-1].strip():
        lines.pop()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
