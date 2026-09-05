"""⛔ P7-4's validity check: is frame AP measuring localization, or frame position?

    python scripts/26_check_frame_confound.py

Phase 7 introduces the project's first localization-relevant number, and section 8.1's
discipline says a metric is worth nothing until you know what a *cheating* model scores on
it. Phases 4-5 learned this the hard way twice: PF-17 found clip length alone worth AUC
0.6157, and PF-19 found a global audio fingerprint worth 0.9728 on clips whose audio was
never touched. Both were caught by `scripts/18_check_confound.py` scoring the shortcut on
its own. This is the same check for the frame head.

**The shortcut being tested: frame index.** Forged spans are not uniformly placed — the
median span starts at 3.1 s in clips averaging ~8 s — so *"frames near the start are more
likely forged"* is a real prior a model could exploit without ever looking at the content.
If that prior alone scored near the model's frame AP, the localization claim would be
hollow in exactly the way PF-17's AUC was.

**Two floors are reported, and they answer different questions:**

* `chance` — score every frame identically. AP equals the forged-frame rate by
  construction, and is the floor *any* number must clear to mean anything.
* `positional` — score frame `t` by P(frame `t` is forged), estimated **on train only** and
  applied to dev. This is the strongest content-free predictor available, and the honest
  denominator for the model's frame AP.

⚠️ **Train-only estimation is not a formality.** Fitting the prior on dev would let it see
the split it is scored against, and it would then look far stronger than a model could
legitimately exploit — inverting the check into a false alarm.

Writes `reports/frame_confound.json`, which `scripts/25_experiment_d.py` reads so the
Experiment D report quotes a measured floor rather than an asserted one.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import AudioPreprocessConfig, VisualFeatureConfig  # noqa: E402
from src.data.dataset import MultimodalFeatureDataset  # noqa: E402
from src.data.manifest import read_manifest  # noqa: E402
from src.data.subset import load_subset  # noqa: E402
from src.evaluation.metrics import frame_metrics  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

# How much of the model's frame AP the positional prior may explain before the
# localization claim stops being credible. Matches PF-17's framing: a shortcut worth most
# of the headline is the headline.
EXPLAINS_THRESHOLD = 0.5


def positional_prior(train_targets: list[np.ndarray], length: int) -> np.ndarray:
    """P(frame t is forged), estimated on train only.

    Clips differ in length, so frame `t` is observed by only the clips at least that long;
    dividing by that count rather than by the clip total keeps the tail from being biased
    toward zero purely because few clips reach it.
    """
    num, den = np.zeros(length), np.zeros(length)
    for t in train_targets:
        num[: len(t)] += t
        den[: len(t)] += 1
    return np.divide(num, den, out=np.zeros(length), where=den > 0)


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Frame-AP confound check (P7-4)")
    ap.add_argument("--subsets", nargs="+", default=["smoke-100", "dev-2k"])
    ap.add_argument("--subset-dir", default="data/manifests/subsets")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--visual-root", default="data/features/visual")
    ap.add_argument("--audio-root", default="data/interim/audio")
    ap.add_argument("--backbone", default="mobilenet_v2")
    ap.add_argument("--feature", default="logmel")
    ap.add_argument("--model-report", default="reports/phase7_bilstm.json")
    ap.add_argument("--out", default="reports/frame_confound.json")
    args = ap.parse_args()

    manifest = read_manifest(args.manifest).set_index("video_id")
    ids: list[str] = []
    for name in args.subsets:
        ids.extend(load_subset(name, args.subset_dir))
    ids = sorted(set(ids))

    ds = MultimodalFeatureDataset(
        ids,
        manifest,
        VisualFeatureConfig(backbone=args.backbone),
        AudioPreprocessConfig(feature=args.feature),
        args.visual_root,
        args.audio_root,
        with_frames=True,
    )
    by_split: dict[str, list[int]] = defaultdict(list)
    for i, split in enumerate(ds.splits):
        by_split[split].append(i)

    print(
        f"{DIM}{'=' * 72}{RESET}\n  ⛔ P7-4 validity -- does frame position alone explain "
        f"frame AP?\n{DIM}{'=' * 72}{RESET}"
    )

    targets = {s: [ds[i].frames for i in idx] for s, idx in by_split.items()}
    if "train" not in targets or "dev" not in targets:
        print(f"{RED}need both a train and a dev split{RESET}")
        return 1

    longest = max(len(t) for group in targets.values() for t in group)
    prior = positional_prior(targets["train"], longest)

    dev = targets["dev"]
    scores = np.concatenate([prior[: len(t)] for t in dev])
    truth = np.concatenate(dev)

    positional = frame_metrics(scores, truth)
    chance = frame_metrics(np.full(truth.size, 0.5), truth)

    print(
        f"\n  dev frames: {positional['n_frames']:,}  "
        f"forged {positional['frame_positive_rate']:.2%}"
    )
    print(f"\n  {'floor':<28}{'frame AP':>10}{'frame AUC':>12}")
    print(
        f"  {'chance (constant score)':<28}{chance['frame_ap']:>10.4f}{chance['frame_auc']:>12.4f}"
    )
    print(
        f"  {'positional prior (train)':<28}{positional['frame_ap']:>10.4f}"
        f"{positional['frame_auc']:>12.4f}"
    )

    facts = {
        "n_dev_frames": positional["n_frames"],
        "forged_frame_rate": positional["frame_positive_rate"],
        "chance_frame_ap": chance["frame_ap"],
        "positional_frame_ap": positional["frame_ap"],
        "positional_frame_auc": positional["frame_auc"],
        "explains_threshold": EXPLAINS_THRESHOLD,
    }

    report = Path(args.model_report)
    if report.exists():
        model_ap = json.loads(report.read_text(encoding="utf-8"))["dev_frame_ap_mean"]
        share = positional["frame_ap"] / model_ap if model_ap else float("nan")
        explained = share >= EXPLAINS_THRESHOLD
        print(f"  {'model (BiLSTM frame head)':<28}{model_ap:>10.4f}")
        print(
            f"\n  positional prior explains {(RED if explained else GREEN)}{share:.1%}{RESET}"
            f" of the model's frame AP"
        )
        print(
            f"  {(RED if explained else GREEN)}"
            + (
                "⛔ frame position is doing most of the work -- the localization claim is "
                "not supported"
                if explained
                else "frame position does not explain the result -- the frame head is "
                "locating forgeries by content"
            )
            + RESET
        )
        facts.update(
            {
                "model_frame_ap": model_ap,
                "positional_share_of_model": float(share),
                "explained_by_position": bool(explained),
            }
        )
    else:
        print(f"\n  {YELLOW}{report} not found -- floors only, no comparison{RESET}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(facts, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
