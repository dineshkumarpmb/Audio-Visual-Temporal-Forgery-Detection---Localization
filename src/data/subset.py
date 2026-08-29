"""P1-15: reproducible stratified development subsets (PROJECT_PLAN section 3.9).

Sampling rules, taken from the plan and enforced here rather than trusted:

- **Preserve official split membership.** A video is never moved between splits; each
  split is sampled independently in proportion to its size. This is section 3.5 RULE 1,
  and violating it would re-introduce exactly the leakage P1-8 exists to prevent.
- **Stratify by the 4-class taxonomy** (real / visual_only / audio_only / both), so a
  subset cannot accidentally drop a fake type and make the section 8.2.4 collapse
  diagnostic unreadable.
- **Fix the seed and commit the id list**, so every experiment runs on literally the
  same videos (P1-16).

`smoke-100` is special-cased to the plan's 50 real / 50 fake, because its purpose is
overfit-a-batch sanity checking rather than distributional fidelity.

Subsets are drawn only from rows with `status == "ok"`: a quarantined video must not
reach a training loop through a side door.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

SUBSET_SIZES = {"smoke-100": 100, "dev-2k": 2_000, "dev-10k": 10_000}
CLASS_ORDER = ("real", "visual_only", "audio_only", "both")


def _take(group: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Deterministic sample of at most `n` rows, ordered for stable output."""
    if n <= 0 or group.empty:
        return group.iloc[0:0]
    g = group.sort_values("video_id")
    if len(g) <= n:
        return g
    return g.sample(n=n, random_state=seed).sort_values("video_id")


def make_subset(
    manifest: pd.DataFrame, name: str, *, seed: int = 1337, size: int | None = None
) -> pd.DataFrame:
    """Build one named subset. Returns the selected manifest rows."""
    if name == "full":
        return manifest[manifest["status"] == "ok"].sort_values("video_id")

    target = size if size is not None else SUBSET_SIZES.get(name)
    if target is None:
        raise ValueError(
            f"unknown subset {name!r}; expected one of {sorted(SUBSET_SIZES)} or 'full'"
        )

    pool = manifest[manifest["status"] == "ok"]
    if pool.empty:
        raise ValueError("no rows with status == 'ok' to sample from")

    if name == "smoke-100":
        # 50 real / 50 fake, spread across splits so the pipeline is exercised on each.
        halves = []
        for label, want in ((0, target // 2), (1, target - target // 2)):
            side = pool[pool["label"] == label]
            per_split = _proportional(side, want, seed)
            halves.append(per_split)
        return pd.concat(halves).sort_values("video_id")

    # dev-2k / dev-10k: proportional over split, then over the 4 classes within split.
    return _proportional(pool, target, seed, by_class=True)


def _proportional(
    pool: pd.DataFrame, target: int, seed: int, *, by_class: bool = False
) -> pd.DataFrame:
    """Allocate `target` rows across splits (and optionally classes) by pool share.

    Largest-remainder allocation, so the total lands on `target` exactly rather than
    drifting by a few rows through repeated rounding.
    """
    chosen: list[pd.DataFrame] = []
    splits = sorted(pool["split"].unique())
    quotas = _largest_remainder({s: int((pool["split"] == s).sum()) for s in splits}, target)

    for split in splits:
        part = pool[pool["split"] == split]
        want = quotas[split]
        if not by_class:
            chosen.append(_take(part, want, seed))
            continue
        classes = [c for c in CLASS_ORDER if (part["class_name"] == c).any()]
        cq = _largest_remainder({c: int((part["class_name"] == c).sum()) for c in classes}, want)
        for c in classes:
            chosen.append(_take(part[part["class_name"] == c], cq[c], seed))

    return pd.concat(chosen).sort_values("video_id") if chosen else pool.iloc[0:0]


def _largest_remainder(weights: dict[str, int], target: int) -> dict[str, int]:
    """Apportion `target` across keys proportionally to `weights`, summing exactly."""
    total = sum(weights.values())
    if total == 0:
        return dict.fromkeys(weights, 0)
    exact = {k: target * v / total for k, v in weights.items()}
    floors = {k: int(v) for k, v in exact.items()}
    remainder = target - sum(floors.values())
    order = sorted(weights, key=lambda k: (-(exact[k] - floors[k]), k))
    for k in order[:remainder]:
        floors[k] = min(floors[k] + 1, weights[k])
    return floors


def subset_fingerprint(video_ids: list[str]) -> str:
    """Content hash of a subset, so a changed id list is detectable in a run log."""
    payload = "\n".join(sorted(video_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def write_subset(df: pd.DataFrame, name: str, out_dir: str | Path, seed: int) -> Path:
    """Write the committed id list (P1-16) plus a small provenance sidecar."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted(df["video_id"].tolist())

    id_path = out / f"{name}.txt"
    id_path.write_text("\n".join(ids) + "\n", encoding="utf-8")

    meta = {
        "name": name,
        "n": len(ids),
        "seed": seed,
        "fingerprint": subset_fingerprint(ids),
        "split_counts": df["split"].value_counts().sort_index().to_dict(),
        "class_counts": df["class_name"].value_counts().sort_index().to_dict(),
        "label_counts": df["label"].value_counts().sort_index().to_dict(),
    }
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return id_path


def load_subset(name: str, out_dir: str | Path) -> list[str]:
    path = Path(out_dir) / f"{name}.txt"
    return path.read_text(encoding="utf-8").split()
