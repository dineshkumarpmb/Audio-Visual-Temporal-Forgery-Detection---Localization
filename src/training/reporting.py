"""Shared run bookkeeping for the Stage-B training scripts (Phases 4, 5, ...).

Extracted from `scripts/14_train.py` when Phase 5 needed the same three things. Keeping one
copy matters more than it looks: `baseline_zero` defines the floor every headline number is
quoted against, and two drifting copies would make Phase 4's and Phase 5's numbers quietly
incomparable — which is exactly what Experiment B has to compare.

Lives in `src/` rather than being imported across scripts because X-3 forbids that
direction, and because a module whose name starts with a digit is not importable anyway.
"""

from __future__ import annotations

import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import collate
from src.evaluation.metrics import summary


def git_sha() -> str:
    """Logged with every run (section 3.8 rule 4) so a number can be traced to code."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        )
        sha = out.stdout.strip()[:12]
        if not sha:
            return "unknown"
        # A run trained from uncommitted code must say so: the bare SHA would point at a
        # commit that does not contain the code that produced the number. Only src/ and
        # scripts/ count -- results written under experiments/ and reports/ are not code.
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "scripts"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return f"{sha}-dirty" if status.stdout.strip() else sha
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def baseline_zero(labels: np.ndarray, seed: int = 1337) -> dict[str, dict]:
    """P4-10 / P5-4: the sanity floors every real number must clear.

    Majority-class emits a constant, so its AUC is exactly 0.5 by construction — that is
    the point. Its *accuracy* is the dataset's fake rate, which is the number that makes
    accuracy an untrustworthy headline here.
    """
    rng = np.random.default_rng(seed)
    majority = float(labels.mean() >= 0.5)
    return {
        "majority_class": summary(np.full(len(labels), majority), labels),
        "random": summary(rng.random(len(labels)), labels),
    }


def loader_for(ds: Dataset, batch_size: int, *, shuffle: bool, seed: int) -> DataLoader:
    """Deterministic shuffling: the seed is the run's, so P4-9/P5-4 seeds really differ."""
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=0,
        generator=g if shuffle else None,
    )
