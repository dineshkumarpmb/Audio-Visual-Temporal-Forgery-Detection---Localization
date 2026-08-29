"""Determinism (P1-18, PROJECT_PLAN §3.8 rule 2).

Preprocessing must be a pure function of (video, config). Every source of
randomness is pinned here, and `seed_everything` is called before any pipeline
stage that could branch on chance.

`PYTHONHASHSEED` is the awkward one: CPython reads it at interpreter start, so
setting it from inside a running process does nothing for *this* process. We set
it anyway so subprocesses and DataLoader workers inherit it, and report whether
the current process actually honoured it.
"""

from __future__ import annotations

import os
import random

DEFAULT_SEED = 1337


def seed_everything(seed: int = DEFAULT_SEED, *, deterministic: bool = True) -> dict[str, object]:
    """Pin every RNG we use. Returns what was actually pinned, for run logs.

    `deterministic=True` forces cuDNN into deterministic algorithms. That costs
    throughput, which is the correct trade for a project whose headline claim is
    reproducibility — Part 7's comparisons are meaningless if two runs of the same
    config disagree.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    state: dict[str, object] = {
        "seed": seed,
        "pythonhashseed_effective": os.environ.get("PYTHONHASHSEED") == str(seed),
        "deterministic": deterministic,
    }

    try:
        import numpy as np

        # Legacy global seeding is deliberate (NPY002 suppressed): section 3.8 requires
        # pinning the *global* RNG because third-party code (albumentations, sklearn,
        # librosa) draws from it. A local Generator would not reach them.
        np.random.seed(seed)  # noqa: NPY002
        state["numpy"] = np.__version__
    except ImportError:  # numpy is a hard dep, but seeding must not be the thing that crashes
        state["numpy"] = None

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        state["torch"] = torch.__version__
        state["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        state["torch"] = None

    return state
