"""P1-8 / R3: the leakage assertions, as build-failing checks.

PROJECT_PLAN section 3.5 RULE 3: assert it in code, do not trust it. These are the
highest-impact correctness checks in the project, because the failure they catch is
silent and flattering -- a leaked split inflates test scores and nothing in a training
curve looks wrong. A model that looks excellent and is worthless is the worst possible
outcome.

They are exported both as a callable (for pipeline use) and as pytest cases in
tests/unit/test_leakage.py, so a violation fails the build rather than printing a
warning nobody reads.

Scope, stated honestly: `source_id` is derived from the `original` chain (P1-7), so
these prove no fake appears in a different split from the real video it was built
from. They cannot prove two *different* real videos of the same VoxCeleb2 speaker are
in the same split -- LAV-DF's minified metadata carries no speaker key. That residual
risk is documented in reports/dataset_statistics.md, not silently ignored.
"""

from __future__ import annotations

from collections.abc import Iterable

SPLITS = ("train", "dev", "test")


class LeakageError(AssertionError):
    """Raised when two splits share a source identity. Never catch this to continue."""


def source_ids_by_split(
    rows: Iterable[tuple[str, str]],
) -> dict[str, set[str]]:
    """Group source_ids by split from an iterable of (split, source_id) pairs."""
    out: dict[str, set[str]] = {s: set() for s in SPLITS}
    for split, source_id in rows:
        out.setdefault(split, set()).add(source_id)
    return out


def check_no_leakage(by_split: dict[str, set[str]], *, raise_on_fail: bool = True) -> dict:
    """The three assertions of section 3.5 RULE 3.

        train ∩ dev  == empty
        train ∩ test == empty
        dev   ∩ test == empty

    Returns a report even when it raises-by-default is off, so callers can log the
    overlap size before failing.
    """
    pairs = [("train", "dev"), ("train", "test"), ("dev", "test")]
    report: dict[str, object] = {"sizes": {k: len(v) for k, v in by_split.items()}}
    overlaps: dict[str, list[str]] = {}

    for a, b in pairs:
        shared = sorted(by_split.get(a, set()) & by_split.get(b, set()))
        overlaps[f"{a}^{b}"] = shared
        report[f"{a}^{b}"] = len(shared)

    report["clean"] = all(not v for v in overlaps.values())

    if raise_on_fail and not report["clean"]:
        detail = "; ".join(f"{k}: {len(v)} shared (e.g. {v[:3]})" for k, v in overlaps.items() if v)
        raise LeakageError(
            "SPLIT LEAKAGE DETECTED (R3, section 3.5 RULE 3) -- "
            f"{detail}. Every metric computed on these splits is invalid."
        )
    return report
