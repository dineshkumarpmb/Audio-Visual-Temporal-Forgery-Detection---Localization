"""P11-1: evaluation as a pure function of saved predictions (`src/evaluation/runs.py`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.runs import (
    classification_metrics,
    compare,
    ground_truth,
    load_predictions,
    localization_metrics,
    per_class_frame_ap,
    per_class_localization,
    summarise,
)
from src.inference.predict import VISUAL_DIM
from src.localization.postprocess import PostProcessConfig

FPS = 25.0


@pytest.fixture
def manifest() -> pd.DataFrame:
    # Three 100-frame clips: a real one, an audio_only fake forged over 1.0-2.0 s, and a
    # visual_only fake forged over 2.0-3.0 s.
    return pd.DataFrame(
        {
            "video_id": ["r", "a", "v"],
            "class_name": ["real", "audio_only", "visual_only"],
            "fake_periods": [[], [[1.0, 2.0]], [[2.0, 3.0]]],
            "fps": [FPS, FPS, FPS],
        }
    ).set_index("video_id")


def _perfect_scores(manifest) -> dict[str, np.ndarray]:
    out = {}
    for vid, row in manifest.iterrows():
        s = np.zeros(100, dtype=np.float32)
        for a, b in row["fake_periods"]:
            s[int(a * FPS) : int(b * FPS)] = 1.0
        out[vid] = s
    return out


@pytest.fixture
def saved(tmp_path, manifest):
    frames = _perfect_scores(manifest)
    ids = list(frames)
    path = tmp_path / "dev_predictions.npz"
    np.savez_compressed(
        path,
        video_ids=np.array(ids),
        labels=np.array([0, 1, 1], dtype=np.int8),
        clip_scores=np.array([0.1, 0.9, 0.8], dtype=np.float32),
        lengths=np.array([100, 100, 100], dtype=np.int32),
        frame_scores=np.concatenate([frames[v] for v in ids]),
        boundary_scores=np.zeros((0, 2), np.float32),
    )
    return path


def test_round_trip_through_npz(saved):
    pred = load_predictions(saved)
    assert list(pred["lengths"]) == ["r", "a", "v"]
    assert pred["bounds"] is None  # an empty boundary array means "no boundary head"
    assert pred["clip_scores"]["a"] == pytest.approx(0.9)
    assert pred["frames"]["a"][30] == 1.0 and pred["frames"]["a"][10] == 0.0


def test_ground_truth_is_clipped_to_what_the_model_saw(manifest):
    gt = ground_truth(manifest, {"r": 100, "a": 100, "v": 60})  # v sees only 2.4 s
    assert gt["r"] == []
    assert gt["a"] == [(1.0, 2.0)]
    assert gt["v"] == [(2.0, 2.4)]


def test_perfect_predictions_score_perfectly(saved, manifest):
    pred = load_predictions(saved)
    gt = ground_truth(manifest, pred["lengths"])
    cfg = PostProcessConfig(threshold=0.5, median_kernel=1, min_duration_s=0.0, merge_gap_s=0.0)
    rep = localization_metrics(pred, gt, cfg)
    assert rep["ap@0.5"] == pytest.approx(1.0)
    assert rep["fp_video_rate"] == 0.0
    cls = classification_metrics(pred, manifest)
    assert cls["auc"] == 1.0 and cls["frame_ap"] == pytest.approx(1.0)
    assert cls["confusion"] == {"tp": 2, "fp": 0, "tn": 1, "fn": 0}
    per = per_class_localization(pred, gt, cfg, manifest)
    assert set(per) == {"audio_only", "visual_only"}
    assert all(v["ap@0.5"] == pytest.approx(1.0) for v in per.values())
    assert per_class_frame_ap(pred, manifest) == {
        "audio_only": pytest.approx(1.0),
        "visual_only": pytest.approx(1.0),
    }


def test_drop_visual_pair_is_reported(saved, manifest):
    pred = load_predictions(saved)
    gt = ground_truth(manifest, pred["lengths"])
    blind = load_predictions(saved)
    blind["frames"] = {v: np.zeros_like(s) for v, s in blind["frames"].items()}
    rep = localization_metrics(pred, gt, PostProcessConfig(median_kernel=1), blind)
    assert rep["ap@0.5_no_refine"] == pytest.approx(1.0)
    assert rep["ap@0.5_drop_visual"] == 0.0


def test_summarise_uses_population_std():
    s = summarise([{"x": 1.0, "tag": "a"}, {"x": 3.0, "tag": "b"}])
    assert s == {"x": {"mean": 2.0, "std": 1.0}}  # strings are skipped


def test_compare_needs_magnitude_and_consistency():
    a = [{"m": 0.50}, {"m": 0.52}, {"m": 0.54}]
    # +0.10 on every seed, far beyond the 0.016 spread -> significant.
    up = compare(a, [{"m": r["m"] + 0.10} for r in a], [0, 1, 2], [0, 1, 2], "m")
    assert up["significant"] and up["consistent"]
    assert up["delta"] == pytest.approx(0.10)
    # Same mean gain but one seed goes the other way -> not significant (criterion 2).
    mixed = compare(a, [{"m": 0.66}, {"m": 0.70}, {"m": 0.50}], [0, 1, 2], [0, 1, 2], "m")
    assert not mixed["consistent"] and not mixed["significant"]
    # Consistent but smaller than the spread -> not significant (criterion 1).
    tiny = compare(a, [{"m": r["m"] + 0.001} for r in a], [0, 1, 2], [0, 1, 2], "m")
    assert tiny["consistent"] and not tiny["significant"]


def test_compare_pairs_by_seed_not_by_position():
    a = [{"m": 0.1}, {"m": 0.2}, {"m": 0.3}]
    b = [{"m": 0.35}, {"m": 0.15}, {"m": 0.25}]  # seeds 2, 0, 1 -> +0.05 each
    c = compare(a, b, [0, 1, 2], [2, 0, 1], "m")
    assert c["per_seed"] == {0: pytest.approx(0.05), 1: pytest.approx(0.05), 2: pytest.approx(0.05)}


def test_visual_dims_match_the_backbones():
    assert VISUAL_DIM == {"resnet18": 512, "mobilenet_v2": 1280}
