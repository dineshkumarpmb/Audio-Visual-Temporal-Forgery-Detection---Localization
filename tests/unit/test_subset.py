"""P1-15 / P1-16: subsets must be stratified, split-preserving and reproducible."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.subset import (
    CLASS_ORDER,
    load_subset,
    make_subset,
    subset_fingerprint,
    write_subset,
)


@pytest.fixture
def manifest() -> pd.DataFrame:
    """A synthetic manifest with the real dataset's split proportions."""
    rows = []
    for split, n in (("train", 600), ("dev", 300), ("test", 100)):
        for i in range(n):
            cls = CLASS_ORDER[i % 4]
            rows.append(
                {
                    "video_id": f"{split}_{i:06d}",
                    "split": split,
                    "class_name": cls,
                    "label": 0 if cls == "real" else 1,
                    "status": "ok",
                }
            )
    # a quarantined row that must never be sampled
    rows.append(
        {
            "video_id": "train_999999",
            "split": "train",
            "class_name": "real",
            "label": 0,
            "status": "bad_label",
        }
    )
    return pd.DataFrame(rows)


class TestSizeAndComposition:
    @pytest.mark.parametrize("name", ["smoke-100", "dev-2k", "dev-10k"])
    def test_never_exceeds_target_or_pool(self, manifest, name):
        sub = make_subset(manifest, name)
        assert len(sub) <= 1000  # the whole usable pool

    def test_exact_size_when_pool_allows(self, manifest):
        assert len(make_subset(manifest, "dev-2k", size=200)) == 200

    def test_smoke_is_half_real(self, manifest):
        sub = make_subset(manifest, "smoke-100")
        assert len(sub) == 100
        assert (sub["label"] == 0).sum() == 50
        assert (sub["label"] == 1).sum() == 50

    def test_all_four_classes_survive(self, manifest):
        """Section 8.2.4's collapse diagnostic is unreadable if a class is dropped."""
        sub = make_subset(manifest, "dev-2k", size=200)
        assert set(sub["class_name"]) == set(CLASS_ORDER)

    def test_split_membership_preserved(self, manifest):
        """Section 3.5 RULE 1 -- a video never moves between splits."""
        sub = make_subset(manifest, "dev-2k", size=200)
        original = manifest.set_index("video_id")["split"]
        for vid, split in zip(sub["video_id"], sub["split"], strict=True):
            assert original[vid] == split

    def test_split_proportions_roughly_preserved(self, manifest):
        sub = make_subset(manifest, "dev-2k", size=200)
        share = sub["split"].value_counts(normalize=True)
        assert share["train"] == pytest.approx(0.6, abs=0.05)
        assert share["dev"] == pytest.approx(0.3, abs=0.05)

    def test_quarantined_rows_never_sampled(self, manifest):
        for name in ("smoke-100", "dev-2k", "full"):
            assert "train_999999" not in set(make_subset(manifest, name)["video_id"])

    def test_full_returns_the_usable_pool(self, manifest):
        assert len(make_subset(manifest, "full")) == 1000

    def test_unknown_name_raises(self, manifest):
        with pytest.raises(ValueError, match="unknown subset"):
            make_subset(manifest, "dev-500")


class TestReproducibility:
    def test_same_seed_same_ids(self, manifest):
        a = make_subset(manifest, "dev-2k", size=200, seed=1337)["video_id"].tolist()
        b = make_subset(manifest, "dev-2k", size=200, seed=1337)["video_id"].tolist()
        assert a == b

    def test_different_seed_different_ids(self, manifest):
        a = make_subset(manifest, "dev-2k", size=200, seed=1337)["video_id"].tolist()
        b = make_subset(manifest, "dev-2k", size=200, seed=7)["video_id"].tolist()
        assert a != b

    def test_fingerprint_is_order_independent(self):
        assert subset_fingerprint(["b", "a"]) == subset_fingerprint(["a", "b"])

    def test_fingerprint_detects_a_changed_id(self):
        assert subset_fingerprint(["a", "b"]) != subset_fingerprint(["a", "c"])

    def test_write_then_load_roundtrip(self, manifest, tmp_path):
        sub = make_subset(manifest, "smoke-100")
        write_subset(sub, "smoke-100", tmp_path, 1337)
        assert sorted(load_subset("smoke-100", tmp_path)) == sorted(sub["video_id"].tolist())

    def test_sidecar_records_provenance(self, manifest, tmp_path):
        import json

        sub = make_subset(manifest, "smoke-100")
        write_subset(sub, "smoke-100", tmp_path, 1337)
        meta = json.loads((tmp_path / "smoke-100.json").read_text(encoding="utf-8"))
        assert meta["seed"] == 1337
        assert meta["n"] == 100
        assert meta["fingerprint"] == subset_fingerprint(sub["video_id"].tolist())
