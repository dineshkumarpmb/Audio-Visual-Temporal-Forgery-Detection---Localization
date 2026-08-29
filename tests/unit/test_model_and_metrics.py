"""Phase 4 unit tests: backbone contract, pooling head, metrics, trainer plumbing.

Metrics are checked against hand-worked cases rather than another library, because the
conventions that matter here (which class is positive, how ties break, what happens with
one class present) are exactly the ones a library call hides.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.dataset import Sample, collate
from src.evaluation.metrics import (
    accuracy,
    average_precision,
    equal_error_rate,
    per_class_breakdown,
    roc_auc,
    summary,
)
from src.models.heads.classification import AttentionPooling, MeanPoolBaseline, VisualBaseline
from src.training.trainer import TrainConfig, Trainer


class TestROCAUC:
    def test_perfect_separation(self):
        assert roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0

    def test_perfect_inversion(self):
        assert roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0

    def test_all_tied_is_exactly_half(self):
        """A constant predictor must score 0.5 -- this is what makes Baseline 0 meaningful."""
        assert roc_auc(np.full(6, 0.7), np.array([0, 1, 0, 1, 1, 0])) == 0.5

    def test_partial_ties_use_average_ranks(self):
        assert roc_auc(np.array([0.5, 0.5, 0.9]), np.array([0, 1, 1])) == pytest.approx(0.75)

    def test_single_class_is_nan_not_half(self):
        """Undefined must look undefined, not like an uninformative model."""
        assert np.isnan(roc_auc(np.array([0.1, 0.9]), np.array([1, 1])))

    def test_invariant_to_monotone_rescaling(self):
        rng = np.random.default_rng(0)
        s, y = rng.random(50), rng.integers(0, 2, 50)
        assert roc_auc(s, y) == pytest.approx(roc_auc(s * 10 + 3, y))

    def test_rejects_non_binary_labels(self):
        with pytest.raises(ValueError, match="0/1"):
            roc_auc(np.array([0.1, 0.9]), np.array([0, 2]))

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            roc_auc(np.array([0.1, 0.9]), np.array([0]))


class TestOtherMetrics:
    def test_average_precision_perfect(self):
        assert average_precision(np.array([0.1, 0.2, 0.9]), np.array([0, 0, 1])) == 1.0

    def test_accuracy_at_threshold(self):
        assert accuracy(np.array([0.4, 0.6]), np.array([0, 1]), 0.5) == 1.0
        assert accuracy(np.array([0.4, 0.6]), np.array([1, 0]), 0.5) == 0.0

    def test_eer_of_a_perfect_model_is_zero(self):
        assert equal_error_rate(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 0.0

    def test_summary_has_every_headline_field(self):
        m = summary(np.array([0.1, 0.9, 0.8, 0.2]), np.array([0, 1, 1, 0]))
        assert {"auc", "ap", "accuracy", "eer", "precision", "recall", "f1"} <= set(m)

    def test_majority_class_scores_the_fake_rate_on_accuracy(self):
        """Why accuracy is not the headline: 73% fake means 73% accuracy for free."""
        labels = np.array([1] * 73 + [0] * 27)
        m = summary(np.ones(100), labels)
        assert m["accuracy"] == pytest.approx(0.73)
        assert m["auc"] == 0.5


class TestPerClassBreakdown:
    def test_detects_a_collapsed_modality(self):
        """Section 8.2.4: aggregate AUC hides a model that ignores one modality."""
        labels = np.array([0] * 20 + [1] * 10 + [1] * 10)
        classes = ["real"] * 20 + ["visual_only"] * 10 + ["audio_only"] * 10
        scores = np.concatenate(
            [
                np.full(20, 0.1),
                np.full(10, 0.11),  # visual fakes barely separated -- near chance
                np.full(10, 0.99),  # audio fakes trivially separated
            ]
        )
        out = per_class_breakdown(scores, labels, classes)
        assert out["audio_only"]["auc"] == 1.0
        assert out["visual_only"]["auc"] == 1.0  # separated, but by a hair
        # The recall at threshold 0.5 is what exposes it:
        assert out["visual_only"]["recall"] == 0.0
        assert out["audio_only"]["recall"] == 1.0

    def test_reports_specificity_for_real(self):
        labels = np.array([0, 0, 1])
        out = per_class_breakdown(np.array([0.1, 0.2, 0.9]), labels, ["real", "real", "both"])
        assert out["real"]["specificity"] == 1.0


class TestAttentionPooling:
    def test_weights_sum_to_one(self):
        pooled, w = AttentionPooling(16)(torch.randn(3, 20, 16))
        assert pooled.shape == (3, 16)
        assert torch.allclose(w.sum(dim=1), torch.ones(3), atol=1e-5)

    def test_masked_positions_get_zero_weight(self):
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[:, 5:] = False
        _, w = AttentionPooling(8)(torch.randn(2, 10, 8), mask)
        assert torch.allclose(w[:, 5:], torch.zeros(2, 5), atol=1e-6)

    def test_fully_masked_row_does_not_produce_nan(self):
        """One bad clip must not poison the whole batch's gradient."""
        pooled, w = AttentionPooling(8)(torch.randn(2, 6, 8), torch.zeros(2, 6, dtype=torch.bool))
        assert torch.isfinite(pooled).all()
        assert torch.isfinite(w).all()

    def test_padding_cannot_change_the_result(self):
        """The core correctness property: padding is inert."""
        torch.manual_seed(0)
        pool = AttentionPooling(8)
        x = torch.randn(1, 6, 8)
        short, _ = pool(x, torch.ones(1, 6, dtype=torch.bool))

        padded = torch.cat([x, torch.randn(1, 9, 8)], dim=1)
        mask = torch.zeros(1, 15, dtype=torch.bool)
        mask[:, :6] = True
        long, _ = pool(padded, mask)
        assert torch.allclose(short, long, atol=1e-5)


class TestVisualBaseline:
    @pytest.mark.parametrize("dim", [512, 1280])
    def test_accepts_both_backbone_widths(self, dim):
        out = VisualBaseline(feature_dim=dim)(torch.randn(2, 30, dim))
        assert out["logit"].shape == (2,)

    def test_wrong_feature_dim_fails_loudly(self):
        with pytest.raises(ValueError, match="resnet18 cache"):
            VisualBaseline(feature_dim=512)(torch.randn(2, 30, 1280))

    def test_rejects_non_3d_input(self):
        with pytest.raises(ValueError, match=r"\(B, T, D\)"):
            VisualBaseline(feature_dim=512)(torch.randn(2, 512))

    def test_returns_a_raw_logit_not_a_probability(self):
        """BCEWithLogitsLoss applies the sigmoid; doing it twice would be silently wrong."""
        torch.manual_seed(0)
        logits = VisualBaseline(feature_dim=512)(torch.randn(64, 30, 512) * 5)["logit"]
        assert logits.min() < 0.0 or logits.max() > 1.0

    def test_mean_pool_ablation_matches_parameter_count(self):
        """The ablation must isolate pooling, not capacity."""
        assert (
            MeanPoolBaseline(feature_dim=512).n_parameters
            == VisualBaseline(feature_dim=512).n_parameters
        )

    def test_mean_pool_ignores_masked_frames(self):
        torch.manual_seed(0)
        model = MeanPoolBaseline(feature_dim=16)
        x = torch.randn(1, 4, 16)
        a = model(x, torch.ones(1, 4, dtype=torch.bool))["pooled"]
        padded = torch.cat([x, torch.full((1, 4, 16), 99.0)], dim=1)
        mask = torch.zeros(1, 8, dtype=torch.bool)
        mask[:, :4] = True
        assert torch.allclose(a, model(padded, mask)["pooled"], atol=1e-5)


class TestCollate:
    def make(self, t: int, label: int = 1) -> Sample:
        rng = np.random.default_rng(t)
        return Sample(
            f"v{t}",
            rng.random((t, 512), dtype=np.float32),
            np.ones(t, dtype=bool),
            label,
            "both",
            "train",
        )

    def test_pads_to_the_longest_clip(self):
        batch = collate([self.make(10), self.make(25), self.make(5)])
        assert batch["features"].shape == (3, 25, 512)
        assert batch["mask"].shape == (3, 25)

    def test_mask_marks_exactly_the_real_frames(self):
        batch = collate([self.make(10), self.make(25)])
        assert batch["mask"][0].sum() == 10
        assert batch["mask"][1].sum() == 25

    def test_padding_region_is_zero(self):
        batch = collate([self.make(4), self.make(9)])
        assert torch.all(batch["features"][0, 4:] == 0)

    def test_labels_and_ids_survive(self):
        batch = collate([self.make(4, 0), self.make(6, 1)])
        assert batch["label"].tolist() == [0.0, 1.0]
        assert batch["video_id"] == ["v4", "v6"]


class TestTrainer:
    def test_overfits_ten_samples(self):
        """The P4-7 contract, on synthetic separable data."""
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        samples = []
        for i in range(10):
            label = i % 2
            feats = rng.normal(label * 3.0, 0.1, (12, 64)).astype(np.float32)
            samples.append(Sample(f"v{i}", feats, np.ones(12, bool), label, "both", "train"))
        batch = collate(samples)

        trainer = Trainer(
            VisualBaseline(feature_dim=64, dropout=0.0), TrainConfig(lr=1e-2), device="cpu"
        )
        losses = trainer.overfit_batch(batch, steps=250, verbose=False)
        assert losses[-1] < 0.01, f"final loss {losses[-1]}"
        assert losses[-1] < losses[0]

    def test_checkpoint_roundtrip(self, tmp_path):
        trainer = Trainer(VisualBaseline(feature_dim=64), TrainConfig(), "cpu", tmp_path)
        trainer.best_auc, trainer.best_epoch = 0.87, 3
        trainer.save("best.pt", epoch=3)

        fresh = Trainer(VisualBaseline(feature_dim=64), TrainConfig(), "cpu", tmp_path)
        assert fresh.resume(tmp_path / "best.pt") == 4
        assert fresh.best_auc == pytest.approx(0.87)
        for a, b in zip(trainer.model.parameters(), fresh.model.parameters(), strict=True):
            assert torch.allclose(a, b)

    def test_pos_weight_reaches_the_loss(self):
        t = Trainer(VisualBaseline(feature_dim=64), TrainConfig(pos_weight=3.0), "cpu")
        assert t.criterion.pos_weight is not None
        assert float(t.criterion.pos_weight) == 3.0
