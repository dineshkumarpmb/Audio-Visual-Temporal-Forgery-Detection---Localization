"""P15-1: the training losses' contracts, in one place.

Each loss that shaped a reported number (focal P10-3, boundary P10-4, InfoNCE sync P8-5) is
held to the properties any correct implementation has: finite and non-negative, lower for
better predictions, blind to padding, and differentiable. Hand-value and ablation-specific
tests stay beside the code they were written for (`test_postprocess.py`, `test_attention.py`);
this file is the cross-cutting floor.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.losses.localization import boundary_loss, boundary_targets, focal_loss_with_logits
from src.losses.sync import InfoNCESyncLoss


def frames(t: int = 40, b: int = 3):
    torch.manual_seed(0)
    targets = torch.zeros(b, t)
    targets[0, 10:20] = 1.0
    targets[2, 5:8] = 1.0  # clip 1 is real: all zeros
    mask = torch.ones(b, t, dtype=torch.bool)
    mask[2, 30:] = False
    return targets, mask


class TestFocal:
    def test_finite_non_negative_and_differentiable(self):
        targets, mask = frames()
        logits = torch.randn_like(targets, requires_grad=True)
        loss = focal_loss_with_logits(logits, targets, mask)
        assert torch.isfinite(loss) and loss >= 0
        loss.backward()
        assert torch.isfinite(logits.grad).all()

    def test_better_predictions_cost_less(self):
        targets, mask = frames()
        good = (targets * 2 - 1) * 4.0  # confident and right
        bad = -good  # confident and wrong
        assert (
            focal_loss_with_logits(good, targets, mask)
            < focal_loss_with_logits(torch.zeros_like(targets), targets, mask)
            < focal_loss_with_logits(bad, targets, mask)
        )

    def test_padding_is_invisible(self):
        targets, mask = frames()
        logits = torch.randn_like(targets)
        poisoned = logits.clone()
        poisoned[2, 30:] = 1e4  # padded frames of clip 2
        torch.testing.assert_close(
            focal_loss_with_logits(logits, targets, mask),
            focal_loss_with_logits(poisoned, targets, mask),
        )

    def test_all_real_batch_does_not_divide_by_zero(self):
        targets = torch.zeros(2, 30)
        mask = torch.ones(2, 30, dtype=torch.bool)
        assert torch.isfinite(focal_loss_with_logits(torch.randn(2, 30), targets, mask))

    def test_shape_mismatch_fails_loudly(self):
        with pytest.raises(ValueError):
            focal_loss_with_logits(torch.zeros(2, 30), torch.zeros(2, 31), torch.ones(2, 30))


class TestBoundary:
    def test_targets_peak_on_the_span_edges_and_skip_real_clips(self):
        targets, mask = frames()
        heat = boundary_targets(targets, mask)
        assert heat.shape == (*targets.shape, 2)
        assert heat[0, 10, 0] == pytest.approx(1.0) and heat[0, 19, 1] == pytest.approx(1.0)
        assert heat[1].max() == 0.0  # a real clip has no boundaries

    def test_finite_non_negative_and_prefers_the_right_peaks(self):
        targets, mask = frames()
        heat = boundary_targets(targets, mask)
        right = torch.where(heat >= 1.0 - 1e-6, 6.0, -6.0)
        wrong = torch.roll(right, shifts=7, dims=1)
        lr, lw = boundary_loss(right, heat, mask), boundary_loss(wrong, heat, mask)
        assert torch.isfinite(lr) and torch.isfinite(lw) and lr >= 0
        assert lr < lw

    def test_padding_is_invisible(self):
        targets, mask = frames()
        heat = boundary_targets(targets, mask)
        logits = torch.randn(*targets.shape, 2)
        poisoned = logits.clone()
        poisoned[2, 30:] = 50.0
        torch.testing.assert_close(
            boundary_loss(logits, heat, mask), boundary_loss(poisoned, heat, mask)
        )


class TestInfoNCE:
    @staticmethod
    def embeds(t: int = 30, b: int = 4, d: int = 16, *, aligned: bool = True):
        torch.manual_seed(0)
        v = nn.functional.normalize(torch.randn(b, t, d), dim=-1)
        a = v.clone() if aligned else nn.functional.normalize(torch.randn(b, t, d), dim=-1)
        return v, a

    def test_aligned_streams_cost_less_than_unrelated_ones(self):
        loss = InfoNCESyncLoss()
        labels = torch.zeros(4)
        g = torch.Generator().manual_seed(0)
        aligned = loss(*self.embeds(aligned=True), labels, generator=g)
        g = torch.Generator().manual_seed(0)
        unrelated = loss(*self.embeds(aligned=False), labels, generator=g)
        assert torch.isfinite(aligned) and torch.isfinite(unrelated)
        assert aligned < unrelated

    def test_differentiable(self):
        v, a = self.embeds(aligned=False)
        v.requires_grad_(True)
        out = InfoNCESyncLoss()(v, a, torch.zeros(4), generator=torch.Generator().manual_seed(0))
        out.backward()
        assert torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0

    @pytest.mark.parametrize("kw", [{"temperature": 0.0}, {"n_negatives": 0}])
    def test_bad_hyperparameters_fail_loudly(self, kw):
        with pytest.raises(ValueError):
            InfoNCESyncLoss(**kw)


def test_classification_bce_pos_weight_balances_the_classes():
    """The clip loss: `pos_weight = n_real / n_fake` makes a constant predictor indifferent."""
    labels = torch.tensor([1.0, 1.0, 1.0, 0.0])  # 3 fakes, 1 real
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1 / 3))
    lo = loss(torch.full((4,), -0.5), labels)
    hi = loss(torch.full((4,), 0.5), labels)
    assert lo == pytest.approx(hi.item(), rel=1e-6)
