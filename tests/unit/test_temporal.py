"""P7-1 .. P7-4: the BiLSTM, the per-frame head, and the targets they are scored against.

Two failure modes drive almost every test here, and neither shows up in a metric:

**Padding leaking into the recurrence.** An LSTM stepped over padding carries that state
into real frames — and for the *backward* direction it does so before ever reaching them.
The result is a model whose prediction for a 60-frame clip depends on whether it shared a
batch with a 500-frame one. Loss curves look normal throughout. So the tests assert the two
things that make it impossible: zero gradient at padded inputs, and bit-identical output for
a clip whether it is batched alone or padded.

**Frame targets off by one.** Section 6.3 calls this the highest-risk step in the project.
A one-frame shift is 40 ms, invisible in AP, and silently wrong in every Part 8 number. The
boundary cases are therefore worked by hand rather than round-tripped.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.metrics import frame_metrics
from src.localization.targets import frame_targets, normalise_periods, target_positive_rate
from src.models.heads.frame import FrameHead
from src.models.temporal.lstm import PackedBiLSTM, TemporalFusionBaseline

VD, AD, D, B, T = 1280, 80, 256, 4, 60
FPS = 25.0  # one frame = 0.04 s


# --------------------------------------------------------------------- targets (P7-3)


def test_real_video_target_is_exactly_zero() -> None:
    """⛔ P10-2 in miniature: not 'almost zero'. One stray frame is a fabricated label."""
    for periods in ([], np.array([], dtype=object), None):
        target = frame_targets(periods, 100, FPS)
        assert target.shape == (100,)
        assert target.sum() == 0.0


def test_span_maps_to_hand_computed_frames() -> None:
    """[0.40, 0.60) s at 25 fps is frames 10..14 inclusive — worked by hand, not round-tripped."""
    target = frame_targets([[0.40, 0.60]], 25, FPS)
    assert target[10:15].all()
    assert not target[:10].any()
    assert not target[15:].any()


def test_partial_overlap_marks_the_frame() -> None:
    """Overlap, not containment. A span covering 30% of a frame still manipulated it.

    [0.41, 0.43) s lies strictly inside frame 10 ([0.40, 0.44)), touching no boundary.
    Containment semantics would label nothing at all and lose the span entirely.
    """
    target = frame_targets([[0.41, 0.43]], 25, FPS)
    assert target.sum() == 1.0
    assert target[10] == 1.0


def test_boundaries_are_half_open() -> None:
    """A span ending exactly on a frame boundary must not claim the next frame.

    [0.00, 0.40) is frames 0..9. Frame 10 starts at exactly 0.40 and shares only a
    zero-width point with the span, so `ceil(0.40 * 25) == 10` must be exclusive.
    """
    target = frame_targets([[0.0, 0.40]], 25, FPS)
    assert target[:10].all()
    assert target[10] == 0.0


def test_multiple_spans_are_unioned() -> None:
    target = frame_targets([[0.2, 0.4], [0.8, 1.0]], 30, FPS)
    assert target[5:10].all() and target[20:25].all()
    assert not target[10:20].any()
    assert target.sum() == 10.0


def test_span_past_the_end_is_clipped_not_fatal() -> None:
    """The caches are truncated to the common audio/video grid; a spilling tail is that."""
    target = frame_targets([[0.8, 5.0]], 25, FPS)
    assert target.shape == (25,)
    assert target[20:].all()


def test_target_length_follows_the_features_not_the_manifest() -> None:
    """The whole reason `n_frames` is a parameter: a target must match what the model saw."""
    assert frame_targets([[0.0, 10.0]], 37, FPS).shape == (37,)


@pytest.mark.parametrize(
    "bad", [[[0.5, 0.5]], [[0.9, 0.2]], [[-0.1, 0.5]], [[0.1]], [[0.1, float("nan")]]]
)
def test_malformed_spans_raise(bad) -> None:
    """A reversed or empty span silently yields an all-real target for a fake video."""
    with pytest.raises((ValueError, TypeError)):
        frame_targets(bad, 50, FPS)


def test_normalise_handles_the_parquet_representation() -> None:
    """Parquet returns ragged object arrays, not lists — the shape the manifest really has."""
    cell = np.array([np.array([2.5, 3.224])], dtype=object)
    assert normalise_periods(cell) == [(2.5, 3.224)]
    assert normalise_periods(np.array([], dtype=object)) == []


def test_positive_rate_is_far_below_the_clip_fake_rate() -> None:
    """The number that sets `frame_pos_weight`. Guessing it from 73% would be 10x wrong."""
    targets = [frame_targets([[1.0, 1.65]], 200, FPS) for _ in range(10)]
    rate = target_positive_rate(targets)
    assert 0.0 < rate < 0.15


# ------------------------------------------------------------------ packing (⛔ P7-2)


def _padded_batch(lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    t_max = max(lengths)
    x = torch.randn(len(lengths), t_max, D)
    mask = torch.zeros(len(lengths), t_max, dtype=torch.bool)
    for i, n in enumerate(lengths):
        mask[i, :n] = True
    return x, mask


def test_no_gradient_flows_through_padding() -> None:
    """⛔ P7-2, the load-bearing test of Phase 7.

    If the LSTM were run over the padded tensor instead of a packed sequence, padded
    positions would receive gradient — they would have influenced the hidden state that
    produced the real frames' outputs. Packing makes that structurally impossible, and this
    asserts the structure rather than the intention.
    """
    torch.manual_seed(0)
    lstm = PackedBiLSTM(d_model=D, hidden=32, layers=2, dropout=0.0)
    x, mask = _padded_batch([60, 25, 47, 12])
    x.requires_grad_(True)

    out = lstm(x, mask)
    # Backprop only from valid positions; padded outputs are zeroed anyway.
    (out * mask.unsqueeze(-1)).sum().backward()

    padded_grad = x.grad[~mask]
    assert padded_grad.numel() > 0, "the fixture must actually contain padding"
    assert torch.count_nonzero(padded_grad) == 0, (
        f"{torch.count_nonzero(padded_grad)} padded positions received gradient — "
        "the recurrence is running over padding"
    )
    # And the valid positions must have received some, or the test proves nothing.
    assert torch.count_nonzero(x.grad[mask]) > 0


def test_padding_does_not_change_a_clip_s_output() -> None:
    """The other half of ⛔ P7-2: batching must not alter a prediction.

    A 12-frame clip alone and the same clip padded to 60 in a mixed batch must produce the
    same 12 outputs. This is what catches a missing `total_length`, `enforce_sorted=True` on
    unsorted input, or a masked-but-not-packed implementation — none of which raise.
    """
    torch.manual_seed(0)
    lstm = PackedBiLSTM(d_model=D, hidden=32, layers=2, dropout=0.0).eval()
    short = torch.randn(1, 12, D)

    alone = lstm(short, torch.ones(1, 12, dtype=torch.bool))

    padded = torch.zeros(3, 60, D)
    mask = torch.zeros(3, 60, dtype=torch.bool)
    padded[0, :12], mask[0, :12] = short[0], True
    padded[1], mask[1] = torch.randn(60, D), True
    padded[2, :47], mask[2, :47] = torch.randn(47, D), True

    batched = lstm(padded, mask)
    torch.testing.assert_close(batched[0, :12], alone[0], rtol=1e-5, atol=1e-6)


def test_temporal_resolution_is_preserved() -> None:
    """P5-2's discipline, carried into Phase 7: T in, T out, or the frame targets misalign."""
    lstm = PackedBiLSTM(d_model=D, hidden=32, layers=2, dropout=0.0).eval()
    for t in (1, 31, 137):
        x, mask = _padded_batch([t, max(1, t // 2)])
        assert lstm(x, mask).shape == (2, t, D)


def test_padded_outputs_are_zeroed() -> None:
    lstm = PackedBiLSTM(d_model=D, hidden=32, layers=2, dropout=0.0).eval()
    x, mask = _padded_batch([60, 20])
    out = lstm(x, mask)
    assert torch.count_nonzero(out[~mask]) == 0


def test_fully_masked_row_does_not_abort_the_batch() -> None:
    """One unreliable clip must not take its batch down — the AttentionPooling guard's twin."""
    lstm = PackedBiLSTM(d_model=D, hidden=32, layers=2, dropout=0.0).eval()
    x, mask = _padded_batch([40, 40])
    mask[1] = False
    out = lstm(x, mask)
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out[1]) == 0


# ------------------------------------------------------------------- frame head (P7-3)


def test_frame_head_emits_one_logit_per_frame() -> None:
    head = FrameHead(D).eval()
    out = head(torch.randn(B, T, D), torch.ones(B, T, dtype=torch.bool))
    assert out.shape == (B, T)


def test_frame_head_is_pointwise() -> None:
    """No temporal mixing of its own — changing frame 0 must not move frame 5.

    If it did, the BiLSTM would no longer be the only thing carrying temporal information
    and Experiment D would be measuring two components.
    """
    torch.manual_seed(0)
    head = FrameHead(D).eval()
    x = torch.randn(1, 10, D)
    before = head(x)
    x2 = x.clone()
    x2[0, 0] = torch.randn(D)
    after = head(x2)
    torch.testing.assert_close(before[0, 1:], after[0, 1:])
    assert not torch.allclose(before[0, 0], after[0, 0])


# ----------------------------------------------------------------- Baseline 4 (P7-1)


def _model(**kw) -> TemporalFusionBaseline:
    return TemporalFusionBaseline(
        visual_dim=VD, feature="logmel", dropout=0.0, lstm_hidden=32, lstm_dropout=0.0, **kw
    )


def _inputs(t: int = T):
    return torch.randn(B, t, VD), torch.randn(B, t, AD), torch.ones(B, t, dtype=torch.bool)


def test_baseline4_output_contract() -> None:
    """Phase 6's keys survive, so the ablation probe and Trainer work unmodified."""
    out = _model().eval()(*_inputs())
    assert out["logit"].shape == (B,)
    assert out["frame_logits"].shape == (B, T)
    assert out["sequence"].shape == (B, T, D)
    assert out["aux_logits"].shape == (2, B)


def test_baseline4_reuses_the_phase6_encoder() -> None:
    """`D - C` must be the recurrence. A re-implemented encoder would confound it."""
    from src.models.fusion.concat import ConcatFusionBaseline

    model = _model()
    assert isinstance(model, ConcatFusionBaseline)
    assert model.encode.__func__ is ConcatFusionBaseline.encode


def test_gradient_reaches_both_streams_from_the_frame_head() -> None:
    """The per-frame loss must train the visual pathway too, not only the audio one."""
    model = _model(modality_dropout=0.0)
    v, a, m = _inputs()
    model(v, a, m)["frame_logits"].sum().backward()
    assert model.visual_proj.weight.grad is not None
    assert model.visual_proj.weight.grad.abs().sum() > 0
    assert model.audio_encoder.blocks[0][0].weight.grad.abs().sum() > 0


def test_frame_logits_are_zero_on_padding() -> None:
    model = _model().eval()
    v, a, _ = _inputs()
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, 40:] = False
    out = model(v, a, mask)
    assert torch.count_nonzero(out["frame_logits"][:, 40:]) == 0


# --------------------------------------------------------------- masked loss (⛔ P7-2)


def test_frame_loss_ignores_padding() -> None:
    """Padding must contribute nothing, or the loss tracks the batch's length mix.

    Same valid content, twice as much padding: the loss must be identical. An unmasked mean
    would halve it, rewarding the model for a property of the collation.
    """
    from src.training.trainer import TrainConfig, Trainer

    trainer = Trainer(torch.nn.Linear(1, 1), TrainConfig(frame_pos_weight=3.0), device="cpu")
    torch.manual_seed(0)
    logits = torch.randn(2, 20)
    targets = (torch.rand(2, 20) > 0.7).float()
    mask = torch.zeros(2, 20, dtype=torch.bool)
    mask[:, :10] = True

    tight = trainer._frame_loss(logits[:, :10], targets[:, :10], mask[:, :10])
    padded = trainer._frame_loss(logits, targets, mask)
    torch.testing.assert_close(tight, padded)

    # And the padded region really did carry loss-worthy content, or this proves nothing.
    unmasked = trainer._frame_loss(logits, targets, torch.ones(2, 20, dtype=torch.bool))
    assert not torch.allclose(tight, unmasked)


def test_frame_pos_weight_penalises_missed_positives() -> None:
    """Without it, 'predict real everywhere' is a strong local optimum at a ~7% rate."""
    from src.training.trainer import TrainConfig, Trainer

    logits = torch.full((1, 10), -5.0)  # confidently "real" everywhere
    targets = torch.zeros(1, 10)
    targets[0, :1] = 1.0  # one forged frame, missed
    mask = torch.ones(1, 10, dtype=torch.bool)

    plain = Trainer(torch.nn.Linear(1, 1), TrainConfig(), "cpu")._frame_loss(logits, targets, mask)
    weighted = Trainer(torch.nn.Linear(1, 1), TrainConfig(frame_pos_weight=9.0), "cpu")._frame_loss(
        logits, targets, mask
    )
    assert weighted > plain


# ------------------------------------------------------------------ frame AP (P7-4)


def test_frame_ap_on_a_hand_worked_case() -> None:
    """Perfect ranking scores 1.0; a constant scorer scores the positive rate."""
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    targets = np.array([1.0, 1.0, 0.0, 0.0])
    assert frame_metrics(scores, targets)["frame_ap"] == pytest.approx(1.0)

    flat = frame_metrics(np.full(4, 0.5), targets)
    assert flat["frame_ap"] == pytest.approx(0.5)
    assert flat["frame_auc"] == pytest.approx(0.5)
    assert flat["frame_positive_rate"] == pytest.approx(0.5)


def test_frame_ap_drops_padding_rather_than_scoring_it() -> None:
    """Padded frames are free true negatives; counting them inflates AP with collation."""
    scores = np.array([0.9, 0.1, 0.0, 0.0])
    targets = np.array([1.0, 0.0, 0.0, 0.0])
    mask = np.array([True, True, False, False])
    m = frame_metrics(scores, targets, mask)
    assert m["n_frames"] == 2
    assert m["frame_positive_rate"] == pytest.approx(0.5)


def test_frame_ap_is_nan_when_no_frame_is_forged() -> None:
    """An all-real dev split has no AP. Returning 0.0 would look like a failed model."""
    assert np.isnan(frame_metrics(np.array([0.1, 0.2]), np.array([0.0, 0.0]))["frame_ap"])


def test_average_precision_is_not_decided_by_row_order() -> None:
    """A constant scorer has ranked nothing, so its AP is the positive rate — always.

    Under order-dependent tie handling this returns 1.0 when the positives happen to sort
    first and 0.5 when they interleave, for the *same* model. `roc_auc` has always resolved
    ties by average rank; this asserts AP holds the same line.
    """
    from src.evaluation.metrics import average_precision

    for labels in ([1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]):
        y = np.array(labels)
        assert average_precision(np.full(4, 0.5), y) == pytest.approx(0.5)

    # And the floor Baseline 0 actually reports: 72% fake -> AP exactly 0.72.
    y = np.array([1] * 72 + [0] * 28)
    assert average_precision(np.ones(100), y) == pytest.approx(0.72)


def test_average_precision_hand_worked_partial_ranking() -> None:
    """One correct hit, then a spurious one: (1/1 + 2/3) / 2 = 0.8333, by hand."""
    from src.evaluation.metrics import average_precision

    scores = np.array([0.9, 0.8, 0.7])
    labels = np.array([1, 0, 1])
    assert average_precision(scores, labels) == pytest.approx(5 / 6)
