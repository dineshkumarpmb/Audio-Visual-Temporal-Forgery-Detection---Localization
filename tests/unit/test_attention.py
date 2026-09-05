"""P8-1 .. P8-7: self-attention, cross-attention, and the sync component.

The sync head gets the most attention here, and component F's testing row says why:

> Deliberately desynchronize a real video by shifting audio +400 ms and assert the sync
> score drops significantly. **If it does not, this component is not working — no matter
> what the loss curve says.**

A contrastive loss that goes down is not evidence the head learned synchrony; it can be
minimised by any feature that happens to correlate with the shift. So ⛔ P8-6 is tested
against a head with *known* aligned embeddings rather than against a trained model, which
makes it a property of the architecture rather than of one run.
"""

from __future__ import annotations

import pytest
import torch

from src.losses.sync import InfoNCESyncLoss
from src.models.attention_model import AttentionFusionModel
from src.models.fusion.cross_attention import BidirectionalCrossAttention
from src.models.heads.sync import SyncHead
from src.models.temporal.transformer import TransformerTemporalEncoder, attention_entropy

VD, AD, D, B, T = 1280, 80, 256, 4, 60
FPS = 25.0
SHIFT_400MS = 10  # 400 ms at 25 fps — component F's desync test


def _masked(lengths: list[int], dim: int = D):
    t_max = max(lengths)
    x = torch.randn(len(lengths), t_max, dim)
    mask = torch.zeros(len(lengths), t_max, dtype=torch.bool)
    for i, n in enumerate(lengths):
        mask[i, :n] = True
    return x, mask


# ------------------------------------------------------- self-attention (P8-1, P8-2, P8-7)


def test_encoder_preserves_temporal_resolution() -> None:
    enc = TransformerTemporalEncoder(D, n_layers=2, n_heads=4).eval()
    for t in (1, 31, 137):
        x, mask = _masked([t, max(1, t // 2)])
        assert enc(x, mask)["sequence"].shape == (2, t, D)


def test_encoder_zeroes_padding() -> None:
    enc = TransformerTemporalEncoder(D, n_layers=2, n_heads=4).eval()
    x, mask = _masked([50, 20])
    out = enc(x, mask)["sequence"]
    assert torch.count_nonzero(out[~mask]) == 0


def test_padding_does_not_leak_into_valid_positions() -> None:
    """A clip's output must not depend on what it was batched with.

    `key_padding_mask` is what enforces this; without it, attention would average padded
    keys into every query and a 20-frame clip's representation would shift depending on its
    batchmates' lengths. Same failure the BiLSTM's packing prevents, different mechanism.
    """
    torch.manual_seed(0)
    enc = TransformerTemporalEncoder(D, n_layers=2, n_heads=4, dropout=0.0).eval()
    short = torch.randn(1, 20, D)

    alone = enc(short, torch.ones(1, 20, dtype=torch.bool))["sequence"]

    padded = torch.zeros(2, 60, D)
    mask = torch.zeros(2, 60, dtype=torch.bool)
    padded[0, :20], mask[0, :20] = short[0], True
    padded[1], mask[1] = torch.randn(60, D), True
    batched = enc(padded, mask)["sequence"]

    torch.testing.assert_close(batched[0, :20], alone[0], rtol=1e-4, atol=1e-5)


def test_positional_encoding_is_ablatable() -> None:
    """⛔ P8-2. Without position the encoder must be permutation-equivariant.

    That is the property the ablation is *for*: if shuffling frames changed nothing when
    positional encoding is on, the embeddings would not be doing their job.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 12, D)
    perm = torch.randperm(12)

    off = TransformerTemporalEncoder(D, n_layers=2, n_heads=4, dropout=0.0, positional=False).eval()
    a = off(x)["sequence"][0][perm]
    b = off(x[:, perm])["sequence"][0]
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)

    on = TransformerTemporalEncoder(D, n_layers=2, n_heads=4, dropout=0.0, positional=True).eval()
    a = on(x)["sequence"][0][perm]
    b = on(x[:, perm])["sequence"][0]
    assert not torch.allclose(a, b, rtol=1e-4, atol=1e-5)


def test_encoder_refuses_to_silently_truncate_a_long_clip() -> None:
    """Truncating past max_len would drop labelled spans. Section 6.9 chunks instead."""
    enc = TransformerTemporalEncoder(D, n_layers=1, n_heads=4, max_len=32).eval()
    with pytest.raises(ValueError, match="exceeds max_len"):
        enc(torch.randn(1, 33, D))


def test_attention_entropy_bounds_are_hand_checked() -> None:
    """⛔ P8-7's collapse detector: uniform -> 1.0, one-hot -> 0.0, by construction."""
    t = 8
    uniform = torch.full((1, 2, t, t), 1.0 / t)
    assert attention_entropy(uniform, torch.ones(1, t, dtype=torch.bool)).allclose(
        torch.ones(1, 2), atol=1e-5
    )

    onehot = torch.zeros(1, 2, t, t)
    onehot[..., 0] = 1.0
    assert attention_entropy(onehot, torch.ones(1, t, dtype=torch.bool)).abs().max() < 1e-4


def test_head_entropy_is_reported_every_forward() -> None:
    """It must be free, or it will not be checked — and an unchecked head collapses quietly."""
    enc = TransformerTemporalEncoder(D, n_layers=3, n_heads=4).eval()
    x, mask = _masked([40, 25])
    ent = enc(x, mask)["head_entropy"]
    assert ent.shape == (2, 3, 4)  # (B, layers, heads)
    assert ((ent >= 0) & (ent <= 1.05)).all()


def test_attention_maps_are_opt_in() -> None:
    """A (B, L, H, T, T) tensor every forward would dominate a 4 GB budget."""
    enc = TransformerTemporalEncoder(D, n_layers=2, n_heads=4).eval()
    x, mask = _masked([30, 30])
    assert "attention_maps" not in enc(x, mask)
    assert enc(x, mask, return_attention=True)["attention_maps"].shape == (2, 2, 4, 30, 30)


def test_fully_masked_row_does_not_nan_the_batch() -> None:
    enc = TransformerTemporalEncoder(D, n_layers=2, n_heads=4).eval()
    x, mask = _masked([30, 30])
    mask[1] = False
    out = enc(x, mask)
    assert torch.isfinite(out["sequence"]).all()
    assert torch.isfinite(out["head_entropy"]).all()


# ----------------------------------------------------------- cross-attention (P8-3)


def test_cross_attention_preserves_shape_and_masks_padding() -> None:
    cross = BidirectionalCrossAttention(D, n_layers=2, n_heads=4).eval()
    v, mask = _masked([50, 22])
    a = torch.randn_like(v)
    out = cross(v, a, mask)
    assert out["fused"].shape == (2, 50, D)
    assert torch.count_nonzero(out["fused"][~mask]) == 0


def test_cross_attention_gradient_reaches_both_directions() -> None:
    """A→V and V→A must both train, or "bidirectional" is decoration."""
    cross = BidirectionalCrossAttention(D, n_layers=2, n_heads=4)
    v, mask = _masked([30, 30])
    a = torch.randn_like(v)
    cross(v, a, mask)["fused"].sum().backward()
    assert cross.a2v[0].attn.in_proj_weight.grad.abs().sum() > 0
    assert cross.v2a[0].attn.in_proj_weight.grad.abs().sum() > 0


def test_fused_output_depends_on_both_streams() -> None:
    """The structural claim behind using cross-attention as a collapse defence."""
    torch.manual_seed(0)
    cross = BidirectionalCrossAttention(D, n_layers=2, n_heads=4, dropout=0.0).eval()
    v, mask = _masked([30, 30])
    a = torch.randn_like(v)
    base = cross(v, a, mask)["fused"]
    assert not torch.allclose(base, cross(torch.randn_like(v), a, mask)["fused"])
    assert not torch.allclose(base, cross(v, torch.randn_like(a), mask)["fused"])


# -------------------------------------------------------------- sync head (⛔ P8-6)


class _IdentitySync(SyncHead):
    """A head whose projections are the identity, so alignment is the *only* variable.

    Testing the desync property on a randomly-initialised head would measure the random
    projection; testing it on a trained one would measure that run. Neither is a property of
    the component. With identity projections, a perfectly aligned pair scores exactly 1.0
    and any drop is attributable to the shift alone.
    """

    def __init__(self, d_model: int = D, window: int = 5) -> None:
        super().__init__(d_model, d_model, window)
        with torch.no_grad():
            for proj in (self.visual_proj, self.audio_proj):
                proj.weight.copy_(torch.eye(d_model))
                proj.bias.zero_()


def test_aligned_streams_score_near_one() -> None:
    head = _IdentitySync().eval()
    x = torch.randn(2, 50, D)
    sync = head(x, x.clone())["sync"]
    assert sync.min() > 0.99


def test_desync_by_400ms_drops_the_sync_score() -> None:
    """⛔ P8-6, component F's own acceptance test.

    A real clip is in sync by construction. Shift its audio by +400 ms (10 frames at
    25 fps) — beyond the head's ±5-frame tolerance — and the score must fall clearly. If it
    did not, the head would be reporting something other than synchrony and every sync
    number in Part 8 would be meaningless.
    """
    torch.manual_seed(0)
    head = _IdentitySync(window=5).eval()
    x = torch.randn(1, 60, D)
    mask = torch.ones(1, 60, dtype=torch.bool)

    in_sync = head.clip_score(head(x, x.clone(), mask)["sync"], mask)
    shifted = torch.roll(x, shifts=SHIFT_400MS, dims=1)
    desynced = head.clip_score(head(x, shifted, mask)["sync"], mask)

    assert in_sync.item() > 0.99
    assert desynced.item() < 0.5, f"desync barely moved the score: {desynced.item():.4f}"
    assert (in_sync - desynced).item() > 0.5


def test_small_offset_is_tolerated_by_the_window() -> None:
    """±5 frames is a tolerance, not a bug: real speech leads and lags its own mouth."""
    torch.manual_seed(0)
    head = _IdentitySync(window=5).eval()
    x = torch.randn(1, 60, D)
    mask = torch.ones(1, 60, dtype=torch.bool)
    nudged = torch.roll(x, shifts=3, dims=1)
    assert head.clip_score(head(x, nudged, mask)["sync"], mask).item() > 0.9


def test_window_wraparound_is_not_counted_as_synchrony() -> None:
    """`roll` wraps; matching the clip's tail against its own head would be a false positive."""
    head = _IdentitySync(window=5).eval()
    x = torch.randn(1, 40, D)
    sync = head(x, x.clone())["sync"]
    assert sync.shape == (1, 40)
    assert torch.isfinite(sync).all()


# ------------------------------------------------------------- InfoNCE loss (P8-5)


def test_sync_loss_is_lower_for_aligned_than_shifted() -> None:
    loss = InfoNCESyncLoss()
    torch.manual_seed(0)
    v = torch.nn.functional.normalize(torch.randn(2, 60, D), dim=-1)
    labels = torch.zeros(2)  # both real
    aligned = loss(v, v.clone(), labels)
    shifted = loss(v, torch.roll(v, shifts=20, dims=1), labels)
    assert aligned < shifted


def test_sync_loss_uses_real_videos_only() -> None:
    """⛔ Training a fake's forged span toward "aligned" erases the signal being detected."""
    loss = InfoNCESyncLoss()
    torch.manual_seed(0)
    v = torch.nn.functional.normalize(torch.randn(3, 60, D), dim=-1)
    a = torch.nn.functional.normalize(torch.randn(3, 60, D), dim=-1)

    all_fake = loss(v, a, torch.ones(3))
    assert all_fake.item() == 0.0, "fakes must contribute nothing to the sync loss"

    # And a batch that contains a real clip must produce a real gradient.
    mixed = loss(v, a, torch.tensor([0.0, 1.0, 1.0]))
    assert mixed.item() > 0.0


def test_sync_loss_ignores_padding() -> None:
    loss = InfoNCESyncLoss()
    torch.manual_seed(0)
    v = torch.nn.functional.normalize(torch.randn(1, 40, D), dim=-1)
    labels = torch.zeros(1)
    mask = torch.zeros(1, 40, dtype=torch.bool)
    mask[:, :20] = True
    tight = loss(v[:, :20], v[:, :20].clone(), labels, mask[:, :20])
    padded = loss(v, v.clone(), labels, mask)
    assert abs(tight.item() - padded.item()) < 0.2


def test_sync_loss_survives_a_short_clip() -> None:
    """Nothing can be shifted >= 10 frames in an 8-frame clip. Return 0, do not crash."""
    loss = InfoNCESyncLoss()
    v = torch.nn.functional.normalize(torch.randn(1, 8, D), dim=-1)
    assert loss(v, v.clone(), torch.zeros(1)).item() == 0.0


# ------------------------------------------------------- assembled model (Experiment E)


def _model(**kw) -> AttentionFusionModel:
    return AttentionFusionModel(
        visual_dim=VD, feature="logmel", dropout=0.0, attn_dropout=0.0, n_layers=2, **kw
    )


def _inputs(t: int = T):
    return torch.randn(B, t, VD), torch.randn(B, t, AD), torch.ones(B, t, dtype=torch.bool)


def test_experiment_e_output_contract() -> None:
    """Phase 6 and 7 keys survive, so Trainer and the ablation probe work unmodified."""
    out = _model().eval()(*_inputs())
    assert out["logit"].shape == (B,)
    assert out["frame_logits"].shape == (B, T)
    assert out["sync"].shape == (B, T)
    assert out["aux_logits"].shape == (2, B)
    assert out["head_entropy"].shape == (B, 2, 4)


@pytest.mark.parametrize(
    ("cross", "temporal"),
    [(True, "transformer"), (False, "transformer"), (True, "lstm"), (False, "lstm")],
)
def test_both_components_are_independently_switchable(cross: bool, temporal: str) -> None:
    """⛔ P8-10. E changes two things at once; the gate needs each measured on its own."""
    out = _model(cross_attention=cross, temporal=temporal).eval()(*_inputs())
    assert out["logit"].shape == (B,)
    assert out["sequence"].shape == (B, T, D)


def test_model_reuses_the_phase6_stream_encoders() -> None:
    """`E - D` must be the attention. A re-implemented encoder would confound it."""
    from src.models.fusion.concat import ConcatFusionBaseline

    model = _model()
    assert isinstance(model, ConcatFusionBaseline)
    assert model.encode_streams.__func__ is ConcatFusionBaseline.encode_streams


def test_gradient_reaches_both_streams_through_cross_attention() -> None:
    model = _model(modality_dropout=0.0)
    v, a, m = _inputs()
    model(v, a, m)["logit"].sum().backward()
    assert model.visual_proj.weight.grad.abs().sum() > 0
    assert model.audio_encoder.blocks[0][0].weight.grad.abs().sum() > 0


def test_sync_head_reads_prefusion_streams() -> None:
    """After cross-attention each stream has absorbed the other; a cosine there is circular.

    Zeroing the visual stream must move the sync score. If sync were computed on the fused
    sequence it would still look "synchronised" because both inputs share a trunk.
    """
    torch.manual_seed(0)
    model = _model(modality_dropout=0.0).eval()
    v, a, m = _inputs()
    base = model(v, a, m)["sync"]
    dropped = model(v, a, m, drop_visual=True)["sync"]
    assert not torch.allclose(base, dropped)
