"""P15-1: shape contracts across every model that feeds a result, in one place.

The rule that matters most is section 7's "assert output length == input length": a
temporal-resolution regression (a stride, a pooling, an off-by-one pad) silently shifts every
frame score against its label. Each model with a frame output is checked at awkward lengths
-- 1 frame, odd, the LAV-DF median, and the positional table's limit -- with ragged batches,
and padding must not change a valid frame's output.
"""

from __future__ import annotations

import pytest
import torch

from src.models.attention_model import AttentionFusionModel
from src.models.heads.classification import AudioBaseline, VisualBaseline
from src.models.temporal.lstm import TemporalFusionBaseline
from src.models.temporal.transformer import MAX_FRAMES

AUDIO_DIM = 80  # log-Mel bins (C-1)
VISUAL_DIM = 1280  # MobileNetV2 (D-1)
LENGTHS = [1, 7, 50, 187, MAX_FRAMES]


def batch(t: int, b: int = 2, *, ragged: bool = True):
    torch.manual_seed(0)
    visual = torch.randn(b, t, VISUAL_DIM)
    audio = torch.randn(b, t, AUDIO_DIM)
    mask = torch.ones(b, t, dtype=torch.bool)
    if ragged and b > 1 and t > 1:
        mask[1, max(1, t // 2) :] = False  # second clip is shorter
    face = torch.ones(b, t, dtype=torch.bool)
    return visual, audio, mask, face


ATTENTION_CONFIGS = [
    pytest.param(
        {"cross_attention": True, "temporal": "transformer", "boundary": True}, id="G-full"
    ),
    pytest.param({"cross_attention": False, "temporal": "transformer"}, id="concat+transformer"),
    pytest.param({"cross_attention": True, "temporal": "lstm"}, id="cross+bilstm"),
]


@pytest.mark.parametrize("cfg", ATTENTION_CONFIGS)
@pytest.mark.parametrize("t", LENGTHS)
def test_attention_model_preserves_length(cfg, t):
    model = AttentionFusionModel(**cfg).eval()
    with torch.no_grad():
        out = model(*batch(t))
    assert out["logit"].shape == (2,)
    assert out["frame_logits"].shape == (2, t), "output length must equal input length"
    if cfg.get("boundary"):
        assert out["boundary_logits"].shape == (2, t, 2)
    assert torch.isfinite(out["frame_logits"]).all() and torch.isfinite(out["logit"]).all()


@pytest.mark.parametrize("cfg", ATTENTION_CONFIGS)
def test_attention_model_padding_cannot_change_valid_frames(cfg):
    """Garbage in the padded tail of clip 2 must not reach clip 2's valid frames."""
    model = AttentionFusionModel(**cfg).eval()
    visual, audio, mask, face = batch(60)
    with torch.no_grad():
        a = model(visual, audio, mask, face)
        visual2, audio2 = visual.clone(), audio.clone()
        visual2[1, 30:] = 1e3
        audio2[1, 30:] = -1e3
        b = model(visual2, audio2, mask, face)
    torch.testing.assert_close(a["frame_logits"][1, :30], b["frame_logits"][1, :30])
    torch.testing.assert_close(a["logit"], b["logit"])


def test_attention_model_refuses_past_the_positional_table():
    """Past MAX_FRAMES it must fail loudly; the pipeline chunks instead (P10-11)."""
    model = AttentionFusionModel().eval()
    with pytest.raises((ValueError, RuntimeError)), torch.no_grad():
        model(*batch(MAX_FRAMES + 1, b=1, ragged=False))


def test_batch_of_one_matches_the_same_clip_in_a_batch():
    model = AttentionFusionModel().eval()
    visual, audio, mask, face = batch(40)
    with torch.no_grad():
        both = model(visual, audio, mask, face)["frame_logits"]
        one = model(visual[:1], audio[:1], mask[:1], face[:1])["frame_logits"]
    torch.testing.assert_close(both[:1], one, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("t", LENGTHS)
def test_bilstm_baseline_preserves_length(t):
    model = TemporalFusionBaseline(visual_dim=VISUAL_DIM, feature="logmel").eval()
    visual, audio, mask, face = batch(t)
    with torch.no_grad():
        out = model(visual, audio, mask, face)
    assert out["frame_logits"].shape == (2, t)
    assert out["logit"].shape == (2,)


@pytest.mark.parametrize("t", [1, 50, 187])
def test_single_modality_baselines_give_one_logit_per_clip(t):
    visual, audio, mask, _ = batch(t)
    with torch.no_grad():
        v = VisualBaseline(feature_dim=VISUAL_DIM).eval()(visual, mask)
        a = AudioBaseline(feature="logmel").eval()(audio, mask)
    assert v["logit"].shape == (2,) and a["logit"].shape == (2,)
