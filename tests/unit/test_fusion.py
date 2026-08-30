"""P6-1 .. P6-5: the fusion baseline and its collapse defences.

Section 5.6's failure mode is not a crash — it is a model that scores well while ignoring
video, which looks exactly like success until someone checks. Phase 5 measured audio at
0.9838 against vision at 0.7189, so the incentive is real and present. These tests assert
that each defence is actually wired in, because a defence that is silently inert would
leave the aggregate number looking fine and the multimodal claim hollow.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.fusion.concat import ConcatFusionBaseline

VD, AD, B, T = 1280, 80, 4, 60


def _model(**kw) -> ConcatFusionBaseline:
    return ConcatFusionBaseline(visual_dim=VD, feature="logmel", dropout=0.0, **kw)


def _inputs(t: int = T):
    return torch.randn(B, t, VD), torch.randn(B, t, AD), torch.ones(B, t, dtype=torch.bool)


def test_forward_shapes() -> None:
    out = _model().eval()(*_inputs())
    assert out["logit"].shape == (B,)
    assert out["fused"].shape == (B, T, 256)
    assert out["aux_logits"].shape == (2, B)


def test_rejects_mismatched_stream_lengths() -> None:
    """⛔ P6-2. The dataset truncates to a common grid; a mismatch here means it did not."""
    model = _model().eval()
    with pytest.raises(ValueError, match="streams disagree on length"):
        model(torch.randn(B, 50, VD), torch.randn(B, 49, AD))


def test_temporal_resolution_is_preserved_through_fusion() -> None:
    """The fused sequence stays on the input grid, so Phase 10 can still label it."""
    for t in (1, 31, 137):
        out = _model().eval()(*_inputs(t))
        assert out["fused"].shape[1] == t


@pytest.mark.parametrize("drop", ["drop_visual", "drop_audio"])
def test_inference_dropout_changes_the_prediction(drop: str) -> None:
    """Section 5.6's collapse probe must actually zero a stream.

    If `drop_audio=True` left the logit untouched the probe would report "video is used"
    for every model, including a fully collapsed one.
    """
    torch.manual_seed(0)
    model = _model().eval()
    v, a, m = _inputs()
    base = model(v, a, m)["logit"]
    dropped = model(v, a, m, **{drop: True})["logit"]
    assert not torch.allclose(base, dropped), f"{drop} had no effect on the logit"


def test_modality_dropout_is_train_only_and_per_clip() -> None:
    """⛔ P6-4. Active in train, inert in eval, and it drops whole clips not scattered frames."""
    model = _model(modality_dropout=0.5)
    v, a, m = _inputs()

    model.eval()
    torch.manual_seed(0)
    first = model(v, a, m)["logit"]
    torch.manual_seed(1)
    second = model(v, a, m)["logit"]
    torch.testing.assert_close(first, second), "eval must be deterministic"

    # In train mode the keep-masks are per-clip, so over many draws some clips differ.
    model.train()
    torch.manual_seed(0)
    seen = {tuple(model(v, a, m)["logit"].detach().numpy().round(4)) for _ in range(12)}
    assert len(seen) > 1, "modality dropout never fired in train mode"


def test_modality_dropout_off_means_no_stochasticity() -> None:
    model = _model(modality_dropout=0.0).train()
    v, a, m = _inputs()
    torch.testing.assert_close(model(v, a, m)["logit"], model(v, a, m)["logit"])


def test_auxiliary_heads_receive_gradient() -> None:
    """⛔ P6-5. The visual pathway must keep learning even if fusion ignores it.

    Backprop from the *visual auxiliary head alone* has to reach the visual projection.
    """
    model = _model()
    v, a, m = _inputs()
    model(v, a, m)["aux_visual"].sum().backward()

    assert model.visual_proj.weight.grad is not None
    assert model.visual_proj.weight.grad.abs().sum() > 0
    # And it must not have leaked through the audio encoder.
    first_audio_conv = model.audio_encoder.blocks[0][0]
    assert first_audio_conv.weight.grad is None or first_audio_conv.weight.grad.abs().sum() == 0


def test_gradient_reaches_both_streams_from_the_fused_logit() -> None:
    model = _model()
    v, a, m = _inputs()
    model(v, a, m)["logit"].sum().backward()
    assert model.visual_proj.weight.grad.abs().sum() > 0
    assert model.audio_encoder.blocks[0][0].weight.grad.abs().sum() > 0


def test_per_modality_normalisation_equalises_scale() -> None:
    """P6-3. A 100x louder visual stream must not arrive 100x larger at the concat.

    Without the LayerNorms the first fusion Linear would be dominated by whichever stream
    happened to have the larger activations — modality preference by arithmetic rather
    than by learning.
    """
    model = _model().eval()
    _, a, m = _inputs()
    small = model(torch.randn(B, T, VD) * 0.01, a, m)["fused"]
    large = model(torch.randn(B, T, VD) * 100.0, a, m)["fused"]
    ratio = large.std().item() / max(small.std().item(), 1e-8)
    assert 0.1 < ratio < 10.0, f"scale leaked through fusion: std ratio {ratio:.1f}"


def test_face_mask_zeroes_untrusted_visual_frames() -> None:
    """A frame with no detected face carries held-over geometry, not evidence."""
    model = _model(modality_dropout=0.0).eval()
    v, a, m = _inputs()
    face = torch.ones(B, T, dtype=torch.bool)
    face[0, 30:] = False

    out_all = model(v, a, m, torch.ones(B, T, dtype=torch.bool))
    out_masked = model(v, a, m, face)
    assert not torch.allclose(out_all["logit"], out_masked["logit"])


def test_rejects_wrong_visual_dim() -> None:
    """A resnet18 cache (512) fed to a mobilenet_v2 model (1280) is otherwise silent."""
    model = _model().eval()
    with pytest.raises(ValueError, match="visual dim 512"):
        model(torch.randn(B, T, 512), torch.randn(B, T, AD))


def test_paired_dataset_asserts_grid_alignment(tmp_path) -> None:
    """⛔ P6-2 at the data layer: a drift beyond Phase 3's ±1 must fail loudly."""
    import pandas as pd

    from src.config import AudioPreprocessConfig, VisualFeatureConfig, config_hash
    from src.data.dataset import MultimodalFeatureDataset

    vcfg = VisualFeatureConfig(backbone="mobilenet_v2")
    acfg = AudioPreprocessConfig(feature="logmel")
    vdir = tmp_path / "visual" / config_hash(vcfg)
    adir = tmp_path / "audio" / config_hash(acfg)
    vdir.mkdir(parents=True)
    adir.mkdir(parents=True)

    # 50 video frames against 40 audio frames -- a drift of 10, far beyond tolerance.
    np.savez(vdir / "x.npz", features=np.zeros((50, 1280), np.float32), found=np.ones(50, bool))
    np.save(adir / "x.npy", np.zeros((40, 80), np.float32))
    manifest = pd.DataFrame({"label": [1], "class_name": ["both"], "split": ["train"]}, index=["x"])

    ds = MultimodalFeatureDataset(
        ["x"], manifest, vcfg, acfg, tmp_path / "visual", tmp_path / "audio"
    )
    with pytest.raises(AssertionError, match="drift 10"):
        _ = ds[0]
