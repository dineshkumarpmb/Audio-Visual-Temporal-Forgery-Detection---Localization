"""P5-1 / P5-2: the audio encoder must preserve temporal resolution exactly.

Section E makes temporal resolution the whole point of this module: "every temporal
downsample you apply is localization precision you can never recover". A regression here
would not crash anything — it would silently shift Phase 10's per-frame targets against the
features they label, and the first symptom would be an unexplainable localization score
weeks later. So the length contract is tested directly, at the sizes most likely to break
it, rather than assumed from the layer definitions.
"""

from __future__ import annotations

import pytest
import torch

from src.models.backbones.audio import DILATIONS, DilatedAudioEncoder, build_audio_encoder
from src.models.heads.classification import AudioBaseline


@pytest.mark.parametrize("t", [1, 2, 15, 30, 31, 32, 146, 500])
@pytest.mark.parametrize("feature,dim", [("logmel", 80), ("mfcc", 40)])
def test_output_length_equals_input_length(t: int, feature: str, dim: int) -> None:
    """⛔ P5-2, the gate item. Includes T below the 31-frame receptive field."""
    enc = build_audio_encoder(feature).eval()
    out = enc(torch.randn(2, t, dim))
    assert out.shape == (2, t, 256), f"{feature} T={t}: got {tuple(out.shape)}"


def test_padding_equals_dilation_not_one() -> None:
    """PF-18: section E's literal `pad=1` would shrink the sequence by 22 frames.

    Guards the specific mistake the plan's text invites, by checking the built layers
    rather than the output — a future refactor could preserve length some other way and
    still deserve to pass, but padding != dilation with k=3 cannot.
    """
    enc = build_audio_encoder("logmel")
    convs = [m for m in enc.modules() if isinstance(m, torch.nn.Conv1d)]
    assert len(convs) == len(DILATIONS)
    for conv, dilation in zip(convs, DILATIONS, strict=True):
        assert conv.dilation == (dilation,)
        assert conv.padding == (dilation,), (
            f"dilation {dilation} needs padding {dilation}, got {conv.padding[0]}"
        )
        assert conv.stride == (1,), "stride must be 1 throughout (section E)"

    # And the arithmetic the plan got wrong, stated once so the reason is visible.
    shrink = sum(2 * d - 2 * 1 for d in DILATIONS)
    assert shrink == 22


def test_receptive_field_matches_section_e() -> None:
    """1 + 2*(1+2+4+8) = 31 frames = 1.24 s at the 40 ms grid."""
    enc = build_audio_encoder("logmel")
    assert enc.receptive_field == 31
    assert enc.receptive_field * 0.04 == pytest.approx(1.24)


def test_padding_does_not_leak_into_valid_positions() -> None:
    """A clip's embeddings must not depend on what else shares its batch.

    Dilation reaches 8 frames, so without re-masking after each block a padded neighbour
    would bleed into the last valid frames — fabricated evidence at exactly the clip
    boundary. Eval mode so BatchNorm uses running statistics and the comparison is fair.
    """
    torch.manual_seed(0)
    enc = build_audio_encoder("logmel").eval()
    short = torch.randn(1, 40, 80)

    alone = enc(short, torch.ones(1, 40, dtype=torch.bool))

    padded = torch.zeros(1, 120, 80)
    padded[:, :40] = short
    mask = torch.zeros(1, 120, dtype=torch.bool)
    mask[:, :40] = True
    batched = enc(padded, mask)

    torch.testing.assert_close(alone, batched[:, :40], rtol=1e-5, atol=1e-5)
    assert batched[:, 40:].abs().max() == 0.0, "padded positions must stay zero"


def test_gradient_reaches_every_block() -> None:
    """Section E's gradient-flow check. A dead block would not show up in the loss curve."""
    model = AudioBaseline("logmel", dropout=0.0)
    out = model(torch.randn(4, 60, 80), torch.ones(4, 60, dtype=torch.bool))
    out["logit"].sum().backward()

    for i, block in enumerate(model.encoder.blocks):
        conv = block[0]
        assert conv.weight.grad is not None, f"block {i} received no gradient"
        assert conv.weight.grad.abs().sum() > 0, f"block {i} gradient is all zero"


def test_rejects_wrong_input_dim() -> None:
    """An MFCC cache fed to a log-mel model is a silent 40-vs-80 disaster otherwise."""
    enc = build_audio_encoder("logmel")
    with pytest.raises(ValueError, match="input dim 40"):
        enc(torch.randn(2, 50, 40))


def test_group_norm_arm_also_preserves_length() -> None:
    """The `--norm group` ablation exists because BatchNorm statistics see padding."""
    enc = DilatedAudioEncoder(80, norm="group").eval()
    assert enc(torch.randn(2, 77, 80)).shape == (2, 77, 256)


def test_audio_baseline_shapes_and_masking() -> None:
    model = AudioBaseline("mfcc", dropout=0.0).eval()
    mask = torch.ones(3, 50, dtype=torch.bool)
    mask[1, 20:] = False
    out = model(torch.randn(3, 50, 40), mask)

    assert out["logit"].shape == (3,)
    assert out["sequence"].shape == (3, 50, 256)
    assert out["attention"].shape == (3, 50)
    # Attention must place no weight on padding.
    assert out["attention"][1, 20:].abs().max() < 1e-6
    torch.testing.assert_close(out["attention"].sum(dim=1), torch.ones(3))
