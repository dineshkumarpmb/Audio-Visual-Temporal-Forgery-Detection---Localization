"""P9-3 / P9-4: the feature-space GAN and the collapse gate that guards its output.

**The gate is the point of these tests, not the architecture.** Section I predicts mode
collapse, and risk R7 rates "the GAN provides no benefit" as *High* likelihood. Both are
survivable — a negative F2 vs F1 is a publishable result. What is not survivable is
*undetected* collapse: near-identical generated vectors are a perfectly learnable marker, so
F2 could beat F1 while the augmentation is worthless. So the tests assert that a degenerate
generator is refused, that a memorising one is refused, and that the refusal is an exception
rather than a warning nobody reads.
"""

from __future__ import annotations

import pytest
import torch

from src.models.gan.discriminator import SequenceDiscriminator
from src.models.gan.generator import SequenceGenerator
from src.training.gan import (
    MIN_STD_RATIO,
    Diversity,
    ModeCollapseError,
    diversity,
    generate_samples,
    train_gan,
)

D, T, B = 256, 40, 8


# ------------------------------------------------------------------- generator (P9-3)


def test_generator_shape_and_variable_length() -> None:
    gen = SequenceGenerator(d_model=D)
    for n_frames in (1, 17, 120):
        out = gen(torch.randn(B, gen.noise_dim), torch.zeros(B, dtype=torch.long), n_frames)
        assert out.shape == (B, n_frames, D)


def test_label_conditioning_changes_the_output() -> None:
    """Without this, the generator has learned one blurred distribution, not two."""
    gen = SequenceGenerator(d_model=D)
    noise = torch.randn(B, gen.noise_dim)
    real_like = gen(noise, torch.zeros(B, dtype=torch.long), T)
    fake_like = gen(noise, torch.ones(B, dtype=torch.long), T)
    assert not torch.allclose(real_like, fake_like)


def test_noise_changes_the_output() -> None:
    gen = SequenceGenerator(d_model=D)
    labels = torch.zeros(B, dtype=torch.long)
    a = gen(torch.randn(B, gen.noise_dim), labels, T)
    b = gen(torch.randn(B, gen.noise_dim), labels, T)
    assert not torch.allclose(a, b)


def test_generator_rejects_wrong_noise_width() -> None:
    gen = SequenceGenerator(d_model=D)
    with pytest.raises(ValueError, match="noise"):
        gen(torch.randn(B, gen.noise_dim + 1), torch.zeros(B, dtype=torch.long), T)


# --------------------------------------------------------------- discriminator (P9-3)


def test_discriminator_is_spectrally_normalised() -> None:
    """§I names spectral norm as the collapse defence; assert it is actually applied."""
    disc = SequenceDiscriminator(d_model=D)
    normalised = [n for n, _ in disc.named_parameters() if n.endswith(("_orig", "_u", "_v"))]
    assert normalised, "no spectrally normalised parameters found"


def test_padding_does_not_vote() -> None:
    """Changing what sits in the padding must not move the logit at all.

    Stated as a comparison between two paddings rather than against an unpadded clip, which
    isolates the property that matters: whatever the padded positions contain, the verdict is
    the same. This started as a tolerance — masking only at the pool left the dilated
    convolutions reading padding near the seam, worth 0.78 of logit against a signal scale of
    ~2 — and became an equality once the discriminator learned to zero its input first.

    The discriminator is warmed up in train mode first. `spectral_norm` estimates the largest
    singular value by power iteration during training; straight out of `__init__` that
    estimate is a random vector, the divisor is far too small, and the logits are absurd
    (~1e6). That is an artefact of never training, not of the architecture.
    """
    torch.manual_seed(0)
    disc = SequenceDiscriminator(d_model=D)
    labels = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        for _ in range(20):  # let power iteration converge
            disc(torch.randn(1, T, D), labels)

    disc.eval()
    seq = torch.randn(1, T, D)
    mask = torch.zeros(1, T + 15, dtype=torch.bool)
    mask[:, :T] = True
    pad_a = torch.cat([seq, torch.randn(1, 15, D) * 50], dim=1)
    pad_b = torch.cat([seq, torch.randn(1, 15, D) * -30], dim=1)
    with torch.no_grad():
        masked = float((disc(pad_a, labels, mask) - disc(pad_b, labels, mask)).abs())
        unmasked = float((disc(pad_a, labels) - disc(pad_b, labels)).abs())
    assert masked == 0.0, f"padding contents moved the masked logit by {masked}"
    assert unmasked > 0.0, "the test's two paddings are indistinguishable -- it proves nothing"


# ------------------------------------------------------------------ collapse gate (P9-4)


def test_constant_output_is_collapsed() -> None:
    real = torch.randn(16, T, D)
    assert diversity(torch.ones(8, T, D) * 0.5, real).collapsed


def test_memorised_samples_are_collapsed() -> None:
    """The opposite failure: perfect diversity, zero novelty."""
    real = torch.randn(16, T, D)
    div = diversity(real[:8].clone(), real)
    assert div.nn_distance < 1e-5
    assert div.collapsed


def test_healthy_samples_pass() -> None:
    real = torch.randn(32, T, D)
    fake = torch.randn(16, T, D)
    div = diversity(fake, real)
    assert not div.collapsed, div.as_dict()


def test_generate_samples_refuses_a_collapsed_generator() -> None:
    """⛔ An untrained generator has a tiny output spread; it must not reach training."""
    real = torch.randn(32, T, D)
    with pytest.raises(ModeCollapseError, match="failed P9-4"):
        generate_samples(SequenceGenerator(d_model=D), real, 8, T)


def test_diversity_needs_more_than_one_sample() -> None:
    with pytest.raises(ValueError, match="at least two"):
        diversity(torch.randn(1, T, D), torch.randn(4, T, D))


def test_thresholds_are_reported_in_the_verdict() -> None:
    div = Diversity(
        pairwise=0.5,
        pairwise_real=0.5,
        std_ratio=MIN_STD_RATIO / 2,
        nn_distance=0.5,
        nn_real=0.5,
        n_samples=8,
    )
    assert div.collapsed and div.as_dict()["collapsed"] is True


def test_thresholds_are_relative_to_the_real_data() -> None:
    """⛔ The calibration bug this project actually hit.

    Real fused embeddings here sit in a narrow cone -- 200 clips measured a mean pairwise
    cosine distance of 0.0264, median 0.0143. An absolute floor of 0.02 would reject a
    generator that matched the data exactly. A generator as varied as its reference must
    pass, however concentrated that reference is.
    """
    narrow = Diversity(
        pairwise=0.026,
        pairwise_real=0.026,
        std_ratio=1.0,
        nn_distance=0.026,
        nn_real=0.026,
        n_samples=64,
    )
    assert not narrow.collapsed, narrow.as_dict()
    assert narrow.pairwise_ratio == pytest.approx(1.0)

    # Same absolute numbers, but the data is ten times more spread out: now it is collapse.
    against_wide = Diversity(
        pairwise=0.026,
        pairwise_real=0.26,
        std_ratio=1.0,
        nn_distance=0.026,
        nn_real=0.026,
        n_samples=64,
    )
    assert against_wide.collapsed


# ----------------------------------------------------------------------- training loop


def test_train_gan_runs_and_logs_diversity() -> None:
    seqs = [torch.randn(60, D) for _ in range(12)]
    labels = torch.randint(0, 2, (12,))
    gen, disc, history = train_gan(
        seqs, labels, d_model=D, steps=4, batch_size=4, n_frames=16, log_every=2, verbose=False
    )
    assert isinstance(gen, SequenceGenerator) and isinstance(disc, SequenceDiscriminator)
    assert history and {"step", "loss_d", "loss_g", "pairwise", "collapsed"} <= set(history[0])
