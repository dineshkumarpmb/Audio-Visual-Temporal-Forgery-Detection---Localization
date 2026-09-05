"""P9-1 / P9-2: splicing, the classical stack, and the boundary contract they rest on.

**The whole value of splicing is that the boundaries are exact.** If they are not, the
augmentation is worse than useless: it manufactures thousands of examples whose labels are
off by a frame and trains the localizer to be wrong in a consistent direction. So the
round-trip through `frame_targets` is asserted here across a wide sweep of positions and
lengths, not spot-checked -- the naive `(i/fps, (i+L)/fps)` encoding fails for roughly 9% of
them, which is exactly the kind of bug that never surfaces in an aggregate metric.

**The second failure mode is leakage.** A splice invents a new `video_id`, so a dev clip
spliced into training would defeat every id-based split check the project has. The guard is
asserted rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.dataset import PairedSample
from src.localization.targets import frame_targets
from src.training.augment import (
    ClassicalConfig,
    SpliceConfig,
    assert_exact_boundaries,
    classical,
    span_to_seconds,
    spec_augment,
    splice,
    temporal_jitter,
)

VD, AD, FPS = 8, 4, 25.0


def make_sample(vid: str, t: int = 120, *, label: int = 0, fill: float = 1.0) -> PairedSample:
    return PairedSample(
        video_id=vid,
        visual=np.full((t, VD), fill, dtype=np.float32),
        audio=np.full((t, AD), fill, dtype=np.float32),
        mask=np.ones(t, dtype=bool),
        face=np.ones(t, dtype=bool),
        label=label,
        class_name="real",
        split="train",
        frames=np.zeros(t, dtype=np.float32),
    )


# ------------------------------------------------------------------ boundaries (P9-1)


def test_span_seconds_round_trip_exactly() -> None:
    """⛔ The contract: seconds -> frame_targets must reproduce the frames it came from.

    Swept rather than sampled. The naive encoding fails 1338 of these 15000 cases at 25 fps,
    every failure a silent 40 ms label error.
    """
    for start in range(0, 3000):
        for length in (1, 3, 16, 47, 120):
            end = start + length
            n = end + 5
            got = frame_targets([span_to_seconds(start, end, FPS)], n, FPS)
            expected = np.zeros(n, dtype=np.float32)
            expected[start:end] = 1.0
            assert np.array_equal(got, expected), f"[{start}, {end}) did not round-trip"


def test_assert_exact_boundaries_returns_the_target() -> None:
    target = assert_exact_boundaries(10, 26, 100, FPS)
    assert target.sum() == 16.0
    assert target[10] == 1.0 and target[25] == 1.0
    assert target[9] == 0.0 and target[26] == 0.0


def test_empty_span_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty span"):
        span_to_seconds(30, 30, FPS)


# --------------------------------------------------------------------- splicing (P9-1)


def test_splice_is_fake_with_exact_frame_targets() -> None:
    rng = np.random.default_rng(0)
    donor, recipient = make_sample("donor", fill=9.0), make_sample("recipient", fill=1.0)
    sample, prov = splice(donor, recipient, SpliceConfig(fps=FPS), rng)

    assert sample.label == 1
    assert sample.frames is not None
    assert sample.frames.sum() == prov.end_frame - prov.start_frame
    # The recorded seconds must reproduce the target the sample carries.
    recovered = frame_targets(prov.fake_periods, len(sample.mask), FPS)
    assert np.array_equal(recovered, sample.frames)


def test_splice_moves_only_the_chosen_modality() -> None:
    """A `visual` splice must leave audio untouched -- that is what makes it audio-proof."""
    for modality, weights in (("visual", (1.0, 0.0, 0.0)), ("audio", (0.0, 1.0, 0.0))):
        rng = np.random.default_rng(3)
        donor, recipient = make_sample("d", fill=9.0), make_sample("r", fill=1.0)
        sample, prov = splice(
            donor, recipient, SpliceConfig(fps=FPS, modality_weights=weights), rng
        )
        assert prov.modality == modality
        span = slice(prov.start_frame, prov.end_frame)
        if modality == "visual":
            assert np.all(sample.visual[span] == 9.0)
            assert np.all(sample.audio == 1.0), "audio must be untouched by a visual splice"
        else:
            assert np.all(sample.audio[span] == 9.0)
            assert np.all(sample.visual == 1.0), "visual must be untouched by an audio splice"


def test_splice_leaves_frames_outside_the_span_alone() -> None:
    rng = np.random.default_rng(11)
    donor, recipient = make_sample("d", fill=9.0), make_sample("r", fill=1.0)
    sample, prov = splice(
        donor, recipient, SpliceConfig(fps=FPS, modality_weights=(0, 0, 1.0)), rng
    )
    outside = np.ones(len(sample.mask), dtype=bool)
    outside[prov.start_frame : prov.end_frame] = False
    assert np.all(sample.visual[outside] == 1.0)
    assert np.all(sample.audio[outside] == 1.0)


def test_splice_refuses_a_fake_source() -> None:
    """⛔ Splicing into a fake unions with its existing spans -- boundaries stop being exact."""
    rng = np.random.default_rng(0)
    real, fake = make_sample("r"), make_sample("f", label=1)
    with pytest.raises(ValueError, match="two real clips"):
        splice(fake, real, SpliceConfig(fps=FPS), rng)
    with pytest.raises(ValueError, match="two real clips"):
        splice(real, fake, SpliceConfig(fps=FPS), rng)


def test_splice_carries_donor_trust_flags() -> None:
    rng = np.random.default_rng(5)
    donor = make_sample("d", fill=9.0)
    donor.face[:] = False
    recipient = make_sample("r")
    sample, prov = splice(
        donor, recipient, SpliceConfig(fps=FPS, modality_weights=(1.0, 0, 0)), rng
    )
    assert not sample.face[prov.start_frame : prov.end_frame].any()
    assert sample.face[: prov.start_frame].all()


# -------------------------------------------------------------------- classical (P9-2)


def test_temporal_jitter_keeps_target_aligned() -> None:
    """The op that would silently desynchronise labels if it rolled instead of trimming."""
    cfg = ClassicalConfig(jitter_frames=6)
    sample = make_sample("v", t=100)
    frames = np.zeros(100, dtype=np.float32)
    frames[40:56] = 1.0
    marked = sample.visual.copy()
    marked[40:56] = 5.0
    sample = PairedSample(
        "v", marked, sample.audio, sample.mask, sample.face, 1, "fake", "train", frames
    )
    for seed in range(40):
        out = temporal_jitter(sample, cfg, np.random.default_rng(seed))
        forged = out.frames > 0
        assert np.all(out.visual[forged] == 5.0), "target and features drifted apart"
        assert np.all(out.visual[~forged] == 1.0)


def test_spec_augment_masks_but_preserves_shape() -> None:
    audio = np.ones((60, 16), dtype=np.float32)
    out = spec_augment(audio, ClassicalConfig(), np.random.default_rng(1))
    assert out.shape == audio.shape
    assert (out == 0.0).any(), "no mask was applied"
    assert audio.sum() == 60 * 16, "input was mutated in place"


def test_time_mask_cannot_swallow_a_whole_splice() -> None:
    """⛔ A mask wider than the shortest forgery would relabel the example as invisible."""
    cfg = ClassicalConfig()
    assert cfg.spec_time_masks * cfg.spec_time_frames < SpliceConfig().min_frames * 2


def test_classical_is_deterministic_given_a_seed() -> None:
    sample = make_sample("v")
    a = classical(sample, ClassicalConfig(), np.random.default_rng(4))
    b = classical(sample, ClassicalConfig(), np.random.default_rng(4))
    assert np.array_equal(a.visual, b.visual) and np.array_equal(a.audio, b.audio)
