"""Augmentation for Experiment F (P9-1, P9-2; plan section I, tier 1).

Everything here operates on **cached features**, never pixels or waveforms. That is forced
by the Stage-A/Stage-B split: `data/features/visual/` holds MobileNetV2 embeddings, not
face crops, so an augmentation that needs the image is an augmentation that needs a full
re-extraction (3.65 s/video, ~48 min per variant of the 786-clip cache).

⚠️ **P9-2 says "crop/colour jitter"; that is a pixel-space operation and it is not
implemented here.** Re-extracting a jittered cache would cost hours and buy robustness in
the one stream the model has repeatedly been shown not to use (PF-20/PF-21: zeroing video
costs +0.0010 to +0.0056 AUC across every Phase 6-8 arm). The feature-space substitutes are
`feature_noise` and `frame_dropout`, and the deviation is recorded here rather than hidden.

**Why splicing comes first (P9-1).** Every other augmentation perturbs an existing example.
Splicing *creates* one, with boundaries known to the frame rather than to the label file,
and boundary precision is exactly what Phase 10's AP@0.75/0.95 will be won or lost on. It is
also the only augmentation here that can change the **class balance by modality**: a
`visual` splice is a fake whose audio is untouched, so unlike anything in LAV-DF's training
mix it cannot be detected from audio at all. See `SpliceConfig.modality_weights`.

⛔ **The quarter-frame inset is not a rounding fudge; it is the fix for a measured bug.**
A spliced span covers frames `[i, i+L)`. The obvious way to record that in the manifest's
units is `(i/fps, (i+L)/fps)` seconds -- and at 25 fps that round-trips *wrongly* through
`frame_targets` for **1338 of 15000** tested `(i, L)` combinations, because `i/25.0*25.0`
lands a few ULP either side of `i` and `floor`/`ceil` amplify it into a whole frame. The
error is silent: the target is one frame wide of the features it labels, training converges,
and every boundary number in Part 8 is off by 40 ms. Insetting each edge by a quarter frame
puts 0.25 of margin between the value and the rounding boundary -- 12 orders of magnitude
more than the error -- and means the *same* frames are selected. A quarter and not a half:
at half, a one-frame span insets to zero width and `normalise_periods` rejects it as
reversed, which the P9-1 sweep caught on its first run. `assert_exact_boundaries`
re-derives the target from the seconds and refuses to return a sample that disagrees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import Field, model_validator

from src.config import LAVDF_FPS, StrictModel
from src.data.dataset import PairedSample
from src.localization.targets import frame_targets

# Section I(a): splice length is drawn to resemble LAV-DF's own spans. P1-11 measured a
# median forged span of ~16 frames (0.65 s); the bounds bracket it rather than centre on it,
# because a model that only ever sees 16-frame forgeries learns the length, not the seam.
MIN_SPLICE_FRAMES = 5
MAX_SPLICE_FRAMES = 50

MODALITIES = ("visual", "audio", "both")


class SpliceConfig(StrictModel):
    """P9-1 manipulation splicing.

    `modality_weights` picks between a fake whose video is swapped, whose audio is swapped,
    or both -- mirroring LAV-DF's own four classes. The default is uniform, which makes
    two thirds of generated fakes undetectable from audio alone. That is deliberate: it is
    the first lever in this project that can *force* the visual pathway to matter, and
    whether it moves the PF-21 collapse is a measurable question rather than a hope.
    """

    min_frames: int = Field(default=MIN_SPLICE_FRAMES, ge=1)
    max_frames: int = Field(default=MAX_SPLICE_FRAMES, ge=1)
    modality_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    fps: float = Field(default=LAVDF_FPS, gt=0)

    @model_validator(mode="after")
    def _check(self) -> SpliceConfig:
        if self.max_frames < self.min_frames:
            raise ValueError(f"max_frames {self.max_frames} < min_frames {self.min_frames}")
        if not np.isclose(sum(self.modality_weights), 1.0):
            raise ValueError(f"modality_weights must sum to 1, got {self.modality_weights}")
        if any(w < 0 for w in self.modality_weights):
            raise ValueError(f"modality_weights must be non-negative, got {self.modality_weights}")
        return self


class ClassicalConfig(StrictModel):
    """P9-2, in the feature space the cache actually holds.

    ⛔ `time_mask_frames` is capped well below the shortest splice on purpose. SpecAugment
    masking that swallowed a whole forged span would relabel the example -- the target still
    says "fake here" while every feature saying so has been zeroed -- which teaches the model
    that forgeries are invisible. Small masks perturb; large masks lie.
    """

    spec_time_masks: int = Field(default=2, ge=0)
    spec_time_frames: int = Field(default=4, ge=0)
    spec_freq_masks: int = Field(default=2, ge=0)
    spec_freq_bins: int = Field(default=8, ge=0)
    jitter_frames: int = Field(default=6, ge=0)
    feature_noise: float = Field(default=0.05, ge=0)
    frame_dropout: float = Field(default=0.05, ge=0, lt=1)
    prob: float = Field(default=0.5, ge=0, le=1)


@dataclass(frozen=True)
class Splice:
    """Provenance for one generated example -- what came from where, in both units."""

    donor_id: str
    recipient_id: str
    modality: str
    start_frame: int
    end_frame: int  # exclusive
    fake_periods: list[tuple[float, float]]


def span_to_seconds(
    start_frame: int, end_frame: int, fps: float = LAVDF_FPS
) -> tuple[float, float]:
    """Frame span `[start, end)` -> the `(start_s, end_s)` that reproduces it exactly.

    Quarter-frame inset on each edge -- see the module docstring. `end_frame` is exclusive,
    so the last forged frame is `end_frame - 1`, and the returned interval runs from just
    inside frame `start` to just inside frame `end - 1`. Both edges stay strictly within the
    frames they name, which is what makes `floor`/`ceil` recover them without float slack.
    """
    if end_frame <= start_frame:
        raise ValueError(f"empty span [{start_frame}, {end_frame})")
    return ((start_frame + 0.25) / fps, (end_frame - 0.25) / fps)


def assert_exact_boundaries(
    start_frame: int, end_frame: int, n_frames: int, fps: float = LAVDF_FPS
) -> np.ndarray:
    """⛔ Round-trip guard: seconds -> `frame_targets` must reproduce `[start, end)` exactly.

    This is P9-1's "exactly-known boundaries" made checkable, and it is the same contract
    P10-1 will assert in the other direction. Raising here costs one sample; not raising
    costs every localization number in Part 8.
    """
    expected = np.zeros(n_frames, dtype=np.float32)
    expected[start_frame:end_frame] = 1.0
    got = frame_targets([span_to_seconds(start_frame, end_frame, fps)], n_frames, fps)
    if not np.array_equal(expected, got):
        raise AssertionError(
            f"splice [{start_frame}, {end_frame}) of {n_frames} at {fps} fps does not "
            f"round-trip: {int(expected.sum())} frames intended, {int(got.sum())} recovered"
        )
    return expected


def splice(
    donor: PairedSample,
    recipient: PairedSample,
    cfg: SpliceConfig,
    rng: np.random.Generator,
) -> tuple[PairedSample, Splice]:
    """Splice a span of `donor` into `recipient`, returning a fake with exact boundaries.

    ⛔ **Both clips must be real.** Splicing into a fake would union the new span with the
    recipient's existing `fake_periods`, and "exactly-known boundaries" would become
    "exactly-known plus whatever was already there" -- which is not what P9-1 asks for and
    not what Phase 10 can be scored against. Real donors also guarantee the *donated* frames
    are genuine, so the only manipulation in the output is the seam this function created.

    The recipient owns the timeline; the donor contributes `end - start` frames of one or
    both streams. Nothing is resampled or interpolated -- both caches are already on the
    same 25 fps grid (P6-2), so a splice is an array slice assignment.
    """
    if donor.label != 0 or recipient.label != 0:
        raise ValueError(
            f"splice needs two real clips, got donor label {donor.label} "
            f"and recipient label {recipient.label}"
        )
    t = len(recipient.mask)
    usable = min(t, len(donor.mask))
    length = int(rng.integers(cfg.min_frames, cfg.max_frames + 1))
    length = min(length, usable)
    if length < 1:
        raise ValueError(f"cannot splice into a {t}-frame clip")

    start = int(rng.integers(0, t - length + 1))
    donor_start = int(rng.integers(0, len(donor.mask) - length + 1))
    end = start + length
    modality = str(rng.choice(MODALITIES, p=np.asarray(cfg.modality_weights, dtype=np.float64)))

    visual = recipient.visual.copy()
    audio = recipient.audio.copy()
    face = recipient.face.copy()
    if modality in ("visual", "both"):
        visual[start:end] = donor.visual[donor_start : donor_start + length]
        # The donor's trust flags travel with the donor's features. Keeping the recipient's
        # would claim held-over geometry is trustworthy, or discard frames that are fine.
        face[start:end] = donor.face[donor_start : donor_start + length]
    if modality in ("audio", "both"):
        audio[start:end] = donor.audio[donor_start : donor_start + length]

    frames = assert_exact_boundaries(start, end, t, cfg.fps)
    period = span_to_seconds(start, end, cfg.fps)
    provenance = Splice(donor.video_id, recipient.video_id, modality, start, end, [period])

    sample = PairedSample(
        video_id=f"splice_{recipient.video_id}_{donor.video_id}_{start}_{end}_{modality}",
        visual=visual,
        audio=audio,
        mask=recipient.mask.copy(),
        face=face,
        label=1,
        class_name=f"spliced_{modality}",
        split=recipient.split,
        frames=frames,
    )
    return sample, provenance


def spec_augment(audio: np.ndarray, cfg: ClassicalConfig, rng: np.random.Generator) -> np.ndarray:
    """SpecAugment time and frequency masking on the `[T, n_mels]` cached log-Mel."""
    out = audio.copy()
    t, bins = out.shape
    for _ in range(cfg.spec_time_masks):
        width = int(rng.integers(0, cfg.spec_time_frames + 1))
        if width and t > width:
            lo = int(rng.integers(0, t - width))
            out[lo : lo + width] = 0.0
    for _ in range(cfg.spec_freq_masks):
        width = int(rng.integers(0, cfg.spec_freq_bins + 1))
        if width and bins > width:
            lo = int(rng.integers(0, bins - width))
            out[:, lo : lo + width] = 0.0
    return out


def temporal_jitter(
    sample: PairedSample, cfg: ClassicalConfig, rng: np.random.Generator
) -> PairedSample:
    """Trim up to `jitter_frames` from one end, carrying every per-frame array with it.

    ⛔ Trim, not roll. A circular shift would wrap the tail of the clip onto its head and
    move forged frames next to frames they never bordered -- inventing a seam the label does
    not describe. Trimming only removes context, and the target follows by construction
    because every array is sliced with the same indices.
    """
    if cfg.jitter_frames <= 0:
        return sample
    t = len(sample.mask)
    shift = int(rng.integers(-cfg.jitter_frames, cfg.jitter_frames + 1))
    if shift == 0 or t - abs(shift) < 2:
        return sample
    lo, hi = (shift, t) if shift > 0 else (0, t + shift)
    return PairedSample(
        video_id=sample.video_id,
        visual=sample.visual[lo:hi],
        audio=sample.audio[lo:hi],
        mask=sample.mask[lo:hi],
        face=sample.face[lo:hi],
        label=sample.label,
        class_name=sample.class_name,
        split=sample.split,
        frames=None if sample.frames is None else sample.frames[lo:hi],
    )


def feature_perturb(
    visual: np.ndarray, face: np.ndarray, cfg: ClassicalConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Feature-space stand-in for crop/colour jitter (see the module docstring).

    Noise is scaled by the clip's own feature scale, so it is a fixed *relative* perturbation
    rather than one that means something different for every backbone. Dropped frames are
    marked untrustworthy in `face` as well as zeroed, which is exactly how a Phase 2
    detection failure presents -- so the model sees a failure mode it will meet at inference.
    """
    out = visual.copy()
    trust = face.copy()
    if cfg.feature_noise > 0:
        scale = float(np.std(out)) * cfg.feature_noise
        if scale > 0:
            out = out + rng.normal(0.0, scale, size=out.shape).astype(np.float32)
    if cfg.frame_dropout > 0:
        drop = rng.random(len(out)) < cfg.frame_dropout
        out[drop] = 0.0
        trust[drop] = False
    return out.astype(np.float32), trust


def classical(sample: PairedSample, cfg: ClassicalConfig, rng: np.random.Generator) -> PairedSample:
    """Apply the P9-2 stack to one sample, each op gated independently by `cfg.prob`."""
    audio = sample.audio
    if rng.random() < cfg.prob:
        audio = spec_augment(audio, cfg, rng)
    visual, face = sample.visual, sample.face
    if rng.random() < cfg.prob:
        visual, face = feature_perturb(visual, face, cfg, rng)
    out = PairedSample(
        video_id=sample.video_id,
        visual=visual,
        audio=audio,
        mask=sample.mask,
        face=face,
        label=sample.label,
        class_name=sample.class_name,
        split=sample.split,
        frames=sample.frames,
    )
    if rng.random() < cfg.prob:
        out = temporal_jitter(out, cfg, rng)
    return out


class AugmentedPairs:
    """`MultimodalFeatureDataset` + P9-1 splices + the P9-2 stack (arms F1 and F2).

    Wraps rather than subclasses, so F0 can use the bare dataset and the two arms differ in
    exactly one object. Indices `[0, len(base))` are the real clips; the tail is synthetic.

    ⛔ **Donors and recipients come only from this dataset**, and this dataset must be the
    *train* split. A splice built from a dev clip would put dev features into training under
    a new id -- leakage that no split check downstream could ever catch, because the id is
    genuinely new. `require_train` makes that a construction-time error rather than a silent
    contamination, in the same spirit as `src.data.leakage`.

    **Reproducibility.** Every synthetic item derives its own generator from
    `seed` and its index, so item *k* is the same splice on every epoch, in every worker, and
    across restarts. The synthetic half is therefore a fixed dataset extension rather than a
    stream of fresh randomness -- which is what makes an F1 vs F2 comparison meaningful, and
    what lets `plan()` write the whole thing to the report as provenance. Classical
    augmentation is deliberately the opposite: it re-randomises per access (`epoch` folds
    into the seed) so the model does not memorise one fixed perturbation.
    """

    def __init__(
        self,
        base,
        *,
        n_spliced: int = 0,
        splice_cfg: SpliceConfig | None = None,
        classical_cfg: ClassicalConfig | None = None,
        seed: int = 0,
        require_train: bool = True,
    ) -> None:
        if not getattr(base, "with_frames", False):
            raise ValueError("AugmentedPairs needs with_frames=True: splices carry frame targets")
        splits = set(getattr(base, "splits", []))
        if require_train and splits - {"train"}:
            raise ValueError(
                f"refusing to augment a dataset containing {sorted(splits - {'train'})} clips -- "
                "splices from dev/test would leak into training under new video ids"
            )
        self.base = base
        self.splice_cfg = splice_cfg or SpliceConfig()
        self.classical_cfg = classical_cfg
        self.seed = seed
        self.epoch = 0

        self.real_indices = [i for i, lab in enumerate(base.labels) if int(lab) == 0]
        if n_spliced and len(self.real_indices) < 2:
            raise ValueError(
                f"splicing needs at least two real clips, this split has {len(self.real_indices)}"
            )
        self.n_spliced = int(n_spliced)

    def set_epoch(self, epoch: int) -> None:
        """Re-randomise classical augmentation per epoch; splices are unaffected."""
        self.epoch = int(epoch)

    def plan(self) -> list[Splice]:
        """The provenance of every synthetic example, without loading any features."""
        out = []
        for k in range(self.n_spliced):
            rng = np.random.default_rng([self.seed, k])
            d, r = self._pair(rng)
            out.append((self.base.video_ids[d], self.base.video_ids[r]))
        return out

    def _pair(self, rng: np.random.Generator) -> tuple[int, int]:
        d, r = rng.choice(self.real_indices, size=2, replace=False)
        return int(d), int(r)

    def __len__(self) -> int:
        return len(self.base) + self.n_spliced

    def __getitem__(self, idx: int) -> PairedSample:
        if idx < len(self.base):
            sample = self.base[idx]
        else:
            k = idx - len(self.base)
            # Fixed across epochs -- the synthetic set is a dataset, not a random stream.
            rng = np.random.default_rng([self.seed, k])
            d, r = self._pair(rng)
            sample, _ = splice(self.base[d], self.base[r], self.splice_cfg, rng)
        if self.classical_cfg is not None:
            rng = np.random.default_rng([self.seed, self.epoch, idx, 7919])
            sample = classical(sample, self.classical_cfg, rng)
        return sample

    @property
    def labels(self) -> np.ndarray:
        return np.concatenate(
            [np.asarray(self.base.labels), np.ones(self.n_spliced, dtype=np.int64)]
        )

    @property
    def label_balance(self) -> dict[str, float]:
        labels = self.labels
        n, pos = len(labels), int(labels.sum())
        return {
            "n": n,
            "positive": pos,
            "negative": n - pos,
            "positive_rate": pos / n if n else 0.0,
        }

    def frame_positive_rate(self) -> float:
        """Fraction of frames that are forged across the augmented split.

        Reads every item, because a splice's length is drawn per item and there is no
        shortcut that is also honest. Costs a few seconds once per run and sets
        `frame_pos_weight`, which a wrong value would quietly saturate the frame head.
        """
        total, positive = 0, 0.0
        for i in range(len(self)):
            frames = self[i].frames
            if frames is None:
                continue
            total += frames.size
            positive += float(frames.sum())
        return positive / total if total else 0.0
