"""Cached-feature dataset and collation (Phase 4).

Reads the `.npz` written by `scripts/13_extract_visual.py` — never video, never a
backbone. That is the whole point of the Stage-A/Stage-B split in section 5.1: a training
run reads a few hundred KB per clip instead of decoding and convolving, so a full sweep
on this 4 GB card takes seconds.

Two details that would quietly corrupt results if got wrong:

**Variable length.** Clips run from 25 to ~500 frames. Batches are padded to the longest
member and a boolean mask marks the real frames, so padding never reaches the pooling
operator. Section 5.5's alternative — cropping every clip to a fixed window — would throw
away exactly the frames a 0.65 s forgery might live in.

**`face_found` folds into the mask.** A frame where Phase 2 found no face carries geometry
held over from a neighbour (see `interpolate_track`), so its features describe the wrong
thing. Masking it out is the difference between "no evidence here" and "fabricated
evidence here".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import AudioPreprocessConfig, VisualFeatureConfig, config_hash


@dataclass(frozen=True)
class Sample:
    video_id: str
    features: np.ndarray  # (T, D) float32
    mask: np.ndarray  # (T,) bool -- True = usable frame
    label: int
    class_name: str
    split: str


class VisualFeatureDataset(Dataset):
    """Cached visual features keyed by `video_id`.

    `require_face` decides whether `face_found` participates in the mask. Turning it off is
    an ablation, not a convenience: it measures how much the detector's failures cost.
    """

    def __init__(
        self,
        video_ids: list[str],
        manifest,  # pandas DataFrame indexed by video_id
        cfg: VisualFeatureConfig,
        root: str | Path = "data/features/visual",
        *,
        require_face: bool = True,
        max_frames: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.root = Path(root) / config_hash(cfg)
        self.require_face = require_face
        self.max_frames = max_frames

        available = [v for v in video_ids if (self.root / f"{v}.npz").exists()]
        if not available:
            raise FileNotFoundError(
                f"no cached features under {self.root}. Run scripts/13_extract_visual.py "
                f"--backbone {cfg.backbone} first."
            )
        self.video_ids = available
        self.missing = [v for v in video_ids if v not in set(available)]

        rows = manifest.loc[available]
        self.labels = rows["label"].to_numpy(dtype=np.int64)
        self.class_names = rows["class_name"].tolist()
        self.splits = rows["split"].tolist()

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Sample:
        vid = self.video_ids[idx]
        with np.load(self.root / f"{vid}.npz") as data:
            features = np.asarray(data["features"], dtype=np.float32)
            found = np.asarray(data["found"], dtype=bool)

        mask = found if self.require_face else np.ones(len(features), dtype=bool)
        if self.max_frames is not None and len(features) > self.max_frames:
            features, mask = features[: self.max_frames], mask[: self.max_frames]

        return Sample(
            vid, features, mask, int(self.labels[idx]), self.class_names[idx], self.splits[idx]
        )

    @property
    def label_balance(self) -> dict[str, float]:
        n = len(self.labels)
        pos = int(self.labels.sum())
        return {
            "n": n,
            "positive": pos,
            "negative": n - pos,
            "positive_rate": pos / n if n else 0.0,
        }


class AudioFeatureDataset(Dataset):
    """Cached audio features keyed by `video_id` (P5-3, Experiment K's two arms).

    Reads the `.npy` written by `scripts/12_extract_audio.py` — `(T, 80)` log-mel or
    `(T, 40)` MFCC, already on the 40 ms video grid, so `T` matches the visual arm's frame
    count for the same clip (Phase 3's alignment gate, 100%).

    **No `face_found` analogue, deliberately.** Every frame of a real waveform is valid
    evidence; a silent stretch is a *finding*, not a hole. So the mask is all-True here and
    only padding is masked, in `collate`. Masking quiet frames would delete exactly the
    evidence an inserted-word forgery leaves behind.

    **Normalisation is per-utterance CMVN, applied at extraction** (`AudioPreprocessConfig.
    cmvn`). That satisfies P5-5's "stats on train only" more strongly than a train-fitted
    scaler would: statistics are computed *within each clip*, so no cross-clip quantity
    exists that could carry dev or test information into training at all.
    """

    def __init__(
        self,
        video_ids: list[str],
        manifest,  # pandas DataFrame indexed by video_id
        cfg: AudioPreprocessConfig,
        root: str | Path = "data/interim/audio",
        *,
        max_frames: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.root = Path(root) / config_hash(cfg)
        self.max_frames = max_frames

        available = [v for v in video_ids if (self.root / f"{v}.npy").exists()]
        if not available:
            raise FileNotFoundError(
                f"no cached audio under {self.root}. Run scripts/12_extract_audio.py "
                f"--feature {cfg.feature} first."
            )
        self.video_ids = available
        self.missing = [v for v in video_ids if v not in set(available)]

        rows = manifest.loc[available]
        self.labels = rows["label"].to_numpy(dtype=np.int64)
        self.class_names = rows["class_name"].tolist()
        self.splits = rows["split"].tolist()

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Sample:
        vid = self.video_ids[idx]
        features = np.asarray(np.load(self.root / f"{vid}.npy"), dtype=np.float32)
        if self.max_frames is not None and len(features) > self.max_frames:
            features = features[: self.max_frames]
        mask = np.ones(len(features), dtype=bool)

        return Sample(
            vid, features, mask, int(self.labels[idx]), self.class_names[idx], self.splits[idx]
        )

    @property
    def label_balance(self) -> dict[str, float]:
        n = len(self.labels)
        pos = int(self.labels.sum())
        return {
            "n": n,
            "positive": pos,
            "negative": n - pos,
            "positive_rate": pos / n if n else 0.0,
        }


@dataclass(frozen=True)
class PairedSample:
    video_id: str
    visual: np.ndarray  # (T, Dv) float32
    audio: np.ndarray  # (T, Da) float32
    mask: np.ndarray  # (T,) bool -- True = a real frame (padding only)
    face: np.ndarray  # (T,) bool -- True = the visual features are trustworthy
    label: int
    class_name: str
    split: str


class MultimodalFeatureDataset(Dataset):
    """Visual and audio features for the same clip, on one shared index grid (P6-2).

    **The grids already agree by construction.** `hop_length=640` is 40 ms, one video frame
    at 25 fps is 40 ms, and Phase 3's gate verified `audio_frames == video_frames ± 1` on
    100% of files. So alignment here is a truncation to `T = min(T_visual, T_audio)`, not an
    interpolation — and the ±1 tolerance is **asserted per clip** rather than assumed,
    because a silent regression upstream would misalign every downstream frame target.

    **Two masks, deliberately.** Phase 4 folded `face_found` into a single mask; that is
    wrong once audio is present, because a frame where the detector found no face still has
    perfectly good audio. Conflating them would discard real acoustic evidence and hand the
    fusion model an artificial reason to prefer video-free solutions.

    * `mask` — the frame exists (padding only). Drives fusion pooling and the audio stream.
    * `face` — the visual features are trustworthy. Drives the visual stream, whose features
      are zeroed where it is False so held-over geometry never reaches the concat.
    """

    def __init__(
        self,
        video_ids: list[str],
        manifest,  # pandas DataFrame indexed by video_id
        visual_cfg: VisualFeatureConfig,
        audio_cfg: AudioPreprocessConfig,
        visual_root: str | Path = "data/features/visual",
        audio_root: str | Path = "data/interim/audio",
        *,
        max_frames: int | None = None,
        align_tolerance: int = 1,
    ) -> None:
        self.visual_cfg = visual_cfg
        self.audio_cfg = audio_cfg
        self.visual_root = Path(visual_root) / config_hash(visual_cfg)
        self.audio_root = Path(audio_root) / config_hash(audio_cfg)
        self.max_frames = max_frames
        self.align_tolerance = align_tolerance

        available = [
            v
            for v in video_ids
            if (self.visual_root / f"{v}.npz").exists() and (self.audio_root / f"{v}.npy").exists()
        ]
        if not available:
            raise FileNotFoundError(
                f"no clip has both {self.visual_root} and {self.audio_root}. Run "
                f"scripts/13_extract_visual.py --backbone {visual_cfg.backbone} and "
                f"scripts/12_extract_audio.py --feature {audio_cfg.feature} first."
            )
        self.video_ids = available
        self.missing = [v for v in video_ids if v not in set(available)]

        rows = manifest.loc[available]
        self.labels = rows["label"].to_numpy(dtype=np.int64)
        self.class_names = rows["class_name"].tolist()
        self.splits = rows["split"].tolist()

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> PairedSample:
        vid = self.video_ids[idx]
        with np.load(self.visual_root / f"{vid}.npz") as data:
            visual = np.asarray(data["features"], dtype=np.float32)
            found = np.asarray(data["found"], dtype=bool)
        audio = np.asarray(np.load(self.audio_root / f"{vid}.npy"), dtype=np.float32)

        # ⛔ P6-2. Phase 3 guarantees this; assert it rather than trust it, because a
        # violation here silently misaligns every per-frame target Phase 10 will build.
        drift = abs(len(visual) - len(audio))
        if drift > self.align_tolerance:
            raise AssertionError(
                f"{vid}: visual has {len(visual)} frames, audio {len(audio)} -- drift {drift} "
                f"exceeds the +/-{self.align_tolerance} Phase 3's gate guarantees"
            )

        t = min(len(visual), len(audio))
        if self.max_frames is not None:
            t = min(t, self.max_frames)
        visual, audio, face = visual[:t], audio[:t], found[:t]
        # Held-over geometry is not evidence. Zero it so it cannot reach the concat.
        visual = visual * face[:, None]

        return PairedSample(
            vid,
            visual,
            audio,
            np.ones(t, dtype=bool),
            face,
            int(self.labels[idx]),
            self.class_names[idx],
            self.splits[idx],
        )

    @property
    def label_balance(self) -> dict[str, float]:
        n = len(self.labels)
        pos = int(self.labels.sum())
        return {
            "n": n,
            "positive": pos,
            "negative": n - pos,
            "positive_rate": pos / n if n else 0.0,
        }


def collate_paired(batch: list[PairedSample]) -> dict[str, torch.Tensor | list[str]]:
    """Pad both streams to the longest clip in the batch, carrying both masks."""
    max_t = max(len(s.mask) for s in batch)
    dv = batch[0].visual.shape[1]
    da = batch[0].audio.shape[1]

    visual = torch.zeros(len(batch), max_t, dv, dtype=torch.float32)
    audio = torch.zeros(len(batch), max_t, da, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_t, dtype=torch.bool)
    face = torch.zeros(len(batch), max_t, dtype=torch.bool)
    for i, s in enumerate(batch):
        t = len(s.mask)
        visual[i, :t] = torch.from_numpy(s.visual)
        audio[i, :t] = torch.from_numpy(s.audio)
        mask[i, :t] = torch.from_numpy(s.mask)
        face[i, :t] = torch.from_numpy(s.face)

    return {
        "features": visual,  # keeps the Trainer's single-stream contract working
        "audio": audio,
        "mask": mask,
        "face": face,
        "label": torch.tensor([s.label for s in batch], dtype=torch.float32),
        "video_id": [s.video_id for s in batch],
        "class_name": [s.class_name for s in batch],
    }


def collate(batch: list[Sample]) -> dict[str, torch.Tensor | list[str]]:
    """Pad to the longest clip in the batch and build the validity mask."""
    max_t = max(len(s.features) for s in batch)
    dim = batch[0].features.shape[1]

    features = torch.zeros(len(batch), max_t, dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_t, dtype=torch.bool)
    for i, s in enumerate(batch):
        t = len(s.features)
        features[i, :t] = torch.from_numpy(s.features)
        mask[i, :t] = torch.from_numpy(s.mask)

    return {
        "features": features,
        "mask": mask,
        "label": torch.tensor([s.label for s in batch], dtype=torch.float32),
        "video_id": [s.video_id for s in batch],
        "class_name": [s.class_name for s in batch],
    }


def split_ids(manifest, video_ids: list[str], split: str) -> list[str]:
    """Official split membership only — never re-derived (section 3.5 RULE 1)."""
    present = [v for v in video_ids if v in manifest.index]
    return [v for v in present if manifest.loc[v, "split"] == split]
