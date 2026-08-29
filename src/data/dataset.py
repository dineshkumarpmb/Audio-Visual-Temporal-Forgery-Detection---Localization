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

from src.config import VisualFeatureConfig, config_hash


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
