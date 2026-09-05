"""Audio-visual synchronisation head (P8-4; PROJECT_PLAN component F, decision F-1).

    visual [T,256] ──► proj ──► normalise ──┐
                                             ├─► cosine over a ±5-frame window ─► [T] sync
    audio  [T,256] ──► proj ──► normalise ──┘

**Why a sync head exists at all: it is section 5.6's defence #2, and the only one that
cannot be satisfied by one modality.** Modality dropout and auxiliary heads both failed to
stop the collapse (PF-20 measured the dropout *backfiring*), and they share a weakness —
both are computed from one stream at a time, so a model that ignores video can still
minimise them. A cosine similarity between two streams is undefined for one stream. There is
no degenerate solution that reads audio alone.

**The ±5-frame window (±200 ms) is the point, not an implementation detail.** A frame-exact
cosine would ask "does audio frame `t` match video frame `t`", which is brittle: real speech
leads and lags its own mouth motion by tens of milliseconds, and the encoders have receptive
fields wider than a frame anyway. Taking the **maximum** similarity over the window asks the
useful question instead — *"is there any nearby audio that explains this video?"* — so a
genuinely in-sync clip scores high even with a small natural offset, and a dubbed segment,
where no nearby audio explains the mouth, scores low.

**Max over the window, not mean.** A mean would be dragged down by the window's far edges on
every clip including real ones, compressing the range the desync test (⛔ P8-6) has to detect
a shift across. Max keeps real clips near 1.0 and lets desynchronisation actually move it.

Scores are cosine similarities in `[-1, 1]`, per frame, and are consumed two ways: as the
`sync` feature the classifier sees, and as the quantity `src.losses.sync.InfoNCESyncLoss`
trains with contrastive supervision that costs nothing to obtain (real videos are in-sync by
construction).
"""

from __future__ import annotations

import torch
from torch import nn

# Component F's key parameter: ±5 frames at 25 fps = ±200 ms.
SYNC_WINDOW = 5


class SyncHead(nn.Module):
    """Two projections into a shared space, then a windowed per-timestep cosine."""

    def __init__(self, d_model: int = 256, d_sync: int = 256, window: int = SYNC_WINDOW) -> None:
        super().__init__()
        if window < 0:
            raise ValueError(f"window must be >= 0, got {window}")
        self.d_model = d_model
        self.window = window
        self.visual_proj = nn.Linear(d_model, d_sync)
        self.audio_proj = nn.Linear(d_model, d_sync)

    def embed(self, visual: torch.Tensor, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project both streams and L2-normalise, so a dot product *is* a cosine.

        Normalising here rather than inside the similarity means `InfoNCESyncLoss` gets unit
        vectors too, and its temperature `τ=0.07` then has its usual meaning instead of
        silently absorbing the embeddings' scale.
        """
        v = nn.functional.normalize(self.visual_proj(visual), dim=-1)
        a = nn.functional.normalize(self.audio_proj(audio), dim=-1)
        return v, a

    def forward(
        self, visual: torch.Tensor, audio: torch.Tensor, mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Both streams `(B, T, d_model)`. Returns per-frame sync scores `(B, T)` in [-1, 1]."""
        if visual.shape != audio.shape:
            raise ValueError(
                f"streams must match: visual {tuple(visual.shape)} vs audio {tuple(audio.shape)}"
            )
        if visual.shape[-1] != self.d_model:
            raise ValueError(f"input dim {visual.shape[-1]} != configured {self.d_model}")

        v, a = self.embed(visual, audio)
        t = v.shape[1]

        # Similarity of video frame i against audio frames i-w .. i+w, then the best match.
        # Built by shifting rather than by a full T x T matrix: only 2w+1 offsets matter, so
        # this is O(T * window) instead of O(T^2) and stays cheap at T=750.
        best = None
        for shift in range(-self.window, self.window + 1):
            shifted = torch.roll(a, shifts=shift, dims=1)
            sim = (v * shifted).sum(dim=-1)  # (B, T) cosine, both unit-norm

            # `roll` wraps, so the |shift| frames at one end were filled from the other end
            # of the clip -- unrelated audio. Exclude them rather than let a wrapped match
            # count as synchrony.
            if shift > 0:
                sim = sim.clone()
                sim[:, :shift] = -1.0
            elif shift < 0:
                sim = sim.clone()
                sim[:, t + shift :] = -1.0

            best = sim if best is None else torch.maximum(best, sim)

        if mask is not None:
            best = best * mask.to(best.dtype)
        return {"sync": best, "visual_embed": v, "audio_embed": a}

    def clip_score(self, sync: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Mean sync over valid frames — the single number ⛔ P8-6's desync test watches."""
        if mask is None:
            return sync.mean(dim=1)
        valid = mask.to(sync.dtype)
        return (sync * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
