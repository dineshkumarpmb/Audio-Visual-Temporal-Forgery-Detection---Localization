"""Optional boundary head (P10-4; section K).

    [T, 256] temporal embeddings  ->  [T, 2] logits: (start-ness, end-ness) per frame

Section 5.4 says the boundary term "improves IoU@0.75+ specifically": the frame head decides
*which* frames are forged, and a boundary head decides *where the span begins and ends*,
which is what the strict IoU thresholds reward. Its output is only useful if post-processing
reads it, so `PostProcessConfig.refine_radius` snaps each segment edge to this head's peak.

Same shape as `FrameHead` apart from the two output channels, for the same reason the frame
head copies the video head: a bigger head would confound "boundary supervision helps" with
"this head has more capacity".
"""

from __future__ import annotations

import torch
from torch import nn


class BoundaryHead(nn.Module):
    def __init__(self, d_model: int = 256, hidden: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """`x` is `(B, T, d_model)`; returns `(B, T, 2)` raw logits, padding zeroed."""
        if x.ndim != 3:
            raise ValueError(f"expected (B, T, D), got {tuple(x.shape)}")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"feature dim {x.shape[-1]} != configured {self.d_model}")
        logits = self.classifier(self.norm(x))
        if mask is not None:
            logits = logits * mask.unsqueeze(-1).to(logits.dtype)
        return logits
