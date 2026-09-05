"""Per-frame forgery head (P7-3, section 5.1's right-hand branch).

    [T, 256] temporal embeddings  ->  [T] logits, one per 40 ms frame

**The first localization-relevant output the project produces.** Phases 4-6 pooled the
whole clip into a single logit; that answers "is this fake" and nothing more. Section 6.1 is
explicit that localization must be *dense per-frame prediction, not sliding-window
classification*, and this head is where that starts.

**Deliberately the same shape as the video-level head** (`Linear(256 -> 64) -> GELU ->
Dropout -> Linear(64 -> 1)`, LayerNorm first). Both branches read the same `[T, 256]`
sequence, so giving the frame branch more capacity would confound "temporal modelling
helps localization" with "this head is bigger". The only difference is where the pooling
happens: the video head pools then classifies, this one classifies then leaves `T` intact.

**One logit per frame + BCE, matching the video head's loss family** (section J deviation
2). Phase 10 swaps BCE for focal loss (P10-3) because the positive rate per frame is far
lower than per clip; Phase 7 stays on weighted BCE so Experiment D measures the BiLSTM
rather than a loss change made at the same time.

**Padding is zeroed, never masked-softmaxed.** There is no softmax here — each frame is an
independent decision — so the mask cannot be folded into the scores the way
`AttentionPooling` folds it. Instead padded positions are zeroed on the way out and the
loss masks them again (`Trainer._frame_loss`). Both, because either alone is a single point
of failure for the one bug that would not show up in any metric.
"""

from __future__ import annotations

import torch
from torch import nn


class FrameHead(nn.Module):
    """Dense per-timestep classifier. `T` in, `T` out, no temporal mixing of its own."""

    def __init__(self, d_model: int = 256, hidden: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """`x` is `(B, T, d_model)`, `mask` is `(B, T)` with True = real frame.

        Returns `(B, T)` raw logits — `BCEWithLogitsLoss` applies the sigmoid itself.
        """
        if x.ndim != 3:
            raise ValueError(f"expected (B, T, D), got {tuple(x.shape)}")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"feature dim {x.shape[-1]} != configured {self.d_model}")

        logits = self.classifier(self.norm(x)).squeeze(-1)  # (B, T)
        if mask is not None:
            logits = logits * mask.to(logits.dtype)
        return logits
