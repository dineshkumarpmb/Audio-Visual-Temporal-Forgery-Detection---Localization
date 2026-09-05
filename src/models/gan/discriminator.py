"""Sequence discriminator with spectral normalisation (P9-3).

Spectral norm on every layer is the plan's named defence against the failure this component
is most likely to hit (§I "Problems": mode collapse). It bounds the discriminator's
Lipschitz constant, which stops it from becoming arbitrarily confident and driving the
generator into a single point that happens to fool it.

**Projection conditioning rather than concatenation.** The label enters as an inner product
against a learned embedding of the pooled features, which is the standard cGAN projection
formulation. Concatenating a label channel lets the discriminator ignore it; the projection
term cannot be ignored because it is the only path by which class information reaches the
logit.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import spectral_norm


class SequenceDiscriminator(nn.Module):
    """`[B, T, 256]` + label -> a realness logit.

    Dilated 1-D convolutions rather than a recurrent net: the discriminator only needs to
    judge local texture and medium-range structure, and a conv stack does that with a fixed
    receptive field, no sequential dependency, and far less time per step.
    """

    def __init__(
        self,
        d_model: int = 256,
        hidden: int = 128,
        n_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            spectral_norm(nn.Conv1d(d_model, hidden, 5, padding=2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            spectral_norm(nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            spectral_norm(nn.Conv1d(hidden, hidden, 5, padding=8, dilation=4)),
            nn.LeakyReLU(0.2),
        )
        self.head = spectral_norm(nn.Linear(hidden, 1))
        self.label_embed = spectral_norm(nn.Embedding(n_classes, hidden))

    def forward(
        self, sequence: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is not None:
            # ⛔ Zero the padding *before* the convolutions, not just before the pool.
            # Masking only at the pool still lets padded positions inside the dilated
            # receptive field (~15 frames) reach real frames near the seam: a unit test
            # measured the logit moving by 0.78 when the padding contents changed, against
            # a signal scale of ~2. Zeroing at the input makes padded positions identical to
            # the zeros `collate_paired` actually writes, so their contents cannot matter.
            sequence = sequence * mask.unsqueeze(-1).to(sequence.dtype)
        h = self.body(sequence.transpose(1, 2)).transpose(1, 2)  # [B, T, H]
        if mask is not None:
            # Padding must not vote. Mean over real frames only, exactly as the pooling
            # operator does everywhere else in this project.
            m = mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        else:
            pooled = h.mean(1)
        logit = self.head(pooled).squeeze(-1)
        return logit + (self.label_embed(labels) * pooled).sum(-1)
