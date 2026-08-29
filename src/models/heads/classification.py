"""Attention-pooled video-level classifier (P4-5, sections 5.1 and J).

    [T, D] features -> Linear(D -> 256) -> attention-pool over T -> LayerNorm
                    -> Linear(256 -> 64) -> GELU -> Dropout -> Linear(64 -> 1)

**Attention pooling, not mean pooling** (section J deviation 1). The mean forged span is
0.650 s = 16 frames (P1-11) against clips averaging ~200 frames, so mean-pooling dilutes
the evidence better than 10:1 — reintroducing precisely the problem section 1.1 describes.
Attention pooling lets the model concentrate on the suspicious span.

**One logit + BCE, not two-way softmax** (section J deviation 2). Mathematically equivalent
for binary classification, but a single logit is simpler to calibrate and threshold, and it
matches the per-frame head Phase 10 adds, so both share one loss family.

**The projection is where D-1 parity comes from.** ResNet-18 gives 512-d and MobileNetV2
1280-d; both are projected to `d_model=256`, so everything downstream is identical between
the two arms and the comparison measures the backbone rather than the head.

**Padding and missing faces are masked, not averaged in.** A frame where the detector found
no face (Phase 2's `face_found=False`) carries geometry held over from a neighbour, and a
padded frame carries nothing at all. Letting either contribute to the pooled representation
would feed the classifier fabricated evidence.
"""

from __future__ import annotations

import torch
from torch import nn

NEG_INF = -1e4  # fp16-safe stand-in for -inf; -inf would produce NaN in a fully-masked row


class AttentionPooling(nn.Module):
    """Single learned query attending over time.

    Scores each frame with a small MLP, softmaxes over valid positions, and returns the
    weighted sum. The attention weights are returned too: they are the model's own claim
    about *where* the forgery is, and comparing them against `fake_periods` is a free
    sanity check long before Phase 10 builds a real localization head.
    """

    def __init__(self, d_model: int = 256, hidden: int = 128) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`x` is `(B, T, D)`, `mask` is `(B, T)` with True = valid.

        Returns `(pooled (B, D), weights (B, T))`.
        """
        scores = self.score(x).squeeze(-1)  # (B, T)

        if mask is not None:
            # A clip with no valid frame at all would softmax over all -inf and yield NaN.
            # Fall back to uniform attention over everything: the `face_found` mask that
            # produced it is itself the signal that this clip is unreliable, and a NaN
            # would poison the whole batch's gradient instead of one sample's.
            empty = ~mask.any(dim=1, keepdim=True)
            mask = mask | empty
            scores = scores.masked_fill(~mask, NEG_INF)

        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bt,btd->bd", weights, x)
        return pooled, weights


class VisualBaseline(nn.Module):
    """Baseline 1 of section 4.2: frozen visual features -> video-level real/fake logit.

    Consumes cached features, so it never touches a backbone at training time. That is what
    makes a full training run on this 4 GB card take seconds rather than hours.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        d_model: int = 256,
        hidden: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model

        self.project = nn.Linear(feature_dim, d_model)
        self.pool = AttentionPooling(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """`features` is `(B, T, feature_dim)`, `mask` is `(B, T)` with True = valid.

        Returns the raw logit — `BCEWithLogitsLoss` applies the sigmoid itself, which is
        numerically stabler than sigmoid-then-BCE.
        """
        if features.ndim != 3:
            raise ValueError(f"expected (B, T, D), got {tuple(features.shape)}")
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"feature dim {features.shape[-1]} != configured {self.feature_dim}. "
                "Mixing a resnet18 cache (512) with a mobilenet_v2 model (1280) does this."
            )

        x = self.project(features)
        pooled, weights = self.pool(x, mask)
        logit = self.classifier(self.norm(pooled)).squeeze(-1)
        return {"logit": logit, "attention": weights, "pooled": pooled}

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MeanPoolBaseline(VisualBaseline):
    """Ablation arm for section J deviation 1: identical, but mean-pools.

    Exists so "attention pooling helps" can be *shown* rather than asserted. Subclassing
    keeps every other parameter and initialisation path identical, so the comparison
    isolates the pooling operator.
    """

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        x = self.project(features)
        if mask is None:
            weights = torch.full(x.shape[:2], 1.0 / x.shape[1], device=x.device, dtype=x.dtype)
        else:
            valid = mask | ~mask.any(dim=1, keepdim=True)
            weights = valid.to(x.dtype) / valid.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = torch.einsum("bt,btd->bd", weights, x)
        logit = self.classifier(self.norm(pooled)).squeeze(-1)
        return {"logit": logit, "attention": weights, "pooled": pooled}
