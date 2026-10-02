"""Localization losses: per-frame focal loss (P10-3) and the boundary loss (P10-4).

Section 5.4's full objective is

    L = λ_cls·BCE(video) + λ_loc·Focal(frame) + λ_sync·InfoNCE + λ_bnd·Boundary

and this module holds the two terms Phase 10 adds. Both mask padding exactly as the Phase 7
frame BCE does (`Trainer._frame_loss`): a padded frame is trivially "real", so letting it into
a loss rewards the batch's length mix rather than the predictions.

**Focal, α = 0.25, γ = 2.0 -- RetinaNet's values, and RetinaNet's normalisation.** The sum
over valid frames is divided by the number of *forged* frames in the batch, not by the number
of frames. With ~6.4% of frames forged (P1-12), a per-frame mean would shrink the term ~15x
relative to the clip BCE beside it and make λ_loc mean something different at every positive
rate. Normalising by positives is what makes `λ_loc = 2.0` a statement about the task rather
than about the batch.

⚠️ α = 0.25 *down*-weights positives. That is not a typo: in the focal-loss paper the γ term
already suppresses the flood of easy negatives, and α then corrects the over-correction. It
is the opposite of the `pos_weight ≈ 14.5` the BCE head uses, which is exactly why Phase 10
treats the loss swap as a variable to measure rather than a free upgrade.

**Boundary loss: penalty-reduced focal loss on Gaussian heatmaps** (CornerNet/CenterNet).
Each ground-truth start and end becomes a Gaussian peak (σ = 2 frames = 80 ms) on its own
channel. Plain BCE against a soft heatmap is minimised almost as well by predicting zero
everywhere -- boundaries are ~2 frames per clip out of ~200 -- so the negatives near a peak
are down-weighted by ``(1 - target)^β`` and the whole term is normalised by the number of
peaks. That is the established loss for exactly this shape of target.
"""

from __future__ import annotations

import math

import torch
from torch import nn

FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
BOUNDARY_SIGMA = 2.0  # frames


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float = FOCAL_ALPHA,
    gamma: float = FOCAL_GAMMA,
) -> torch.Tensor:
    """⛔ P10-3. Masked binary focal loss, normalised by the number of positive frames."""
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError(
            f"shape mismatch: logits {tuple(logits.shape)}, targets {tuple(targets.shape)}, "
            f"mask {tuple(mask.shape)}"
        )
    valid = mask.to(logits.dtype)
    ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t).pow(gamma) * ce
    n_pos = (targets * valid).sum().clamp(min=1.0)
    return (loss * valid).sum() / n_pos


def boundary_targets(frames: torch.Tensor, mask: torch.Tensor, sigma: float = BOUNDARY_SIGMA):
    """`(B, T)` binary frame targets -> `(B, T, 2)` Gaussian start / end heatmaps.

    Start = the first forged frame of a run; end = the *last* forged frame, so both peaks sit
    on frames that belong to the span. A run that starts at frame 0 still has a start: a
    forgery that opens the clip begins there. Padding is zero.
    """
    if frames.ndim != 2:
        raise ValueError(f"expected (B, T), got {tuple(frames.shape)}")
    b, t = frames.shape
    f = (frames > 0.5) & mask.bool()
    prev = torch.zeros_like(f)
    prev[:, 1:] = f[:, :-1]
    nxt = torch.zeros_like(f)
    nxt[:, :-1] = f[:, 1:]
    starts = f & ~prev
    ends = f & ~nxt

    idx = torch.arange(t, device=frames.device, dtype=torch.float32)
    radius = max(1, int(math.ceil(3 * sigma)))
    out = torch.zeros(b, t, 2, device=frames.device, dtype=torch.float32)
    for ch, peaks in enumerate((starts, ends)):
        for bi, ti in peaks.nonzero(as_tuple=False).tolist():
            lo, hi = max(0, ti - radius), min(t, ti + radius + 1)
            g = torch.exp(-((idx[lo:hi] - ti) ** 2) / (2 * sigma**2))
            out[bi, lo:hi, ch] = torch.maximum(out[bi, lo:hi, ch], g)
    return out * mask.unsqueeze(-1).to(out.dtype)


def boundary_loss(
    logits: torch.Tensor,
    heatmap: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """⛔ P10-4. Penalty-reduced focal loss over `(B, T, 2)` start/end heatmaps."""
    if logits.shape != heatmap.shape:
        raise ValueError(f"shape mismatch: {tuple(logits.shape)} vs {tuple(heatmap.shape)}")
    valid = mask.unsqueeze(-1).to(logits.dtype)
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    is_peak = (heatmap >= 1.0 - 1e-6).to(logits.dtype)
    pos = -((1 - p) ** alpha) * torch.log(p) * is_peak
    neg = -((1 - heatmap) ** beta) * (p**alpha) * torch.log(1 - p) * (1 - is_peak)
    n_peaks = (is_peak * valid).sum().clamp(min=1.0)
    return ((pos + neg) * valid).sum() / n_peaks
