"""Auxiliary InfoNCE synchronisation loss (P8-5; decision F-1, section 5.4).

    positives = temporally aligned (audio_t, visual_t) pairs from **real** videos
    negatives = the same visual frame against audio shifted by >= 10 frames
    temperature = 0.07

**The supervision is free, which is the whole argument for decision F-1.** A real LAV-DF
video is in-sync by construction — nobody had to label it — so every real clip in the batch
is a fully-supervised sync example at zero annotation cost. That is why the plan recommends a
learned contrastive sync over bolting on a pretrained SyncNet: no extra dependency, no second
model in a 4 GB VRAM budget, and it adapts to this dataset's actual audio pipeline.

⛔ **Real videos only, and this is not a detail.** A fake clip's forged span is *supposed* to
be out of sync — that desynchrony is the signal the classifier is looking for. Training the
sync head to call those frames "aligned" would teach it to erase the very evidence the model
needs, and the damage would show up as a mysteriously weak classifier rather than as an
obviously broken loss. So fakes are excluded from this term entirely; they are still scored
by the sync head at inference, just never used as positives.

**Negatives are shifts of ≥10 frames (400 ms) within the same clip**, not other clips'
audio. A cross-clip negative is trivially separable — different speaker, different room,
different noise floor — so the head would learn speaker identity and score a perfect loss
having learned nothing about synchrony. A within-clip shift holds every one of those
nuisance variables fixed and varies only alignment, which is the one thing being asked
about. 10 frames is also comfortably outside `SyncHead`'s ±5 window, so a "negative" cannot
be a pair the head is designed to accept.
"""

from __future__ import annotations

import torch
from torch import nn

# Component F's key parameter.
TEMPERATURE = 0.07
# Section 5.4 / decision F-1: negatives are shifted by at least this many frames.
MIN_NEGATIVE_SHIFT = 10


class InfoNCESyncLoss(nn.Module):
    """Contrastive alignment loss over per-frame audio-visual embeddings.

    Expects the **L2-normalised** embeddings `SyncHead.embed` returns, so similarities are
    cosines and `temperature` keeps its usual meaning.
    """

    def __init__(
        self,
        temperature: float = TEMPERATURE,
        min_shift: int = MIN_NEGATIVE_SHIFT,
        n_negatives: int = 8,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if n_negatives < 1:
            raise ValueError(f"need at least one negative, got {n_negatives}")
        self.temperature = temperature
        self.min_shift = min_shift
        self.n_negatives = n_negatives

    def forward(
        self,
        visual_embed: torch.Tensor,
        audio_embed: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Embeddings `(B, T, D)` unit-norm, `labels` `(B,)` with 0 = real, `mask` `(B, T)`.

        Returns a scalar. A batch with no real clip, or clips too short to shift by
        `min_shift`, returns 0 rather than NaN — early batches in a small subset can be all
        fake, and one such batch must not destroy the run.
        """
        if visual_embed.shape != audio_embed.shape:
            raise ValueError(
                f"embeddings must match: {tuple(visual_embed.shape)} vs {tuple(audio_embed.shape)}"
            )
        device = visual_embed.device
        zero = torch.zeros((), device=device, dtype=visual_embed.dtype)

        # ⛔ real clips only
        is_real = labels < 0.5
        if not bool(is_real.any()):
            return zero

        v = visual_embed[is_real]
        a = audio_embed[is_real]
        m = mask[is_real] if mask is not None else None
        t = v.shape[1]
        if t <= self.min_shift:
            # Nothing can be shifted far enough to be a legitimate negative.
            return zero

        # Positive: the aligned pair at each timestep.
        pos = (v * a).sum(dim=-1, keepdim=True)  # (n, T, 1)

        # Negatives: the same visual frame against audio rolled by >= min_shift. Sampling a
        # few offsets rather than using all of them keeps this O(T * n_negatives) and makes
        # the negative set differ across steps, which is what stops the head from fitting
        # one particular offset.
        span = t - self.min_shift
        n_neg = min(self.n_negatives, span)
        offsets = torch.randperm(span, generator=generator, device="cpu")[:n_neg] + self.min_shift

        negs = []
        for shift in offsets.tolist():
            rolled = torch.roll(a, shifts=int(shift), dims=1)
            negs.append((v * rolled).sum(dim=-1, keepdim=True))
        neg = torch.cat(negs, dim=-1)  # (n, T, n_neg)

        # Standard InfoNCE: the aligned pair must beat every shifted one. Logit index 0 is
        # the positive, so the target is 0 everywhere.
        logits = torch.cat([pos, neg], dim=-1) / self.temperature  # (n, T, 1 + n_neg)
        flat = logits.reshape(-1, logits.shape[-1])
        target = torch.zeros(flat.shape[0], dtype=torch.long, device=device)
        per_frame = nn.functional.cross_entropy(flat, target, reduction="none").view(
            logits.shape[:2]
        )

        # Padded frames carry no waveform and no mouth; averaging them in would dilute the
        # loss by the batch's length mix, exactly as the frame loss would (P7-2).
        if m is None:
            return per_frame.mean()
        valid = m.to(per_frame.dtype)
        total = valid.sum()
        if total <= 0:
            return zero
        return (per_frame * valid).sum() / total
