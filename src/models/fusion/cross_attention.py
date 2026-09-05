"""Bidirectional cross-modal attention fusion (P8-3; PROJECT_PLAN section G, decision G-1).

    visual [T,256] ──► V attends to A ──┐
                                         ├─► concat ─► project ─► [T,256]
    audio  [T,256] ──► A attends to V ──┘
              (2 layers, pre-norm, 4 heads)

**Why this and not concat, in one sentence:** decision G-1 says cross-attention *"directly
models 'does audio at time t explain video at time t?' — which is the task"*, whereas early
concat can only see the two streams side by side and hope an MLP notices the disagreement.

**This is also the last untried defence against modality collapse.** Phase 6 measured the
collapse (PF-20), Phase 7 showed a BiLSTM does not undo it — recurrence over a representation
that already ignores video cannot un-ignore it. Cross-attention is structurally different:
the V→A stream's *queries* come from video, so if the visual pathway carries nothing, those
queries are constant and the attention has no information to route. The model cannot reach a
good fused representation while ignoring video in the way a concat MLP can, because video is
what decides *where it looks*. Whether that is enough is Experiment E's question, not a claim
to make in advance.

**Pre-norm, residual, 2 layers** (section G's key parameters: `d_model=256`, `n_heads=4`,
`dropout=0.1`, pre-norm). Each layer is cross-attention then a feed-forward, both residual,
so a layer that learns nothing degrades to the identity and fusion falls back on the streams
it was given rather than destroying them.

**Both directions are kept and concatenated, not averaged.** A→V and V→A answer different
questions — "which video frames explain this audio" versus "which audio explains this video"
— and averaging them would discard exactly the asymmetry that makes a dubbed segment look
different from a face-swapped one.
"""

from __future__ import annotations

import torch
from torch import nn


class CrossAttentionLayer(nn.Module):
    """One pre-norm cross-attention block: `q + attn(norm(q), norm(kv))`, then a residual FF."""

    def __init__(
        self, d_model: int = 256, n_heads: int = 4, d_ff: int = 512, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        *,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attended, weights = self.attn(
            self.norm_q(query),
            self.norm_kv(context),
            self.norm_kv(context),
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = query + self.dropout(attended)
        x = x + self.dropout(self.ff(self.norm_ff(x)))
        return x, weights


class BidirectionalCrossAttention(nn.Module):
    """A→V and V→A for `n_layers`, then concat and project back to `d_model` (P8-3).

    Consumes the two *already-encoded* streams — `(B, T, 256)` each, as
    `ConcatFusionBaseline.encode` produces them — so Experiment E differs from Experiment C
    only in how the streams are combined, never in how they were embedded.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.v2a = nn.ModuleList(
            CrossAttentionLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        )
        self.a2v = nn.ModuleList(
            CrossAttentionLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        )
        self.project = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Both streams are `(B, T, d_model)`; `mask` is `(B, T)` with True = real frame."""
        if visual.shape != audio.shape:
            raise ValueError(
                f"streams must match after encoding: visual {tuple(visual.shape)} vs "
                f"audio {tuple(audio.shape)}"
            )
        if visual.shape[-1] != self.d_model:
            raise ValueError(f"input dim {visual.shape[-1]} != configured {self.d_model}")
        t_in = visual.shape[1]

        key_padding = None
        if mask is not None:
            # Same all-masked guard as the self-attention encoder: one unusable clip must
            # not NaN its whole batch.
            guard = ~mask.any(dim=1, keepdim=True)
            key_padding = ~(mask | guard)

        v, a = visual, audio
        maps: dict[str, torch.Tensor] = {}
        for i, (layer_v, layer_a) in enumerate(zip(self.v2a, self.a2v, strict=True)):
            # Both directions read the *same* inputs, so layer i is genuinely symmetric
            # rather than V→A seeing an audio stream that A→V has already rewritten.
            new_v, w_a2v = layer_a(v, a, key_padding, need_weights=return_attention)
            new_a, w_v2a = layer_v(a, v, key_padding, need_weights=return_attention)
            v, a = new_v, new_a
            if return_attention:
                maps[f"a2v_layer{i}"] = w_a2v
                maps[f"v2a_layer{i}"] = w_v2a

        fused = self.project(torch.cat([v, a], dim=-1))
        if mask is not None:
            fused = fused * mask.unsqueeze(-1).to(fused.dtype)

        if fused.shape[1] != t_in:
            raise AssertionError(f"temporal resolution changed: {t_in} in, {fused.shape[1]} out")

        out = {"fused": fused, "visual": v, "audio": a}
        if return_attention:
            out.update(maps)
        return out
