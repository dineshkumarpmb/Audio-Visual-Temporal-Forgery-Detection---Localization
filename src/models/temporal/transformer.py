"""Self-attention temporal encoder (P8-1, P8-2, P8-7; PROJECT_PLAN section H).

    [T, 256] fused  ->  4-layer pre-norm Transformer, 4 heads, d_ff=512  ->  [T, 256]

**This module carries the project's strongest theoretical claim, and Experiment E is what
makes it a measurement.** Section H:

> A forged span is not anomalous in absolute terms — the generated face is photorealistic
> and the TTS is intelligible. It is anomalous **relative to the rest of the same video**.
> Self-attention computes exactly this relative comparison — every timestep against every
> other. An LSTM cannot: its hidden state is a lossy summary of the past, so it cannot
> directly compare frame 200 against frame 12.

Phase 7's BiLSTM is the incumbent that claim has to beat. `E > D` is the whole point.

**Pre-norm, not post-norm** (P8-7). `x + attn(norm(x))` keeps an identity path from input to
output, so gradients reach layer 1 without passing through four LayerNorms. Post-norm
Transformers need a warmup schedule to train at all at this depth; pre-norm tolerates a
constant LR, which matters when a full sweep on this 4 GB card is measured in minutes and
nobody wants to debug a warmup schedule to find out whether attention helps.

**Hand-rolled rather than `nn.TransformerEncoder`, for one reason: attention weights.**
The stock module discards them, and Phase 8 needs them twice — P8-7's head-entropy collapse
check and P8-8's attention-map figure, which is the qualitative evidence the gate requires.
Everything else here is the standard block.

⛔ **Attention weights are opt-in, and on a 4 GB card that is not an optimisation.**
Asking `nn.MultiheadAttention` for weights (`need_weights=True`) does two things: it
materialises a `(B, heads, T, T)` tensor, and it disables PyTorch's fused
scaled-dot-product kernel, which never forms that matrix at all. At batch 32 and this
subset's real T (median 187, **max 497**) that is ~126 MB per layer forward, kept for
backward, across 4 layers — and it OOM'd the GTX 1650 on the first Experiment E run.

So training uses the fused path, and the analysis quantities are computed at eval time:

* `compute_entropy=True` — P8-7's `(B, layers, heads)` collapse check. Cheap, but it needs
  the weights, so it runs during evaluation rather than every training step. Head collapse
  is a property of the trained model, so measuring it at eval loses nothing.
* `return_attention=True` — P8-8's full maps, for figures only, one clip at a time.
"""

from __future__ import annotations

import torch
from torch import nn

# Section H: 750 frames = 30 s at 25 fps. Longer clips are Phase 10's chunking problem
# (P10-11), not something to silently truncate here.
MAX_FRAMES = 750
NEG_INF = -1e4  # fp16-safe; -inf would NaN a fully-masked row


class LearnedPositionalEncoding(nn.Module):
    """Learned absolute position embeddings (P8-2).

    Learned rather than sinusoidal because the forgery prior is *not* translation-invariant
    in an interesting way — but that is a hypothesis, which is why P8-2 also asks for the
    ablation. `enabled=False` is that ablation: it removes position entirely, leaving
    attention permutation-equivariant, and the Experiment E matrix reports both.
    """

    def __init__(self, d_model: int = 256, max_len: int = MAX_FRAMES, enabled: bool = True) -> None:
        super().__init__()
        self.max_len = max_len
        self.enabled = enabled
        self.embedding = nn.Embedding(max_len, d_model) if enabled else None
        if enabled:
            # Small init: positions should nudge the representation, not dominate the
            # content features they are added to.
            nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        t = x.shape[1]
        if t > self.max_len:
            raise ValueError(
                f"{t} frames exceeds max_len={self.max_len} ({self.max_len / 25:.0f} s at 25 fps). "
                "Long videos are chunked with overlap and stitched (section 6.9 / P10-11), "
                "never truncated here -- truncation would silently drop labelled spans."
            )
        pos = torch.arange(t, device=x.device)
        return x + self.embedding(pos).unsqueeze(0)


def attention_entropy(weights: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """⛔ P8-7's collapse detector: normalised entropy of each head's attention.

    `weights` is `(B, heads, T, T)` — row `i` is where query `i` looked. Returns
    `(B, heads)` in `[0, 1]`, normalised by `log(valid positions)`:

    * **≈ 1.0** — the head attends uniformly. It has learned nothing selective, and a whole
      layer of these is the collapse section H's testing row warns about.
    * **≈ 0.0** — the head attends to a single frame. Sharp, and on this task plausibly
      correct: a forged span is a small part of a clip.

    Normalising matters because raw entropy grows with `T`, so a long clip would look more
    "collapsed" than a short one for reasons having nothing to do with the model.
    """
    if weights.ndim != 4:
        raise ValueError(f"expected (B, heads, T, T), got {tuple(weights.shape)}")
    probs = weights.clamp_min(1e-12)
    entropy = -(probs * probs.log()).sum(dim=-1)  # (B, heads, T)

    if mask is None:
        n_valid = torch.full(
            (weights.shape[0],), weights.shape[-1], device=weights.device, dtype=weights.dtype
        )
        return (entropy.mean(dim=-1) / n_valid.log().clamp_min(1e-12)).unsqueeze(-1).squeeze(-1)

    valid = mask.unsqueeze(1).to(entropy.dtype)  # (B, 1, T)
    n_query = valid.sum(dim=-1).clamp_min(1.0)  # (B, 1)
    mean_entropy = (entropy * valid).sum(dim=-1) / n_query  # (B, heads)
    n_valid = mask.sum(dim=1).clamp_min(2).to(entropy.dtype).unsqueeze(-1)  # (B, 1)
    return mean_entropy / n_valid.log()


class PreNormEncoderLayer(nn.Module):
    """One pre-norm block: `x + attn(norm(x))`, then `x + ff(norm(x))`."""

    def __init__(
        self, d_model: int = 256, n_heads: int = 4, d_ff: int = 512, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        *,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.norm1(x)
        attended, weights = self.attn(
            h,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = x + self.dropout(attended)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, weights


class TransformerTemporalEncoder(nn.Module):
    """4-layer pre-norm Transformer over the fused sequence. `T` in, `T` out.

    Drop-in replacement for `PackedBiLSTM` — same `(B, T, 256)` contract in both directions,
    so Experiment E swaps one for the other and changes nothing else. That is what keeps
    `E - D` interpretable.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        *,
        max_len: int = MAX_FRAMES,
        positional: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.positional = positional

        self.pos = LearnedPositionalEncoding(d_model, max_len, enabled=positional)
        self.layers = nn.ModuleList(
            PreNormEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        )
        # Pre-norm leaves the output un-normalised; the final norm is standard and keeps the
        # scale reaching the heads comparable with the BiLSTM arm's projected output.
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
        compute_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
        """`x` is `(B, T, d_model)`, `mask` is `(B, T)` with True = real frame.

        Leave both flags off in the training loop: that is the fused-kernel path, and the
        only one that fits a 4 GB card at this subset's clip lengths.
        """
        if x.ndim != 3:
            raise ValueError(f"expected (B, T, D), got {tuple(x.shape)}")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"input dim {x.shape[-1]} != configured {self.d_model}")
        t_in = x.shape[1]

        key_padding = None
        guard = None
        if mask is not None:
            # A clip with no valid frame would have every key masked and softmax over all
            # -inf -> NaN, poisoning the whole batch's gradient for one bad sample. Same
            # guard as AttentionPooling: fall back to attending over everything.
            guard = ~mask.any(dim=1, keepdim=True)
            key_padding = ~(mask | guard)  # True = ignore this key

        need_weights = return_attention or compute_entropy
        h = self.pos(x)
        entropies, maps = [], []
        for layer in self.layers:
            h, weights = layer(h, key_padding, need_weights=need_weights)
            if weights is not None:
                if compute_entropy:
                    entropies.append(attention_entropy(weights, mask))
                if return_attention:
                    maps.append(weights)

        h = self.norm(h)
        if mask is not None:
            h = h * mask.unsqueeze(-1).to(h.dtype)

        if h.shape[1] != t_in:
            raise AssertionError(f"temporal resolution changed: {t_in} in, {h.shape[1]} out")

        out = {"sequence": h}
        if entropies:
            # (B, layers, heads) -- P8-7's collapse check, computed at eval time.
            out["head_entropy"] = torch.stack(entropies, dim=1)
        if return_attention:
            out["attention_maps"] = torch.stack(maps, dim=1)  # (B, layers, heads, T, T)
        return out

    @property
    def receptive_field(self) -> str:
        """Every frame sees every other in one hop — the property section H argues for."""
        return "full sequence (global self-attention, 1 hop)"
