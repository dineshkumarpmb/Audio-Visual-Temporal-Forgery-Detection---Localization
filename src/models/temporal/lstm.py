"""Baseline 4 — audio-visual fusion + BiLSTM (P7-1 .. P7-4; section 4.2, Experiment D).

    visual [T,Dv] ─┐                                    ┌─► attn-pool ─► [1] video logit
                   ├─ ConcatFusion.encode ─► [T,256] ─► BiLSTM ─► [T,256] ─┤
    audio  [T,Da] ─┘        (Phase 6, verbatim)          2×256              └─► [T] frame logits

**This is the report-faithful architecture** (section 5.3 row 1: "concat + LSTM + softmax"),
kept as Baseline 4 so Phase 8's Transformer has to *beat* it rather than replace it by
assertion. Section H's argument is that global comparison beats sequential memory for
detecting a 0.65 s inconsistency in an 8 s clip; that argument is worth nothing until the
sequential-memory number exists. This module is that number.

**Everything before the BiLSTM is Phase 6's, called not copied.** `TemporalFusionBaseline`
subclasses `ConcatFusionBaseline` and reuses `encode()` — the same projections, the same
per-modality LayerNorms, the same modality dropout, the same face masking, the same
auxiliary heads. So Experiment D's `D − C` isolates one added component, which is exactly
what P7-6 means by "temporal contribution measured **independently of attention**".

⚠️ "Independently of attention" means self- and cross-attention (Phase 8). *Attention
pooling* is present in every arm since Phase 4 and is therefore a constant, not a confound —
it is how all four arms collapse `[T,256]` to one logit, D included.

**⛔ P7-2 — packing, not masking, is what keeps padding out of the recurrence.** A batch
mixes clips from 25 to ~500 frames. An LSTM run over the padded tensor would keep stepping
its hidden state through the padding, so the *backward* pass of a BiLSTM would begin at
frame 499 of a 60-frame clip and arrive at the real content having already integrated 439
frames of zeros. Zeroing the output afterwards does not undo that — the contamination is in
the state, not the output. `pack_padded_sequence` never presents those positions to the
recurrence at all, which is why the unit test asserts *zero gradient at padded inputs*
rather than merely zero activations.

**Length is preserved exactly** (`total_length=T`), for the same reason the audio encoder
asserts it (P5-2, PF-18): the frame head's output must stay index-aligned with the per-frame
targets `src.localization.targets` builds, or every localization number is off by a
silent offset.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models.fusion.concat import ConcatFusionBaseline
from src.models.heads.frame import FrameHead


class PackedBiLSTM(nn.Module):
    """2-layer bidirectional LSTM over a padded batch, `T` in and `T` out.

    Projects the `2 × hidden` bidirectional output back to `d_model` so everything
    downstream — the head widths, the frame head, Phase 8's encoder — keeps operating on
    256-d, and Baseline 4 differs from Baseline 3 by the recurrence alone rather than by
    also doubling the head's input width.
    """

    def __init__(
        self,
        d_model: int = 256,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}")
        self.d_model = d_model
        self.hidden = hidden
        self.layers = layers

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            # PyTorch applies inter-layer dropout only when there is more than one layer,
            # and warns if given a non-zero value with one. Pass 0 rather than be warned.
            dropout=dropout if layers > 1 else 0.0,
        )
        self.project = nn.Linear(2 * hidden, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """`x` is `(B, T, d_model)`, `mask` is `(B, T)` with True = real frame."""
        if x.ndim != 3:
            raise ValueError(f"expected (B, T, D), got {tuple(x.shape)}")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"input dim {x.shape[-1]} != configured {self.d_model}")
        t_in = x.shape[1]

        if mask is None:
            out, _ = self.lstm(x)
        else:
            # A fully-masked row would give length 0, which `pack_padded_sequence` rejects
            # outright. Clamp to 1 and let the output mask discard it, mirroring
            # `AttentionPooling`'s empty-row guard: one unreliable clip must not abort the
            # batch it happens to share.
            lengths = mask.sum(dim=1).clamp(min=1)
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            # ⛔ total_length: without it the batch is truncated to its longest *sequence*,
            # which equals T only when the longest clip has no padding. Any shorter and
            # every downstream frame index silently shifts.
            out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=t_in)

        out = self.project(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)

        if out.shape[1] != t_in:
            raise AssertionError(
                f"temporal resolution changed: {t_in} frames in, {out.shape[1]} out. "
                "pad_packed_sequence needs total_length to restore the padded length."
            )
        return out

    @property
    def receptive_field(self) -> str:
        """Unbounded in both directions — the property Phase 8 has to justify replacing."""
        return "full sequence (bidirectional recurrence)"


class TemporalFusionBaseline(ConcatFusionBaseline):
    """Baseline 4: Phase 6's fusion, a BiLSTM, and the first per-frame output (P7-1..P7-3).

    Returns the Phase 6 keys unchanged — `logit`, `attention`, `aux_logits` — plus
    `frame_logits` `(B, T)` and `sequence` `(B, T, 256)`. Keeping the old keys means
    `Trainer`, the modality-ablation probe and the 4-class breakdown all work on this model
    without a branch, so Experiment D reuses Experiment C's measurement code rather than a
    parallel copy of it that could drift.
    """

    def __init__(
        self,
        visual_dim: int = 1280,
        feature: str = "logmel",
        d_model: int = 256,
        hidden: int = 64,
        dropout: float = 0.3,
        *,
        modality_dropout: float = 0.2,
        norm: str = "batch",
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
    ) -> None:
        super().__init__(
            visual_dim=visual_dim,
            feature=feature,
            d_model=d_model,
            hidden=hidden,
            dropout=dropout,
            modality_dropout=modality_dropout,
            norm=norm,
        )
        self.temporal = PackedBiLSTM(
            d_model=d_model, hidden=lstm_hidden, layers=lstm_layers, dropout=lstm_dropout
        )
        self.frame_head = FrameHead(d_model, hidden, dropout)

    def forward(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        mask: torch.Tensor | None = None,
        face: torch.Tensor | None = None,
        *,
        drop_visual: bool = False,
        drop_audio: bool = False,
    ) -> dict[str, torch.Tensor]:
        enc = self.encode(visual, audio, mask, face, drop_visual=drop_visual, drop_audio=drop_audio)
        sequence = self.temporal(enc["fused"], mask)

        logit, weights = self.head(sequence, mask)
        frame_logits = self.frame_head(sequence, mask)

        # The auxiliary heads stay on the *pre-temporal* streams, deliberately. Their job
        # (P6-5) is to keep gradient flowing into the visual projection; routing them
        # through the shared BiLSTM would let the recurrence satisfy them from audio alone
        # and quietly disarm the defence.
        aux_visual, _ = self.visual_head(enc["visual"], enc["visual_mask"])
        aux_audio, _ = self.audio_head(enc["audio"], mask)

        return {
            "logit": logit,
            "frame_logits": frame_logits,
            "attention": weights,
            "sequence": sequence,
            "fused": enc["fused"],
            "aux_logits": torch.stack([aux_visual, aux_audio], dim=0),
            "aux_visual": aux_visual,
            "aux_audio": aux_audio,
        }
