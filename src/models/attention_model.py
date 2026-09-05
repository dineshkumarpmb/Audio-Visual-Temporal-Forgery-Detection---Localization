"""The section 5.1 model — cross-attention fusion + Transformer + dual heads (Experiment E).

    visual [T,Dv] ─┐                          ┌─► sync head ──► [T] sync score
                   ├─ encode_streams ─► ┌─────┤                 (aux InfoNCE, P8-5)
    audio  [T,Da] ─┘   (Phase 6, shared) │    │
                                         ▼    ▼
                       BIDIRECTIONAL CROSS-ATTENTION (2 layers, P8-3)
                                         │
                       TRANSFORMER ENCODER (4 layers, 4 heads, P8-1)
                                         │
                          ┌──────────────┴──────────────┐
                          ▼                             ▼
                   attention-pool                  per-frame head
                          │                             │
                     [1] video logit            [T] frame logits

**This is the architecture section 5.3 recommends and the one the whole plan builds toward**
— the "middle row" that is feasible on 4 GB, explainable through attention maps, and strong
enough to defend. Every earlier phase exists to make its components individually measured
rather than jointly assumed.

**Two components change at once versus Phase 7, and that is deliberate — but it is also why
`E - D` alone would be uninterpretable.** Experiment E swaps the concat for cross-attention
*and* the BiLSTM for a Transformer. So the model exposes both independently:

* `temporal="transformer", cross_attention=False` — self-attention only, isolating P8-1
* `temporal="lstm", cross_attention=True` — cross-attention only, isolating P8-3
* both — the full section 5.1 model

P8-10's gate reads those arms, not just the headline, so the credit for any gain is
assignable. This is the same discipline P7-6 imposed on the BiLSTM.

**The sync head reads the pre-fusion streams, not the fused sequence** (P8-4/P8-5). After
cross-attention each stream has already absorbed the other, so a cosine between them would
be measuring how well the fusion mixed rather than whether audio and video agree — and it
would be trivially satisfiable. Reading `encode_streams`' output keeps the two views
genuinely separate, which is the property that makes the sync loss impossible to satisfy
with one modality (section 5.6 defence #2).
"""

from __future__ import annotations

import torch

from src.models.fusion.concat import ConcatFusionBaseline
from src.models.fusion.cross_attention import BidirectionalCrossAttention
from src.models.heads.frame import FrameHead
from src.models.heads.sync import SYNC_WINDOW, SyncHead
from src.models.temporal.lstm import PackedBiLSTM
from src.models.temporal.transformer import MAX_FRAMES, TransformerTemporalEncoder


class AttentionFusionModel(ConcatFusionBaseline):
    """Experiment E. Reuses Phase 6's stream encoders; everything after them is new."""

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
        # --- the two components under test, independently switchable (P8-10) -----------
        cross_attention: bool = True,
        temporal: str = "transformer",
        # --- section H / G key parameters ---------------------------------------------
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 512,
        attn_dropout: float = 0.1,
        cross_layers: int = 2,
        max_len: int = MAX_FRAMES,
        positional: bool = True,
        sync_window: int = SYNC_WINDOW,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
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
        if temporal not in ("transformer", "lstm", "none"):
            raise ValueError(f"unknown temporal {temporal!r}")
        self.cross_attention_enabled = cross_attention
        self.temporal_kind = temporal
        self.positional = positional

        self.cross = (
            BidirectionalCrossAttention(d_model, cross_layers, n_heads, d_ff, attn_dropout)
            if cross_attention
            else None
        )
        if temporal == "transformer":
            self.temporal = TransformerTemporalEncoder(
                d_model,
                n_layers,
                n_heads,
                d_ff,
                attn_dropout,
                max_len=max_len,
                positional=positional,
            )
        elif temporal == "lstm":
            self.temporal = PackedBiLSTM(d_model, lstm_hidden, lstm_layers, attn_dropout)
        else:
            self.temporal = None

        self.sync_head = SyncHead(d_model, d_model, sync_window)
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
        return_attention: bool = False,
        compute_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
        enc = self.encode_streams(
            visual, audio, mask, face, drop_visual=drop_visual, drop_audio=drop_audio
        )
        v, a = enc["visual"], enc["audio"]

        # ⛔ Sync is measured on the separate streams, before any mixing (P8-4).
        sync = self.sync_head(v, a, mask)

        if self.cross is not None:
            cross = self.cross(v, a, mask, return_attention=return_attention)
            fused = cross["fused"]
        else:
            cross = {}
            fused = self.fuse(torch.cat([v, a], dim=-1))

        if self.temporal_kind == "transformer":
            temporal = self.temporal(
                fused, mask, return_attention=return_attention, compute_entropy=compute_entropy
            )
            sequence = temporal["sequence"]
            head_entropy = temporal.get("head_entropy")
            maps = temporal.get("attention_maps")
        elif self.temporal_kind == "lstm":
            sequence = self.temporal(fused, mask)
            head_entropy, maps = None, None
        else:
            sequence, head_entropy, maps = fused, None, None

        logit, weights = self.head(sequence, mask)
        frame_logits = self.frame_head(sequence, mask)

        # Auxiliary heads stay on the pre-fusion streams for the same reason as Phase 7:
        # routed through the shared trunk they could be satisfied from audio alone, which
        # would silently disarm P6-5.
        aux_visual, _ = self.visual_head(v, enc["visual_mask"])
        aux_audio, _ = self.audio_head(a, mask)

        out = {
            "logit": logit,
            "frame_logits": frame_logits,
            "attention": weights,
            "sequence": sequence,
            "fused": fused,
            "sync": sync["sync"],
            "sync_visual_embed": sync["visual_embed"],
            "sync_audio_embed": sync["audio_embed"],
            "aux_logits": torch.stack([aux_visual, aux_audio], dim=0),
            "aux_visual": aux_visual,
            "aux_audio": aux_audio,
        }
        if head_entropy is not None:
            out["head_entropy"] = head_entropy
        if return_attention:
            if maps is not None:
                out["attention_maps"] = maps
            out.update({k: val for k, val in cross.items() if k.endswith(tuple("0123456789"))})
        return out

    @torch.no_grad()
    def sync_score(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        mask: torch.Tensor | None = None,
        face: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean per-clip sync — the quantity ⛔ P8-6's desync test watches."""
        out = self.forward(visual, audio, mask, face)
        return self.sync_head.clip_score(out["sync"], mask)
