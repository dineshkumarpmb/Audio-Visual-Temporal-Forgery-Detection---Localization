"""Baseline 3 — early-concat audio-visual fusion (P6-1 .. P6-5, section G).

    visual [T, Dv] --proj--> [T, 256] --\\
                                         concat -> MLP -> [T, 256] -> attn-pool -> logit
    audio  [T, Da] --1D-CNN-> [T, 256] --/

Section G recommends bidirectional cross-attention and keeps early concat as **Baseline 3
so the gain is measured, not assumed**. This module is that baseline; cross-attention
arrives in Phase 8 as Experiment E and has to beat this to justify itself.

**⛔ Built against modality collapse from the first run, not retrofitted.** Section 5.6
warns that on LAV-DF audio forgeries are far easier to detect than visual ones, so a joint
model can converge to reading audio and ignoring video — "a decent aggregate number and a
model whose multimodal claim is hollow". Phase 5 turned that from a worry into a
measurement: audio scores **0.9838** dev AUC against the visual arm's **0.7189**, and
PF-19 found part of the audio margin is a global processing artefact. So all three of
section 5.6's defences are present here by construction:

1. **Modality dropout** (P6-4, `p=0.2` per stream) — "the single most effective
   intervention". Each stream is independently zeroed for a whole clip during training, so
   neither pathway can be relied on alone and both must carry signal.
2. **Per-modality auxiliary heads** (P6-5) — small classifiers on each stream, so gradients
   reach the visual pathway even when the fused head has learned to ignore it.
3. The sync loss is component F, and arrives with Phase 8.

**Per-modality normalisation** (P6-3). The two streams arrive on wildly different scales —
frozen ImageNet activations against per-utterance CMVN'd log-mel — and a raw concat would
let the larger-norm stream dominate the first Linear purely by magnitude. Each is projected
then LayerNorm'd, so fusion sees two comparably-scaled streams and any dominance that
remains is learned rather than arithmetic.

**Two masks.** `mask` marks real frames; `face` marks frames whose visual features are
trustworthy. A frame with no detected face still has valid audio, so the streams are masked
independently — see `MultimodalFeatureDataset`.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.backbones.audio import NATIVE_DIM, build_audio_encoder
from src.models.heads.classification import AttentionPooling


class _Head(nn.Module):
    """Attention-pool -> LayerNorm -> MLP -> one logit. The Phase 4/5 head, verbatim.

    Shared by the fused pathway and both auxiliary heads so an auxiliary arm is a like-for-
    like unimodal comparison rather than a differently-shaped one.
    """

    def __init__(self, d_model: int = 256, hidden: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.pool = AttentionPooling(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled, weights = self.pool(x, mask)
        return self.classifier(self.norm(pooled)).squeeze(-1), weights


class ConcatFusionBaseline(nn.Module):
    """Baseline 3: early concat of the two streams, then an MLP and one video-level logit."""

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
    ) -> None:
        super().__init__()
        if feature not in NATIVE_DIM:
            raise ValueError(f"unknown feature {feature!r}; expected one of {sorted(NATIVE_DIM)}")
        self.visual_dim = visual_dim
        self.audio_dim = NATIVE_DIM[feature]
        self.d_model = d_model
        self.modality_dropout = modality_dropout

        # --- per-stream encoders, each ending at d_model on a comparable scale (P6-3) ---
        self.visual_proj = nn.Linear(visual_dim, d_model)
        self.visual_norm = nn.LayerNorm(d_model)
        self.audio_encoder = build_audio_encoder(feature, norm=norm)
        self.audio_norm = nn.LayerNorm(d_model)

        # --- fusion: concat -> MLP (section G, Baseline 3) ------------------------------
        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = _Head(d_model, hidden, dropout)

        # --- collapse defence #3: a head on each stream on its own (P6-5) ---------------
        self.visual_head = _Head(d_model, hidden, dropout)
        self.audio_head = _Head(d_model, hidden, dropout)

    def _drop_modalities(self, batch: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        """⛔ P6-4. Zero a whole stream for a whole clip, independently per stream.

        Per *clip*, not per frame: the failure being defended against is the model learning
        a global preference for one modality, and dropping scattered frames would not
        disturb that. Both streams may be dropped at once — the model then sees a clip with
        no evidence, which is rare (`p^2` = 4%) and teaches it to be uncertain rather than
        confident, which is the right behaviour on a clip where the detector failed.
        """
        if not self.training or self.modality_dropout <= 0:
            ones = torch.ones(batch, 1, 1, device=device)
            return ones, ones
        keep_v = (torch.rand(batch, 1, 1, device=device) >= self.modality_dropout).float()
        keep_a = (torch.rand(batch, 1, 1, device=device) >= self.modality_dropout).float()
        return keep_v, keep_a

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
        """`visual` is `(B, T, Dv)`, `audio` is `(B, T, Da)`, masks are `(B, T)`.

        `drop_visual` / `drop_audio` zero a stream at *inference*, which is section 5.6's
        collapse probe: "zero out the audio stream at inference; if performance barely
        drops, video is unused".
        """
        if visual.ndim != 3 or audio.ndim != 3:
            raise ValueError(
                f"expected (B, T, D) pairs, got {tuple(visual.shape)} / {tuple(audio.shape)}"
            )
        if visual.shape[1] != audio.shape[1]:
            raise ValueError(
                f"streams disagree on length: visual {visual.shape[1]}, audio {audio.shape[1]}. "
                "MultimodalFeatureDataset truncates to the common grid (P6-2)."
            )
        if visual.shape[-1] != self.visual_dim:
            raise ValueError(f"visual dim {visual.shape[-1]} != configured {self.visual_dim}")

        v = self.visual_norm(self.visual_proj(visual))
        a = self.audio_norm(self.audio_encoder(audio, mask))

        # The visual stream is only trustworthy where a face was found.
        if face is not None:
            v = v * face.unsqueeze(-1).to(v.dtype)

        keep_v, keep_a = self._drop_modalities(visual.shape[0], visual.device)
        if drop_visual:
            keep_v = torch.zeros_like(keep_v)
        if drop_audio:
            keep_a = torch.zeros_like(keep_a)
        v, a = v * keep_v, a * keep_a

        fused = self.fuse(torch.cat([v, a], dim=-1))
        logit, weights = self.head(fused, mask)

        visual_mask = mask if face is None else (mask & face if mask is not None else face)
        aux_visual, _ = self.visual_head(v, visual_mask)
        aux_audio, _ = self.audio_head(a, mask)

        return {
            "logit": logit,
            "attention": weights,
            "fused": fused,
            "aux_logits": torch.stack([aux_visual, aux_audio], dim=0),
            "aux_visual": aux_visual,
            "aux_audio": aux_audio,
        }

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
