"""Dilated stride-1 1D-CNN audio encoder (P5-1, P5-2; PROJECT_PLAN section E).

    [T, 80] log-mel  ->  [T, 256] per-timestep embeddings
    [T, 40] MFCC     ->  [T, 256]        (Experiment K's other arm)

4 x [Conv1d(k=3) -> BatchNorm1d -> ReLU], channels 80 -> 128 -> 256 -> 256 -> 256,
dilations `[1, 2, 4, 8]`, **stride 1 throughout**. Receptive field 1 + 2*(1+2+4+8) = **31
frames = 1.24 s** at the 40 ms grid, with no temporal downsampling anywhere.

**Why stride 1 and dilation, not pooling.** Section E: every temporal downsample is
localization precision that cannot be recovered. A stride-2 stack of 4 layers gives 16x
coarser time -- 640 ms per output step -- which makes IoU@0.75 against a ~0.65 s span
unreachable. This is the audio analogue of the `hop_length=640` discipline, and P5-2 exists
to assert it rather than trust it.

⛔ **`padding` must equal `dilation`, not the plan's literal `pad=1`.** For `k=3` a Conv1d
outputs `L + 2*padding - 2*dilation`, so `pad=1` preserves length **only at dilation 1**.
Section E's literal spec -- "Conv1d(k=3, pad=1)" alongside dilations `[1,2,4,8]` -- would
shrink the sequence by `2*(0+1+3+7) = 22` frames, silently destroying the frame-for-frame
alignment Phase 3 spent its whole gate establishing, and making Phase 10's per-frame
targets off-by-22 against the features. Recorded as **PF-18**. P5-2's assertion is exactly
the check that catches it, which is why the plan asks for the assertion.

**Padding is re-zeroed after every block.** A batch mixes clip lengths, so short clips are
padded. Dilation means a padded position influences valid positions up to 8 frames away by
the last layer, so leaving activations to grow there feeds the pooling operator fabricated
evidence -- the same failure `face_found` masking prevents on the visual side. Zeroing
after each block keeps padding inert.

⚠️ **Residual caveat: BatchNorm statistics still see padded positions.** Keeping BatchNorm
is report-faithful (section E names it), but its running mean/var are computed over the
whole padded tensor, so they shift with the batch's length mix. The masking above stops
padding from corrupting *neighbouring* activations; it does not make BN length-invariant.
`--norm group` is available as the ablation arm that removes the effect entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

DILATIONS = (1, 2, 4, 8)
CHANNELS = (128, 256, 256, 256)
NATIVE_DIM = {"logmel": 80, "mfcc": 40}


@dataclass(frozen=True)
class EncoderInfo:
    name: str
    in_dim: int
    feature_dim: int
    receptive_field: int
    n_params: int


def _norm(kind: str, channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm1d(channels)
    if kind == "group":
        # 32 is the usual default; clamp so it always divides the channel count.
        groups = 32 if channels % 32 == 0 else 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"unknown norm {kind!r}; expected 'batch' or 'group'")


class DilatedAudioEncoder(nn.Module):
    """Per-timestep audio embeddings at the input frame rate.

    Input:  `(B, T, in_dim)` float32 -- log-mel or MFCC on the 40 ms video grid.
    Output: `(B, T, 256)` float32 -- **T is preserved exactly** (P5-2).
    """

    def __init__(
        self,
        in_dim: int = 80,
        channels: tuple[int, ...] = CHANNELS,
        dilations: tuple[int, ...] = DILATIONS,
        *,
        norm: str = "batch",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(channels) != len(dilations):
            raise ValueError(f"{len(channels)} channel widths vs {len(dilations)} dilations")
        self.in_dim = in_dim
        self.feature_dim = channels[-1]
        self.dilations = tuple(dilations)

        blocks = []
        prev = in_dim
        for width, dilation in zip(channels, dilations, strict=True):
            blocks.append(
                nn.Sequential(
                    # padding == dilation is what preserves length for k=3; see PF-18.
                    nn.Conv1d(prev, width, kernel_size=3, padding=dilation, dilation=dilation),
                    _norm(norm, width),
                    nn.ReLU(inplace=True),
                    *([nn.Dropout(dropout)] if dropout > 0 else []),
                )
            )
            prev = width
        self.blocks = nn.ModuleList(blocks)

    @property
    def receptive_field(self) -> int:
        """Frames of context each output position sees. k=3 adds 2*d per layer."""
        return 1 + 2 * sum(self.dilations)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (B, T, F), got {tuple(x.shape)}")
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"input dim {x.shape[-1]} != configured {self.in_dim}. Mixing an 80-bin "
                "log-mel cache with a 40-coefficient MFCC model does this."
            )
        t_in = x.shape[1]

        # (B, T, F) -> (B, F, T): Conv1d convolves over the last axis.
        h = x.transpose(1, 2)
        m = None if mask is None else mask.unsqueeze(1).to(h.dtype)  # (B, 1, T)
        if m is not None:
            h = h * m
        for block in self.blocks:
            h = block(h)
            if m is not None:
                h = h * m

        out = h.transpose(1, 2)

        # ⛔ P5-2. Cheap, and the one regression that would silently invalidate Phase 10 --
        # a per-frame target array would no longer line up with the features it labels.
        if out.shape[1] != t_in:
            raise AssertionError(
                f"temporal resolution changed: {t_in} frames in, {out.shape[1]} out. "
                "Stride must be 1 and padding must equal dilation (PF-18)."
            )
        return out

    def info(self, name: str = "audio") -> EncoderInfo:
        return EncoderInfo(
            name=name,
            in_dim=self.in_dim,
            feature_dim=self.feature_dim,
            receptive_field=self.receptive_field,
            n_params=sum(p.numel() for p in self.parameters()),
        )


def build_audio_encoder(
    feature: str = "logmel", *, norm: str = "batch", dropout: float = 0.0
) -> DilatedAudioEncoder:
    """Encoder sized for one of Experiment K's two feature arms."""
    if feature not in NATIVE_DIM:
        raise ValueError(f"unknown feature {feature!r}; expected one of {sorted(NATIVE_DIM)}")
    return DilatedAudioEncoder(NATIVE_DIM[feature], norm=norm, dropout=dropout)
