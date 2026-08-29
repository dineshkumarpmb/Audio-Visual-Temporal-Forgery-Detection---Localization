"""Frozen visual backbones for Decision D-1 (P4-1).

Both arms of Experiment J live here: **ResNet-18** (512-d) and **MobileNetV2** (1280-d).
The plan flags D-1 as a genuine contradiction to be settled by measurement at the Phase 4
gate, so this module deliberately makes the two interchangeable — same input contract,
same output contract, same call site — and lets the numbers decide.

**Frozen, not fine-tuned.** Weights are ImageNet-pretrained, `requires_grad=False`, and
the module is permanently in `eval()`. That is what makes the cached-feature design work:
extraction runs once offline, and every subsequent training run reads `.npy` instead of
touching a GPU-bound conv stack. It also means BatchNorm uses running statistics rather
than batch statistics, which matters on a 4 GB card where batches are small.

**Parity for the D-1 comparison.** MobileNetV2 emits 1280-d against ResNet-18's 512-d, so
the plan calls for a learned linear projection to 512 "for parity". That projection is
*learned*, so it cannot live in the frozen offline extractor — it belongs to the trained
head. The extractor therefore caches each backbone's **native** dimension, and
`src/models/heads/classification.py` projects to a common width. Both arms get that
projection, including ResNet-18's 512→512: giving it only to MobileNetV2 would hand one
arm extra capacity and confound the very comparison D-1 exists to make.

**Precision (PF-4).** The plan says "autocast fp16" for extraction. On this GTX 1650
(TU117, the one Turing die with no tensor cores) fp16 is 5-8x *slower* than fp32, measured
in Phase 0. So compute is fp32 and only the cached output is cast to fp16 — that is
storage, not arithmetic, and halves the feature cache for free.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

# ImageNet normalisation, as the pretrained weights expect. Applied inside the module so a
# caller cannot forget it -- feeding raw uint8 would silently produce garbage features.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BACKBONES = ("resnet18", "mobilenet_v2")
NATIVE_DIM = {"resnet18": 512, "mobilenet_v2": 1280}


@dataclass(frozen=True)
class BackboneInfo:
    name: str
    feature_dim: int
    n_params: int
    n_trainable: int


class FrozenVisualBackbone(nn.Module):
    """ImageNet-pretrained 2D CNN with the classifier removed, frozen in eval mode.

    Input:  float tensor `(B, 3, H, W)` in [0, 1] (or uint8, which is scaled here).
    Output: float32 `(B, feature_dim)` — 512 for resnet18, 1280 for mobilenet_v2.
    """

    def __init__(self, name: str = "resnet18", *, pretrained: bool = True) -> None:
        super().__init__()
        if name not in BACKBONES:
            raise ValueError(f"unknown backbone {name!r}; expected one of {BACKBONES}")
        self.name = name

        self.body, self.feature_dim = _build(name, pretrained)

        # Normalisation as buffers so they move with .to(device) and are saved in the
        # state dict -- a device mismatch here is an obscure runtime error otherwise.
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        self.freeze()

    def freeze(self) -> None:
        """Disable gradients and pin eval mode. Idempotent."""
        for p in self.body.parameters():
            p.requires_grad_(False)
        self.body.eval()

    def train(self, mode: bool = True) -> FrozenVisualBackbone:  # noqa: FBT001, FBT002
        """Stay in eval whatever the caller does.

        `Trainer.train()` walks the whole module tree; without this override it would flip
        the backbone's BatchNorm layers into batch-statistics mode mid-training, silently
        changing the features under the head and making cached and live features disagree.
        """
        super().train(False)
        self.body.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"expected (B, 3, H, W), got {tuple(x.shape)}")
        x = (x - self.mean) / self.std
        return self.body(x).flatten(1)

    def info(self) -> BackboneInfo:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return BackboneInfo(self.name, self.feature_dim, total, trainable)


def _build(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Return the feature trunk (classifier stripped) and its output width."""
    import torchvision.models as tvm

    if name == "resnet18":
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.resnet18(weights=weights)
        # Everything up to and including avgpool; drop `fc`.
        body = nn.Sequential(*list(model.children())[:-1])
        return body, 512

    weights = tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.mobilenet_v2(weights=weights)
    body = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1))
    return body, 1280


def build_backbone(name: str = "resnet18", *, pretrained: bool = True) -> FrozenVisualBackbone:
    return FrozenVisualBackbone(name, pretrained=pretrained)
