"""Conditional feature-space generator (P9-3; plan section I, tier 1b).

    G: (noise, label, T) -> [B, T, 256]

**It generates fused embeddings, not video.** Section I is explicit that full audio-visual
synthesis is a multi-GPU, multi-week project and is out of scope on a 4 GB GTX 1650. What is
in scope is generating sequences in the 256-d space `AttentionFusionModel` produces *after*
fusion -- vectors, not pixels -- which costs almost nothing to train and is still honestly
describable as GAN augmentation in the report's sense.

**Conditioning is on the clip label**, so the generator learns two distributions (real-like
and fake-like) rather than one blurred average. Without it, every generated sample would sit
between the classes and the classifier would be trained on contradictory supervision.

⚠️ **The known limitation, stated up front.** The target distribution is captured from a
*trained* checkpoint's fused layer. During F2 that layer keeps training, so generated
samples describe where the trunk used to be, not where it is. This is inherent to augmenting
in a learned space and it is one concrete reason F2 may not beat F1 -- exactly the outcome
P9-6 pre-commits to reporting honestly. It is recorded here so the result is not
rationalised after the fact.
"""

from __future__ import annotations

import torch
from torch import nn

NOISE_DIM = 64


class SequenceGenerator(nn.Module):
    """Noise + label -> a variable-length sequence of fused features.

    A GRU rather than transposed convolutions: the output length must vary per batch to
    match real clips (25 to ~500 frames), and a recurrent decoder handles that natively
    while a deconvolution stack would need a fixed length or a resize.

    ⛔ **The seed enters at every step, not only as the initial hidden state, and that is a
    fix for a measured P9-4 failure.** The first version routed all per-sample information
    through `h0` alone. A GRU driven by a constant input forgets its initial state, so by
    frame 64 every sample had relaxed to the same trajectory: measured on 592 real fused
    embeddings, `pairwise_ratio` was **0.01** against a 0.50 floor, while `std_ratio` stayed
    healthy -- the outputs differed frame to frame but every *clip mean* pointed the same way,
    which is the quantity P9-4 scores. Adding the seed to each step's input took that to 0.29
    on its own, and to 0.61-0.80 once the target space was standardised (see `train_gan`).

    **Outputs live in a standardised space.** `feat_mean` / `feat_std` are set from the real
    embeddings by `train_gan`; `forward` works in the whitened space the discriminator sees,
    and `sample` un-whitens by default so callers get vectors in the model's own fused space.
    They default to 0 / 1, so an untrained generator behaves exactly as before.
    """

    def __init__(
        self,
        d_model: int = 256,
        noise_dim: int = NOISE_DIM,
        hidden: int = 256,
        n_layers: int = 2,
        n_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.label_embed = nn.Embedding(n_classes, noise_dim)
        self.project = nn.Linear(noise_dim * 2, hidden)
        self.rnn = nn.GRU(
            hidden,
            hidden,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden, d_model)
        # A learned constant that the per-sample seed is added to at every step, so the
        # sequence is a decode of the noise rather than a transform of some other signal.
        self.step = nn.Parameter(torch.zeros(1, 1, hidden))
        self.n_layers = n_layers
        self.hidden = hidden
        # Buffers, not parameters: they describe the data, not the model, and they must
        # travel with the checkpoint so a reloaded generator samples into the same space.
        self.register_buffer("feat_mean", torch.zeros(d_model))
        self.register_buffer("feat_std", torch.ones(d_model))

    def forward(self, noise: torch.Tensor, labels: torch.Tensor, n_frames: int) -> torch.Tensor:
        if noise.shape[-1] != self.noise_dim:
            raise ValueError(f"expected noise of {self.noise_dim} dims, got {noise.shape[-1]}")
        if n_frames < 1:
            raise ValueError(f"n_frames must be >= 1, got {n_frames}")
        b = noise.shape[0]
        seed = self.project(torch.cat([noise, self.label_embed(labels)], dim=-1))
        h0 = seed.unsqueeze(0).expand(self.n_layers, b, self.hidden).contiguous()
        # The seed is added to every step, so the sample cannot be forgotten by the recurrence
        # -- see the class docstring for the measurement that forced this.
        steps = self.step.expand(b, n_frames, self.hidden) + seed.unsqueeze(1)
        seq, _ = self.rnn(steps, h0)
        return self.out(seq)

    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Record the real embeddings' per-dimension mean and std (see `train_gan`)."""
        self.feat_mean.copy_(mean.detach().to(self.feat_mean.device))
        self.feat_std.copy_(std.detach().to(self.feat_std.device).clamp_min(1e-6))

    def denormalize(self, sequence: torch.Tensor) -> torch.Tensor:
        """Whitened space -> the model's own fused space."""
        return sequence * self.feat_std + self.feat_mean

    @torch.no_grad()
    def sample(
        self,
        n: int,
        n_frames: int,
        labels: torch.Tensor | None = None,
        device=None,
        denormalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(sequences, labels)` for augmentation. Labels default to a balanced draw.

        `denormalize=False` returns the whitened output, which is the space the
        discriminator and the training-time diversity log both work in.
        """
        device = device or next(self.parameters()).device
        if labels is None:
            labels = torch.randint(0, 2, (n,), device=device)
        noise = torch.randn(n, self.noise_dim, device=device)
        out = self.forward(noise, labels, n_frames)
        return (self.denormalize(out) if denormalize else out), labels
