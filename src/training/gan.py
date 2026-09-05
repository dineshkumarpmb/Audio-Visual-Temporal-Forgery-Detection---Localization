"""Training and collapse-monitoring for the feature-space GAN (P9-3, P9-4).

⛔ **P9-4 is a gate on using the samples at all, not a chart to admire afterwards.** Section
I lists mode collapse as the expected failure, and a collapsed generator is worse than no
generator: it injects thousands of near-identical vectors that act as a spurious, perfectly
learnable class marker, and F2 could then *beat* F1 for a reason that has nothing to do with
augmentation quality. So `diversity()` runs before any sample reaches training, and
`ModeCollapseError` is raised rather than warned.

Three numbers, because collapse has three distinct signatures:

* **`pairwise`** -- mean cosine distance between generated clips. Near zero means the
  generator emits one sequence regardless of noise: classic collapse.
* **`std_ratio`** -- generated per-dimension spread over real. Near zero means the samples
  occupy a subspace far thinner than the data, the partial collapse that survives a
  pairwise check because the few directions left still differ.
* **`nn_distance`** -- how far each generated clip sits from its nearest *real* neighbour.
  Near zero is the opposite failure: memorisation. The generator is reproducing training
  embeddings, so "augmentation" is duplication and F2 measures nothing.

⛔ **Every threshold is relative to the real data, and the first version was not.** Measured
on this project's own fused embeddings, 200 real clips have a mean pairwise cosine distance
of **0.0264** and a *median* of **0.0143** -- they sit in a very narrow cone, which is what
one should expect given the section 5.6 collapse: a representation the model builds mostly
from audio, over clips that are acoustically similar. An absolute floor of 0.02 on pairwise
distance therefore asked the generator to be more diverse than the data it is imitating, and
would have rejected a *perfect* generator about half the time. Ratios against the reference
batch are the only defensible form: collapse means "far less varied than the data", not
"less varied than a number chosen in advance".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from src.models.gan.discriminator import SequenceDiscriminator
from src.models.gan.generator import SequenceGenerator

# Below these, samples are refused. All three are fractions of what the *real* reference
# batch scores, so they adapt to however concentrated the embedding space happens to be.
# Deliberately loose: they catch a broken generator, not a mediocre one.
MIN_PAIRWISE_RATIO = 0.5
MIN_STD_RATIO = 0.10
MIN_NN_RATIO = 0.5


class ModeCollapseError(RuntimeError):
    """Raised when generated samples fail P9-4 and must not be used."""


@dataclass(frozen=True)
class Diversity:
    """Raw measurements plus the reference values they are judged against."""

    pairwise: float
    pairwise_real: float
    std_ratio: float
    nn_distance: float
    nn_real: float
    n_samples: int

    @property
    def pairwise_ratio(self) -> float:
        """Generated spread as a fraction of the real data's own spread."""
        return self.pairwise / max(self.pairwise_real, 1e-8)

    @property
    def nn_ratio(self) -> float:
        """Novelty: distance to the nearest real clip, against real-to-real spacing.

        Below 1 the samples sit closer to the training embeddings than those embeddings sit
        to each other -- at the limit, memorisation rather than generation.
        """
        return self.nn_distance / max(self.nn_real, 1e-8)

    @property
    def collapsed(self) -> bool:
        return (
            self.pairwise_ratio < MIN_PAIRWISE_RATIO
            or self.std_ratio < MIN_STD_RATIO
            or self.nn_ratio < MIN_NN_RATIO
        )

    def as_dict(self) -> dict[str, float | bool]:
        return {
            **asdict(self),
            "pairwise_ratio": self.pairwise_ratio,
            "nn_ratio": self.nn_ratio,
            "collapsed": self.collapsed,
        }


def _clip_means(batch: torch.Tensor) -> torch.Tensor:
    """`[N, T, D]` -> L2-normalised `[N, D]`, so distances are cosines."""
    pooled = batch.mean(dim=1)
    return pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def diversity(fake: torch.Tensor, real: torch.Tensor) -> Diversity:
    """P9-4's three measurements over generated `fake` against reference `real`."""
    if fake.ndim != 3 or real.ndim != 3:
        raise ValueError(f"expected [N, T, D] tensors, got {tuple(fake.shape)}/{tuple(real.shape)}")
    if len(fake) < 2:
        raise ValueError("diversity needs at least two generated samples")

    f, r = _clip_means(fake.float()), _clip_means(real.float())

    def mean_pairwise(x: torch.Tensor) -> float:
        sim = x @ x.T
        off = ~torch.eye(len(x), dtype=torch.bool, device=x.device)
        return float((1.0 - sim[off]).mean())

    pairwise, pairwise_real = mean_pairwise(f), mean_pairwise(r)
    std_ratio = float(fake.float().std(dim=(0, 1)).mean() / real.float().std(dim=(0, 1)).mean())
    nn_distance = float((1.0 - (f @ r.T).max(dim=1).values).mean())
    # Real-to-real nearest neighbour, self excluded -- the natural spacing of the data, which
    # is what "too close to the training set" has to be measured against.
    real_sim = (r @ r.T) - 2.0 * torch.eye(len(r), device=r.device)
    nn_real = float((1.0 - real_sim.max(dim=1).values).mean())
    return Diversity(pairwise, pairwise_real, std_ratio, nn_distance, nn_real, len(fake))


@torch.no_grad()
def collect_fused(model: nn.Module, loader, device: str = "cpu", max_batches: int | None = None):
    """Cache the `fused` [T, 256] sequences a trained model produces, plus their labels.

    This is the GAN's training set. It comes from a checkpoint rather than from a live model
    so the target distribution is fixed while the GAN trains -- see the generator's docstring
    for why that is also the component's main limitation.
    """
    model.eval()
    seqs, labels = [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        face = batch.get("face")
        out = model(
            batch["features"].to(device),
            batch["audio"].to(device),
            batch["mask"].to(device),
            None if face is None else face.to(device),
        )
        fused = out["fused"].detach().cpu()
        mask = batch["mask"].cpu()
        for j in range(len(fused)):
            seqs.append(fused[j][mask[j]])
            labels.append(int(batch["label"][j]))
    return seqs, torch.tensor(labels, dtype=torch.long)


def _crop(seqs: list[torch.Tensor], idx: np.ndarray, n_frames: int, rng) -> torch.Tensor:
    """Random fixed-length crops, so a batch is rectangular without padding the GAN."""
    out = []
    for i in idx:
        s = seqs[int(i)]
        if len(s) < n_frames:
            pad = s.new_zeros(n_frames - len(s), s.shape[-1])
            out.append(torch.cat([s, pad], dim=0))
        else:
            lo = int(rng.integers(0, len(s) - n_frames + 1))
            out.append(s[lo : lo + n_frames])
    return torch.stack(out)


def train_gan(
    seqs: list[torch.Tensor],
    labels: torch.Tensor,
    *,
    d_model: int = 256,
    steps: int = 1500,
    batch_size: int = 32,
    n_frames: int = 64,
    lr: float = 2e-4,
    device: str = "cpu",
    log_every: int = 250,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[SequenceGenerator, SequenceDiscriminator, list[dict]]:
    """Non-saturating GAN training on cached fused embeddings.

    `betas=(0.5, 0.999)` is the long-standing GAN default; the higher default beta1 makes the
    discriminator's momentum fight the moving target. Fixed-length crops keep every batch
    rectangular -- the generator is asked for `n_frames` and the real batch is cropped to
    match, so neither side can distinguish real from fake by length alone.
    """
    gen = SequenceGenerator(d_model=d_model).to(device)
    disc = SequenceDiscriminator(d_model=d_model).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    history: list[dict] = []
    rng = np.random.default_rng(seed)

    for step in range(steps):
        idx = rng.integers(0, len(seqs), size=batch_size)
        real = _crop(seqs, idx, n_frames, rng).to(device)
        real_labels = labels[torch.from_numpy(idx)].to(device)

        noise = torch.randn(batch_size, gen.noise_dim, device=device)
        fake_labels = torch.randint(0, 2, (batch_size,), device=device)
        fake = gen(noise, fake_labels, n_frames)

        d_real = disc(real, real_labels)
        d_fake = disc(fake.detach(), fake_labels)
        loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()

        # Non-saturating: the generator maximises log D(G(z)) instead of minimising
        # log(1 - D(G(z))), which has usable gradients while the discriminator is winning.
        d_fake2 = disc(fake, fake_labels)
        loss_g = bce(d_fake2, torch.ones_like(d_fake2))
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()

        if (step + 1) % log_every == 0 or step == 0:
            gen.eval()
            sample, _ = gen.sample(64, n_frames, device=device)
            div = diversity(sample, real)
            gen.train()
            row = {
                "step": step + 1,
                "loss_d": float(loss_d),
                "loss_g": float(loss_g),
                **div.as_dict(),
            }
            history.append(row)
            if verbose:
                print(
                    f"    step {step + 1:>5}  D {row['loss_d']:.3f}  G {row['loss_g']:.3f}  "
                    f"pairwise {row['pairwise']:.4f}  std {row['std_ratio']:.3f}  "
                    f"nn {row['nn_distance']:.4f}",
                    flush=True,
                )
    return gen, disc, history


def generate_samples(
    gen: SequenceGenerator,
    reference: torch.Tensor,
    n: int,
    n_frames: int,
    *,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, Diversity]:
    """⛔ P9-4 gate: generate, measure, and refuse to return collapsed samples."""
    gen.eval()
    samples, labels = gen.sample(n, n_frames, device=device)
    div = diversity(samples, reference.to(device))
    if div.collapsed:
        raise ModeCollapseError(
            f"generator failed P9-4 -- pairwise {div.pairwise:.4f} vs real "
            f"{div.pairwise_real:.4f} (ratio {div.pairwise_ratio:.2f}, min "
            f"{MIN_PAIRWISE_RATIO}), std_ratio {div.std_ratio:.3f} (min {MIN_STD_RATIO}), "
            f"nn {div.nn_distance:.4f} vs real spacing {div.nn_real:.4f} "
            f"(ratio {div.nn_ratio:.2f}, min {MIN_NN_RATIO}). "
            "Samples refused: a collapsed generator is a spurious learnable marker, not data."
        )
    return samples.detach(), labels, div
