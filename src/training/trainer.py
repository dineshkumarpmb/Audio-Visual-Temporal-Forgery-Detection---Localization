"""Training loop for the cached-feature Stage-B models (P4-6).

Deliberately small. Stage A did the expensive work, so this is a few hundred KB per clip
through a ~180 k-parameter head — an epoch over dev-2k is seconds, not minutes, and the
4 GB card is nowhere near the limit. The complexity budget goes into *correctness*
instead: checkpoint every epoch, resume cleanly, log everything, and never let the dev
split influence a weight update.

**Precision (PF-4).** Section 12.3 asks for AMP fp16. Phase 0 measured fp16 arithmetic at
**0.13x** fp32 on this GTX 1650 (TU117, no tensor cores), so AMP is off by default and
exposed as a flag for the Kaggle side, where P100/T4 tensor cores make it a real win.

**Checkpoint every epoch (X-4, R13).** Assume interruption. `resume()` restores model,
optimiser, epoch and best-so-far, so a killed run continues rather than restarts.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import frame_metrics, per_class_breakdown, summary
from src.losses.sync import InfoNCESyncLoss


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    grad_accum: int = 1
    pos_weight: float | None = None
    patience: int = 8
    amp: bool = False  # PF-4: off locally, worth enabling on Kaggle
    grad_clip: float = 1.0
    seed: int = 1337
    num_workers: int = 0
    # P6-5: weight on the per-modality auxiliary heads. 0 keeps the Phase 4/5 objective
    # exactly as it was; the fusion runs set it.
    aux_weight: float = 0.0
    # P7-3: weight on the per-frame head. 0 keeps Phases 4-6 unchanged; Phase 7 sets it.
    # Phase 10 replaces this BCE term with focal loss (P10-3) and adds the boundary and
    # sync terms -- doing that here would confound Experiment D with a loss change.
    frame_weight: float = 0.0
    # Frames are far more imbalanced than clips: ~5-10% of frames are forged against 73%
    # of clips. Without this the frame head predicts "real" everywhere and looks fine.
    frame_pos_weight: float | None = None
    # P8-5: weight on the auxiliary InfoNCE sync loss. Section 5.4 starts it at 0.3.
    # 0 keeps Phases 4-7 unchanged, so every earlier arm's objective is untouched.
    sync_weight: float = 0.0


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    metrics: dict = field(default_factory=dict)
    seconds: float = 0.0


class Trainer:
    """Single-model trainer. One instance per (backbone, seed) run."""

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainConfig,
        device: str = "cuda",
        run_dir: str | Path | None = None,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.run_dir = Path(run_dir) if run_dir else None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)

        pw = (
            torch.tensor([cfg.pos_weight], device=device, dtype=torch.float32)
            if cfg.pos_weight
            else None
        )
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        # Built only when used, so Phases 4-7 construct exactly what they did before.
        self.sync_criterion = InfoNCESyncLoss() if cfg.sync_weight else None
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")
        self.history: list[EpochResult] = []
        self.best_auc = -np.inf
        self.best_epoch = -1
        self.start_epoch = 0

    # ---------------------------------------------------------------- one pass

    def _frame_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """⛔ P7-2's other half: masked BCE, so padding contributes nothing to the loss.

        The plan's Phase 7 "Problems" note names padding contaminating the loss as the
        failure to design against, and it is a quiet one — a padded frame's target is 0 and
        its logit is ~0, so it *looks* like an easy correct prediction. Averaging over the
        padded tensor would therefore make the loss shrink as batches got raggeder, and the
        model would be rewarded for the batch's length mix rather than its predictions.

        The mean is over valid positions only, so the loss is comparable across batches
        regardless of how many frames were padding. `pos_weight` re-balances the ~5-10%
        positive frame rate; unweighted, "real everywhere" is a strong local optimum.
        """
        pw = (
            torch.tensor(self.cfg.frame_pos_weight, device=logits.device, dtype=logits.dtype)
            if self.cfg.frame_pos_weight
            else None
        )
        per_frame = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=pw
        )
        valid = mask.to(per_frame.dtype)
        return (per_frame * valid).sum() / valid.sum().clamp(min=1.0)

    def _forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """One batch -> (loss, model output), for a single-stream or a fused model.

        The two-stream branch is selected by the *batch* carrying an `audio` key, which only
        `collate_paired` produces. Phase 4 and Phase 5 batches take the original path
        unchanged, and with `aux_weight` and `frame_weight` defaulting to 0 their objective
        is byte-identical.
        """
        mask = batch["mask"].to(self.device, non_blocking=True)
        labels = batch["label"].to(self.device, non_blocking=True)

        if "audio" in batch:
            out = self.model(
                batch["features"].to(self.device, non_blocking=True),
                batch["audio"].to(self.device, non_blocking=True),
                mask,
                batch["face"].to(self.device, non_blocking=True) if "face" in batch else None,
            )
        else:
            out = self.model(batch["features"].to(self.device, non_blocking=True), mask)

        logits = out["logit"]
        loss = self.criterion(logits, labels)

        # P6-5: auxiliary per-modality heads. Their gradients are the reason the visual
        # pathway keeps learning even once the fused head can score well on audio alone.
        aux = out.get("aux_logits")
        if aux is not None and self.cfg.aux_weight:
            aux_loss = torch.stack([self.criterion(a, labels) for a in aux]).mean()
            loss = loss + self.cfg.aux_weight * aux_loss

        # P7-3: the per-frame head. Needs both a model that emits frame logits and a batch
        # carrying targets, so a Phase 7 model evaluated on a Phase 6 loader still runs.
        frame_logits, frames = out.get("frame_logits"), batch.get("frames")
        if frame_logits is not None and frames is not None and self.cfg.frame_weight:
            loss = loss + self.cfg.frame_weight * self._frame_loss(
                frame_logits, frames.to(self.device, non_blocking=True), mask
            )

        # P8-5: the auxiliary sync loss. ⛔ It reads `labels` because it is computed on
        # **real clips only** -- training a fake's forged span toward "aligned" would teach
        # the head to erase the very desynchrony the classifier is looking for.
        v_embed = out.get("sync_visual_embed")
        if self.sync_criterion is not None and v_embed is not None:
            loss = loss + self.cfg.sync_weight * self.sync_criterion(
                v_embed, out["sync_audio_embed"], labels, mask
            )
        return loss, out

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total, n = 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            with torch.autocast("cuda", enabled=self.cfg.amp and self.device == "cuda"):
                loss, _ = self._forward(batch)
            self.scaler.scale(loss / self.cfg.grad_accum).backward()

            if (step + 1) % self.cfg.grad_accum == 0:
                if self.cfg.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total += float(loss.item()) * len(batch["label"])
            n += len(batch["label"])

        return total / max(n, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, dict, np.ndarray, np.ndarray, list[str]]:
        self.model.eval()
        total, n = 0.0, 0
        scores, labels, classes = [], [], []
        frame_scores, frame_targets = [], []

        for batch in loader:
            loss, out = self._forward(batch)
            logits = out["logit"]
            total += float(loss.item()) * len(batch["label"])
            n += len(batch["label"])
            scores.append(torch.sigmoid(logits).float().cpu().numpy())
            labels.append(batch["label"].numpy())
            classes.extend(batch["class_name"])

            # P7-4. Flatten to valid positions here rather than storing ragged (B, T)
            # blocks: frame AP pools every frame in the split into one ranking, so the
            # clip a frame came from stops mattering the moment it is masked.
            fl, ft = out.get("frame_logits"), batch.get("frames")
            if fl is not None and ft is not None:
                keep = batch["mask"].cpu().numpy().ravel().astype(bool)
                frame_scores.append(torch.sigmoid(fl).float().cpu().numpy().ravel()[keep])
                frame_targets.append(ft.numpy().ravel()[keep])

        s, y = np.concatenate(scores), np.concatenate(labels)
        metrics = summary(s, y)
        if frame_scores:
            metrics.update(
                frame_metrics(np.concatenate(frame_scores), np.concatenate(frame_targets))
            )
        return total / max(n, 1), metrics, s, y, classes

    # ---------------------------------------------------------------- driving

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, *, verbose: bool = True):
        """Train with early stopping on **dev** AUC. Test is never touched here."""
        for epoch in range(self.start_epoch, self.cfg.epochs):
            t0 = time.time()
            # P9-2: a dataset that re-randomises per epoch has to be told which epoch it is.
            # `AugmentedPairs` folds this into its per-item seed so classical augmentation
            # varies across epochs while the spliced half stays fixed; without the call the
            # model would see one frozen perturbation of every clip for the whole run.
            # Generic by design -- datasets without the hook are untouched.
            set_epoch = getattr(train_loader.dataset, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch)
            train_loss = self.train_epoch(train_loader)
            val_loss, metrics, *_ = self.evaluate(val_loader)
            result = EpochResult(epoch, train_loss, val_loss, metrics, time.time() - t0)
            self.history.append(result)

            auc = metrics["auc"]
            improved = not np.isnan(auc) and auc > self.best_auc
            if improved:
                self.best_auc, self.best_epoch = auc, epoch
                self.save("best.pt", epoch)
            self.save("last.pt", epoch)  # X-4: every epoch, not just the best

            if verbose:
                star = " *" if improved else ""
                print(
                    f"    epoch {epoch:>3}  train {train_loss:.4f}  val {val_loss:.4f}  "
                    f"AUC {auc:.4f}  acc {metrics['accuracy']:.3f}  "
                    f"{result.seconds:.1f}s{star}",
                    flush=True,
                )

            if epoch - self.best_epoch >= self.cfg.patience:
                if verbose:
                    print(f"    early stop: no dev AUC gain in {self.cfg.patience} epochs")
                break

        return self.history

    def overfit_batch(self, batch: dict, steps: int = 300, verbose: bool = True) -> list[float]:
        """⛔ P4-7: drive a handful of samples to ~zero loss before any real training.

        If this fails the model is broken — a detached graph, a frozen parameter, a
        learning rate that cannot move anything — and no amount of real training will
        reveal which. Cheap to run, and it fails loudly.
        """
        self.model.train()
        losses = []
        for step in range(steps):
            self.optimizer.zero_grad(set_to_none=True)
            loss, _ = self._forward(batch)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.item()))
            if verbose and (step % 50 == 0 or step == steps - 1):
                print(f"    step {step:>4}  loss {losses[-1]:.6f}", flush=True)
        return losses

    # ---------------------------------------------------------------- state

    def save(self, name: str, epoch: int) -> Path | None:
        if not self.run_dir:
            return None
        path = self.run_dir / name
        tmp = path.with_suffix(".pt.tmp")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "best_auc": self.best_auc,
                "best_epoch": self.best_epoch,
                "config": asdict(self.cfg),
            },
            tmp,
        )
        tmp.replace(path)
        return path

    def resume(self, path: str | Path) -> int:
        """Restore a checkpoint. Returns the epoch to continue from."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.best_auc = ckpt.get("best_auc", -np.inf)
        self.best_epoch = ckpt.get("best_epoch", -1)
        self.start_epoch = int(ckpt["epoch"]) + 1
        return self.start_epoch

    def write_history(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps([asdict(h) for h in self.history], indent=2, default=float) + "\n",
            encoding="utf-8",
        )


def evaluate_split(trainer: Trainer, loader: DataLoader) -> dict:
    """Metrics plus the section 8.2.4 four-class breakdown."""
    loss, metrics, scores, labels, classes = trainer.evaluate(loader)
    return {
        "loss": loss,
        **metrics,
        "per_class": per_class_breakdown(scores, labels, classes),
    }
