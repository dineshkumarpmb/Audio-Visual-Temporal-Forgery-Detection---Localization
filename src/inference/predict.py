"""Turn a trained checkpoint into saved predictions -- the only step of evaluation that needs a model.

`collect_predictions` writes the `.npz` layout `src/evaluation/runs.py` reads; everything
after it (post-processing, AP, the evidence table) is a pure function of that file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.models.attention_model import AttentionFusionModel

VISUAL_DIM = {"resnet18": 512, "mobilenet_v2": 1280}


@torch.no_grad()
def collect_predictions(model, loader, device, *, drop_visual: bool = False) -> dict:
    """Per-clip frame (and boundary) probabilities over valid frames, flattened + offsets."""
    model.eval()
    ids, labels, clip, frames, bounds, lengths = [], [], [], [], [], []
    for batch in loader:
        mask = batch["mask"].to(device)
        out = model(
            batch["features"].to(device),
            batch["audio"].to(device),
            mask,
            batch["face"].to(device),
            drop_visual=drop_visual,
        )
        fp = torch.sigmoid(out["frame_logits"]).float().cpu().numpy()
        bp = (
            torch.sigmoid(out["boundary_logits"]).float().cpu().numpy()
            if "boundary_logits" in out
            else None
        )
        cp = torch.sigmoid(out["logit"]).float().cpu().numpy()
        m = batch["mask"].numpy().astype(bool)
        for i, vid in enumerate(batch["video_id"]):
            t = int(m[i].sum())
            # Valid frames are a prefix: collate_paired pads at the end only.
            ids.append(vid)
            labels.append(int(batch["label"][i]))
            clip.append(float(cp[i]))
            frames.append(fp[i, :t])
            if bp is not None:
                bounds.append(bp[i, :t])
            lengths.append(t)
    return {
        "video_ids": np.array(ids),
        "labels": np.array(labels, dtype=np.int8),
        "clip_scores": np.array(clip, dtype=np.float32),
        "lengths": np.array(lengths, dtype=np.int32),
        "frame_scores": np.concatenate(frames).astype(np.float32),
        "boundary_scores": (
            np.concatenate(bounds).astype(np.float32) if bounds else np.zeros((0, 2), np.float32)
        ),
    }


def load_attention_run(run_dir: str | Path, device: str) -> tuple[AttentionFusionModel, dict]:
    """Rebuild a section 5.1 model (Phases 8-11) from its `result.json` and load `best.pt`.

    Only arms built as cross-attention + Transformer + positional encoding qualify -- the
    configuration Phase 8's `full`, every Phase 9 arm and every Phase 10/11 arm share.
    Dropout rates are irrelevant in eval mode, and `load_state_dict` is strict, so a
    mismatched architecture fails loudly rather than loading half a model.
    """
    run_dir = Path(run_dir)
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    for key, want in (("cross_attention", True), ("temporal", "transformer"), ("positional", True)):
        if key in result and result[key] != want:
            raise ValueError(f"{run_dir}: {key}={result[key]!r}, not the section 5.1 model")
    model = AttentionFusionModel(
        visual_dim=VISUAL_DIM[result["backbone"]],
        feature=result["feature"],
        cross_attention=True,
        temporal="transformer",
        positional=True,
        boundary=bool(result.get("boundary_head", False)),
    )
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), result
