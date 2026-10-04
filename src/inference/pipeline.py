"""One video file -> verdict + forged segments, end to end (P12-1, P12-3).

    video.mp4 ─► detect + track ─► align + MobileNetV2 ─► log-Mel ─► section 5.1 model ─► segments
                 (MediaPipe, CPU)    (streamed, GPU)       (ffmpeg)   (GPU)                (frozen §6.4)

Every training-time number in this project was produced from *cached* features: Stage A
(`scripts/12`, `scripts/13`) wrote `.npz`/`.npy`, Stage B read them. This module is the first
code that goes from a raw file to segments in one call, which is what Phase 13's API serves.

**Parity with training is the design constraint, not speed.** A demo that disagrees with the
results table is a failure (P12-5), so every step calls the *same* functions the caches were
built with -- `choose_face`, `interpolate_track`, `align_face`, the frozen backbone,
`extract_audio`, `scores_to_segments` -- and two training-time details are reproduced on
purpose rather than "fixed":

* visual features are rounded through **fp16**, because the cache stored them that way
  (`VisualFeatureConfig.store_fp16`) and the model was trained on the rounded values;
* visual features are **zeroed where no face was found**, as `MultimodalFeatureDataset` does.

`scripts/41_phase12_gate.py` checks the parity numerically against the saved dev predictions.

**Memory (P13-7's concern, applied here first).** Pass 1 decodes and detects; pass 2 aligns
and featurises in batches as crops accumulate, so only one batch of 112x112 crops is ever
alive. Pass 2 re-uses pass 1's decoded frames when they fit in `frame_budget_mb` (a 10 s
LAV-DF clip is 37 MB at 224x224) and otherwise decodes again -- the two-pass trade
`src/preprocessing/pipeline.py` made for this 6 GB machine, now paid only when a large upload
actually needs it. Decode was ~20% of the warm per-clip time before this (P12-1).

**Precision.** fp32 by default. PF-4 measured fp16 arithmetic at 0.13x fp32 on this GTX 1650
(TU117 has no tensor cores); the fp16 switches exist so `scripts/40_benchmark.py` can
measure that again on the real pipeline instead of assuming it (P12-3).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.config import AudioPreprocessConfig, VideoPreprocessConfig, VisualFeatureConfig
from src.localization.chunking import stitch_scores
from src.localization.postprocess import PostProcessConfig, scores_to_segments
from src.models.attention_model import AttentionFusionModel
from src.models.backbones.visual import build_backbone
from src.models.temporal.transformer import MAX_FRAMES
from src.preprocessing.align import align_face
from src.preprocessing.audio import extract_audio
from src.preprocessing.face import DEFAULT_MODEL, FaceLandmarkerPool, choose_face, interpolate_track
from src.preprocessing.pipeline import DEFAULT_MAX_GAP
from src.preprocessing.video import DecodeError, iter_frames

BUNDLE_FORMAT = 1
STAGES = ("detect", "align", "visual", "audio", "model", "postprocess")


@dataclass
class AnalysisResult:
    """What one call returns. Seconds are on the model's own frame grid (`T / fps`)."""

    n_frames: int
    duration_s: float
    fps: float
    clip_score: float
    is_fake: bool
    segments: list[tuple[float, float, float]]
    frame_scores: np.ndarray  # (T,) float32 probabilities
    face_found_fraction: float
    audio_silent: bool
    timings: dict[str, float] = field(default_factory=dict)

    def as_dict(self, *, include_frames: bool = True) -> dict:
        out = {
            "n_frames": self.n_frames,
            "duration_s": round(self.duration_s, 3),
            "fps": self.fps,
            "clip_score": round(self.clip_score, 6),
            "is_fake": self.is_fake,
            "segments": [
                {"start_s": round(s, 3), "end_s": round(e, 3), "confidence": round(c, 4)}
                for s, e, c in self.segments
            ],
            "face_found_fraction": round(self.face_found_fraction, 4),
            "audio_silent": self.audio_silent,
            "timings_s": {k: round(v, 4) for k, v in self.timings.items()},
        }
        if include_frames:
            out["frame_scores"] = [round(float(x), 5) for x in self.frame_scores]
        return out


# ---------------------------------------------------------------- the deployable artefact


def model_kwargs_from_result(result: dict) -> dict:
    """The constructor arguments `load_attention_run` uses, made explicit and storable."""
    return {
        "visual_dim": VisualFeatureConfig(backbone=result["backbone"]).feature_dim,
        "feature": result["feature"],
        "cross_attention": True,
        "temporal": "transformer",
        "positional": True,
        "boundary": bool(result.get("boundary_head", False)),
    }


def build_bundle(run_dir: str | Path, postprocess: dict, *, source: dict | None = None) -> dict:
    """Weights-only, self-contained model file: Stage-B + frozen backbone + frozen §6.4 config.

    `best.pt` also carries optimizer state (Adam's two moments), which is why it is ~3x the
    weights. The backbone's ImageNet weights are included so inference never depends on the
    torchvision download cache.
    """
    run_dir = Path(run_dir)
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    backbone = build_backbone(result["backbone"])
    return {
        "format": BUNDLE_FORMAT,
        "model_kwargs": model_kwargs_from_result(result),
        "model": {k: v.clone() for k, v in ckpt["model"].items()},
        "backbone_name": result["backbone"],
        "backbone": {k: v.clone() for k, v in backbone.state_dict().items()},
        "postprocess": dict(postprocess),
        "source": {"run_dir": run_dir.as_posix(), "seed": result.get("seed"), **(source or {})},
    }


def bundle_parameter_count(bundle: dict) -> dict[str, int]:
    def count(sd: dict) -> int:
        return int(sum(v.numel() for v in sd.values() if torch.is_tensor(v)))

    return {"model": count(bundle["model"]), "backbone": count(bundle["backbone"])}


# ---------------------------------------------------------------- the analyzer


class VideoAnalyzer:
    """Loads every model **once** and analyses any number of videos (P12-3, P13-8).

    Not thread-safe -- MediaPipe's landmarker and the CUDA models are shared state. Phase 13
    serialises calls with a semaphore of 1 (P13-9), which this class assumes.
    """

    def __init__(
        self,
        bundle: dict | str | Path,
        *,
        device: str | None = None,
        visual_batch: int = 64,
        backbone_fp16: bool = False,
        model_fp16: bool = False,
        landmarker_path: str | Path = DEFAULT_MODEL,
        threshold: float = 0.5,
        frame_budget_mb: float = 256.0,
    ) -> None:
        if not isinstance(bundle, dict):
            bundle = torch.load(Path(bundle), map_location="cpu", weights_only=False)
        if bundle.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"unsupported bundle format {bundle.get('format')!r}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_batch = visual_batch
        self.backbone_fp16 = backbone_fp16
        self.model_fp16 = model_fp16
        self.threshold = threshold
        self.frame_budget = int(frame_budget_mb * 1024**2)
        self.source = bundle.get("source", {})

        self.video_cfg = VideoPreprocessConfig()
        self.audio_cfg = AudioPreprocessConfig(feature=bundle["model_kwargs"]["feature"])
        self.postprocess = PostProcessConfig(**bundle["postprocess"])

        self.backbone = build_backbone(bundle["backbone_name"], pretrained=False)
        self.backbone.load_state_dict(bundle["backbone"])
        self.backbone = self.backbone.to(self.device)
        if backbone_fp16:
            self.backbone = self.backbone.half()

        self.model = AttentionFusionModel(**bundle["model_kwargs"])
        self.model.load_state_dict(bundle["model"])  # strict: a mismatch fails loudly
        self.model = self.model.to(self.device).eval()

        self.landmarker = FaceLandmarkerPool(landmarker_path)

    @torch.no_grad()
    def warmup(self) -> float:
        """Pay the one-off costs at startup, not on the first request (P12-3, P13-8).

        The first clip after construction measured ~40 s against ~3 s warm: CUDA context and
        cuDNN autotuning in the backbone (~23 s) and librosa/numba JIT in the log-Mel (~20 s).
        One dummy pass through every stage moves all of it here. Returns the seconds spent.
        """
        t0 = time.perf_counter()
        size = self.video_cfg.crop_size
        dummy = np.zeros((self.visual_batch, size, size, 3), dtype=np.uint8)
        x = torch.from_numpy(dummy).to(self.device).permute(0, 3, 1, 2)
        self.backbone(x.half() / 255.0 if self.backbone_fp16 else x)
        from src.preprocessing.audio import log_mel

        t = 50
        rng = np.random.default_rng(0)
        audio = log_mel(
            rng.standard_normal(t * self.audio_cfg.hop_length).astype(np.float32),
            self.audio_cfg,
            t,
        )
        visual = np.zeros((t, self.backbone.feature_dim), dtype=np.float32)
        self._forward(visual, audio[:, : self.model.audio_dim], np.ones(t, dtype=bool))
        self.landmarker.detect(np.zeros((224, 224, 3), dtype=np.uint8))
        self._sync()
        return time.perf_counter() - t0

    def close(self) -> None:
        self.landmarker.close()

    def __enter__(self) -> VideoAnalyzer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- timing ------------------------------------------------------------

    def _sync(self) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()

    # -- stages ------------------------------------------------------------

    def _track(self, path: Path):
        """Pass 1: landmarks every `detect_every_n` frames, counting frames as it goes.

        Same keyframe policy as `detect_track`; the only difference is that the frame count
        comes from the decode itself, because an upload has no manifest `n_frames` (PF-7's
        `video_frames` equals the decoded count on every file it was checked against).
        """
        keyframes: dict[int, np.ndarray] = {}
        previous = None
        n = 0
        kept: list[np.ndarray] | None = []
        kept_bytes = 0
        for i, frame in enumerate(iter_frames(path, target_fps=self.video_cfg.target_fps)):
            n = i + 1
            if kept is not None:
                kept_bytes += frame.nbytes
                if kept_bytes <= self.frame_budget:
                    kept.append(frame)
                else:  # over budget: drop the buffer and let pass 2 decode again
                    kept = None
            if i % self.video_cfg.detect_every_n:
                continue
            candidates = self.landmarker.detect(frame)
            if not candidates:
                continue
            chosen = choose_face(candidates, previous)
            keyframes[i] = chosen
            previous = chosen
        if n == 0:
            raise DecodeError(f"no frames decoded from {path}")
        return interpolate_track(keyframes, n, max_gap=DEFAULT_MAX_GAP), n, kept

    @torch.no_grad()
    def _visual(
        self, path: Path, track, n_frames: int, frames: list[np.ndarray] | None = None
    ) -> tuple[np.ndarray, float]:
        """Pass 2: align and featurise in streamed batches. Returns features + backbone seconds.

        `frames` is pass 1's buffer when it fitted the budget; otherwise the file is decoded
        again. Both sources yield the identical frame sequence.
        """
        size = self.video_cfg.crop_size
        dim = self.backbone.feature_dim
        feats = np.zeros((n_frames, dim), dtype=np.float32)
        batch = np.empty((self.visual_batch, size, size, 3), dtype=np.uint8)
        fill, start, backbone_s = 0, 0, 0.0

        def flush(k: int) -> None:
            nonlocal backbone_s, start
            t0 = time.perf_counter()
            x = torch.from_numpy(np.ascontiguousarray(batch[:k])).to(self.device)
            x = x.permute(0, 3, 1, 2)
            if self.backbone_fp16:
                x = x.half() / 255.0
            out = self.backbone(x).float().cpu().numpy()
            self._sync()
            backbone_s += time.perf_counter() - t0
            # The cache stored fp16 and the model trained on the rounded values (store_fp16).
            feats[start : start + k] = out.astype(np.float16).astype(np.float32)
            start += k

        source = (
            frames
            if frames is not None
            else iter_frames(path, target_fps=self.video_cfg.target_fps)
        )
        for i, frame in enumerate(source):
            if i >= n_frames:
                break
            batch[fill] = align_face(
                frame, track.points[i], crop_size=size, margin=self.video_cfg.face_margin
            )
            fill += 1
            if fill == self.visual_batch:
                flush(fill)
                fill = 0
        if fill:
            flush(fill)
        if start != n_frames:
            raise DecodeError(f"{path}: pass 2 decoded {start} frames, pass 1 decoded {n_frames}")
        return feats, backbone_s

    @torch.no_grad()
    def _forward(self, visual: np.ndarray, audio: np.ndarray, face: np.ndarray):
        """Stage B on one clip. Clips past the positional table are chunked (P10-11)."""
        t = len(audio)

        def run(a: int, b: int) -> tuple[np.ndarray, float]:
            v = torch.from_numpy(visual[a:b]).to(self.device).unsqueeze(0)
            au = torch.from_numpy(audio[a:b]).to(self.device).unsqueeze(0)
            mask = torch.ones(1, b - a, dtype=torch.bool, device=self.device)
            f = torch.from_numpy(face[a:b]).to(self.device).unsqueeze(0)
            # fp16 via autocast, weights stay fp32: LayerNorm/softmax keep fp32 accumulation.
            with torch.autocast(
                device_type="cuda" if self.device == "cuda" else "cpu",
                dtype=torch.float16 if self.device == "cuda" else torch.bfloat16,
                enabled=self.model_fp16,
            ):
                out = self.model(v, au, mask, f)
            frames = torch.sigmoid(out["frame_logits"].float())[0].cpu().numpy()
            return frames, float(torch.sigmoid(out["logit"].float())[0])

        if t <= MAX_FRAMES:
            frames, clip = run(0, t)
            return frames, clip

        # Long upload: average overlapping frame scores (section 6.9); the clip verdict is
        # the most suspicious chunk, since a forgery anywhere makes the video fake. LAV-DF
        # never reaches this branch (longest clip 497 frames).
        clips: list[float] = []

        def predict(a: int, b: int) -> np.ndarray:
            frames, clip = run(a, b)
            clips.append(clip)
            return frames

        return stitch_scores(t, predict).astype(np.float32), max(clips)

    # -- public ------------------------------------------------------------

    def analyze(self, path: str | Path) -> AnalysisResult:
        path = Path(path)
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        track, n_frames, frames = self._track(path)
        timings["detect"] = time.perf_counter() - t0  # decode + landmarks

        t0 = time.perf_counter()
        visual, backbone_s = self._visual(path, track, n_frames, frames)
        del frames
        pass2 = time.perf_counter() - t0
        timings["align"] = pass2 - backbone_s  # decode + warp
        timings["visual"] = backbone_s

        t0 = time.perf_counter()
        audio_res = extract_audio(path, n_frames, self.audio_cfg)
        timings["audio"] = time.perf_counter() - t0

        # Same alignment and masking as MultimodalFeatureDataset (P6-2).
        t = min(len(visual), len(audio_res.features))
        found = track.found[:t]
        visual = visual[:t] * found[:, None]
        audio = audio_res.features[:t]

        t0 = time.perf_counter()
        frame_scores, clip_score = self._forward(visual, audio, found)
        self._sync()
        timings["model"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        segments = scores_to_segments(frame_scores, self.postprocess)
        timings["postprocess"] = time.perf_counter() - t0
        timings["total"] = sum(timings[s] for s in STAGES)

        fps = self.video_cfg.target_fps
        return AnalysisResult(
            n_frames=t,
            duration_s=t / fps,
            fps=fps,
            clip_score=clip_score,
            is_fake=clip_score >= self.threshold,
            segments=segments,
            frame_scores=frame_scores.astype(np.float32),
            face_found_fraction=float(found.mean()) if t else 0.0,
            audio_silent=audio_res.is_silent,
            timings=timings,
        )
