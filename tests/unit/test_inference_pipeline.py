"""Phase 12: the single-video pipeline (`src.inference.pipeline`).

The numerical parity against real dev predictions is `scripts/42_phase12_gate.py`'s job (it
needs the trained checkpoint and LAV-DF). These tests pin the contracts that must hold for
*any* weights: the bundle format, the single-decode path agreeing exactly with the two-pass
one, chunking for long uploads, and the output shape the API will serialise.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest
import torch

from src.inference.pipeline import (
    BUNDLE_FORMAT,
    STAGES,
    VideoAnalyzer,
    bundle_parameter_count,
    model_kwargs_from_result,
)
from src.localization.postprocess import PostProcessConfig, assert_segment_contract
from src.models.attention_model import AttentionFusionModel
from src.models.backbones.visual import build_backbone
from src.preprocessing.face import DEFAULT_MODEL

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
HAVE_LANDMARKER = DEFAULT_MODEL.exists()
needs_media = pytest.mark.skipif(
    not (HAVE_FFMPEG and HAVE_LANDMARKER),
    reason="needs ffmpeg and models/face_landmarker.task (scripts/08_fetch_models.py)",
)


def _bundle() -> dict:
    torch.manual_seed(0)
    kwargs = model_kwargs_from_result(
        {"backbone": "mobilenet_v2", "feature": "logmel", "boundary_head": True}
    )
    return {
        "format": BUNDLE_FORMAT,
        "model_kwargs": kwargs,
        "model": AttentionFusionModel(**kwargs).state_dict(),
        "backbone_name": "mobilenet_v2",
        "backbone": build_backbone("mobilenet_v2", pretrained=False).state_dict(),
        "postprocess": PostProcessConfig(threshold=0.5).as_dict(),
        "source": {"test": True},
    }


@pytest.fixture(scope="module")
def bundle() -> dict:
    return _bundle()


def _make_video(path, seconds: float, *, silent: bool = False) -> None:
    # anullsrc is endless; -shortest cuts it to the video's length.
    audio = (
        "anullsrc=r=16000:cl=mono"
        if silent
        else f"sine=frequency=440:sample_rate=16000:duration={seconds}"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=224x224:rate=25:duration={seconds}",
            "-f",
            "lavfi",
            "-i",
            audio,
            "-shortest",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


class TestBundle:
    def test_model_kwargs_match_the_section_5_1_model(self):
        kw = model_kwargs_from_result({"backbone": "mobilenet_v2", "feature": "logmel"})
        assert kw == {
            "visual_dim": 1280,
            "feature": "logmel",
            "cross_attention": True,
            "temporal": "transformer",
            "positional": True,
            "boundary": False,
        }

    def test_parameter_count(self, bundle):
        counts = bundle_parameter_count(bundle)
        assert counts["model"] > 1_000_000
        assert 2_000_000 < counts["backbone"] < 2_500_000  # MobileNetV2 trunk, no classifier

    def test_unknown_format_is_refused(self, bundle):
        with pytest.raises(ValueError, match="format"):
            VideoAnalyzer({**bundle, "format": 999}, device="cpu")


@pytest.fixture(scope="module")
def analyzer(bundle):
    with VideoAnalyzer(bundle, device="cpu") as a:
        yield a


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("media") / "clip.mp4"
    _make_video(path, 2.0)
    return path


@needs_media
class TestAnalyze:
    def test_result_contract(self, analyzer, clip):
        res = analyzer.analyze(clip)
        assert res.n_frames == 50
        assert res.duration_s == pytest.approx(2.0)
        assert res.frame_scores.shape == (50,)
        assert np.all((res.frame_scores >= 0) & (res.frame_scores <= 1))
        assert 0.0 <= res.clip_score <= 1.0
        assert res.is_fake == (res.clip_score >= 0.5)
        assert_segment_contract(res.segments, res.duration_s)
        assert set(STAGES) | {"total"} == set(res.timings)
        # A test pattern has no face: every visual frame is masked, audio still drives it.
        assert res.face_found_fraction == 0.0
        d = res.as_dict()
        assert len(d["frame_scores"]) == 50 and "timings_s" in d

    def test_single_decode_equals_two_pass(self, bundle, analyzer, clip):
        """The RAM-budgeted frame buffer must not change a single number."""
        a = analyzer.analyze(clip)
        with VideoAnalyzer(bundle, device="cpu", frame_budget_mb=0.0) as two_pass:
            b = two_pass.analyze(clip)
        np.testing.assert_array_equal(a.frame_scores, b.frame_scores)
        assert a.clip_score == b.clip_score

    def test_repeat_calls_are_deterministic(self, analyzer, clip):
        a, b = analyzer.analyze(clip), analyzer.analyze(clip)
        np.testing.assert_array_equal(a.frame_scores, b.frame_scores)

    def test_silent_audio_is_flagged(self, analyzer, tmp_path):
        path = tmp_path / "silent.mp4"
        _make_video(path, 1.0, silent=True)
        res = analyzer.analyze(path)
        assert res.audio_silent
        assert np.all(np.isfinite(res.frame_scores))

    def test_warmup_runs(self, analyzer):
        assert analyzer.warmup() >= 0.0


@needs_media
def test_long_input_is_chunked_not_truncated(bundle):
    """Past the 750-frame positional table the model must stitch chunks (P10-11)."""
    with VideoAnalyzer(bundle, device="cpu") as a:
        t = 900
        rng = np.random.default_rng(0)
        frames, clip = a._forward(
            rng.standard_normal((t, 1280)).astype(np.float32),
            rng.standard_normal((t, 80)).astype(np.float32),
            np.ones(t, dtype=bool),
        )
    assert frames.shape == (t,)
    assert np.all(np.isfinite(frames)) and 0.0 <= clip <= 1.0
