"""The report's derived evidence (`src/inference/explain.py`) and keyframes (`keyframes.py`).

Both feed the page *and* the PDF, so a wrong number here is wrong twice. The attribution
tests pin the wording's honesty as much as the arithmetic: the caveat must always travel
with the reading.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.inference.explain import (
    attribute,
    key_figures,
    modality_summary,
    segment_evidence,
    summary_sentence,
)
from src.inference.keyframes import (
    MAX_SEGMENTS,
    SAMPLES_FOR_REAL,
    face_box,
    render_thumbnail,
    select_keyframes,
)

FPS = 25.0


class TestAttribution:
    @pytest.mark.parametrize(
        ("audio", "visual", "want"),
        [
            (0.8, 0.1, "voice"),
            (0.1, 0.8, "face"),
            (0.7, 0.6, "both"),
            (0.1, 0.12, "neither"),
            (0.18, 0.01, "voice"),  # neither crosses 0.2, but the voice leads by > 0.15
            (None, 0.5, "unknown"),
        ],
    )
    def test_cases(self, audio, visual, want):
        assert attribute(audio, visual, threshold=0.2) == want

    def test_segment_evidence_reads_the_right_frames(self):
        scores = np.zeros(100)
        scores[25:50] = 0.9
        audio = np.zeros(100)
        audio[25:50] = 0.8
        visual = np.full(100, 0.1)
        face = np.ones(100)
        face[40:50] = 0
        bnd = np.zeros((100, 2))
        bnd[25, 0], bnd[49, 1] = 0.7, 0.6
        (ev,) = segment_evidence(
            [(1.0, 2.0, 0.9)], scores, FPS, 0.2,
            audio_only=audio, visual_only=visual, boundary=bnd, face_found=face,
        )  # fmt: skip
        assert ev["frames"] == [25, 50]
        assert ev["peak_score"] == pytest.approx(0.9)
        assert ev["audio_only_mean"] == pytest.approx(0.8)
        assert ev["visual_only_mean"] == pytest.approx(0.1)
        assert ev["face_coverage"] == pytest.approx(0.6)
        assert ev["driver"] == "voice"
        assert ev["start_confidence"] == pytest.approx(0.7)
        assert ev["end_confidence"] == pytest.approx(0.6)

    def test_without_single_stream_runs_the_driver_is_unknown(self):
        (ev,) = segment_evidence([(0.0, 0.4, 0.8)], np.full(10, 0.8), FPS, 0.2)
        assert ev["driver"] == "unknown" and ev["audio_only_mean"] is None


class TestModality:
    @pytest.mark.parametrize(
        ("both", "audio", "visual", "phrase"),
        [
            (0.95, 0.94, 0.40, "rests on the voice"),
            (0.95, 0.40, 0.94, "rests on the face"),
            (0.95, 0.93, 0.93, "Either stream alone"),
            (0.95, 0.70, 0.70, "Both the voice and the face"),
        ],
    )
    def test_readings(self, both, audio, visual, phrase):
        m = modality_summary(both, audio, visual)
        assert phrase in m["reading"]
        assert "not how the video was edited" in m["caveat"]  # the caveat always travels

    def test_unavailable_without_reruns(self):
        assert modality_summary(0.9, None, None) == {"available": False}


class TestFiguresAndWords:
    def test_key_figures(self):
        kf = key_figures(
            [(1.0, 2.0, 0.8), (3.0, 3.5, 0.7)], np.array([0.1, 0.9, 0.3]), 10.0, 0.97, 0.5
        )
        assert kf["n_segments"] == 2
        assert kf["forged_s"] == pytest.approx(1.5)
        assert kf["forged_fraction"] == pytest.approx(0.15)
        assert kf["peak_frame_score"] == pytest.approx(0.9)

    def test_summary_matches_the_frontend_wording(self):
        """Verdict.tsx's `summarise` says the same four things in the same words."""
        assert (
            summary_sentence(False, [], 5.0) == "No part of this video was flagged as manipulated."
        )
        assert summary_sentence(True, [], 5.0).startswith("The video as a whole looks manipulated")
        assert summary_sentence(True, [(6.96, 8.48, 0.7), (10.08, 10.92, 0.7)], 14.4) == (
            "2 manipulated segments found, 2.36 s of the 14.4 s video."
        )
        assert "1 short moment was flagged" in summary_sentence(False, [(1, 1.4, 0.3)], 5.0)


class TestKeyframes:
    def test_real_video_gets_evenly_spaced_samples(self):
        picks = select_keyframes(100, [], FPS)
        assert [p[1] for p in picks] == ["sample"] * SAMPLES_FOR_REAL
        assert [p[0] for p in picks] == [12, 37, 62, 87]

    def test_a_span_gets_context_start_middle_end_context(self):
        picks = select_keyframes(250, [(4.0, 6.0, 0.8)], FPS)
        assert [(f, role) for f, role, _ in picks] == [
            (88, "before"), (100, "start"), (124, "middle"), (149, "end"), (161, "after"),
        ]  # fmt: skip
        assert {seg for *_, seg in picks} == {0}

    def test_context_frames_are_dropped_at_the_clip_edges(self):
        picks = select_keyframes(50, [(0.0, 2.0, 0.9)], FPS)
        assert [role for _, role, _ in picks] == ["start", "middle", "end"]

    def test_only_the_most_confident_spans_get_thumbnails(self):
        segs = [(i * 2.0, i * 2.0 + 0.5, 0.1 * (i + 1)) for i in range(6)]
        used = {seg for *_, seg in select_keyframes(400, segs, FPS)}
        assert used == {5, 4, 3, 2} and len(used) == MAX_SEGMENTS

    def test_a_spans_own_frame_wins_over_a_neighbours_context(self):
        # span 1 starts exactly where span 0's "after" frame falls
        picks = dict(
            (f, (role, seg))
            for f, role, seg in select_keyframes(300, [(1.0, 2.0, 0.9), (2.48, 3.0, 0.8)], FPS)
        )
        assert picks[62] == ("start", 1)

    def test_face_box_stays_in_frame(self):
        pts = np.array([[5, 5], [20, 5], [12, 12], [6, 20], [18, 20]], dtype=float)
        x0, y0, x1, y1 = face_box(pts, 30, 30)
        assert 0 <= x0 < x1 <= 29 and 0 <= y0 < y1 <= 29

    def test_thumbnail_is_a_small_jpeg(self):
        import cv2

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pts = np.array([[280, 200], [360, 200], [320, 240], [290, 290], [350, 290]], dtype=float)
        jpeg = render_thumbnail(frame, pts, highlight=True)
        assert jpeg[:2] == b"\xff\xd8"
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[1] == 320 and img.shape[0] == 240
        assert img[:, :, 2].max() > 150  # the red box is drawn
        assert render_thumbnail(frame, None, highlight=False)[:2] == b"\xff\xd8"
