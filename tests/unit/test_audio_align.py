"""P3-8: audio preprocessing, with the alignment contract front and centre.

The gate these support (P3-4/P3-9) is `audio_frames == video_frames ± 1` on 100% of
files. The plan is blunt about why it is strict: "every alignment bug you let through
here becomes an unexplainable localization failure in Phase 10."

Synthetic signals throughout, so these run without the dataset. The end-to-end check on
real audio is `scripts/12_extract_audio.py`, which is the gate itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import AudioPreprocessConfig
from src.preprocessing.audio import (
    AudioError,
    assert_aligned,
    cmvn,
    extract_audio,
    fit_to_video,
    is_silent,
    log_mel,
    mfcc,
    preemphasis,
)


@pytest.fixture
def cfg() -> AudioPreprocessConfig:
    return AudioPreprocessConfig()


def tone(freq: float, n_frames: int, cfg: AudioPreprocessConfig, amp: float = 0.5) -> np.ndarray:
    t = np.arange(n_frames * cfg.hop_length) / cfg.sample_rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestGrid:
    """The 40 ms coincidence that the whole design rests on."""

    def test_one_hop_is_one_video_frame(self, cfg):
        assert cfg.hop_length / cfg.sample_rate == 0.040
        assert cfg.frames_per_second == 25.0


class TestFitToVideo:
    def test_exact_length_untouched(self, cfg):
        y = np.ones(10 * cfg.hop_length, dtype=np.float32)
        assert np.array_equal(fit_to_video(y, 10, cfg), y)

    def test_short_audio_is_zero_padded(self, cfg):
        y = np.ones(10 * cfg.hop_length - 100, dtype=np.float32)
        out = fit_to_video(y, 10, cfg)
        assert len(out) == 10 * cfg.hop_length
        assert np.all(out[-100:] == 0.0)

    def test_long_audio_is_trimmed(self, cfg):
        y = np.ones(10 * cfg.hop_length + 500, dtype=np.float32)
        assert len(fit_to_video(y, 10, cfg)) == 10 * cfg.hop_length

    def test_padding_only_touches_the_tail(self, cfg):
        y = np.arange(10 * cfg.hop_length - 50, dtype=np.float32)
        out = fit_to_video(y, 10, cfg)
        assert np.array_equal(out[: len(y)], y)

    @pytest.mark.parametrize("n_frames", [1, 7, 103, 297, 500])
    def test_any_length_yields_exactly_n_frames(self, cfg, n_frames):
        """The reason the gate hits 100% rather than 88%."""
        for offset in (-1000, -137, 0, 137, 1000):
            length = max(1, n_frames * cfg.hop_length + offset)
            y = np.zeros(length, dtype=np.float32)
            assert len(fit_to_video(y, n_frames, cfg)) == n_frames * cfg.hop_length


class TestFrameCount:
    # n_frames=1 is degenerate -- 640 samples is shorter than n_fft, so librosa warns
    # about it. Kept because the invariant must hold even there; the real floor is
    # min_duration_s=1.0, i.e. 25 frames.
    @pytest.mark.filterwarnings("ignore:n_fft=1024 is too large:UserWarning")
    @pytest.mark.parametrize("n_frames", [1, 5, 50, 103, 297])
    def test_log_mel_returns_exactly_n_frames(self, cfg, n_frames):
        y = fit_to_video(tone(440.0, n_frames, cfg), n_frames, cfg)
        assert log_mel(y, cfg, n_frames).shape == (n_frames, cfg.n_mels)

    @pytest.mark.parametrize("n_frames", [5, 50, 103])
    def test_mfcc_returns_exactly_n_frames(self, cfg, n_frames):
        y = fit_to_video(tone(440.0, n_frames, cfg), n_frames, cfg)
        assert mfcc(y, cfg, n_frames).shape == (n_frames, cfg.n_mfcc)

    def test_both_arms_share_the_time_axis(self, cfg):
        """A Phase 3 success criterion: log-mel and MFCC must be interchangeable in T."""
        n = 40
        y = fit_to_video(tone(440.0, n, cfg), n, cfg)
        assert log_mel(y, cfg, n).shape[0] == mfcc(y, cfg, n).shape[0] == n


class TestToneRoundTrip:
    """P3-6: a pure tone must land in the mel bin that covers it."""

    @pytest.mark.parametrize("freq", [220.0, 440.0, 1000.0, 4000.0])
    def test_peak_lands_in_the_right_bin(self, cfg, freq):
        import librosa

        n = 50
        y = fit_to_video(tone(freq, n, cfg), n, cfg)
        spectrum = log_mel(y, cfg, n).mean(axis=0)
        peak = int(np.argmax(spectrum))

        centres = librosa.mel_frequencies(n_mels=cfg.n_mels + 2, fmin=cfg.fmin, fmax=cfg.fmax)[1:-1]
        expected = int(np.argmin(np.abs(centres - freq)))
        assert abs(peak - expected) <= 1, (
            f"{freq} Hz peaked at bin {peak} ({centres[peak]:.0f} Hz), "
            f"expected bin {expected} ({centres[expected]:.0f} Hz)"
        )

    def test_higher_tone_peaks_in_a_higher_bin(self, cfg):
        n = 50
        peaks = []
        for freq in (300.0, 1500.0, 5000.0):
            y = fit_to_video(tone(freq, n, cfg), n, cfg)
            peaks.append(int(np.argmax(log_mel(y, cfg, n).mean(axis=0))))
        assert peaks[0] < peaks[1] < peaks[2]


class TestNumericalGuards:
    def test_silence_does_not_produce_negative_infinity(self, cfg):
        """log(0) is -inf and would surface as NaN in the first batch-norm."""
        n = 20
        out = log_mel(np.zeros(n * cfg.hop_length, dtype=np.float32), cfg, n)
        assert np.all(np.isfinite(out))
        assert np.allclose(out, np.log(cfg.eps))

    def test_loud_signal_stays_finite(self, cfg):
        n = 20
        y = np.ones(n * cfg.hop_length, dtype=np.float32)
        assert np.all(np.isfinite(log_mel(y, cfg, n)))

    def test_mfcc_of_silence_is_finite(self, cfg):
        n = 20
        assert np.all(np.isfinite(mfcc(np.zeros(n * cfg.hop_length, dtype=np.float32), cfg, n)))


class TestPreemphasis:
    def test_zero_coefficient_is_identity(self):
        y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert np.array_equal(preemphasis(y, 0.0), y)

    def test_first_sample_is_preserved(self):
        y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert preemphasis(y, 0.97)[0] == 1.0

    def test_difference_relation(self):
        y = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        out = preemphasis(y, 0.5)
        assert out[1] == pytest.approx(2.0 - 0.5 * 1.0)
        assert out[2] == pytest.approx(4.0 - 0.5 * 2.0)

    def test_attenuates_a_constant_offset(self):
        """A DC signal is exactly what a high-pass should remove."""
        y = np.ones(100, dtype=np.float32)
        assert abs(preemphasis(y, 0.97)[1:].mean()) < 0.05

    def test_length_preserved(self):
        y = np.random.default_rng(0).normal(size=257).astype(np.float32)
        assert len(preemphasis(y, 0.97)) == 257


class TestCMVN:
    def test_zero_mean_unit_variance_per_coefficient(self):
        x = np.random.default_rng(0).normal(5.0, 3.0, (100, 80)).astype(np.float32)
        out = cmvn(x)
        assert np.allclose(out.mean(axis=0), 0.0, atol=1e-5)
        assert np.allclose(out.std(axis=0), 1.0, atol=1e-3)

    def test_normalises_each_coefficient_independently(self):
        x = np.zeros((50, 3), dtype=np.float32)
        x[:, 0] = np.linspace(0, 1, 50)
        x[:, 1] = np.linspace(100, 200, 50)
        out = cmvn(x)
        assert np.allclose(out[:, 0], out[:, 1], atol=1e-4)

    def test_constant_channel_does_not_divide_by_zero(self):
        x = np.ones((10, 4), dtype=np.float32)
        assert np.all(np.isfinite(cmvn(x)))

    def test_removes_a_channel_offset(self):
        """Why CMVN matters here: without it, recording conditions are a free shortcut."""
        base = np.random.default_rng(1).normal(size=(80, 20)).astype(np.float32)
        assert np.allclose(cmvn(base), cmvn(base + 7.5), atol=1e-4)


class TestSilenceDetection:
    def test_all_zero_is_silent(self):
        assert is_silent(np.zeros(1000, dtype=np.float32))

    def test_speech_is_not_silent(self):
        assert not is_silent(np.random.default_rng(0).normal(0, 0.1, 1000).astype(np.float32))

    def test_dither_below_threshold_is_silent(self):
        assert is_silent(np.full(1000, 1e-9, dtype=np.float32))


class TestAlignmentAssertion:
    def test_exact_passes(self):
        assert_aligned(103, 103, "x.mp4")

    @pytest.mark.parametrize("delta", [-1, 0, 1])
    def test_within_tolerance_passes(self, delta):
        assert_aligned(103 + delta, 103, "x.mp4")

    @pytest.mark.parametrize("delta", [-2, 2, 5])
    def test_outside_tolerance_raises(self, delta):
        with pytest.raises(AudioError, match="audio frames vs"):
            assert_aligned(103 + delta, 103, "x.mp4")

    def test_message_points_at_the_cause(self):
        with pytest.raises(AudioError, match="fit_to_video"):
            assert_aligned(100, 103, "x.mp4")


class TestExtractAudio:
    def test_missing_file_raises_audio_error(self, cfg, tmp_path):
        with pytest.raises(AudioError):
            extract_audio(tmp_path / "nope.mp4", 10, cfg)
