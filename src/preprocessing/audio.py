"""Audio on exactly the same temporal grid as video (P3-1 .. P3-7, section C).

The whole point of this module is one invariant:

    audio frame t  ==  video frame t

`hop_length=640` at 16 kHz is 40 ms, and one video frame at 25 fps is 40 ms, so the two
grids coincide *by construction* and cross-modal alignment is an array index rather than
an interpolation. Section C calls this "the single most important number in the
preprocessing".

⛔ **Getting to 100% needs more than the right hop.** P3-4 demands
`audio_frames == video_frames ± 1` on *every* file, with no rounding fudge. Neither
standard framing achieves that on LAV-DF, measured over all 136,304 entries:

| framing | within ±1 |
|---|---|
| `center=True` | 119,823 / 136,304 (87.9%) |
| `center=False` | 86,440 / 136,304 (63.4%) |

The residue is not rounding — the decoded track genuinely is not `video_frames × 640`
samples long, drifting from 0 to about +2.4 frames. So `fit_to_video()` pads or trims the
waveform to *exactly* `n_video_frames × hop_length` samples before the STFT. The frame
count is then correct by construction on 100% of files rather than by luck, and the
adjustment is recorded per file so the size of the lie stays visible.

Fixing it here is deliberate: section 3 of Phase 3's "expected problems" says to handle
the off-by-one at this boundary, and every alignment bug let through becomes an
unexplainable localization failure in Phase 10.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import AudioPreprocessConfig


class AudioError(RuntimeError):
    """Raised when audio cannot be decoded. Callers quarantine, never skip silently."""


@dataclass(frozen=True)
class AudioResult:
    """Features plus the provenance needed to audit the alignment."""

    features: np.ndarray  # (T, n_mels) or (T, n_mfcc), float32
    n_frames: int
    decoded_samples: int
    target_samples: int
    is_silent: bool

    @property
    def pad_samples(self) -> int:
        """Positive = zeros appended, negative = samples trimmed."""
        return self.target_samples - self.decoded_samples

    @property
    def pad_frames(self) -> float:
        return self.pad_samples / 640.0


def decode_audio(path: str | Path, cfg: AudioPreprocessConfig) -> np.ndarray:
    """P3-1: demux → forced mono → resample, via ffmpeg.

    ffmpeg rather than librosa/audioread: `-ac 1` and `-ar` make the mono downmix and the
    resample explicit and reproducible, instead of depending on whichever backend the
    library happens to select. Returns float32 in [-1, 1).
    """
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",  # forced mono downmix
        "-ar",
        str(cfg.sample_rate),  # explicit resample
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
    except (subprocess.SubprocessError, OSError) as exc:
        raise AudioError(f"ffmpeg failed for {path}: {type(exc).__name__}: {exc}") from exc

    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()[:200]
        raise AudioError(f"ffmpeg exit {proc.returncode} for {path}: {detail}")
    if not proc.stdout:
        raise AudioError(f"no audio stream decoded from {path}")

    pcm = np.frombuffer(proc.stdout, dtype="<i2")
    return (pcm.astype(np.float32) / 32768.0).copy()


def fit_to_video(y: np.ndarray, n_video_frames: int, cfg: AudioPreprocessConfig) -> np.ndarray:
    """Pad or trim the waveform to exactly `n_video_frames * hop_length` samples.

    This is what makes P3-4 hold on 100% of files instead of 88%. The adjustment is
    sub-frame in practice (0 to ~2.4 frames, i.e. under 0.1 s) and always at the tail.
    """
    target = n_video_frames * cfg.hop_length
    if len(y) == target:
        return y
    if len(y) < target:
        return np.pad(y, (0, target - len(y)), mode="constant")
    return y[:target]


def preemphasis(y: np.ndarray, coeff: float) -> np.ndarray:
    """First-order high-pass: y[n] - a*y[n-1]. Flattens the spectral tilt of voiced speech.

    Applied before the STFT so the mel bands carry comparable energy across frequency;
    without it the low bands dominate and the high-frequency roll-off that betrays a
    vocoder is buried.
    """
    if coeff <= 0:
        return y
    out = np.empty_like(y)
    out[0] = y[0]
    out[1:] = y[1:] - coeff * y[:-1]
    return out


def cmvn(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-utterance cepstral mean and variance normalisation (P3-3).

    Normalises each coefficient over time *within one clip*, which removes the recording
    channel and the speaker's average spectrum. That matters here beyond the usual
    reasons: without it a model can separate real from fake by recognising recording
    conditions rather than artefacts, and it would score well while learning nothing.
    """
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / (std + eps)


def is_silent(y: np.ndarray, threshold: float = 1e-6) -> bool:
    """P3-7: an all-zero (or effectively all-zero) waveform is `no_audio`, not data."""
    return bool(np.max(np.abs(y)) < threshold)


def log_mel(y: np.ndarray, cfg: AudioPreprocessConfig, n_frames: int) -> np.ndarray:
    """STFT → mel filterbank → log, trimmed to exactly `n_frames`.

    `center=True` with a waveform already fitted to `n_frames * hop` yields `n_frames + 1`
    frames; the trailing one is dropped. Frame `t` is therefore centred on the *start* of
    video frame `t`, with a 64 ms window overlapping its neighbours symmetrically — which
    is what you want for detecting a transition at a frame boundary.
    """
    import librosa

    spec = librosa.stft(
        y, n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.n_fft, center=True
    )
    power = np.abs(spec) ** 2
    mel_fb = librosa.filters.mel(
        sr=cfg.sample_rate, n_fft=cfg.n_fft, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax
    )
    mel = mel_fb @ power
    # eps *before* the log: a silent frame is exactly 0 power and log(0) is -inf, which
    # would propagate as NaN through the first batch-norm and be very hard to trace back.
    out = np.log(mel + cfg.eps).T.astype(np.float32)
    return out[:n_frames]


def mfcc(y: np.ndarray, cfg: AudioPreprocessConfig, n_frames: int) -> np.ndarray:
    """P3-5: the parallel MFCC path for Experiment K (decision C-1).

    Built by DCT-II on the same log-mel, so the two arms differ by exactly one transform
    and the comparison is clean — anything else would confound the experiment.
    """
    import scipy.fftpack

    mel = log_mel(y, cfg, n_frames)  # (T, n_mels)
    coeffs = scipy.fftpack.dct(mel, axis=1, type=2, norm="ortho")
    return np.ascontiguousarray(coeffs[:, : cfg.n_mfcc]).astype(np.float32)


def extract_audio(path: str | Path, n_video_frames: int, cfg: AudioPreprocessConfig) -> AudioResult:
    """Full pipeline: decode → fit → pre-emphasis → features → CMVN.

    `n_video_frames` must be the manifest's `n_frames` (metadata `video_frames`), the same
    column Phase 2 uses — see decision PF-7.
    """
    raw = decode_audio(path, cfg)
    decoded = len(raw)
    silent = is_silent(raw)

    y = fit_to_video(raw, n_video_frames, cfg)
    y = preemphasis(y, cfg.preemphasis)

    features = (
        mfcc(y, cfg, n_video_frames) if cfg.feature == "mfcc" else log_mel(y, cfg, n_video_frames)
    )
    if cfg.cmvn and not silent:
        # A silent clip has zero variance; normalising it divides noise by noise.
        features = cmvn(features)

    if not np.all(np.isfinite(features)):
        raise AudioError(f"non-finite values in features for {path} -- check the eps guard")

    return AudioResult(
        features=features.astype(np.float32),
        n_frames=int(features.shape[0]),
        decoded_samples=decoded,
        target_samples=n_video_frames * cfg.hop_length,
        is_silent=silent,
    )


def assert_aligned(n_audio: int, n_video: int, path: str | Path, *, tol: int = 1) -> None:
    """⛔ P3-4/P3-9: the gate. Strict by design — no rounding fudge.

    Section 3's Phase 3 gate note: "every alignment bug you let through here becomes an
    unexplainable localization failure in Phase 10."
    """
    if abs(n_audio - n_video) > tol:
        raise AudioError(
            f"{path}: {n_audio} audio frames vs {n_video} video frames (tolerance +/-{tol}). "
            "fit_to_video() should make this exact -- if it fires, the waveform was not "
            "fitted to n_video_frames * hop_length before the STFT."
        )
