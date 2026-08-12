# Audio-Visual Temporal Forgery Detection & Localization

Detect whether a video has been manipulated — and, more usefully, output **which time intervals**
were manipulated, by exploiting the fact that audio and video are two views of one physical event.

> **Status: Phase 0 of 16 (environment).** No model exists yet. Every results table in
> `reports/` is an empty template by design — see the honesty rules in `PROJECT_PLAN.md` §0.3.
> No number appears in this repository until it has been measured.

---

## Why localization rather than classification

Modern attacks don't replace a whole video. They alter a few words — turning *"I do not support
this policy"* into *"I do support this policy"* — by regenerating a ~2 second span of lip motion
and matching synthetic speech. The other 95% is genuine.

That breaks whole-video classifiers two ways: a 2 s forgery in a 20 s clip is diluted ~10:1 into
the noise floor, and a binary "fake" verdict tells a journalist or moderator nothing actionable.
This project answers **where**.

## Documents

| File | What it is |
|---|---|
| [`PROJECT_PLAN.md`](PROJECT_PLAN.md) | The full engineering plan — architecture, experiment matrix, 16 phases. Treated as a signed artifact; not edited after approval |
| [`TASKS.md`](TASKS.md) | 201-task implementation breakdown with per-phase gates |
| [`reports/decision_log.md`](reports/decision_log.md) | Living decision log (the working version of Plan Appendix B) |
| [`reports/dataset_access.md`](reports/dataset_access.md) | LAV-DF access, size, and confirmed metadata schema |
| [`reports/report_audit.md`](reports/report_audit.md) | Audit of the original project report |
| [`reports/hardware_report.md`](reports/hardware_report.md) | Measured capability of the development machine |

---

## Setup

Requires **Python 3.10** (not 3.11+ — mediapipe has wheel gaps, and the LAV-DF reference toolchain
pins `<3.11`), **ffmpeg** on PATH, and an NVIDIA GPU.

```bash
make venv        # create .venv with Python 3.10
make torch       # PyTorch from the CUDA 11.8 index — see the warning below
make install-dev # project + dev dependencies + pre-commit hooks
make check-bench # Phase 0 gate; writes reports/hardware_report.md
```

`make check` exits non-zero if anything is missing, so it works as a CI gate too.

> ### ⚠️ Never `pip install torch` from PyPI here
>
> PyPI serves **CPU-only** wheels on Windows. `torch` is deliberately excluded from
> `pyproject.toml`'s dependencies so that a later `pip install -e .` cannot silently replace a
> working CUDA build with a CPU one.
>
> This machine's driver (526.47) supports **CUDA 11.8** (needs ≥522.06) but **not CUDA 12.x**
> (needs ≥527.41), so `cu118` is required, not merely preferred. torch **2.7.1** is the last
> release with `cu118` wheels.

---

## Architecture

Two stages, because **5.94 GB of system RAM is the binding constraint — not the 4 GB of VRAM**.
Decoding video on the fly inside a training loop exhausts it. So feature extraction runs once,
offline, and is cached:

```
STAGE A — offline, run once, resumable
  video → face crops → frozen visual backbone → [T,512] embeddings → .npy on D:
  audio → log-mel (hop 640 @ 16 kHz = 40 ms = exactly 1 frame @ 25 fps) → [T,80] → .npy

STAGE B — trained many times, fits in 4 GB VRAM, ~10-45 min per run
  cached features → cross-modal attention → Transformer → {classification, localization} heads
```

This is not a compromise. Stage B training in minutes is what makes the full ablation matrix
runnable at all — and an ablation matrix is the difference between *"I built a thing"* and
*"I measured which parts matter"*.

The `hop_length=640` choice is load-bearing: at 16 kHz that is exactly 40 ms, which is exactly one
frame at 25 fps, so cross-modal alignment is an array index rather than an interpolation.

## Dataset

**LAV-DF** (Localized Audio Visual DeepFake) — 136,304 videos built on VoxCeleb2, where
manipulations are content-driven and temporally localized, with per-video ground-truth intervals.
25.6 GB. CC BY-NC 4.0, non-commercial research use only.

Cite the dataset authors if you use this work:
Cai et al., *"Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal
Method for Temporal Forgery Localization"* (DICTA 2022) and the CVIU 2023 journal version.
See [`reports/dataset_access.md`](reports/dataset_access.md).

## Repository layout

```
configs/     YAML experiment configs, validated by pydantic
data/        gitignored.  raw/ is immutable; interim/ and features/ are derived
src/         core library — never imports from api/ or scripts/
api/         FastAPI service
frontend/    React + Vite; the timeline visualization is the centrepiece
scripts/     numbered pipeline entrypoints (00_check_env → 10_export_model)
tests/       unit / integration / property
experiments/ append-only run directories
reports/     written deliverables
```

Three rules worth stating: `data/raw/` is never modified, `experiments/` is never overwritten, and
dependencies point one way so the core is testable without a web server.

## License

Code: MIT. The LAV-DF dataset is CC BY-NC 4.0 and is **not** redistributed here.
