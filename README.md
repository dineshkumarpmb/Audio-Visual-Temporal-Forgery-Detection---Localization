# Audio-Visual Temporal Forgery Detection & Localization

[![CI](https://github.com/dineshkumarpmb/Audio-Visual-Temporal-Forgery-Detection---Localization/actions/workflows/ci.yml/badge.svg)](https://github.com/dineshkumarpmb/Audio-Visual-Temporal-Forgery-Detection---Localization/actions/workflows/ci.yml)

Detects whether a talking-head video has been manipulated — and **where**: it outputs the forged
time segments in seconds, not just a fake/real verdict. Audio and video are treated as two views
of one event, fused with cross-modal attention and a Transformer, on a 4 GB laptop GPU.

A from-scratch rebuild of a final-year project whose source code was lost. Every number in this
README is measured and traced to the JSON an experiment wrote — `make check-docs` recomputes them.

![The result page for a partially fake video: verdict, key figures, a timeline with two forged spans, the per-frame score, the segment table, and the voice-or-face evidence](docs/figures/result_page.png)

> **Demo capture:** pending (task P14-11). The screenshot above is the real result page on a
> real model output, rendered without the video.

---

## Results

LAV-DF, dev-2k subset: trained on **592**<!--n:reports/experiment_f.json#arms/F0/n_clips_train|int--> clips,
tuned on 175<!--n:reports/evidence_table.json#n_dev_clips|int-->, scored **once** on a held-out test split of
392<!--n:reports/test_once.json#n_clips|int--> clips. Mean of 3 seeds.

| | test |
|---|---|
| clip ROC-AUC | **0.984**<!--n:reports/test_once.json#summary/auc/mean|3--> |
| clip F1 | 0.967<!--n:reports/test_once.json#summary/f1/mean|3--> |
| **forged-segment AP@0.5** | **0.801**<!--n:reports/test_once.json#summary/ap@0.5/mean|3--> |
| segment AP@0.75 | 0.539<!--n:reports/test_once.json#summary/ap@0.75/mean|3--> |
| mean boundary error | 97<!--n:reports/test_once.json#summary/boundary_error_ms/mean|int--> ms |
| real videos given a false segment | 2.6<!--n:reports/test_once.json#summary/fp_video_rate/mean|pct1-->% |
| end to end, raw video → segments | 4.45<!--n:reports/benchmark.json#variants/optimized/s_per_10s/median|2--> s per 10 s clip (GTX 1650) |
| model file | 31.1<!--n:reports/model_export.json#size_mb|1--> MB |

**What the numbers do not say.** Read [`reports/final_report.md` §5](reports/final_report.md#5-limitations)
before quoting any of them:

- **The model reads the soundtrack and ignores the face.** Removing the whole video stream costs
  0.0016<!--n:reports/test_once.json#summary/visual_reliance_auc/mean|4--> AUC. Six defences against this collapse were tried; none worked.
- **Part of what it reads is a dataset artefact.** LAV-DF re-synthesises the entire soundtrack of
  every fake, so absolute numbers are upper bounds on this data.
- **Subset scale, one dataset.** 592 of 136,304<!--n:reports/dataset_facts.json#n_total|int--> clips; nothing here measures
  generalization to other generators or real-world deepfakes.
- **Not comparable to the original project's 80%** — its protocol is unknown and its report is not
  available.

**What the ablation found** (13 claims, 3 seeds each, failures reported): only the visual backbone
(MobileNetV2, AUC +0.087<!--n:reports/evidence_table.json#claims/0/metrics/auc/delta|sign3-->), the audio features (log-Mel,
+0.216<!--n:reports/evidence_table.json#claims/1/metrics/auc/delta|sign3-->) and temporal context (+0.084<!--n:reports/evidence_table.json#claims/3/metrics/auc/delta|sign3-->)
clearly helped. Early fusion *lost* to audio alone (−0.080<!--n:reports/evidence_table.json#claims/2/metrics/auc/delta|sign3-->); the GAN
augmentation added nothing (+0.001<!--n:reports/evidence_table.json#claims/8/metrics/ap@0.5/delta|sign3--> AP@0.5).

Full tables: [`reports/results.md`](reports/results.md) · per-run detail with git SHAs:
[`reports/evidence_table.md`](reports/evidence_table.md).

## How it works

![Two-stage architecture](docs/figures/architecture.svg)

**Two stages**, because 6 GB of RAM — not 4 GB of VRAM — is the binding constraint. Stage A runs
once per video: MediaPipe face landmarks → aligned 112×112 crops → a frozen MobileNetV2 (1280-d per
frame), and audio → 80-bin log-Mel with a 40 ms hop, which is exactly one video frame at 25 fps, so
the streams align by array index. Stage B is the trained model: cross-attention, a Transformer, and
three heads — clip verdict, per-frame score, segment boundaries. Post-processing tuned on dev turns
frame scores into segments.

## Run it

Requirements: **Python 3.10**, **ffmpeg** on PATH, **Node 22**, and an NVIDIA GPU for speed (CPU
works, slower). Windows paths in the docs; everything also runs on Linux (CI does).

```bash
make venv            # .venv with Python 3.10
make torch           # PyTorch 2.7.1 from the CUDA 11.8 index (never plain `pip install torch`)
make install-dev     # project + dev dependencies + pre-commit hooks
make check           # environment gate: exits non-zero if anything is missing
make models          # MediaPipe weights (hash-pinned)
make frontend-install
```

**The trained weights are not in this repository** (`models/avtfd_g_seed0.pt`, 31 MB, derived
from non-commercial data). Produce them with the pipeline below (`make export-model`), then:

```bash
make demo            # API on :8000 + web app on :5173, opens the browser; Ctrl+C stops both
```

Upload a talking-head clip with sound. The result page shows the verdict, a timeline of forged
spans (click one to play it), per-frame scores, keyframes, which stream drove each span, and
**PDF / JSON / CSV** downloads. The API alone is `make api` (docs at
`http://127.0.0.1:8000/docs`).

## Reproduce the results

The dataset is LAV-DF (CC BY-NC 4.0), read from a public Kaggle mirror — never redistributed here.
You need a Kaggle account with an API token (`make check-kaggle` verifies it).

```bash
make phase1     # metadata → manifest (leakage assertions are fatal) → dev-2k subset → stats
make phase2     # face tracking + alignment + label verification
make phase3     # log-Mel and MFCC on the video frame grid; alignment gate
make phase4     # visual features, both backbones → decision D-1
make phase5     # audio arms → decision C-1
make phase6     # early fusion (Experiment C)
make phase7     # BiLSTM + per-frame head (Experiment D)
make phase8     # attention arms (Experiment E)
make phase9     # augmentation + GAN (Experiment F)
make phase10    # localization: losses, λ, post-processing frozen on dev (Experiment G)
make phase11    # ablations H, I and the evidence table
make phase12    # export the model, benchmark, end-to-end parity check
```

Every stage is resumable and writes its report under `reports/`; every training run logs its config,
history, git SHA and predictions under `experiments/`. New training should use
`--strict-determinism` for bit-identical re-runs (PF-27).

⚠️ **The test split is scored exactly once.** This repository's single test run is recorded in
[`reports/test_once.md`](reports/test_once.md), and `reports/test_once.lock` makes
`make test-once` refuse a second one. In a fresh reproduction, delete the lock once your own model
is frozen — that run is then *your* single look at test.

Expect hours, not minutes: feature extraction dominates (the dev-2k subset is ~1,200 clips), and
each of the 66 training runs takes 1–30 minutes on a GTX 1650.

## Tests

```bash
make phase15    # exactly what CI runs: ruff, mypy, the test suite with the 70% coverage gate, front end
make check-docs # every number in the docs, recomputed from the JSON it came from
```

The ground-truth alignment and the AP metric — the two things every reported number depends on — are
checked against independent references (exact rational arithmetic; scikit-learn; the dataset
authors' evaluator). Coverage is 82% of `src/`.

## Documents

| | |
|---|---|
| [`reports/final_report.md`](reports/final_report.md) | the write-up: method, results, ablations, **limitations** |
| [`reports/results.md`](reports/results.md) | every measured number, with what it means |
| [`reports/evidence_table.md`](reports/evidence_table.md) | all arms × seeds, generated, with git SHAs |
| [`reports/decision_log.md`](reports/decision_log.md) | 33 recorded decisions, including the ones that went wrong |
| [`reports/portfolio.md`](reports/portfolio.md) · [`reports/interview_prep.md`](reports/interview_prep.md) | resume text and interview answers, from measured numbers only |
| [`PROJECT_PLAN.md`](PROJECT_PLAN.md) · [`TASKS.md`](TASKS.md) | the original plan and the 203-task record of executing it |

## Repository layout

```
src/         the core: data, preprocessing, models, losses, evaluation, inference
api/         FastAPI service (uploads, job queue, PDF report)
frontend/    React + Vite result page
scripts/     numbered pipeline entry points, 00 → 47
tests/       unit, integration, property, regression; committed tiny fixtures
configs/     frozen configurations (post-processing frozen on dev)
experiments/ one directory per training run (configs and results tracked; weights ignored)
reports/     every report, table and decision; docs/figures/ for the diagrams
```

## License and citation

Code: MIT. LAV-DF is CC BY-NC 4.0 and is not redistributed. If you use this work, cite the dataset:
Z. Cai, K. Stefanov, A. Dhall, M. Hayat, *"Do You Really Mean That? Content Driven Audio-Visual
Deepfake Dataset and Multimodal Method for Temporal Forgery Localization,"* DICTA 2022, and its
CVIU 2023 journal version.
