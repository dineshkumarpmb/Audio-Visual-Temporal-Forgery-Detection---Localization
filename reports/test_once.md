# ⛔ P11-8 — the single test run

**Test split, scored once.** Arm `full` (Experiment G's headline), 3 seeds, 392 clips (291 fake / 101 real).

- scored at 2026-10-03T21:49:22, git `679be7bd9e26`
- post-processing: `configs\postprocess_frozen.json`, frozen on dev at 2026-10-02T23:29:02 — `{"threshold": 0.2, "median_kernel": 5, "closing": 3, "min_duration_s": 0.2, "merge_gap_s": 0.0, "merge_first": false, "refine_radius": 0, "fps": 25.0}`
- model selection, λ, post-processing and every Phase 11 decision were made on dev
  before this ran; nothing was changed after it.

| metric | test (mean ± std over seeds) |
|---|---|
| auc | 0.9842 ± 0.0013 |
| ap | 0.9952 ± 0.0003 |
| accuracy | 0.9524 ± 0.0064 |
| precision | 0.9846 ± 0.0060 |
| recall | 0.9507 ± 0.0043 |
| f1 | 0.9674 ± 0.0043 |
| frame_ap | 0.8903 ± 0.0037 |
| ap@0.5 | 0.8005 ± 0.0167 |
| ap@0.75 | 0.5392 ± 0.0299 |
| ap@0.95 | 0.0236 ± 0.0212 |
| ar@100 | 0.5878 ± 0.0418 |
| ar@10 | 0.5878 ± 0.0418 |
| boundary_error_ms | 96.7 ± 14.2 |
| fp_video_rate | 0.026 ± 0.017 |
| visual_reliance_auc | 0.0016 ± 0.0034 |
| visual_reliance_ap50 | -0.0096 ± 0.0058 |

## 4-class breakdown (test)

| class | N | video AUC | frame AP | AP@0.5 |
|---|---|---|---|---|
| visual_only | 95 | 0.9660 ± 0.0043 | 0.7057 ± 0.0273 | 0.7125 ± 0.0812 |
| audio_only | 99 | 0.9869 ± 0.0054 | 0.8993 ± 0.0100 | 0.8483 ± 0.0085 |
| both | 97 | 0.9994 ± 0.0006 | 0.9210 ± 0.0084 | 0.8524 ± 0.0245 |

## Prediction files (sha256)

| seed | file | sha256 |
|---|---|---|
| seed0 | `experiments/phase10_full/seed0/test_predictions.npz` | `e27f30e211823e4b…` |
| seed1 | `experiments/phase10_full/seed1/test_predictions.npz` | `77d7fec2252c47e3…` |
| seed2 | `experiments/phase10_full/seed2/test_predictions.npz` | `da9a3d90019b032e…` |

> ⚠️ PF-17 / PF-19 / PF-22 apply to test as to dev: the absolute numbers are upper
> bounds on this data. The test split is drawn from the same LAV-DF release, so this
> measures held-out *clips*, not generalisation to unseen generators.
