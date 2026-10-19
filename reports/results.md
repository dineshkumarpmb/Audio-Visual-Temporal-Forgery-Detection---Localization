# Results

Every measured number in the project, in one place (P16-2). Each figure is traced: an
invisible tag after it names the JSON an experiment script wrote, and
`scripts/46_check_docs.py` recomputes it from that file. The full per-arm tables, with
every run's git SHA, are generated in [`evidence_table.md`](evidence_table.md); this page
selects from them and says what they mean.

## Read this first

- **Scale.** Everything was trained on **592**<!--n:reports/experiment_f.json#arms/F0/n_clips_train|int-->
  clips of LAV-DF's dev-2k subset, tuned on **175**<!--n:reports/evidence_table.json#n_dev_clips|int--> dev clips and scored once on
  **392**<!--n:reports/test_once.json#n_clips|int--> test clips — out of
  **136,304**<!--n:reports/dataset_facts.json#n_total|int--> in the full dataset. The plan's 10k-clip scale-up (R12) was not run.
- **Protocol.** Identity-disjoint official splits; every arm trained with 3 seeds and reported as
  mean ± std; every model choice, λ and post-processing parameter fixed on dev; the test split
  scored **exactly once** (`reports/test_once.lock`, P11-8).
- **Upper bounds.** Clip length (PF-17) and an audio processing fingerprint (PF-19, PF-22) are
  both partly readable in LAV-DF. Absolute numbers — test included — are upper bounds on this data;
  the paired deltas in the ablation table are the trustworthy quantities.

## Headline — the final model on the held-out test split

Experiment G (section 5.1 model + focal frame loss + boundary head + sync loss), 3 seeds.

| | test | |
|---|---|---|
| clip ROC-AUC | **0.984**<!--n:reports/test_once.json#summary/auc/mean|3--> ± 0.001<!--n:reports/test_once.json#summary/auc/std|3--> | the primary classification metric (§8.1) |
| clip F1 / accuracy | 0.967<!--n:reports/test_once.json#summary/f1/mean|3--> / 95.2<!--n:reports/test_once.json#summary/accuracy/mean|pct1-->% | at threshold 0.5 |
| precision / recall | 0.985<!--n:reports/test_once.json#summary/precision/mean|3--> / 0.951<!--n:reports/test_once.json#summary/recall/mean|3--> | |
| **segment AP@0.5** | **0.801**<!--n:reports/test_once.json#summary/ap@0.5/mean|3--> ± 0.017<!--n:reports/test_once.json#summary/ap@0.5/std|3--> | localization — the central claim |
| segment AP@0.75 / AP@0.95 | 0.539<!--n:reports/test_once.json#summary/ap@0.75/mean|3--> / 0.024<!--n:reports/test_once.json#summary/ap@0.95/mean|3--> | tight boundaries are the weak point |
| AR@100 | 0.588<!--n:reports/test_once.json#summary/ar@100/mean|3--> | |
| frame AP | 0.890<!--n:reports/test_once.json#summary/frame_ap/mean|3--> | per 40 ms frame |
| mean boundary error | 97<!--n:reports/test_once.json#summary/boundary_error_ms/mean|int--> ms | about 2.4 frames |
| false segments on real videos | 2.6<!--n:reports/test_once.json#summary/fp_video_rate/mean|pct1-->% of real clips | |
| AUC lost with the video removed | 0.0016<!--n:reports/test_once.json#summary/visual_reliance_auc/mean|4--> | the collapse onto audio (PF-21) |

Dev → test: AUC 0.9889<!--n:reports/evidence_table.json#summary/G/auc/mean|4--> → 0.9842<!--n:reports/test_once.json#summary/auc/mean|4-->,
AP@0.5 0.852<!--n:reports/evidence_table.json#summary/G/ap@0.5/mean|3--> → 0.801<!--n:reports/test_once.json#summary/ap@0.5/mean|3-->.
The gap is held-out clips from the same release — not generalization to other generators.

### By manipulation type (test)

| class | N | clip AUC | frame AP | AP@0.5 |
|---|---|---|---|---|
| face only (`visual_only`) | 95<!--n:reports/test_once.json#four_class/visual_only/n|int--> | 0.966<!--n:reports/test_once.json#four_class/visual_only/auc/mean|3--> | 0.706<!--n:reports/test_once.json#four_class/visual_only/frame_ap/mean|3--> | **0.712**<!--n:reports/test_once.json#four_class/visual_only/ap@0.5/mean|3--> |
| voice only (`audio_only`) | 99<!--n:reports/test_once.json#four_class/audio_only/n|int--> | 0.987<!--n:reports/test_once.json#four_class/audio_only/auc/mean|3--> | 0.899<!--n:reports/test_once.json#four_class/audio_only/frame_ap/mean|3--> | 0.848<!--n:reports/test_once.json#four_class/audio_only/ap@0.5/mean|3--> |
| both | 97<!--n:reports/test_once.json#four_class/both/n|int--> | 0.999<!--n:reports/test_once.json#four_class/both/auc/mean|3--> | 0.921<!--n:reports/test_once.json#four_class/both/frame_ap/mean|3--> | 0.852<!--n:reports/test_once.json#four_class/both/ap@0.5/mean|3--> |

Face-only edits are found worst. Their soundtrack is untouched, so the 0.966<!--n:reports/test_once.json#four_class/visual_only/auc/mean|3--> AUC on them is
largely the audio fingerprint (PF-19), not the face being read.

## Table 8.2.1 — classification by architecture (dev, 3 seeds)

| Exp | model | ROC-AUC | F1 |
|---|---|---|---|
| A | visual only (MobileNetV2) | 0.719<!--n:reports/evidence_table.json#summary/A/auc/mean|3--> | 0.524<!--n:reports/evidence_table.json#summary/A/f1/mean|3--> |
| B | audio only (log-Mel) | 0.984<!--n:reports/evidence_table.json#summary/B/auc/mean|3--> | 0.938<!--n:reports/evidence_table.json#summary/B/f1/mean|3--> |
| C | audio + video, early concat | 0.904<!--n:reports/evidence_table.json#summary/C/auc/mean|3--> | 0.832<!--n:reports/evidence_table.json#summary/C/f1/mean|3--> |
| D | + BiLSTM | 0.988<!--n:reports/evidence_table.json#summary/D/auc/mean|3--> | 0.980<!--n:reports/evidence_table.json#summary/D/f1/mean|3--> |
| E | cross-attention + Transformer (§5.1) | 0.993<!--n:reports/evidence_table.json#summary/E/auc/mean|3--> | 0.966<!--n:reports/evidence_table.json#summary/E/f1/mean|3--> |
| G | final (focal + boundary + sync) | 0.989<!--n:reports/evidence_table.json#summary/G/auc/mean|3--> | 0.974<!--n:reports/evidence_table.json#summary/G/f1/mean|3--> |

## Table 8.2.2 — localization (dev unless marked)

| Exp | AP@0.5 | AP@0.75 | boundary error | false segments on real |
|---|---|---|---|---|
| E | 0.802<!--n:reports/evidence_table.json#summary/E/ap@0.5/mean|3--> | 0.581<!--n:reports/evidence_table.json#summary/E/ap@0.75/mean|3--> | 73<!--n:reports/evidence_table.json#summary/E/boundary_error_ms/mean|int--> ms | 0.0<!--n:reports/evidence_table.json#summary/E/fp_video_rate/mean|pct1-->% |
| G-bce (control) | 0.846<!--n:reports/evidence_table.json#summary/G-bce/ap@0.5/mean|3--> | 0.642<!--n:reports/evidence_table.json#summary/G-bce/ap@0.75/mean|3--> | 69<!--n:reports/evidence_table.json#summary/G-bce/boundary_error_ms/mean|int--> ms | 2.0<!--n:reports/evidence_table.json#summary/G-bce/fp_video_rate/mean|pct1-->% |
| G-focal | 0.844<!--n:reports/evidence_table.json#summary/G-focal/ap@0.5/mean|3--> | 0.679<!--n:reports/evidence_table.json#summary/G-focal/ap@0.75/mean|3--> | 61<!--n:reports/evidence_table.json#summary/G-focal/boundary_error_ms/mean|int--> ms | 4.8<!--n:reports/evidence_table.json#summary/G-focal/fp_video_rate/mean|pct1-->% |
| G (final) | 0.852<!--n:reports/evidence_table.json#summary/G/ap@0.5/mean|3--> | 0.634<!--n:reports/evidence_table.json#summary/G/ap@0.75/mean|3--> | 78<!--n:reports/evidence_table.json#summary/G/boundary_error_ms/mean|int--> ms | 0.0<!--n:reports/evidence_table.json#summary/G/fp_video_rate/mean|pct1-->% |
| **G (test)** | **0.801**<!--n:reports/test_once.json#summary/ap@0.5/mean|3--> | 0.539<!--n:reports/test_once.json#summary/ap@0.75/mean|3--> | 97<!--n:reports/test_once.json#summary/boundary_error_ms/mean|int--> ms | 2.6<!--n:reports/test_once.json#summary/fp_video_rate/mean|pct1-->% |

Tuning post-processing on dev was the largest single lever for localization:
+0.241<!--n:reports/experiment_g.json#tuning_gain_ap50/full|3--> AP@0.5 for G over the plan's defaults.

## Section 7.3 — which components earned their place

Paired by seed. *Helps / hurts* means |Δ| beats the seed spread **and** all three seeds agree.
Every claim is listed, including the ones that failed.

| claim | deciding metric | Δ | verdict |
|---|---|---|---|
| J — MobileNetV2 over ResNet-18 (D-1) | AUC | +0.087<!--n:reports/evidence_table.json#claims/0/metrics/auc/delta|sign3--> | ✅ helps |
| K — log-Mel over MFCC (C-1) | AUC | +0.216<!--n:reports/evidence_table.json#claims/1/metrics/auc/delta|sign3--> | ✅ helps |
| C > B — fusion beats audio alone | AUC | −0.080<!--n:reports/evidence_table.json#claims/2/metrics/auc/delta|sign3--> | ❌ **hurts** |
| D > C — temporal context (BiLSTM) | AUC | +0.084<!--n:reports/evidence_table.json#claims/3/metrics/auc/delta|sign3--> | ✅ helps |
| E > D — the §5.1 attention model | frame AP | −0.018<!--n:reports/evidence_table.json#claims/4/metrics/frame_ap/delta|sign3--> | ❌ **hurts** |
| G-1 — cross-attention over concat | frame AP | +0.039<!--n:reports/evidence_table.json#claims/5/metrics/frame_ap/delta|sign3--> | not significant |
| H-1 — Transformer over BiLSTM | frame AP | +0.015<!--n:reports/evidence_table.json#claims/6/metrics/frame_ap/delta|sign3--> | not significant |
| F1 > F0 — splice + classical augmentation | AP@0.5 | −0.051<!--n:reports/evidence_table.json#claims/7/metrics/ap@0.5/delta|sign3--> | not significant (frame AP −0.053<!--n:reports/evidence_table.json#claims/7/metrics/frame_ap/delta|sign3--> hurt) |
| F2 > F1 — GAN augmentation | AP@0.5 | +0.001<!--n:reports/evidence_table.json#claims/8/metrics/ap@0.5/delta|sign3--> | not significant — the GAN added nothing |
| focal over BCE frame loss | AP@0.5 | −0.002<!--n:reports/evidence_table.json#claims/9/metrics/ap@0.5/delta|sign3--> | not significant (AP@0.75 +0.037<!--n:reports/evidence_table.json#claims/9/metrics/ap@0.75/delta|sign3--> on all seeds) |
| boundary head | AP@0.5 | +0.008<!--n:reports/evidence_table.json#claims/10/metrics/ap@0.5/delta|sign3--> | not significant |
| H — sync loss | AP@0.5 | +0.002<!--n:reports/evidence_table.json#claims/11/metrics/ap@0.5/delta|sign3--> | not significant (AP@0.75 −0.096<!--n:reports/evidence_table.json#claims/11/metrics/ap@0.75/delta|sign3--> hurt) |
| I — modality dropout | AP@0.5 | +0.017<!--n:reports/evidence_table.json#claims/12/metrics/ap@0.5/delta|sign3--> | not significant |

**What the table says, plainly:** two input choices (D-1, C-1) and temporal context are the
components with a clear, significant benefit. Fusion as such *lost* to the audio arm; the
attention model, augmentation, the GAN, the boundary head, the sync loss and modality dropout
are all within seed noise at AP@0.5. The final configuration keeps them because the
pre-committed selection rule (highest mean dev AP@0.5) picked it, not because each one helps.

## Table 8.2.3 — performance (GTX 1650, 4 GB)

| | |
|---|---|
| end to end, raw video → segments | **4.45**<!--n:reports/benchmark.json#variants/optimized/s_per_10s/median|2--> s per 10 s clip (median; p90 5.26<!--n:reports/benchmark.json#variants/optimized/s_per_10s/p90|2--> s) |
| where the time goes | decode + face landmarks 3.84<!--n:reports/benchmark.json#variants/optimized/median_stage_s_per_10s/detect|2--> s; MobileNetV2 0.31<!--n:reports/benchmark.json#variants/optimized/median_stage_s_per_10s/visual|2--> s and the §5.1 model 0.05<!--n:reports/benchmark.json#variants/optimized/median_stage_s_per_10s/model|2--> s on the GPU |
| model file | 31.1<!--n:reports/model_export.json#size_mb|1--> MB (+ 3.6 MB face landmarker) |
| accuracy after optimization | unchanged on all 175 dev clips end to end (`reports/phase12_gate.md`) |

## Table 8.2.5 — the two decisions the original report left open

| decision | options | winner | margin |
|---|---|---|---|
| D-1 visual backbone | ResNet-18 vs MobileNetV2 | **MobileNetV2** | AUC +0.087<!--n:reports/evidence_table.json#claims/0/metrics/auc/delta|sign3--> |
| C-1 audio features | MFCC vs log-Mel | **log-Mel** | AUC +0.216<!--n:reports/evidence_table.json#claims/1/metrics/auc/delta|sign3--> |

Both were settled by experiment (J, K), as decision PF-3 required — not by preference.

## Against the original report (Part 9)

The original report claimed **80.00%** accuracy for its proposed system and 53.35% for the
existing one. The rebuild's test accuracy is 95.2<!--n:reports/test_once.json#summary/accuracy/mean|pct1-->% — but the two are
**not comparable**, and this page does not claim an improvement:

- the original's protocol (split, threshold, video- or frame-level) is unknown — the report
  itself is not available (PRE-1, `reports/report_audit.md`);
- this accuracy is measured at subset scale and is an upper bound (PF-17, PF-19);
- the original reported classification only. Localization — AP@0.5 0.801<!--n:reports/test_once.json#summary/ap@0.5/mean|3--> — has no counterpart to compare.

When the report is located, this section is where the comparison will be made.
