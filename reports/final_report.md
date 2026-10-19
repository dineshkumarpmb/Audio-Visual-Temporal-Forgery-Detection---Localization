# Audio-Visual Temporal Forgery Detection & Localization — Final Report

*A rebuild, from zero, of a final-year project whose source code was lost. Every number below
is measured and traced to the JSON an experiment script wrote (`scripts/46_check_docs.py`);
negative and null results are reported with the same weight as positive ones.*

## Abstract

Most audio-visual deepfakes do not replace a whole video; they change a word or two. A clip-level
"fake" verdict is then both diluted and useless to the person who has to act on it. This project
detects manipulation in talking-head video **and localizes it in time**, outputting forged
segments in seconds. Trained and evaluated on LAV-DF at subset scale
(592<!--n:reports/experiment_f.json#arms/F0/n_clips_train|int--> training clips, a
392<!--n:reports/test_once.json#n_clips|int-->-clip test split scored once), the final model reaches
**ROC-AUC 0.984**<!--n:reports/test_once.json#summary/auc/mean|3--> and **segment AP@0.5 0.801**<!--n:reports/test_once.json#summary/ap@0.5/mean|3-->
with a 97<!--n:reports/test_once.json#summary/boundary_error_ms/mean|int--> ms mean boundary error, and runs end to end in
4.45<!--n:reports/benchmark.json#variants/optimized/s_per_10s/median|2--> s per 10 s clip on a 4 GB laptop GPU. The more important findings are
negative: on this data **the model reads the soundtrack and ignores the face** (removing the
video costs 0.0016<!--n:reports/test_once.json#summary/visual_reliance_auc/mean|4--> AUC), nothing tried changed that, and of
thirteen tested design claims only three helped.

![Two-stage architecture](../docs/figures/architecture.svg)

---

## 1. Problem and scope

The original project (a final-year report, not available locally — PRE-1) classified whole videos
and reported 80.00% accuracy. Its protocol is unknown, its 53.35% baseline sits 3.35 points
above chance, and it measured nothing about *where* a forgery is. The rebuild was therefore
framed around method, not a number to beat (plan §9.4):

1. a leak-free protocol with every number reproducible from a committed config;
2. temporal **localization**, with AP/AR metrics the original never reported;
3. an ablation of every component, with failures reported;
4. a working local system — API and browser front end — inside a 6 GB RAM / 4 GB VRAM budget.

Deployment was explicitly out of scope.

## 2. Data and protocol

**LAV-DF** — 136,304<!--n:reports/dataset_facts.json#n_total|int--> talking-head clips derived from VoxCeleb2, in which a word or
two of real speech has its face, its voice or both replaced; each fake carries labelled
`fake_periods` in seconds. Only the **dev-2k subset** was used locally: 592 train,
175<!--n:reports/evidence_table.json#n_dev_clips|int--> dev, 392 test clips, fetched from a Kaggle mirror proven byte-equivalent to
the authors' release (CL-7).

- **Splits** are the official, identity-disjoint ones; overlap of speaker identities between train
  and dev is asserted to be zero on every build (R3).
- **Labels were checked against the media**, not trusted: divergence onset between each fake and
  its original matched the labelled start on 7/7 pairs, and every one of the
  114,253<!--n:reports/gt_alignment.json#n_spans|int--> labelled spans survives the seconds → frames → seconds round trip within
  one frame — after fixing a float-rounding bug (`1.16 × 25 = 28.999…`) that had shifted
  2,938 span boundaries by a frame (P10-1).
- **Seeds.** Every arm was trained with 3 seeds; results are mean ± std and claims are judged on
  paired, per-seed deltas.
- **Test once.** Model choice, loss weights and the post-processing configuration were frozen on
  dev; the test split was scored a single time and a lock file refuses a second run (P11-8).

## 3. Architecture

**Two stages**, because 6 GB of system RAM — not 4 GB of VRAM — is the binding constraint.
Decoding video inside a training loop exhausts it, so feature extraction runs once and is cached.

**Stage A (offline).** MediaPipe face landmarks on the CPU every fifth frame, tracked between;
a 5-point similarity alignment to a 112×112 crop; a frozen ImageNet **MobileNetV2** giving a 1280-d
vector per frame. Audio is decoded by ffmpeg to 16 kHz mono and turned into an **80-bin log-Mel**
spectrogram with a hop of 640 samples — exactly 40 ms, exactly one video frame at 25 fps — so the
two streams align by array index.

**Stage B (trained, 5.85<!--n:reports/model_export.json#params_stage_b|mega2--> M parameters).** A linear projection of the visual features and a dilated
1D-CNN over the audio, both to 256-d; two layers of bidirectional cross-attention; a 4-layer
pre-norm Transformer with learned positions (clips over 750 frames are chunked and stitched);
three heads — an attention-pooled clip verdict, a per-frame score trained with focal loss, and a
start/end boundary heatmap. Training adds an InfoNCE audio-visual sync loss, auxiliary
single-modality heads and modality dropout (p = 0.2), all aimed at the collapse risk of §5.6.

**Post-processing**, tuned on dev and frozen: median filter (5 frames) → threshold 0.2 → morphological
closing (3) → drop spans under 0.2 s → segments `(start, end, confidence)`. Tuning it was the single
largest localization lever: +0.241<!--n:reports/experiment_g.json#tuning_gain_ap50/full|3--> AP@0.5 over the plan's defaults.

## 4. Results

Full tables: [`results.md`](results.md). Per-arm detail with git SHAs: [`evidence_table.md`](evidence_table.md).

### 4.1 The final model on test

| metric | test (3 seeds) |
|---|---|
| clip ROC-AUC | **0.984**<!--n:reports/test_once.json#summary/auc/mean|3--> ± 0.001<!--n:reports/test_once.json#summary/auc/std|3--> |
| F1 · accuracy | 0.967<!--n:reports/test_once.json#summary/f1/mean|3--> · 95.2<!--n:reports/test_once.json#summary/accuracy/mean|pct1-->% |
| segment AP@0.5 · AP@0.75 · AP@0.95 | **0.801**<!--n:reports/test_once.json#summary/ap@0.5/mean|3--> · 0.539<!--n:reports/test_once.json#summary/ap@0.75/mean|3--> · 0.024<!--n:reports/test_once.json#summary/ap@0.95/mean|3--> |
| AR@100 | 0.588<!--n:reports/test_once.json#summary/ar@100/mean|3--> |
| frame AP | 0.890<!--n:reports/test_once.json#summary/frame_ap/mean|3--> |
| mean boundary error | 97<!--n:reports/test_once.json#summary/boundary_error_ms/mean|int--> ms |
| real clips given a false segment | 2.6<!--n:reports/test_once.json#summary/fp_video_rate/mean|pct1-->% |

Localization is good at loose overlap and weak at tight overlap: AP falls from 0.801<!--n:reports/test_once.json#summary/ap@0.5/mean|3--> at IoU 0.5 to
0.024<!--n:reports/test_once.json#summary/ap@0.95/mean|3--> at IoU 0.95. Spans are found; their edges are typically two to three frames off.

### 4.2 By manipulation type

Face-only edits are the weak class on test — AP@0.5 0.712<!--n:reports/test_once.json#four_class/visual_only/ap@0.5/mean|3-->
against 0.848<!--n:reports/test_once.json#four_class/audio_only/ap@0.5/mean|3--> for voice-only and
0.852<!--n:reports/test_once.json#four_class/both/ap@0.5/mean|3--> for both. Section 5 explains why their clip AUC is still high.

### 4.3 Ablation — which components earned their place

Of thirteen claims tested under the §7.3 criteria (|Δ| beats the seed spread *and* all three seeds
agree), three **helped**:

- **MobileNetV2 over ResNet-18** (D-1): AUC +0.087<!--n:reports/evidence_table.json#claims/0/metrics/auc/delta|sign3-->.
- **log-Mel over MFCC** (C-1): AUC +0.216<!--n:reports/evidence_table.json#claims/1/metrics/auc/delta|sign3-->.
- **temporal context** (BiLSTM over early fusion): AUC +0.084<!--n:reports/evidence_table.json#claims/3/metrics/auc/delta|sign3-->.

Two **hurt**: early audio-visual fusion scored *below* the audio arm alone
(AUC −0.080<!--n:reports/evidence_table.json#claims/2/metrics/auc/delta|sign3-->), and the full attention model lowered frame AP against
the BiLSTM (−0.018<!--n:reports/evidence_table.json#claims/4/metrics/frame_ap/delta|sign3-->). Eight were **within seed noise**: cross-attention,
the Transformer, splice + classical augmentation, the **GAN** (F2 − F1 AP@0.5
+0.001<!--n:reports/evidence_table.json#claims/8/metrics/ap@0.5/delta|sign3-->), focal loss, the boundary head, the sync loss and modality dropout.
Two secondary effects were significant: focal loss sharpened boundaries (AP@0.75
+0.037<!--n:reports/evidence_table.json#claims/9/metrics/ap@0.75/delta|sign3-->) and the sync loss blurred them (AP@0.75
−0.096<!--n:reports/evidence_table.json#claims/11/metrics/ap@0.75/delta|sign3-->).

The plan predicted augmentation would be "likely the biggest single win" and that global attention
would beat sequential memory. At this scale both predictions were wrong, and the report says so.

### 4.4 Efficiency

End to end from a raw `.mp4`: **4.45**<!--n:reports/benchmark.json#variants/optimized/s_per_10s/median|2--> s per 10 s clip (median; p90
5.26<!--n:reports/benchmark.json#variants/optimized/s_per_10s/p90|2--> s) on a GTX 1650, with a 31.1<!--n:reports/model_export.json#size_mb|1--> MB model file. The
time is decode plus CPU face landmarks (3.84<!--n:reports/benchmark.json#variants/optimized/median_stage_s_per_10s/detect|2--> s); the model itself takes
0.05<!--n:reports/benchmark.json#variants/optimized/median_stage_s_per_10s/model|2--> s. So ONNX export was dropped and fp16 was measured slower
on this GPU (PF-29). Running the shipped pipeline end to end on all 175 dev clips changed no metric
(largest frame-score difference 5.4e-7, P12-5).

## 5. Limitations

This section is the most important one in the report.

**1. The model does not use the face.** Removing the entire video stream costs
0.0016<!--n:reports/test_once.json#summary/visual_reliance_auc/mean|4--> test AUC. Modality dropout, auxiliary heads, recurrence,
cross-attention, self-attention and a sync loss were each tried against this collapse (§5.6), and
none broke it (PF-20, PF-21). Localization is audio-driven too. A product built on this model is
an audio forgery detector with a face tracker attached.

**2. Part of what it reads is a dataset artefact.** In LAV-DF the whole soundtrack of every fake is
re-synthesised — divergence from the original starts at t = 0 on 8 of 8 checked pairs
(PF-19) — so a global audio fingerprint flags *face-only* fakes whose audio was never edited. The
log-Mel audio arm scores 0.973<!--n:reports/confound_check.json#arms/audio:logmel/blind_auc|3--> AUC on that class while being blind to
faces by construction. Clip **length** is a second shortcut: real clips are shorter, and length alone
scores AUC 0.616<!--n:reports/confound_check.json#length_auc_overall|3--> (PF-17). Every absolute number here is therefore an upper
bound on this data. The paired deltas are the trustworthy part.

**3. Subset scale.** Training used 592 of LAV-DF's 136,304 clips. The plan's 10k-clip scale-up (R12)
was not run: the full dataset is 25.6 GB against a machine with 6 GB of RAM, and the Kaggle track was
taken only as far as data access. Whether any null result above turns positive with 100× the data is
unknown.

**4. One dataset, one kind of forgery.** The test split is held-out *clips* from the same release:
same generators, same speakers' dataset, same processing. Nothing here measures generalization to
other manipulation methods, languages, compression or recording conditions. That is the largest
open question, and the honest answer to "does it work on real deepfakes?" is *not measured*.

**5. Tight boundaries.** AP@0.95 is 0.024<!--n:reports/test_once.json#summary/ap@0.95/mean|3-->. Spans are located; their edges are not
precise to the frame.

**6. Faces the tracker misses.** MediaPipe finds no face at all on 4 of 1,159 cached clips; the API
refuses those by default (PF-30).

**7. No comparison with the original.** Without the original report, its 80% cannot be compared:
protocol, split and threshold are unknown. The rebuild's 95.2% accuracy is *not* a claimed improvement.

**8. Determinism.** Default-mode training reproduces to about one seed-std, not bit for bit; strict
mode (`--strict-determinism`) is bit-identical and is the default for new runs (PF-27).

## 6. What was built

- **Pipeline** — `src/`: manifest and leakage checks, streamed preprocessing, cached features,
  every model of Experiments A–I, losses, the localization metrics (cross-checked against the
  dataset authors' evaluator), post-processing, and `VideoAnalyzer` (raw video → verdict + segments).
- **Experiments** — 66 logged training runs across 26 arms, each with its config, history, git SHA
  and saved predictions under `experiments/`; every table is regenerated from them by script.
- **API** — FastAPI with streamed uploads, PyAV-verified inputs, one analysis at a time, structured
  errors, and a PDF report (`api/`).
- **Front end** — React: the verdict, a timeline of forged spans, per-frame scores, keyframes,
  the voice-or-face evidence and downloads (`frontend/`).
- **Tests** — over 600 Python and 63 front-end tests; the ground-truth alignment and AP metric
  are checked against independent references; coverage 82% of `src/`; `mypy` clean; CI on
  GitHub Actions, which also re-checks every traced number in these documents.

## 7. If this work continued

1. **Break the audio shortcut before anything else**: re-encode real and fake audio identically
   (or use a dataset without whole-track re-synthesis) and re-measure. Every other result depends on it.
2. **Scale up** to the 10k subset on Kaggle and re-run the ablation; null results at 592 clips are weak evidence.
3. **Force the visual stream to matter**: a held-out face-only evaluation view (PF-24) and training
   signals the soundtrack cannot satisfy.
4. **Test generalization** on a second dataset before claiming anything about real-world use.

## 8. Reproducing this

`README.md` has the commands. In short: `make check`, fetch the dev-2k subset (`make fetch-meta`),
then `make phase1` … `make phase12` re-run every stage and regenerate every table; `make phase15`
runs the test suite CI runs; `make demo` serves the model. The test split is spent — its numbers are
recorded in `reports/test_once.md` and are not re-scored.

## References

- Z. Cai, K. Stefanov, A. Dhall, M. Hayat. *Do You Really Mean That? Content Driven Audio-Visual
  Deepfake Dataset and Multimodal Method for Temporal Forgery Localization.* DICTA 2022; extended in
  *Computer Vision and Image Understanding*, 2023. (LAV-DF, CC BY-NC 4.0.)
- M. Sandler et al. *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018.
- A. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
- T.-Y. Lin et al. *Focal Loss for Dense Object Detection.* ICCV 2017.
- A. van den Oord et al. *Representation Learning with Contrastive Predictive Coding.* 2018 (InfoNCE).
