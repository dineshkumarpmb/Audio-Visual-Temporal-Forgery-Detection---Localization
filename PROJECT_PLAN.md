# Audio-Visual Temporal Forgery Detection & Localization
## Complete Engineering / Research Implementation Plan (Rebuild from Zero)

**Author:** Dineshkumar K
**Plan date:** 2026-08-11
**Status:** PLANNING — no code to be written until this document is approved.

---

## 0. READ THIS FIRST — Provenance, Scope, and Honesty Rules

### 0.1 Source-of-truth caveat

**The original project report was NOT readable during the drafting of this plan.** The working
directory was empty and a filesystem search of `D:\`, Downloads, Desktop and Documents found no
matching PDF/DOCX.

Therefore **Part 1 of this plan is a reconstruction** built from two sources only:

| Source | What it gave us | Reliability |
|---|---|---|
| Your typed project summary in the request | Title, component list, tech stack, the two accuracy figures, the ResNet-18/MobileNetV2 contradiction | High — your own words |
| `D:\portfolio\src\locales\en.json` (portfolio entry, found on disk) | "TensorFlow, OpenCV, GANs, ResNet, LSTM, MFCC", "identify and localize forged segments in video streams" | Medium — a marketing summary, not the report |

Everything in Part 1 that could not be sourced from those two is marked **`TO VERIFY`**. When the
report becomes available, Part 1 must be re-audited against it before Phase 1 begins. Parts 2–15
are forward-looking design and are **not** affected by the missing report.

### 0.2 Scope change applied

**Deployment / containerization has been removed at your request.** Concretely:

- Phase 16 (Docker/deployment) is deleted; the old Phase 17 (Documentation) becomes **Phase 16**.
- `docker-compose.yml`, `Dockerfile`, and `deploy/` are removed from the repository structure.
- Part 15 deliverable #13 ("Deployment strategy") is replaced by **"Local run strategy"**.
- **The API and the frontend are retained** — Phases 13 and 14 stay. They run locally
  (`uvicorn` + Vite). Removing deployment removes *shipping*, not the *interface*.
- Part 1 §12 is retained but explicitly retitled, because it documents what the *original report*
  specified — it is historical analysis, not a build target.

### 0.3 Honesty rules binding this document and everything that follows

1. No fabricated dataset statistics. Published figures are labelled as such and still require
   verification against your actual download.
2. No fabricated experimental results. Every results table is a **template with empty cells**.
3. The report's 80.00% / 53.35% figures are treated as **historical claims to be independently
   reproduced or beaten**, never as assumed outcomes.
4. Where the report is ambiguous or the design has a real fork, the fork is **surfaced as a
   decision**, not silently resolved.

---

## 0.4 THE THREE THINGS THAT WILL DECIDE THIS PROJECT

Read these before anything else. They are the load-bearing findings.

### FINDING 1 — Your hardware makes the naive architecture impossible, and that is fine

Measured on this machine:

```
GPU     : NVIDIA GeForce GTX 1650, 4096 MiB VRAM, driver 526.47
CPU     : AMD Ryzen 5 3550H, 4 cores / 8 threads (mobile)
RAM     : 5.94 GB total system memory
Disk    : D: 167 GB free  |  C: 27 GB free
Python  : NOT INSTALLED
ffmpeg  : NOT INSTALLED
```

**5.94 GB of system RAM is the binding constraint — not the 4 GB of VRAM.** Windows 11 idles at
roughly 2.5–3.5 GB, leaving you ~2.5–3 GB of headroom. An end-to-end training loop that decodes
video on the fly (`ResNet-18 → LSTM` over 100+ frames, several DataLoader workers each holding
decoded frame buffers) will exhaust that and either OOM-kill or thrash the pagefile on C:, which
has only 27 GB free.

**The plan therefore mandates a two-stage, feature-cached architecture:**

```
STAGE A (run once, offline, CPU/GPU, resumable)
  video → face crops → frozen visual backbone → 512-d/frame embeddings → .npy on D:
  audio → log-mel → frozen/light audio encoder → embeddings        → .npy on D:

STAGE B (run many times, fast, fits easily in 4 GB VRAM)
  cached embeddings → fusion → self-attention → temporal → heads
```

This is not a compromise forced by poverty — it is standard practice for temporal localization,
and it buys you something valuable: **Stage B trains in minutes, so you can actually run the
full 7-experiment ablation matrix**, which is the entire point of Parts 4 and 7. An end-to-end
model you can train twice proves nothing. A cached model you can train forty times proves a lot.

Full-network fine-tuning (unfreezing the backbone) is deferred to **cloud GPU** — see Part 12.

### FINDING 2 — The visual-backbone contradiction is a real decision, and I am not resolving it by fiat

You flagged that the report says ResNet-18 in the body and MobileNetV2 in the conclusion. Correct
call. This plan does **not** pick one. It converts the contradiction into **Experiment B4a vs B4b**:
both backbones trained on identical splits, identical seeds, identical schedules, compared on
frame-level AUC *and* on extraction throughput and VRAM. See Part 5 §5.2 for the decision
framework and my *prior* (with reasoning), and Part 7 for the experiment.

The decision is made **at the Phase 4 exit gate, from measurements** — not now, and not from the
report.

### FINDING 3 — Two numbers in the report need provenance before they can be used

**The 53.35% "existing system" figure.** On a balanced binary task, chance is 50%. 53.35% is
3.35 points above chance. That is not a working system; it is a system that barely functions. Three
possible explanations, with very different consequences:

| If it was... | Then "80% vs 53.35%" is... | Action |
|---|---|---|
| A number *cited from a prior paper* on a different dataset/protocol | **Not a valid comparison** — different test set | Replace with a baseline you measure yourself |
| A model the original team *implemented and measured* | A valid but very weak baseline | Reproduce it as Baseline 0, then beat it |
| A *different task formulation* (e.g. localization scored as classification) | Apples-to-oranges | Re-derive under one protocol |

**The 80.00% figure.** A suspiciously round number next to precision 0.80 / recall 0.79 / F1 0.79.
Needs its evaluation protocol identified: video-level or frame-level? which split? what threshold?

**Both are marked `TO VERIFY` and are blocking items for Phase 1.** This plan's target-setting
(Part 9) explicitly handles the case where the comparison turns out to be invalid.

---

# PART 1 — UNDERSTANDING THE ORIGINAL PROJECT

> Reconstructed per §0.1. `TO VERIFY` = could not be sourced; confirm against the report.

## 1.1 Problem statement

Deepfake generation has moved from whole-video synthesis to **partial, localized, content-driven
manipulation**. A modern attack does not replace an entire video; it alters a few words — swapping
"I do not support this policy" into "I do support this policy" — by regenerating a two-second span
of lip motion and a matching span of synthetic speech. The other 95% of the video is genuine.

This breaks the assumption behind first-generation detectors. A whole-video classifier averages
evidence over time, so a 2-second forgery inside a 20-second clip is diluted roughly 10:1 and
disappears into the noise floor. Worse, a binary "fake" verdict is not actionable for a
journalist, moderator, or forensic analyst — they need to know **which span** to examine.

The problem this project addresses: **given a video with audio, decide whether it has been
manipulated, and if so, output the time intervals that were manipulated** — using the fact that
audio and video are two views of one physical event, so a manipulation of either leaves a
detectable inconsistency *between* them that is harder to hide than an artifact *within* either.

## 1.2 Existing-system limitations

Per the report `TO VERIFY`, augmented with the standard critique the report is presumed to make:

| # | Limitation | Why it matters |
|---|---|---|
| 1 | **Unimodal analysis** — visual-only or audio-only | Discards the cross-modal correlation that is the strongest and most generator-agnostic signal |
| 2 | **Whole-video classification only** | Cannot answer "where"; dilutes short forgeries to invisibility |
| 3 | **Frame-independent scoring** | Ignores temporal structure; a per-frame CNN cannot see that lip motion *stopped tracking* the audio |
| 4 | **Poor generalization to unseen generators** | Models latch onto generator-specific pixel artifacts that vanish on the next generator |
| 5 | **Weak/absent audio modelling** | Audio deepfakes (TTS/vocoder) go entirely undetected |
| 6 | **Reported accuracy 53.35%** | `TO VERIFY` — near chance; see §0.4 Finding 3 |
| 7 | **No confidence/uncertainty output** | A bare label is not usable evidence |
| 8 | **No deployable interface** | Research script, not a usable tool |

## 1.3 Proposed-system solution (as specified in the report)

A multimodal pipeline that:

1. Extracts **visual** features per frame with a CNN backbone (ResNet-18 / MobileNetV2 — *contested*).
2. Extracts **audio** features as MFCCs, encoded by a **1D-CNN**.
3. **Fuses** the two streams into a joint audio-visual representation.
4. Performs **lip-sync verification** — checking whether mouth motion is consistent with speech.
5. Applies **self-attention** so the model can compare every timestep against every other and
   surface locally anomalous regions.
6. Applies an **LSTM** for sequential temporal modelling.
7. Classifies real/fake via **softmax**.
8. Emits **temporal localization** — timestamps of forged spans.
9. Uses **GAN-based augmentation** to expand training diversity and improve generalization.
10. Serves results through a **Flask** web interface.

## 1.4 Functional requirements

| ID | Requirement |
|---|---|
| FR-01 | Accept an uploaded video file containing both video and audio streams |
| FR-02 | Validate the file (container, codecs, presence of both streams, duration bounds) |
| FR-03 | Decode video to frames; detect, crop and align faces |
| FR-04 | Demux and resample audio to a canonical rate |
| FR-05 | Compute visual embeddings per frame |
| FR-06 | Compute audio embeddings per audio frame |
| FR-07 | Align the two streams on a common temporal grid |
| FR-08 | Fuse modalities and model cross-modal consistency |
| FR-09 | Model temporal context across the clip |
| FR-10 | Output a video-level real/fake decision with a calibrated confidence score |
| FR-11 | Output per-frame forgery probabilities |
| FR-12 | Convert per-frame probabilities into merged forged segments with `[start, end]` timestamps |
| FR-13 | Report evaluation metrics on held-out data |
| FR-14 | Expose an HTTP API for upload → analysis → results |
| FR-15 | Provide a web UI showing the verdict, a timeline visualization, and segment timestamps |
| FR-16 | Persist per-run artifacts (config, metrics, checkpoint) for reproducibility |

## 1.5 Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-01 | Reproducibility | Fixed seeds; config-driven runs; identical results on re-run |
| NFR-02 | Inference latency | `TO VERIFY` — report target unknown. Plan target: **< 10 s for a 10 s clip** on the GTX 1650 |
| NFR-03 | Memory ceiling | Must run within **4 GB VRAM / 6 GB RAM** |
| NFR-04 | Model size | Stage-B head **< 50 MB**; full pipeline **< 200 MB** |
| NFR-05 | Modularity | Every block independently testable and swappable |
| NFR-06 | Robustness | Graceful failure on no-face, no-audio, corrupt, or too-short input |
| NFR-07 | Observability | Structured logs; all experiments tracked |
| NFR-08 | Maintainability | Typed interfaces, unit tests, documented configs |
| NFR-09 | Usability | Non-expert can upload and read the result unaided |

## 1.6 Dataset requirements (as specified)

Primary dataset: **LAV-DF** (Localized Audio Visual DeepFake). Required properties:

- Contains both real and fake videos, each with synchronized audio and video.
- Fakes are **temporally localized** — only sub-spans are manipulated.
- Provides **per-video ground-truth intervals** of manipulation.
- Distinguishes **which modality** was manipulated (video only / audio only / both).
- Provides an **official train/dev/test split** with no identity leakage.

Full treatment in Part 3.

## 1.7 Model architecture (as specified in the report)

```
                 ┌──────────────────────────────────────────────┐
   video stream  │  frames → face detect/align → ResNet-18*     │→ visual features
                 └──────────────────────────────────────────────┘        │
                                                                          ▼
                                                                    ┌──────────┐
                                                                    │  FUSION  │
                                                                    └──────────┘
                 ┌──────────────────────────────────────────────┐        ▲
   audio stream  │  waveform → MFCC → 1D-CNN                    │→ audio features
                 └──────────────────────────────────────────────┘

     FUSION → lip-sync verification → SELF-ATTENTION → LSTM → softmax → real/fake
                                                          └→ temporal localization → timestamps

     GAN ──(synthetic training samples)──▶ training set
```

`*` ResNet-18 **or** MobileNetV2 — **unresolved, see §0.4 Finding 2**.

## 1.8 Data flow (as specified)

`Upload → validate → demux → [video: decode/detect/align/embed | audio: resample/MFCC/encode]
→ align → fuse → attend → temporal → {classify, localize} → post-process → JSON + UI`

## 1.9 Training pipeline (as specified)

Load LAV-DF → preprocess → GAN-augment → batch → forward → loss (cross-entropy `TO VERIFY`
whether localization loss was included) → backprop → validate → checkpoint best → repeat.

Hyperparameters (optimizer, LR, batch size, epochs, schedule): **`TO VERIFY`** — none supplied.

## 1.10 Inference pipeline (as specified)

Upload → preprocess (same code path as training) → forward → softmax → threshold → segments →
render. Whether the report used sliding windows or dense per-frame prediction: **`TO VERIFY`**.

## 1.11 Evaluation methodology (as specified)

Accuracy, Precision, Recall, F1 are confirmed present. Whether ROC-AUC, confusion matrix, or any
**localization** metric (AP/AR at IoU thresholds) was reported: **`TO VERIFY`**.

> **Analytical note.** If the report evaluated a *localization* system using only *classification*
> metrics, the headline 80% does not measure the project's central claim at all. This is a
> priority question for the report audit.

## 1.12 Serving architecture as described in the original report *(historical — not a build target)*

Flask app, local server, HTML form upload, synchronous inference, rendered result page.
Retained here only to document the original specification; per §0.2 the rebuild is not deployed.

---

# PART 2 — THE REBUILT SYSTEM: COMPONENT ARCHITECTURE

Fifteen components, A–O. Each specifies purpose, I/O, technology, architecture, parameters,
compute, and test strategy.

---

### A. Data Ingestion

| | |
|---|---|
| **Purpose** | Get LAV-DF onto disk, verify integrity, parse metadata into a queryable manifest |
| **Input** | LAV-DF archive + `metadata.json`; or a single uploaded video at inference |
| **Output** | `data/raw/` video files + `manifest.parquet` (one row per video, all labels resolved) |
| **Technology** | Python 3.10, `pandas`, `pyarrow`, `ffprobe`, `tqdm`, `hashlib` |
| **Architecture** | Stateless ETL script. Read metadata → probe every file → join → validate → write parquet |
| **Key parameters** | `min_duration=1.0s`, `max_duration=30.0s`, `require_audio=True`, `require_video=True` |
| **Compute** | CPU only; I/O-bound. ~1–3 ms per `ffprobe` call. Full dataset scan `TO VERIFY` (size unknown) — budget 1–3 h |
| **Testing** | Manifest row count == metadata entry count minus quarantined; zero nulls in label columns; every `fake_periods` interval satisfies `0 ≤ start < end ≤ duration`; hash a 20-file sample twice for determinism |

**Manifest schema (the contract every downstream stage depends on):**

```
video_id        str    unique key
path            str    absolute path on D:
split           str    train | dev | test        (official split — never re-derived)
label           int    0 = real, 1 = fake
modify_video    bool
modify_audio    bool
fake_periods    list[[float,float]]  seconds; empty for real
duration        float  seconds
fps             float
n_frames        int
sample_rate     int
source_id       str    identity / source-video key — CRITICAL for leakage checks
status          str    ok | no_audio | no_video | corrupt | too_short | too_long
```

---

### B. Video Preprocessing

| | |
|---|---|
| **Purpose** | Turn a video into a fixed-rate sequence of aligned face crops |
| **Input** | Video file path + manifest row |
| **Output** | `uint8` array `[T, 112, 112, 3]` + per-frame `face_found` mask, written to `data/interim/faces/{video_id}.npy` |
| **Technology** | `PyAV` or `decord` for decoding; **face detection: see decision below**; `OpenCV` for warping |
| **Architecture** | decode → sample to fixed fps → detect face → 5-point landmark similarity-transform align → crop → resize → cache |
| **Key parameters** | `target_fps=25`, `crop=112×112`, `detect_every_n=5` frames + track between, `margin=0.25`, `align=True` |
| **Compute** | **The dominant cost of the whole project.** CPU-bound decode + GPU-light detection |
| **Testing** | Golden-file test on 3 fixed videos (byte-identical output on re-run); assert `T == round(duration × 25)` ± 1; visual contact sheet for 20 random samples; assert `face_found.mean() > 0.9` on a clean sample |

**DECISION B-1 — Face detector.** Three options, materially different on your hardware:

| Option | Speed (CPU) | Accuracy | VRAM | Verdict |
|---|---|---|---|---|
| MediaPipe Face Mesh | Very fast | Good, gives 468 landmarks free | 0 (CPU) | **Recommended** — CPU-only keeps VRAM free, landmarks enable both alignment and the mouth-ROI crop needed by component F |
| MTCNN (`facenet-pytorch`) | Slow | Good | ~400 MB | Fallback |
| RetinaFace | Slow | Best | ~800 MB | Overkill; VRAM you don't have |

**Recommendation: MediaPipe.** It runs on CPU (leaving all 4 GB VRAM for the backbone), returns
landmarks for alignment *and* a mouth ROI in one pass, and is deterministic.

> **Why alignment matters more than it looks.** Without similarity-transform alignment, the network
> spends capacity learning to be invariant to head pose. With it, the mouth sits at approximately
> the same pixels in every frame — so *changes* at those pixels across time become the signal.
> For a lip-sync-driven task this is not cosmetic; it is most of the battle.

---

### C. Audio Preprocessing

| | |
|---|---|
| **Purpose** | Turn the audio track into a time-frequency representation on the **same temporal grid as video** |
| **Input** | Video file path |
| **Output** | `float32 [T, n_mels]` log-mel array → `data/interim/audio/{video_id}.npy` |
| **Technology** | `ffmpeg` demux, `librosa` / `torchaudio` |
| **Architecture** | demux → mono → resample 16 kHz → pre-emphasis → STFT → mel filterbank → log → per-utterance CMVN |
| **Key parameters** | **`sr=16000`, `n_fft=1024`, `hop_length=640`, `n_mels=80`, `fmin=20`, `fmax=7600`** |
| **Compute** | Fast. ~50–150 ms per 10 s clip, CPU |
| **Testing** | Assert audio frame count == video frame count ± 1; round-trip a synthetic 440 Hz tone and assert peak lands in the correct mel bin; assert no NaN/Inf after log |

> **The `hop_length=640` trick — this is the single most important number in the preprocessing.**
> At 16 kHz, a hop of 640 samples is exactly **40 ms**. At 25 fps, one video frame is exactly
> **40 ms**. So one audio frame maps to exactly one video frame, and cross-modal alignment becomes
> an array index rather than an interpolation. Every resampling/interpolation you avoid here is a
> class of alignment bug you never have to debug. Lock this in Phase 3 and never change it.

**DECISION C-1 — MFCC vs log-Mel.** The report specifies MFCC. **I recommend log-Mel as the
default, with MFCC retained as an ablation arm.** Reasoning: MFCC applies a DCT and keeps ~13–40
coefficients, which decorrelates and *compresses away* fine spectral structure. That structure is
precisely where TTS/vocoder artifacts live — phase incoherence, over-smoothed harmonics,
characteristic high-frequency roll-off. MFCC was designed for *speech recognition*, where speaker
and channel detail is nuisance; here it is the signal.

This is not a silent replacement — it becomes **Experiment B2a (MFCC) vs B2b (log-Mel)**, so you
can *demonstrate* the improvement rather than assert it. That is a stronger portfolio story than
either choice alone.

---

### D. Visual Feature Extraction

| | |
|---|---|
| **Purpose** | Map each aligned face crop to a compact embedding |
| **Input** | `[T, 112, 112, 3]` face crops |
| **Output** | `float16 [T, 512]` embeddings → `data/features/visual/{backbone}/{video_id}.npy` |
| **Technology** | PyTorch, `torchvision` pretrained weights |
| **Architecture** | ImageNet-pretrained backbone, classifier head removed, global-average-pooled |
| **Key parameters** | ResNet-18 → 512-d; MobileNetV2 → 1280-d (project to 512 with a learned linear layer for parity) |
| **Compute** | **Frozen, batched, run once.** Batch 64 crops at 112² fits in ~1.2 GB VRAM |
| **Testing** | Determinism (same input → identical output, `model.eval()`, no dropout); embedding norms in a sane range; a t-SNE sanity plot separating identities |

**Storage estimate:** 512 dims × 2 bytes = 1 KB/frame. A 10 s clip at 25 fps = 250 frames = 250 KB.
For a 10,000-video development subset: **~2.5 GB**. For the full dataset (size `TO VERIFY`):
**~25–35 GB**. Both fit comfortably in your 167 GB free on D:.

**DECISION D-1 — the ResNet-18 / MobileNetV2 contradiction.** See Part 5 §5.2.

---

### E. Audio Feature Extraction

| | |
|---|---|
| **Purpose** | Map the log-mel sequence to per-timestep audio embeddings |
| **Input** | `[T, 80]` log-mel |
| **Output** | `float16 [T, 256]` → `data/features/audio/{encoder}/{video_id}.npy` |
| **Technology** | PyTorch |
| **Architecture** | **1D-CNN** (report-faithful): 4 × [Conv1d(k=3, pad=1) → BN → ReLU], channels 80→128→256→256→256. **Stride 1 throughout — temporal resolution must be preserved for localization.** |
| **Key parameters** | kernel 3, dilation `[1,2,4,8]` to widen receptive field to ~31 frames (1.24 s) without downsampling |
| **Compute** | Trivial — < 100 MB VRAM, trains in seconds |
| **Testing** | Output length == input length exactly (assert, do not assume); gradient-flow check; overfit a 10-sample batch to ~zero loss |

> **Why stride 1 and dilation instead of pooling.** Every temporal downsample you apply is
> localization precision you can never recover. A stride-2 stack of 4 layers gives you 16× coarser
> time — 640 ms per output step — which makes an IoU@0.75 score against a 2 s ground-truth span
> essentially unreachable. Dilation widens context at *no cost in resolution*. This is the audio
> analogue of the `hop_length=640` discipline.

**Stretch option (cloud only):** replace with frozen **WavLM-Base+** or **Wav2Vec2** features.
Expect a large gain — but read the warning in Part 5 §5.6 first, because on LAV-DF a strong audio
encoder can make the model ignore video entirely, which quietly defeats the project's premise.

---

### F. Audio-Visual Synchronization

| | |
|---|---|
| **Purpose** | Explicitly measure whether mouth motion agrees with speech at each instant |
| **Input** | Visual embeddings `[T,512]` (or dedicated mouth-ROI crops) + audio embeddings `[T,256]` |
| **Output** | Per-timestep sync score `[T,1]` + a sync-aware joint embedding |
| **Technology** | PyTorch; optionally pretrained SyncNet |
| **Architecture** | Two projection heads to a shared 256-d space → cosine similarity per timestep, computed over a sliding ±5-frame window → sync curve |
| **Key parameters** | window `±5` frames (±200 ms), temperature `τ=0.07` for the contrastive loss |
| **Compute** | Negligible |
| **Testing** | **Deliberately desynchronize** a real video by shifting audio +400 ms and assert the sync score drops significantly. If it does not, this component is not working — no matter what the loss curve says |

**DECISION F-1 — learned vs pretrained sync.**

| Approach | Pros | Cons |
|---|---|---|
| **Auxiliary contrastive loss, learned jointly** *(recommended)* | No extra dependency; adapts to the dataset; supervision is free (real videos are in-sync by construction) | Needs careful negative sampling |
| Pretrained SyncNet as a frozen feature extractor | Strong prior, immediately meaningful | Extra dependency; trained on different data; another model in VRAM |

**Recommendation: learned, via an auxiliary InfoNCE loss.** Positives = temporally aligned
(audio_t, visual_t) pairs from **real** videos; negatives = pairs shifted by ≥10 frames. The
supervision costs nothing, and the desync test above gives you a direct, honest way to verify the
component actually learned sync rather than a shortcut.

---

### G. Multimodal Fusion

| | |
|---|---|
| **Purpose** | Combine modalities so cross-modal *inconsistency* becomes linearly detectable |
| **Input** | `[T,512]` visual + `[T,256]` audio |
| **Output** | `[T, d_model=256]` fused sequence |
| **Technology** | PyTorch |
| **Architecture** | **Bidirectional cross-modal attention** — 2 layers, A→V and V→A, then concatenate and project |
| **Key parameters** | `d_model=256`, `n_heads=4`, `dropout=0.1`, pre-norm |
| **Compute** | O(T²) attention. At T=250, a 250×250 matrix per head is trivial |
| **Testing** | Ablate to concat-only and confirm the metric moves (Experiment C vs later); inspect attention maps on a known forged span |

**DECISION G-1 — fusion strategy.**

| Strategy | Description | When it wins |
|---|---|---|
| Early concat | `[visual ‖ audio]` → MLP | Baseline 3. Cheap, surprisingly decent |
| Gated fusion | Learned per-timestep modality weights | When one modality is unreliable |
| **Bidirectional cross-attention** | Each modality attends to the other | **Recommended.** Directly models "does audio at time *t* explain video at time *t*?" — which *is* the task |

**Recommendation: cross-attention**, with early concat retained as Baseline 3 so the gain is
measured, not assumed.

---

### H. Self-Attention / Temporal Modeling

| | |
|---|---|
| **Purpose** | Give every timestep global context so *locally anomalous* regions stand out |
| **Input** | `[T,256]` fused |
| **Output** | `[T,256]` context-enriched |
| **Technology** | PyTorch `nn.TransformerEncoder` |
| **Architecture** | 4-layer pre-norm Transformer encoder, learned positional encoding, `d_ff=512` |
| **Key parameters** | `n_layers=4`, `n_heads=4`, `d_model=256`, `d_ff=512`, `dropout=0.1`, `max_T=750` (30 s) |
| **Compute** | ~1.6 M params. Trains on 4 GB VRAM with batch 16 at T=250 |
| **Testing** | Attention-entropy check (heads should not all collapse to uniform); positional-encoding ablation; overfit test |

> **Why self-attention is the right tool for *this* task specifically.** A forged span is not
> anomalous in absolute terms — the generated face is photorealistic and the TTS is intelligible.
> It is anomalous **relative to the rest of the same video**: a different noise floor, a slightly
> different skin-texture statistic, a lip-sync offset that the genuine portions do not have.
> Self-attention computes exactly this relative comparison — every timestep against every other.
> An LSTM cannot: its hidden state is a lossy summary of the past, so it cannot directly compare
> frame 200 against frame 12. This is the strongest theoretical argument in the whole project, and
> **Experiment D vs E is designed to demonstrate it empirically.**

---

### I. GAN-Based Augmentation

> **This is the weakest-specified part of the original report and needs honest reframing.**

Training a GAN to synthesize *convincing audio-visual deepfakes from scratch* is a multi-GPU,
multi-week research project. On a 4 GB GTX 1650 with 6 GB RAM it is not achievable, and attempting
it will consume the entire project timeline and produce nothing usable. Rather than quietly drop
the requirement or pretend otherwise, here is a three-tier reframing with an explicit recommendation.

| Tier | Method | Feasible here? | Expected value |
|---|---|---|---|
| **1** | **Feature-space augmentation + manipulation splicing** | ✅ Yes | **High** |
| 2 | Face-region GAN (StyleGAN2-ADA / pix2pix on 112² crops) | ⚠️ Marginal — days of training | Medium |
| 3 | Full AV deepfake synthesis (Wav2Lip + TTS) | ❌ No | Out of scope |

**Recommendation: Tier 1**, with two mechanisms:

**(a) Manipulation splicing (do this first — likely the biggest real win).** Using the manifest,
splice a real span from video X into the timeline of video Y, generating a *new* training example
with **exactly known** ground-truth boundaries. This costs nothing, is label-exact, and directly
attacks the model's weakest point — boundary precision — by multiplying the number of distinct
boundary events the model sees. It is augmentation with perfect supervision.

**(b) Conditional GAN on the fused feature sequence.** A small generator
`G: (noise, label) → [T,256]` with a sequence discriminator, trained on cached embeddings. Cheap
(operates on 256-d vectors, not pixels), and defensible as "GAN augmentation" in the report's sense.

> **CRITICAL EXPERIMENTAL-DESIGN WARNING.** If you compare *"GAN augmentation"* against
> *"no augmentation"*, any gain is uninterpretable — you cannot tell whether the GAN helped or
> whether *augmentation in general* helped. **Experiment F must have three arms:**
>
> - **F0** no augmentation
> - **F1** strong classical augmentation (spec-augment, temporal jitter, crop/colour jitter, splicing)
> - **F2** F1 **+** GAN-generated samples
>
> The claim "the GAN helped" is only supported by **F2 > F1**, not by F2 > F0. Most published
> ablations get this wrong; getting it right is a genuine differentiator in an interview.

---

### J. Classification

| | |
|---|---|
| **Purpose** | Video-level real/fake verdict with a calibrated confidence |
| **Input** | `[T,256]` |
| **Output** | Scalar probability ∈ [0,1] |
| **Technology** | PyTorch |
| **Architecture** | Attention-pooling over time → LayerNorm → Linear(256→64) → GELU → Dropout → Linear(64→1) → sigmoid |
| **Key parameters** | Single logit + `BCEWithLogitsLoss` (not 2-way softmax — see note) |
| **Compute** | Negligible |
| **Testing** | Calibration via reliability diagram + Expected Calibration Error; threshold swept on **dev**, frozen before touching test |

> **Two deliberate deviations from the report.**
> **(1) Attention pooling, not mean pooling.** Mean-pooling a 250-frame sequence where 50 frames
> are fake dilutes the signal 5:1 — reintroducing the exact problem in §1.1. Attention pooling
> lets the model concentrate on the suspicious span. **Max-pooling is the alternative** and is
> worth an ablation; it is more sensitive but noisier.
> **(2) Sigmoid + BCE, not softmax.** For binary classification these are mathematically
> equivalent, but a single logit is simpler to calibrate and threshold, and matches the per-frame
> head so both heads share one loss family.

---

### K. Temporal Localization

| | |
|---|---|
| **Purpose** | Convert per-frame scores into merged, timestamped forged segments |
| **Input** | `[T,256]` |
| **Output** | `[{start_sec, end_sec, confidence}]` |
| **Technology** | PyTorch + NumPy/SciPy post-processing |
| **Architecture** | Per-frame head → `[T,1]` logits; optional boundary head → `[T,2]` (start-ness, end-ness) |
| **Key parameters** | See Part 6 for the full parameter set and tuning protocol |
| **Compute** | Negligible |
| **Testing** | Synthetic sequence with a known injected span → assert recovered IoU > 0.9; property test that outputs are sorted, non-overlapping, within `[0, duration]` |

Full treatment in **Part 6**.

---

### L. Evaluation

| | |
|---|---|
| **Purpose** | Compute all metrics reproducibly from saved predictions |
| **Input** | `predictions.parquet` + `manifest.parquet` |
| **Output** | `metrics.json`, confusion matrix, ROC/PR curves, per-IoU AP/AR table |
| **Technology** | `scikit-learn`, NumPy, Matplotlib |
| **Architecture** | **Pure function** — decoupled from training. Reads saved predictions, writes metrics |
| **Key parameters** | IoU thresholds `[0.5, 0.75, 0.95]`; AR@`[100, 50, 20, 10]` |
| **Compute** | Seconds |
| **Testing** | **Unit-test the AP implementation against hand-computed toy cases.** A silently wrong AP will invalidate every experiment in Part 7 |

> Decoupling evaluation from training is not stylistic. It means you can recompute every metric,
> add a metric you forgot, or fix a metric bug **without retraining anything**.

---

### M. API / Backend

| | |
|---|---|
| **Purpose** | HTTP interface for upload → analysis → results |
| **Input** | `multipart/form-data` video upload |
| **Output** | JSON verdict + segments + per-frame scores |
| **Technology** | **FastAPI** + `uvicorn` (see Part 13 for why over Flask) |
| **Architecture** | `POST /api/v1/analyze` → job id; `GET /api/v1/jobs/{id}` → status/result; `GET /health`; `GET /api/v1/model-info`. Background task queue (in-process; single-worker given RAM) |
| **Key parameters** | `max_upload=200 MB`, `allowed=[mp4, avi, mov, mkv]`, `timeout=300 s`, model loaded **once** at startup |
| **Compute** | Model resident ~200 MB VRAM; **one concurrent inference max** on 6 GB RAM |
| **Testing** | `pytest` + `httpx` — happy path, oversized file, wrong MIME, no-audio video, no-face video, corrupt file |

---

### N. Frontend

| | |
|---|---|
| **Purpose** | Make the output legible to a non-expert |
| **Input** | API JSON |
| **Output** | Rendered verdict, timeline, timestamps |
| **Technology** | **React + Vite + TailwindCSS**, or a single-file HTML+Alpine.js if you want zero build step |
| **Architecture** | Upload (drag-drop) → progress → results: verdict badge, confidence, **timeline strip with forged spans highlighted**, video player that seeks to a span on click, per-frame score line chart, JSON download |
| **Key parameters** | Poll interval 1 s; timeline colour-coded green/red |
| **Compute** | Browser only |
| **Testing** | Component tests (Vitest); manual cross-browser; test with a **real** video, a **fully fake** video, and a **partially fake** video |

> The timeline is the centrepiece. It is the visual proof that this project does localization and
> not just classification — and it is what a recruiter or interviewer will actually remember.

---

### O. Logging & Experiment Tracking

| | |
|---|---|
| **Purpose** | Make every number in Parts 7–9 traceable to the exact code and config that produced it |
| **Input** | Config, metrics, artifacts |
| **Output** | Per-run directory + a queryable comparison table |
| **Technology** | **MLflow** (local, file-backed) or **Weights & Biases** (free tier). `structlog` for app logs |
| **Architecture** | Per run log: git commit SHA, full resolved config, all seeds, metrics per epoch, final test metrics, checkpoint path, hardware, wall-clock |
| **Key parameters** | `experiments/{exp_id}/{run_id}/` |
| **Testing** | Re-run one experiment from its logged config and confirm metrics match to ~1e-4 |

**Recommendation: MLflow local.** No account, no upload, no network dependency, and the run store
is just files on D: — which suits a project you may work on offline.

---

# PART 3 — DATASET

## 3.1 What LAV-DF is

**LAV-DF** = *Localized Audio Visual DeepFake*, introduced in Cai et al., *"Do You Really Mean
That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery
Localization"* (DICTA 2022), built on top of **VoxCeleb2**.

Its defining property, and the reason it is the right dataset here: manipulations are
**content-driven and temporally localized**. Specific *words* are replaced to alter the sentiment
or meaning of the utterance, with the corresponding visual span regenerated. Most of each fake
video is genuine. That is precisely the threat model in §1.1.

## 3.2 Published statistics — `TO VERIFY` against your actual download

> These figures come from the **published paper**, not from your report and not from a download I
> have inspected. **Treat as provisional and re-derive from your manifest in Phase 1.**

| Statistic | Published value | Status |
|---|---|---|
| Total videos | 136,304 | `TO VERIFY` |
| Real videos | 36,431 | `TO VERIFY` |
| Fake videos | 99,873 | `TO VERIFY` |
| Base dataset | VoxCeleb2 | `TO VERIFY` |
| Frame rate | 25 fps | `TO VERIFY` |
| Audio sample rate | 16 kHz | `TO VERIFY` |
| Download size (GB) | **unknown** | `TO VERIFY` — **check before downloading; C: has only 27 GB free** |
| Mean/median duration | **unknown** | `TO VERIFY` — compute in Phase 1 |
| Mean forged-span duration | **unknown** | `TO VERIFY` — compute in Phase 1; **drives window size in Part 6** |
| Fake-frame fraction | **unknown** | `TO VERIFY` — compute in Phase 1; **drives loss weighting** |

**Phase 1 must produce a `dataset_statistics.md` containing the real, measured versions of every
row above.** Nothing downstream should cite the provisional column.

## 3.3 Expected structure — `TO VERIFY`

```
LAV-DF/
├── metadata.json
├── train/  *.mp4
├── dev/    *.mp4
└── test/   *.mp4
```

Expected `metadata.json` fields per entry (`TO VERIFY` — confirm exact names on download):

```json
{
  "file": "train/000001.mp4",
  "original": "train/000000.mp4",
  "split": "train",
  "modify_video": true,
  "modify_audio": false,
  "fake_periods": [[2.44, 3.88]],
  "duration": 8.52,
  "video_frames": 213,
  "audio_frames": 136320
}
```

`fake_periods` is the ground truth for the entire localization task: a list of `[start_sec,
end_sec]` intervals, **empty for real videos**.

## 3.4 Label taxonomy

Four classes exist, and collapsing them to binary too early throws away useful signal:

| Class | `modify_video` | `modify_audio` | Binary label |
|---|---|---|---|
| Real | false | false | 0 |
| Visual-only fake | true | false | 1 |
| Audio-only fake | false | true | 1 |
| Both fake | true | true | 1 |

**Recommendation:** train the primary head as binary, but **always report metrics broken down by
these four classes.** This breakdown is diagnostic gold — if visual-only fakes score near chance
while audio-only fakes score near-perfect, your model is ignoring video, and no aggregate number
will tell you that. It is also exactly the kind of analysis that distinguishes a real engineer
from someone who reports one accuracy figure.

## 3.5 Splits and data-leakage prevention

**RULE 1 — Use the official `train`/`dev`/`test` split. Never re-derive it by random shuffling.**

**RULE 2 — Understand *why*.** LAV-DF fakes are derived from real source videos, and both derive
from VoxCeleb2 **identities**. A random clip-level shuffle puts:
- the *same speaker* in train and test → the model memorizes faces/voices;
- a *fake* in test whose *real original* is in train → near-duplicate leakage.

Either inflates test scores substantially and silently. A model that looks excellent and is
worthless is the worst possible outcome, and it is the single most common failure in student
deepfake projects.

**RULE 3 — Assert it in code, do not trust it.** Phase 1 must include a test that fails the build:

```
assert set(train.source_id) ∩ set(test.source_id) == ∅
assert set(train.source_id) ∩ set(dev.source_id)  == ∅
assert set(dev.source_id)   ∩ set(test.source_id) == ∅
```

`TO VERIFY` — confirm whether `metadata.json` exposes a usable identity key. If not, derive
`source_id` from the `original` field (following the chain to its root) and document the derivation.

**RULE 4 — The test split is touched exactly once**, at the very end, after all hyperparameters
and thresholds are frozen on `dev`. Every intermediate decision uses `dev`. Repeatedly evaluating
on test and picking the best is a slow-motion form of overfitting that produces numbers you cannot
defend.

## 3.6 Class imbalance — two distinct problems

**Video level.** Published counts suggest ≈73% fake / 27% real (`TO VERIFY`). Mitigation:
`WeightedRandomSampler` or `pos_weight` in BCE; **report ROC-AUC and PR-AUC, not accuracy**.

**Frame level — much more severe.** If fake spans average ~1.5 s in ~8 s videos, fake frames are
a small minority *even within fake videos*. A naive per-frame BCE will happily predict "real"
everywhere and score well on accuracy while localizing nothing.

Mitigations, in order of preference:
1. **Focal loss** (`α=0.25, γ=2.0`) on the per-frame head — the standard fix for dense prediction.
2. `pos_weight = n_neg/n_pos` in BCE, computed from the **train split only**.
3. Sample clips so that fake-containing clips are over-represented (but never oversample the
   *same* clip so heavily it memorizes).
4. **Always report AP, never frame accuracy.** Frame accuracy on this task is a meaningless number
   that will look impressive and mean nothing.

## 3.7 Corrupt / missing file handling

Fail loudly, quarantine, never silently skip:

| Condition | Detection | Action |
|---|---|---|
| File missing | `os.path.exists` | `status=missing`, exclude, **log the id** |
| Unreadable | `ffprobe` non-zero exit | `status=corrupt`, quarantine |
| No audio stream | no audio in `ffprobe` | `status=no_audio`, exclude |
| No video stream | no video in `ffprobe` | `status=no_video`, exclude |
| Zero faces detected | `face_found.sum() == 0` | `status=no_face`, exclude, **inspect a sample manually** |
| Too short (< 1 s) | duration | `status=too_short`, exclude |
| `fake_periods` out of bounds | interval vs duration | `status=bad_label`, **inspect — this may indicate a parsing bug on your side, not bad data** |

**Report the exclusion count and reasons in `dataset_statistics.md`.** If you exclude 30% of the
data, that is a finding, not a footnote — and it changes how your results should be read.

## 3.8 Reproducible preprocessing

1. **Content-hash the config.** Cache path = `features/{sha256(preprocess_config)[:8]}/`. Change a
   parameter → new cache directory → no possibility of mixing feature versions. This single trick
   eliminates an entire category of "why did my numbers change?" debugging.
2. **Seed everything**: `random`, `numpy`, `torch`, `torch.cuda`, `PYTHONHASHSEED`,
   `torch.backends.cudnn.deterministic=True`.
3. **Version the manifest** — write `manifest_v1.parquet`, never mutate in place.
4. **Log the git SHA** in every run directory.
5. **Preprocessing must be a pure function of (video, config).** No wall-clock, no unsedeed
   randomness, no machine-dependent branches.

## 3.9 Development subset — start here

**Do not begin with the full dataset.** Build a stratified subset first:

| Stage | Videos | Purpose | Feature size (est.) |
|---|---|---|---|
| **Smoke** | **100** (50 real / 50 fake) | Prove the pipeline runs end-to-end; overfit-a-batch tests | ~25 MB |
| **Dev** | **2,000** (stratified by split and by the 4-class taxonomy) | All architecture decisions, Baselines 1–4, Experiments A–E | ~500 MB |
| **Scale** | **10,000** | Experiments F–G; the numbers you report | ~2.5 GB |
| **Full** | all (`TO VERIFY`) | Final run, cloud GPU only | ~25–35 GB |

Sampling rules: preserve the **official split membership** (never move a video between splits);
stratify by the 4-class taxonomy; **fix the subset seed and commit the resulting id list to git**
so the subset itself is reproducible and every experiment runs on literally the same videos.

## 3.10 Scaling up later

Because features are cached and content-hashed, scaling is purely additive: extract features for
the new ids, append to the manifest, retrain Stage B. **No re-extraction of existing videos.**
Extraction is resumable — check for the output `.npy` before processing, so an interrupted run
resumes rather than restarts. On a 6 GB machine, an 8-hour extraction job *will* be interrupted;
plan for it from day one rather than discovering it at hour seven.

---

# PART 4 — BASELINE MODELS

## 4.1 Why baselines are non-negotiable

The final architecture has six components stacked together. If you build it all at once and it
reaches, say, 82% — **you have learned nothing about why**. You cannot tell whether self-attention
contributed, whether the GAN helped, or whether a visual-only model would have got 81% and every
other component is decoration.

Baselines convert the project from *"I built a thing"* into *"I measured which parts matter"*.
That difference is the entire gap between a student project and engineering, and it is what an
interviewer will probe.

They also serve three practical purposes: they **catch bugs** (an audio-only model at 50% means
your audio pipeline is broken, and you find out in Phase 5 instead of Phase 10); they **bound
expectations**; and they **de-risk the schedule** — you have a working, reportable system after
Phase 4 rather than nothing until Phase 10.

## 4.2 Baseline definitions

All baselines share the identical data subset, splits, seeds, optimizer, schedule, and evaluation
code. **Only the model changes.** Anything else varying makes the comparison meaningless.

### Baseline 0 — Trivial / sanity floors
- **Majority class**: always predict "fake". Accuracy = the fake ratio.
- **Random**: AUC ≈ 0.5.
- **Purpose:** the floor every real model must clear. Also catches label-pipeline bugs — if your
  trained model matches the majority-class baseline exactly, it has learned nothing and collapsed.
- **Cost:** minutes.

### Baseline 1 — Visual only
- Face crops → visual backbone → mean/attention pool → classifier. **No audio at all.**
- **Purpose:** how much of the task is solvable from pixels alone? Isolates the visual pipeline.
- **Also serves as the arena for the ResNet-18 vs MobileNetV2 decision** (B1a vs B1b).
- **Failure signal:** AUC ≈ 0.5 → your face crops or labels are wrong. Do not proceed.

### Baseline 2 — Audio only
- Log-mel → 1D-CNN → pool → classifier. **No video at all.**
- **Purpose:** how much is solvable from audio alone? Also the MFCC-vs-log-Mel arena (B2a vs B2b).
- **Expect this to be strong on LAV-DF** — TTS artifacts are quite detectable. **This is important
  and slightly uncomfortable: if audio-only nearly matches your full multimodal model, the
  multimodal story is weak and you need to say so honestly** (and see §5.6 for how to fight it).

### Baseline 3 — Simple audio-visual fusion
- Concatenate pooled visual + pooled audio → MLP. **No temporal modelling, no attention.**
- **Purpose:** does *any* multimodal combination beat the better single modality? This is the
  minimum bar the whole premise must clear.
- **Failure signal:** if B3 ≤ max(B1, B2), fusion is destroying information — check normalization
  and feature scaling before adding complexity on top of a broken foundation.

### Baseline 4 — Audio-visual + temporal
- Concat per timestep → **BiLSTM** → pool → classifier. Report-faithful.
- **Purpose:** isolates the contribution of temporal modelling from the contribution of attention.
  Without this, any gain from the Transformer could be "temporal modelling helps", not
  "*attention* helps". This baseline is what makes the Experiment D vs E claim valid.

### Final model
Cross-attention fusion + sync loss + self-attention + temporal + dual heads (classification +
localization) + augmentation. Built only after every baseline above is measured and recorded.

## 4.3 Baseline results template *(EMPTY — fill only after execution)*

| ID | Model | Subset | Video AUC | Video F1 | Frame AP | Params | Train time | Notes |
|---|---|---|---|---|---|---|---|---|
| B0-maj | Majority class | dev-2k | | | | 0 | — | |
| B1a | Visual only (ResNet-18) | dev-2k | | | | | | |
| B1b | Visual only (MobileNetV2) | dev-2k | | | | | | |
| B2a | Audio only (MFCC + 1D-CNN) | dev-2k | | | | | | |
| B2b | Audio only (log-Mel + 1D-CNN) | dev-2k | | | | | | |
| B3 | Concat fusion | dev-2k | | | | | | |
| B4 | Concat + BiLSTM | dev-2k | | | | | | |

---

# PART 5 — FINAL MODEL ARCHITECTURE

## 5.1 Recommended architecture

```
════════════════════════ STAGE A — OFFLINE, RUN ONCE, CACHED ════════════════════════

  VIDEO ──► decode @25fps ──► MediaPipe detect+align ──► 112×112 crops
                                                              │
                                                              ▼
                                              frozen visual backbone (D-1)
                                                              │
                                                    [T, 512] float16 ──► D:\features\visual\

  AUDIO ──► demux ──► 16 kHz mono ──► log-mel (n_fft 1024, hop 640, 80 mels)
                                                              │
                                                    [T, 80] float32 ──► D:\features\audio\

              hop 640 @ 16 kHz = 40 ms  ≡  1 frame @ 25 fps   ► indices align exactly

════════════════════════ STAGE B — TRAINED, FITS IN 4 GB VRAM ═══════════════════════

  [T,512] visual ──► Linear 512→256 ──┐
                                       ├──► BIDIRECTIONAL CROSS-ATTENTION (2 layers)
  [T,80] mel ──► 1D-CNN (dilated,     │         V attends to A  ‖  A attends to V
                 stride 1) ──► [T,256]┘                   │
                                                          ├──► sync head ──► [T,1] sync score
                                                          │                  (aux InfoNCE loss)
                                                          ▼
                                        TRANSFORMER ENCODER — 4 layers, 4 heads, d=256
                                                          │
                                        ┌─────────────────┴──────────────────┐
                                        ▼                                     ▼
                              attention-pool over T                    per-frame head
                                        │                                     │
                                        ▼                                     ▼
                              [1] video logit                          [T,1] frame logits
                                        │                                     │
                                        ▼                                     ▼
                                 sigmoid → verdict          threshold → smooth → merge → segments
                                                                                       │
                                                                                       ▼
                                                                        [{start, end, confidence}]

  LOSS = λ_cls · BCE(video)  +  λ_loc · Focal(frame)  +  λ_sync · InfoNCE  +  λ_bnd · BoundaryLoss
         λ_cls = 1.0            λ_loc = 2.0             λ_sync = 0.3        λ_bnd = 0.5   (tune on dev)
```

**Estimated Stage-B size:** ~3.5 M parameters, ~14 MB fp32. Comfortably within NFR-04.

## 5.2 DECISION D-1 — ResNet-18 vs MobileNetV2 *(the flagged contradiction)*

**This is resolved by measurement at the Phase 4 exit gate, not by this document.**

| Criterion | ResNet-18 | MobileNetV2 |
|---|---|---|
| Parameters | 11.7 M | 3.4 M |
| GFLOPs @ 224² | ~1.8 | ~0.3 |
| GFLOPs @ 112² | ~0.45 | ~0.08 |
| Output dim | 512 | 1280 |
| Fine-tuning stability on small data | **Good** | Sensitive |
| Small-batch behaviour | **Robust** | Fragile (depthwise + BN) |
| Extraction speed on GTX 1650 | Slower | **Faster** |
| Suitability for cached frozen extraction | **Fine** — cost paid once | Fine |

**My prior (to be confirmed or overturned by the experiment): ResNet-18 at 112×112.**

Reasoning, in order of weight:
1. **The efficiency argument mostly evaporates under the cached architecture.** MobileNetV2's
   advantage is inference cost — but extraction runs **once, offline**. Paying 4× the FLOPs one
   time to get better features every subsequent training run is a good trade.
2. **At 112² instead of 224², ResNet-18 costs ~0.45 GFLOPs** — already cheap in absolute terms.
3. **MobileNetV2's depthwise-separable convolutions + BatchNorm are notoriously unstable at small
   batch sizes and low learning rates**, which is exactly the regime a 4 GB GPU forces you into if
   you ever fine-tune. ResNet-18's plain residual + BN blocks tolerate it far better.
4. ResNet-18's 512-d output is a natural size; MobileNetV2's 1280-d needs projection for parity
   anyway, adding a confound.

**When MobileNetV2 should win, and you should let it:** if Phase 4 shows the AUC gap is within
noise (< 1 point, overlapping seed variance) **and** extraction is ≥2× faster, take MobileNetV2 —
faster extraction directly buys you more experiments, which is worth more than a fractional AUC
point. Let the measurement decide.

**Option C — worth knowing you rejected it.** Neither backbone models *motion*; both are per-frame
2D. A (2+1)D stem or a small video transformer would capture temporal texture directly. **Rejected
for now** because it multiplies extraction cost and does not fit the cached-feature design — but
mentioning that you considered and rejected it, with reasons, is a strong interview answer.

## 5.3 Architecture alternatives considered

| Alternative | Accuracy | Compute | Train complexity | Explainability | Repro | Portfolio fit | Verdict |
|---|---|---|---|---|---|---|---|
| Report-faithful (concat + LSTM + softmax) | Low-Mid | Low | Low | Mid | High | Low | Keep as **Baseline 4** |
| **Recommended (cross-attn + Transformer + dual heads)** | **Mid-High** | **Low-Mid** | **Mid** | **High** (attention maps) | **High** | **High** | ✅ **ADOPT** |
| BA-TFD replication (boundary-aware, contrastive) | High | High | High | Mid | Mid | Mid | Stretch goal |
| End-to-end fine-tuned AV-Transformer | Highest | **Very high** | High | Low | Mid | Mid | ❌ Infeasible on 4 GB |
| Pretrained WavLM + CLIP features | High | Mid | Low | Low | High | Mid | Stretch (see §5.6) |

**Recommendation: the middle row.** It is the only option that is simultaneously feasible on your
hardware, explainable (attention maps over the timeline are directly visualizable in the frontend),
and strong enough to be worth defending.

## 5.4 Loss design

```
L = λ_cls·BCE(video) + λ_loc·Focal(frame) + λ_sync·InfoNCE + λ_bnd·BoundaryLoss
```

- **BCE(video)** — video-level, `pos_weight` from train split.
- **Focal(frame)** — `α=0.25, γ=2.0`; the workhorse for imbalanced dense prediction.
- **InfoNCE(sync)** — auxiliary; computed on **real** videos only (where in-sync is guaranteed).
- **BoundaryLoss** — optional; per-frame start-ness/end-ness against a Gaussian-smoothed target
  centred on ground-truth boundaries. Improves IoU@0.75+ specifically.

Start with `λ = (1.0, 2.0, 0.3, 0.5)` and tune **on dev**. `λ_loc > λ_cls` deliberately — the
localization head is the harder task and the project's actual differentiator.

## 5.5 Handling variable-length video

- **Training:** random-crop a fixed `T=250` frames (10 s). Pad shorter clips with a mask.
- **Inference:** full-length with a padding mask; if `T > 750` (30 s), process in overlapping
  chunks and stitch (Part 6 §6.9).
- **Never** resize the time axis by interpolation — it corrupts ground-truth timestamps and is a
  subtle bug that will silently degrade every localization metric.

## 5.6 ⚠️ The modality-collapse risk — design against it from the start

**On LAV-DF, audio forgeries (TTS/vocoder) are substantially easier to detect than visual ones.**
The consequence: a jointly-trained multimodal model can converge to a solution that reads audio and
**ignores video almost entirely**, because that minimizes training loss fastest. You get a decent
aggregate number and a model whose "multimodal" claim is hollow — the exact failure the whole
project premise is meant to avoid.

**Detect it:**
- The 4-class breakdown from §3.4 — if **visual-only fakes** score near chance while audio-only
  fakes score near-perfect, collapse has occurred.
- Zero out the audio stream at inference; if performance barely drops, video is unused.

**Prevent it:**
1. **Modality dropout** — randomly zero the entire audio stream (p=0.2) or visual stream (p=0.2)
   during training. Forces both pathways to carry independent signal. **The single most effective
   intervention.**
2. **The sync loss (component F)** — it cannot be satisfied by either modality alone, by construction.
3. **Per-modality auxiliary heads** — small classifiers on each stream, ensuring gradients reach both.

This risk is worth stating explicitly in your final report. Identifying a failure mode, designing
against it, and *measuring* whether the defence worked is exactly the kind of thing that reads as
genuine engineering maturity.

---

# PART 6 — TEMPORAL LOCALIZATION

The central capability. Not *"is this fake?"* but *"which 1.8 seconds are fake?"*

```
Video:      00:00 ══════════════════════════════════════════ 00:10
Ground truth:  REAL          FAKE            REAL
Prediction:    ├──────────────┤─────────────┤────────────────┤
               00:00–03:20    03:20–06:10    06:10–10:00
```

## 6.1 Dense per-frame prediction, not sliding-window classification

**Two paradigms:**

| | Sliding window | **Dense per-frame** *(recommended)* |
|---|---|---|
| Method | Chop into windows, classify each | One score per frame, in one pass |
| Resolution | Window size (coarse) | 40 ms (one frame) |
| Compute | Re-encodes overlapping regions | Single pass |
| Boundary precision | Poor — quantized to stride | **Good** |
| Complexity | Simple | Moderate |

**Recommendation: dense per-frame.** The Transformer already produces `[T, 256]` — one linear
layer gives `[T, 1]`. You get 40 ms resolution for essentially free, whereas a 1 s window with
0.5 s stride caps your boundary precision at ±250 ms no matter how good the model is. Sliding
window is retained only as a **fallback** if dense training proves unstable.

## 6.2 Window/segment parameters — for the *chunking* of long videos only

| Parameter | Value | Rationale |
|---|---|---|
| Training crop `T` | 250 frames (10 s) | Fits 4 GB VRAM at batch 16 |
| Inference chunk | 750 frames (30 s) | Full-length for typical clips |
| Chunk overlap | 125 frames (5 s) | Only for videos > 30 s |
| Frame resolution | 40 ms | = 1/25 s |

> **`TO VERIFY` — Phase 1 must compute the mean and distribution of forged-span durations.** If
> spans average ~1.5 s, a 10 s crop comfortably contains whole spans plus real context on both
> sides, which is what the model needs to detect *relative* anomaly. If spans turn out much longer,
> `T` must increase. **Do not finalize `T` before measuring this.**

## 6.3 Ground-truth alignment — the highest-risk step in the project

Converting `fake_periods` (seconds) into a per-frame `[T]` binary target:

```
target = zeros(T)
for (start_sec, end_sec) in fake_periods:
    i0 = floor(start_sec * fps)
    i1 = ceil (end_sec   * fps)
    target[i0:i1] = 1
```

**This is where off-by-one errors hide and quietly ruin everything.** A one-frame shift costs you
little; a systematic half-second shift makes IoU@0.75 unreachable while the loss curve looks
perfectly healthy and every unit test passes.

**Mandatory verification (do not skip):**
1. Invert the transform — convert `target` back to intervals and assert it matches `fake_periods`
   to within one frame.
2. **Render 20 videos with the forged span burned in as a red overlay and watch them.** Confirm
   the highlighted mouth motion actually looks manipulated. This 30-minute manual check has caught
   more label bugs than any automated test, and it is the reason to do it in Phase 2, not Phase 10.
3. Assert `target.sum() == 0` for every real video, exactly.

## 6.4 Post-processing: scores → segments

```
1. SMOOTH      median filter, kernel = 5 frames (200 ms)     → remove single-frame flicker
2. THRESHOLD   binary = (smoothed > τ),  τ tuned on dev       → default τ = 0.5
3. MORPHOLOGY  binary closing, kernel = 3                     → fill 1-2 frame holes
4. EXTRACT     contiguous runs of 1 → candidate segments
5. MIN-DURATION  discard segments < 0.4 s (10 frames)         → kill noise blips
6. MERGE-GAPS  merge segments separated by < 0.3 s            → rejoin split spans
7. CONFIDENCE  per segment = mean score over its frames
8. TO SECONDS  start = i0/fps,  end = i1/fps
```

## 6.5 Parameters to tune — **on dev only, then freeze**

| Parameter | Default | Search range | Effect |
|---|---|---|---|
| `τ` threshold | 0.5 | 0.1 → 0.9 | Precision/recall trade-off |
| Median kernel | 5 | 1, 3, 5, 9 | Noise vs boundary sharpness |
| Min duration | 0.4 s | 0.1 → 1.0 s | Kills false positives; may kill short true spans |
| Merge gap | 0.3 s | 0.1 → 1.0 s | Rejoins split detections; may merge distinct spans |

Tune by grid search maximizing **AP@0.5 on dev**. Then **freeze and never touch again.** Tuning
these on test is the most common way to produce a number you cannot defend in an interview.

## 6.6 Localization metrics

**Primary — the numbers that define success:**
- **AP@IoU=0.5** — main headline metric
- **AP@IoU=0.75** — strict; measures genuine boundary precision
- **AP@IoU=0.95** — very strict; expect low values, and say so
- **AR@100 / @50 / @20 / @10** — average recall at N proposals

**Secondary — diagnostic:**
- Frame-level AP and ROC-AUC (threshold-free; isolates the model from post-processing)
- **Boundary error**: mean |predicted_start − true_start| and |predicted_end − true_end| in
  milliseconds. **Highly interpretable and excellent for a resume line** — "localizes boundaries
  to within X ms" communicates far more than an AP number to a non-specialist.
- Segment count error: predicted vs true number of spans

**IoU for 1-D intervals:**
```
IoU = |A ∩ B| / |A ∪ B|
    = max(0, min(a_end,b_end) − max(a_start,b_start)) / (max(a_end,b_end) − min(a_start,b_start))
```

## 6.7 ⚠️ Unit-test the AP implementation before trusting a single result

Temporal AP has real subtleties: matching predictions to ground truth greedily by confidence,
handling multiple predictions matching one ground truth (only the highest-confidence counts;
the rest are false positives), and interpolating the precision-recall curve.

**A wrong AP implementation invalidates every experiment in Part 7 while looking entirely
plausible.** Write hand-computed toy cases first:
- one perfect prediction → AP = 1.0
- one prediction with IoU = 0.4, threshold 0.5 → AP = 0.0
- two predictions, one correct one spurious → verify the exact expected value by hand

Where possible, cross-check against an established implementation.

## 6.8 Real videos

A real video should produce **zero** segments. Include a dedicated metric: **false-positive
segment rate on real videos** — the fraction of genuine videos where the system flags any span.
For a tool a journalist might use, this is arguably the metric that matters most: falsely accusing
a real video is a worse failure than missing a fake one.

## 6.9 Long videos

For `duration > 30 s`: chunk with 5 s overlap, run inference per chunk, average per-frame scores in
overlap regions, then run post-processing **once on the stitched full-length score sequence** — not
per chunk. Post-processing per chunk creates artificial boundaries at every chunk edge.

---

# PART 7 — EXPERIMENTAL DESIGN

## 7.1 Protocol — the rules that make comparisons valid

1. **One variable at a time.** Change the model; hold data, split, seed, optimizer, schedule,
   and evaluation code fixed.
2. **Three seeds minimum** (0, 1, 2) for any headline claim. Report **mean ± std**.
3. **A difference smaller than the seed-to-seed standard deviation is not a result.** Say so.
4. **Dev for all decisions. Test once, at the end.**
5. **Every run logged** to MLflow with its git SHA and full config.
6. **Negative results are reported.** If the GAN does not help, that is a finding — and reporting
   it honestly is more impressive than hiding it.

## 7.2 Experiment matrix

| ID | Model | Subset | Key config | Hypothesis | Records |
|---|---|---|---|---|---|
| **A** | Visual only | dev-2k | ResNet-18 / MobileNetV2, frozen, attn-pool | Beats chance; moderate. Establishes visual pipeline works | Video AUC/F1, params, extract time |
| **B** | Audio only | dev-2k | log-mel + 1D-CNN | Beats chance; **likely strong** — TTS is detectable | Video AUC/F1 |
| **C** | Audio + visual (concat) | dev-2k | Early concat + MLP | **C > max(A,B)** — modalities are complementary | Video AUC/F1, Δ vs best unimodal |
| **D** | A+V + BiLSTM | dev-2k | 2-layer BiLSTM, 256 hidden | **D > C** — temporal context helps | + frame AP |
| **E** | A+V + self-attention | dev-2k | 4-layer Transformer | **E > D** — global comparison beats sequential memory | + frame AP, attention maps |
| **F0** | E, no augmentation | dev-10k | — | Reference point | all |
| **F1** | E + classical aug | dev-10k | spec-augment, jitter, splicing | **F1 > F0** | all |
| **F2** | E + classical + GAN aug | dev-10k | + conditional feature GAN | **F2 > F1** ← *the only valid GAN claim* | all |
| **G** | Final localization model | dev-10k → full | Dual heads + sync + boundary loss | Best AP@0.5 | **AP@{0.5,0.75,0.95}, AR@{100,50,20,10}**, boundary error, FP-rate on real |
| **H** | Ablation: −sync loss | dev-10k | G with λ_sync=0 | G > H | AP@0.5 |
| **I** | Ablation: −modality dropout | dev-10k | G without dropout | Detects collapse (§5.6) | AP@0.5 + **4-class breakdown** |
| **J** | Backbone: ResNet-18 vs MobileNetV2 | dev-2k | Only backbone varies | Resolves **D-1** | AUC, extract time, VRAM |
| **K** | Audio: MFCC vs log-Mel | dev-2k | Only audio features vary | Resolves **C-1** | AUC |

## 7.3 How to decide whether a component genuinely helps

For each claimed improvement, all four must hold:

1. **Magnitude** — the mean improvement exceeds the pooled seed std across 3 seeds.
2. **Consistency** — it holds on **all three** seeds, not just on average.
3. **Direction on the right metric** — a component justified for localization must improve
   **AP**, not just video AUC.
4. **Cost is stated** — report the parameter and latency cost alongside. A +0.3 AUC point for 3×
   the inference time is a *negative* result for a practical system, and should be reported as one.

**If a component fails these tests, report it as not helping.** The ablation's job is to find out,
not to confirm what you hoped.

## 7.4 Compute budget

| Stage | Runs | Est. time/run (GTX 1650) | Total |
|---|---|---|---|
| Feature extraction, dev-2k | 1 | 1.5–3 h `TO VERIFY` | 3 h |
| Feature extraction, dev-10k | 1 | 8–15 h `TO VERIFY` | 15 h |
| Experiments A–E × 3 seeds | 15 | 10–25 min | ~5 h |
| Experiments F0–F2 × 3 seeds | 9 | 25–45 min | ~6 h |
| Experiment G × 3 seeds | 3 | 45–90 min | ~4 h |
| Ablations H–K × 3 seeds | 12 | 15–30 min | ~5 h |

**Feature extraction dominates by an order of magnitude — which is exactly why caching is the
right architecture.** Extract once (overnight), then run 40+ experiments cheaply.

---

# PART 8 — EVALUATION

## 8.1 Metric definitions

**Classification (video level)**
| Metric | Formula | Note |
|---|---|---|
| Accuracy | (TP+TN)/N | **Misleading under imbalance — never report alone** |
| Precision | TP/(TP+FP) | Of flagged fakes, how many were fake |
| Recall | TP/(TP+FN) | Of real fakes, how many were caught |
| F1 | 2PR/(P+R) | Harmonic mean |
| **ROC-AUC** | area under TPR-FPR | **Threshold-free — primary** |
| **PR-AUC** | area under P-R | **Better than ROC under imbalance** |
| Confusion matrix | 2×2 | Always include |

**Localization** — AP@IoU{0.5, 0.75, 0.95}; AR@{100, 50, 20, 10}; segment precision/recall at
IoU 0.5; mean boundary error (ms); FP-segment rate on real videos.

**Performance** — inference latency (mean/p95, per 10 s clip, broken into preprocess / extract /
model / postprocess); peak VRAM; peak RAM; model size (MB); preprocessing throughput (videos/min).

## 8.2 Results table templates — **EMPTY UNTIL EXPERIMENTS RUN**

### Table 8.2.1 — Classification

| Exp | Model | Split | Acc | P | R | F1 | ROC-AUC | PR-AUC | Seeds |
|---|---|---|---|---|---|---|---|---|---|
| A | Visual only | dev | | | | | | | 3 |
| B | Audio only | dev | | | | | | | 3 |
| C | A+V concat | dev | | | | | | | 3 |
| D | A+V + LSTM | dev | | | | | | | 3 |
| E | A+V + self-attn | dev | | | | | | | 3 |
| F2 | + GAN aug | dev | | | | | | | 3 |
| G | **Final** | **test** | | | | | | | 3 |

### Table 8.2.2 — Localization

| Exp | AP@0.5 | AP@0.75 | AP@0.95 | AR@100 | AR@50 | AR@20 | AR@10 | Boundary err (ms) | FP-seg rate (real) |
|---|---|---|---|---|---|---|---|---|---|
| E | | | | | | | | | |
| F2 | | | | | | | | | |
| G (test) | | | | | | | | | |

### Table 8.2.3 — Performance (GTX 1650, 4 GB)

| Component | Latency (10 s clip) | Peak VRAM | Peak RAM | Notes |
|---|---|---|---|---|
| Decode + face align | | | | |
| Visual extraction | | | | |
| Audio extraction | | | | |
| Stage-B forward | | | | |
| Post-processing | | | | |
| **End-to-end** | | | | |

### Table 8.2.4 — 4-class diagnostic breakdown *(collapse detector, §5.6)*

| Class | N | Video AUC | Frame AP |
|---|---|---|---|
| Real | | — | — |
| Visual-only fake | | | |
| Audio-only fake | | | |
| Both fake | | | |

### Table 8.2.5 — Decision resolutions

| Decision | Option A | Option B | Winner | Margin | Basis |
|---|---|---|---|---|---|
| **D-1** Visual backbone | ResNet-18 | MobileNetV2 | | | Exp J |
| **C-1** Audio features | MFCC | log-Mel | | | Exp K |
| G-1 Fusion | Concat | Cross-attention | | | Exp C vs E |
| H-1 Temporal | BiLSTM | Transformer | | | Exp D vs E |

---

# PART 9 — TARGET IMPROVEMENT

## 9.1 The report's historical claims

| Quantity | Existing system | Proposed system |
|---|---|---|
| Accuracy | **53.35%** | **80.00%** |
| Precision | not reported | 0.80 |
| Recall | not reported | 0.79 |
| F1 | not reported | 0.79 |

## 9.2 Improvements as claimed — arithmetic

**Accuracy:**
- **Absolute improvement** = 80.00 − 53.35 = **+26.65 percentage points**
- **Relative improvement** = 26.65 / 53.35 = **+49.95%** (≈ a 50% relative increase)
- **Relative error reduction** = (46.65 − 20.00) / 46.65 = **57.13%** of remaining error eliminated
- **Improvement over chance**: existing = +3.35 pts above 50%; proposed = +30.00 pts above 50%,
  i.e. **~9× the above-chance margin**

**Precision / Recall / F1:**
> **NOT COMPUTABLE.** The report supplies these only for the proposed system. With no
> existing-system precision, recall, or F1, **no improvement can be calculated** for these three
> metrics. Marked **`TO VERIFY`** — check whether the report contains a comparison table that was
> not included in the summary. **Do not compute these against an assumed baseline.**

## 9.3 Interpretation before adopting these as targets

Three observations that must be resolved in the Phase 1 report audit:

1. **53.35% is 3.35 points above chance.** See §0.4 Finding 3. Depending on provenance, "80% vs
   53.35%" may not be a valid comparison at all.
2. **The evaluation protocol is unspecified** — video-level or frame-level? which split? what
   threshold? An 80% that is not attached to a protocol cannot be reproduced by definition.
3. **Only classification metrics are quoted.** If the project's central claim is *localization*,
   then **none of these four numbers measure the central claim.** This is the most important
   observation in Part 9.

## 9.4 Realistic targets for the rebuild

Structured as commitments about **method**, not promises about numbers.

| Tier | Target | Rationale |
|---|---|---|
| **T0 — Validity** | Every number produced under a documented, frozen, leak-free protocol with 3 seeds | **Non-negotiable.** A defensible 74% beats an indefensible 84% |
| **T1 — Match** | Video-level accuracy/F1 ≥ the report's 80% / 0.79 **on your own protocol** | Demonstrates the rebuild is at least as good |
| **T2 — Exceed classification** | Beat T1 by a margin exceeding seed variance | The modernizations (log-mel, cross-attention, Transformer) should deliver this |
| **T3 — The real goal: localization** | Report AP@0.5, AP@0.75, AR@100 — **metrics the original never reported** | **This is where the rebuild's genuine contribution lies** |
| **T4 — Ablation evidence** | Quantified per-component contribution across Experiments A–K | Proves *which* parts matter |
| **T5 — Efficiency** | End-to-end inference < 10 s per 10 s clip on GTX 1650; model < 50 MB | Demonstrates practical engineering |

**No target number in T2–T5 is predicted here.** They are filled after execution.

> **The honest framing for your resume and interviews:** the strongest claim is *not* "I beat 80%".
> It is **"the original reported classification metrics only; I rebuilt it with a validated
> leak-free protocol, added temporal localization with AP/AR metrics the original never measured,
> and ablated every component to show which ones actually contribute."** That is a substantially
> better story, and it is one you can defend line by line under questioning.

---

# PART 10 — REPOSITORY STRUCTURE

Improvements over your draft: `docker-compose.yml`/`Dockerfile`/`deploy/` removed (per §0.2);
`data/` split into `raw`/`interim`/`features`/`external` with a strict "never edit raw" rule;
`experiments/` separated from `logs/`; `Makefile` added; `pyproject.toml` replaces bare
`requirements.txt`; `reports/` added for the written deliverable.

```
deepfake-detection/
│
├── README.md                       # what, why, results, how to run
├── pyproject.toml                  # deps + tool config (ruff, mypy, pytest)
├── requirements.txt                # pinned, generated from pyproject
├── Makefile                        # make setup / features / train / test / api
├── .env.example
├── .gitignore                      # data/, checkpoints/, logs/, *.npy
├── .pre-commit-config.yaml
│
├── configs/
│   ├── default.yaml                # base config; all others inherit
│   ├── preprocess/
│   │   ├── video_25fps_112.yaml
│   │   └── audio_16k_mel80.yaml
│   ├── model/
│   │   ├── baseline_visual.yaml
│   │   ├── baseline_audio.yaml
│   │   ├── fusion_concat.yaml
│   │   ├── fusion_crossattn.yaml
│   │   └── final.yaml
│   └── experiment/
│       ├── exp_a_visual.yaml  …  exp_k_audiofeat.yaml
│
├── data/                           # gitignored
│   ├── raw/                        # LAV-DF as downloaded — NEVER MODIFIED
│   ├── interim/faces/  interim/audio/
│   ├── features/{hash}/visual/  {hash}/audio/
│   └── manifests/manifest_v1.parquet  subset_dev2k.txt  subset_dev10k.txt
│
├── src/
│   ├── config.py                   # pydantic schemas — configs are validated, not dicts
│   ├── seed.py                     # seed_everything()
│   ├── data/
│   │   ├── download.py  manifest.py  validate.py  subset.py
│   │   ├── dataset.py              # torch Dataset over cached features
│   │   └── collate.py              # variable-length padding + masks
│   ├── preprocessing/
│   │   ├── video.py  audio.py  face.py  align.py  cache.py
│   ├── models/
│   │   ├── backbones/visual.py  audio_encoder.py
│   │   ├── fusion/{concat,cross_attention,gated}.py
│   │   ├── temporal/{lstm,transformer}.py
│   │   ├── heads/{classification,localization,sync}.py
│   │   ├── gan/{generator,discriminator}.py
│   │   └── build.py                # config → assembled model
│   ├── losses/{focal,infonce,boundary,combined}.py
│   ├── training/{trainer,loop,optim,callbacks,augment}.py
│   ├── localization/{decode,postprocess,merge}.py
│   ├── evaluation/{classification,localization,ap,performance,report}.py
│   ├── inference/{pipeline,predictor}.py
│   └── utils/{io,logging,timing,viz}.py
│
├── api/
│   ├── main.py  routes/{analyze,health}.py  schemas.py
│   ├── services/{inference_service,job_store}.py  dependencies.py
│
├── frontend/
│   ├── package.json  vite.config.ts  index.html
│   └── src/{App.tsx, components/{Upload,Timeline,Verdict,ScoreChart,SegmentList}.tsx, api/client.ts}
│
├── scripts/
│   ├── 00_check_env.py  01_build_manifest.py  02_validate_dataset.py
│   ├── 03_make_subset.py  04_extract_visual.py  05_extract_audio.py
│   ├── 06_train.py  07_evaluate.py  08_run_ablations.py
│   ├── 09_benchmark.py  10_export_model.py
│   └── viz_overlay.py               # burn ground-truth spans onto video — §6.3 check
│
├── tests/
│   ├── unit/{test_manifest,test_audio_align,test_gt_alignment,test_ap_metric,
│   │         test_postprocess,test_model_shapes,test_losses}.py
│   ├── integration/{test_pipeline_e2e,test_api}.py
│   └── fixtures/{tiny_real.mp4, tiny_fake.mp4, expected_*.json}
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb  02_feature_sanity.ipynb
│   ├── 03_attention_viz.ipynb        04_results_analysis.ipynb
│
├── checkpoints/{exp_id}/{run_id}/best.pt      # gitignored
├── experiments/{exp_id}/{run_id}/{config.yaml, metrics.json, predictions.parquet, plots/}
├── logs/
└── reports/{figures/, dataset_statistics.md, results.md, final_report.md}
```

**Three structural rules worth stating explicitly:**
1. **`data/raw/` is immutable.** Every transformation writes elsewhere. This makes the whole
   pipeline re-runnable from a known state.
2. **`experiments/` is append-only.** Never overwrite a run directory. Disk is cheap; a lost
   result is not.
3. **`src/` never imports from `api/` or `scripts/`.** Dependencies point one way, so the core is
   testable without a web server.

---

# PART 11 — DEVELOPMENT PHASES

> **Deployment phase removed per §0.2.** Phases run 0 → 16; old Phase 16 (Docker) is deleted and
> old Phase 17 (Documentation) becomes Phase 16.

**Universal exit gate — applies to every phase.** A phase is complete only when: tests pass, the
output is reproducible from a committed config, the work is committed with a descriptive message,
and results are recorded in `reports/`.

---

### PHASE 0 — Environment & Hardware Validation

| | |
|---|---|
| **Objective** | A working, verified toolchain and an honest, measured picture of what this machine can do |
| **Depends on** | Nothing |
| **Input** | Bare Windows 11 machine |
| **Output** | Working env; `reports/hardware_report.md` |

**Tasks**
1. **Install Python 3.10** — *currently not installed*. Use the python.org installer, not the
   Microsoft Store alias.
2. **Install ffmpeg** — *currently not installed*. Add to PATH; verify `ffmpeg -version`.
3. Create venv on **D:** (C: has only 27 GB free).
4. Install PyTorch with **CUDA 11.8** — verify `torch.cuda.is_available() == True` on the GTX 1650.
5. Install: `torchvision, torchaudio, librosa, opencv-python, mediapipe, pandas, pyarrow, scikit-learn, matplotlib, mlflow, fastapi, uvicorn, pytest, ruff, pydantic, tqdm, av`.
6. **Benchmark honestly**: matmul throughput; peak allocatable VRAM before OOM; decode speed for one video; free RAM under load.
7. `scripts/00_check_env.py` — asserts every dependency and prints the hardware table.

**Files:** `pyproject.toml`, `requirements.txt`, `scripts/00_check_env.py`, `Makefile`, `.gitignore`, `reports/hardware_report.md`

**Success criteria**
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → `True`
- [ ] `ffmpeg -version` works
- [ ] A 4 GB VRAM ceiling test runs without a driver crash
- [ ] `scripts/00_check_env.py` exits 0
- [ ] **Measured** free RAM under a simulated load is recorded

**Expected problems & debugging**

| Problem | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` False | CPU-only wheel installed | Reinstall with the explicit CUDA index URL |
| Driver 526.47 too old for a CUDA build | Driver predates the toolkit | Use CUDA 11.8 wheels (compatible), or update the driver |
| `mediapipe` install fails | Python version mismatch | Pin Python 3.10 — 3.12+ has wheel gaps |
| Out of space on C: | pip caches to C: | Set `PIP_CACHE_DIR` to D: |
| ffmpeg not found | PATH not refreshed | Restart the shell after PATH edit |

**Gate:** CUDA works, ffmpeg works, hardware report written. **Do not proceed without CUDA** —
CPU-only training changes the entire plan's timeline and you need to know now.

---

### PHASE 1 — Dataset Acquisition & Validation

| | |
|---|---|
| **Objective** | LAV-DF on disk, fully validated, with **real measured statistics** replacing every `TO VERIFY` in Part 3 |
| **Depends on** | Phase 0 |
| **Input** | LAV-DF source |
| **Output** | `data/raw/`, `manifest_v1.parquet`, `reports/dataset_statistics.md`, subset id lists |

**Tasks**
1. **Check the download size before downloading** — C: has 27 GB free; download to **D:** (167 GB).
2. Obtain LAV-DF (request access per the authors' terms). Verify checksums.
3. **Audit the report against this plan** — resolve §0.4 Finding 3 (provenance of 53.35% and 80%)
   and every Part 1 `TO VERIFY`. Record findings in `reports/report_audit.md`.
4. Parse `metadata.json`; confirm actual field names against §3.3.
5. `ffprobe` every file → build the manifest.
6. **Compute and write real statistics**: counts, 4-class breakdown, duration distribution,
   **forged-span duration distribution** (drives Part 6 `T`), fake-frame fraction (drives loss
   weighting), fps/sample-rate consistency.
7. **Implement and run the leakage assertions** (§3.5 Rule 3). Build fails on violation.
8. Quarantine bad files per §3.7; report exclusion counts and reasons.
9. Build reproducible subsets (100 / 2k / 10k); **commit the id lists to git**.

**Files:** `src/data/{download,manifest,validate,subset}.py`, `scripts/01–03`, `tests/unit/test_manifest.py`, `reports/{dataset_statistics.md, report_audit.md}`

**Success criteria**
- [ ] Manifest covers every file with zero null labels
- [ ] **All three leakage assertions pass**
- [ ] Every `fake_periods` interval within `[0, duration]`
- [ ] Real videos have empty `fake_periods`, exactly
- [ ] `dataset_statistics.md` written — **every `TO VERIFY` in Part 3 now has a real number**
- [ ] Subset lists committed and reproducible

**Expected problems**

| Problem | Debug |
|---|---|
| Access request pending | Start Phases 2–3 on any small AV video set; swap in LAV-DF later |
| Download exceeds disk | Download split-by-split; extract features and delete raw per split |
| Field names differ from §3.3 | Print one raw entry; update the parser; this is *expected* |
| Leakage assertion fails | **Stop.** Investigate `source_id` derivation before anything else |
| Many `no_face` videos | Inspect 10 manually — detector misconfiguration or genuinely hard footage? |

**Gate:** manifest validated, leakage assertions green, real statistics written, report audit
complete. **The `TO VERIFY` markers in Part 3 and Part 9 must be resolved here or explicitly
escalated as unresolvable.**

---

### PHASE 2 — Video Preprocessing

| | |
|---|---|
| **Objective** | Deterministic video → aligned face crops, verified by eye |
| **Depends on** | Phase 1 |
| **Output** | `data/interim/faces/*.npy`, contact sheets, overlay videos |

**Tasks**
1. Decoder wrapper (PyAV) with exact fps resampling to 25.
2. MediaPipe detection + 5-point similarity alignment → 112×112.
3. Detect every 5th frame, interpolate/track between; handle detection gaps.
4. Content-hashed caching (§3.8); **resumable** (skip if output exists).
5. Contact-sheet visualizer for 20 random videos.
6. **`scripts/viz_overlay.py` — burn `fake_periods` onto video as a red overlay and watch 20 of
   them** (§6.3). This is a mandatory manual check, not optional polish.

**Files:** `src/preprocessing/{video,face,align,cache}.py`, `scripts/viz_overlay.py`, `tests/unit/test_video_preprocessing.py`

**Success criteria**
- [ ] Frame count == `round(duration × 25)` ± 1 for 100 sampled videos
- [ ] Byte-identical output across two runs (determinism)
- [ ] `face_found.mean() > 0.9` on a clean sample
- [ ] Contact sheets show correctly cropped, aligned, upright faces
- [ ] **You have personally watched 20 overlay videos and confirmed the highlighted spans look manipulated**
- [ ] Interrupt-and-resume verified

**Expected problems**

| Problem | Debug |
|---|---|
| **RAM exhaustion (6 GB!)** | Never hold all frames — stream and write incrementally. Cap `num_workers=2` |
| Very slow | Profile decode vs detect. Consider `decord`; lower `detect_every_n` |
| Faces misaligned/rotated | Check landmark index order and the similarity transform; visualize immediately |
| Multiple faces | Pick largest, or the most temporally consistent track — document the choice |
| fps mismatch | Some videos are not 25 fps — resample explicitly, assert afterwards |

**Gate:** determinism verified, visual inspection done, resumability confirmed.

---

### PHASE 3 — Audio Preprocessing

| | |
|---|---|
| **Objective** | Audio on the exact same temporal grid as video |
| **Depends on** | Phase 1 |
| **Output** | `data/interim/audio/*.npy` |

**Tasks**
1. ffmpeg demux → mono → 16 kHz.
2. Log-mel with **`n_fft=1024, hop_length=640, n_mels=80`** (§C — the 40 ms alignment).
3. Per-utterance CMVN.
4. **Assert** `audio_frames == video_frames ± 1` for every file.
5. MFCC path in parallel, for Experiment K.
6. Synthetic-tone round-trip test.

**Files:** `src/preprocessing/audio.py`, `tests/unit/test_audio_align.py`

**Success criteria**
- [ ] **Frame-count assertion passes on 100% of files** — no exceptions, no rounding fudge
- [ ] 440 Hz tone lands in the correct mel bin
- [ ] No NaN/Inf after log (epsilon guard present)
- [ ] Both log-mel and MFCC paths produce identical shapes

**Expected problems**

| Problem | Debug |
|---|---|
| Off-by-one frame counts | `center=True` pads by `n_fft//2` — set `center=False` or trim consistently. **Fix here, not downstream** |
| Silent audio | Check for all-zero waveforms; flag as `no_audio` |
| log(0) → −inf | Add `1e-10` before log |
| Multi-channel audio | Force mono downmix explicitly |

**Gate:** alignment assertion green on the whole subset. **This gate is strict — every alignment
bug you let through here becomes an unexplainable localization failure in Phase 10.**

---

### PHASE 4 — Visual Baseline *(Experiment A + Decision D-1)*

| | |
|---|---|
| **Objective** | First trained model; resolve the ResNet-18/MobileNetV2 contradiction by measurement |
| **Depends on** | Phase 2 |
| **Output** | Visual features cached; B1a/B1b results; **D-1 resolved** |

**Tasks**
1. Frozen-backbone feature extractor for **both** ResNet-18 and MobileNetV2.
2. Extract over dev-2k; record throughput and peak VRAM for each.
3. Attention-pool + classifier head; train both; 3 seeds each.
4. **Overfit-a-batch test first** — 10 samples to ~zero loss. If this fails, the model is broken;
   fix before any real training.
5. Record Experiment J; write the D-1 decision with its justification.

**Files:** `src/models/backbones/visual.py`, `src/models/heads/classification.py`, `src/training/*`, `scripts/{04_extract_visual,06_train}.py`

**Success criteria**
- [ ] Overfit-a-batch reaches near-zero loss
- [ ] Both backbones beat the majority-class baseline by a clear margin
- [ ] Extraction throughput and VRAM measured for both
- [ ] **D-1 resolved and documented with the measurements that decided it**
- [ ] MLflow runs logged with git SHA

**Expected problems**

| Problem | Debug |
|---|---|
| AUC ≈ 0.5 | **Stop.** Labels or crops are wrong. Re-verify Phase 2 before touching the model |
| Overfit test fails | LR too low, frozen params, or a detached graph. Check `requires_grad` |
| CUDA OOM during extraction | Reduce batch to 32/16; use `torch.no_grad()` + `autocast` |
| Train AUC ≫ dev AUC | Expected with a 2k subset — proceed, but note it |

**Gate:** a working trained model, D-1 resolved by data.

---

### PHASE 5 — Audio Baseline *(Experiment B + Decision C-1)*

| | |
|---|---|
| **Objective** | Audio pipeline validated; resolve MFCC vs log-Mel |
| **Depends on** | Phase 3 |

**Tasks**
1. Dilated stride-1 1D-CNN encoder (§E).
2. Train on MFCC (B2a) and log-Mel (B2b); 3 seeds each.
3. **Assert output length == input length** — a temporal-resolution regression here silently
   destroys Phase 10.
4. Record Experiment K; resolve C-1.

**Success criteria**
- [ ] Output length == input length, asserted in code
- [ ] Both variants beat majority-class
- [ ] **C-1 resolved and documented**
- [ ] Overfit test passes

**Expected problems:** audio-only unexpectedly strong (→ good, but read §5.6 now, not later);
padding leaking into loss (use masks); normalization computed on train only.

**Gate:** audio pipeline validated, C-1 resolved.

> **Milestone.** After Phase 5 you have two working models and two report contradictions resolved
> by measurement. **This is already a defensible, presentable project.** Everything after this
> adds capability on a foundation you know is sound.

---

### PHASE 6 — Multimodal Fusion *(Experiment C)*

**Objective:** prove modalities are complementary.
**Tasks:** concat fusion → MLP; align feature streams; verify C > max(A, B); implement modality
dropout early (§5.6); add the 4-class diagnostic breakdown.
**Success:** [ ] C > max(A,B) beyond seed variance — **or** an honest investigation of why not.
[ ] 4-class breakdown table produced. [ ] Feature-scale mismatch ruled out.
**Problems:** if C ≤ max(A,B) → check per-modality normalization first (unequal scales let one
modality dominate the linear layer), then check for collapse (§5.6).
**Gate:** the multimodal premise is empirically supported, or the failure is understood and documented.

---

### PHASE 7 — Temporal Modeling *(Experiment D)*

**Objective:** BiLSTM baseline; isolate temporal contribution from attention contribution.
**Tasks:** 2-layer BiLSTM, 256 hidden; `pack_padded_sequence` for variable length; add the
**per-frame head** and frame-level AP here (first localization-relevant metric).
**Success:** [ ] D > C. [ ] Packing verified — no gradient flows through padding.
[ ] Frame AP computed. **Problems:** LSTM slow on long sequences (reduce `T` or hidden size);
padding contaminating the loss (masked loss, verified by unit test).
**Gate:** temporal contribution measured independently of attention.

---

### PHASE 8 — Self-Attention *(Experiment E)*

**Objective:** demonstrate the §H argument — global comparison beats sequential memory.
**Tasks:** 4-layer pre-norm Transformer; learned positional encoding; **upgrade fusion to
bidirectional cross-attention**; attention-map visualization over the timeline.
**Success:** [ ] E > D beyond seed variance. [ ] Attention maps show elevated attention on forged
spans (**qualitative evidence — this becomes a figure in your report and a talking point in
interviews**). [ ] Positional encoding ablated. **Problems:** training instability → warmup +
pre-norm + gradient clipping (1.0); attention collapse → check head entropy; overfitting on 2k →
expected, move to 10k.
**Gate:** attention contribution measured, attention maps produced.

---

### PHASE 9 — GAN Augmentation *(Experiments F0/F1/F2)*

**Objective:** determine whether GAN augmentation genuinely helps **over strong classical
augmentation** (§I warning).
**Tasks:** implement **manipulation splicing** first (likely the biggest win); spec-augment,
temporal jitter, crop/colour jitter; then the conditional feature-space GAN; run **all three arms**.
**Success:** [ ] F1 > F0. [ ] **F2 vs F1 measured and reported honestly, whichever way it goes.**
[ ] GAN outputs inspected for mode collapse before use.
**Problems:** GAN mode collapse (monitor output diversity; use spectral norm); GAN samples hurting
(likely — report it); **F2 ≈ F1 → the correct conclusion is "the GAN did not add value beyond
classical augmentation," and stating that is a strength, not a failure.**
**Gate:** three arms run, conclusion recorded honestly.

---

### PHASE 10 — Temporal Localization *(Experiment G)* ⭐

**Objective:** the project's headline capability.
**Tasks:** per-frame head with focal loss; optional boundary head; full post-processing chain
(§6.4); **unit-tested AP/AR implementation (§6.7)**; tune τ and post-processing on **dev**, then
freeze; measure boundary error and FP-segment rate on real videos.
**Success:** [ ] AP unit tests pass against hand-computed cases. [ ] Synthetic known-span test
recovers IoU > 0.9. [ ] AP@{0.5,0.75,0.95} + AR@{100,50,20,10} on dev. [ ] Post-processing frozen
before test. [ ] **Predicted timelines visually compared against ground truth for 20 videos.**
**Problems:** low AP but healthy frame AUC → post-processing is the culprit, tune it before
touching the model; systematically shifted predictions → **an alignment bug from Phase 2/3, go
back**; fragmented segments → increase merge gap and smoothing.
**Gate:** localization metrics computed with a **verified** AP implementation.

---

### PHASE 11 — Ablation Experiments

**Objective:** the full evidence table.
**Tasks:** run A–K × 3 seeds; compute mean ± std; apply the four §7.3 criteria to every claimed
improvement; fill Tables 8.2.1–8.2.5; **run the test split exactly once, at the very end.**
**Success:** [ ] All experiments logged. [ ] Every claim passes all four criteria or is reported
as not significant. [ ] **Negative results included.** [ ] Test touched once.
**Gate:** the complete evidence table exists and every number is traceable to a logged run.

---

### PHASE 12 — Model Optimization

**Objective:** meet the latency and size targets.
**Tasks:** profile end-to-end; ONNX export; fp16 inference; batch the extraction; cache the model
in memory; measure Table 8.2.3.
**Success:** [ ] < 10 s per 10 s clip. [ ] Model < 50 MB. [ ] **Accuracy after optimization
re-verified — quantization can degrade it, and shipping an unverified optimized model is how you
end up with a demo that contradicts your results table.**
**Gate:** performance targets met with accuracy confirmed unchanged.

---

### PHASE 13 — FastAPI Backend

**Objective:** a working local API.
**Tasks:** `POST /api/v1/analyze`, `GET /api/v1/jobs/{id}`, `GET /health`, `GET /api/v1/model-info`;
pydantic schemas; background job execution; **strict input validation** (size, MIME, codec,
duration); structured error responses; **model loaded once at startup**; OpenAPI docs auto-generated.
**Success:** [ ] All endpoints tested with `pytest` + `httpx`. [ ] Oversized/corrupt/no-audio/
no-face inputs return clean structured errors, never a 500 stack trace. [ ] **Concurrency limited
to 1** — 6 GB RAM cannot support more. [ ] `/docs` renders.
**Problems:** OOM on concurrent requests (enforce a semaphore of 1); model reloading per request
(load at startup, verify by timing the second request); large uploads (stream to disk, never read
into memory on a 6 GB machine).
**Gate:** API tested and stable under the error cases.

---

### PHASE 14 — Frontend

**Objective:** make the result legible — and make localization *visible*.
**Tasks:** drag-drop upload with progress; polling; **timeline strip with forged spans highlighted**;
video player seeking to a span on click; per-frame score chart; JSON download; graceful error states.
**Success:** [ ] Full flow works against the local API. [ ] **Timeline correctly reflects returned
segments — verified against a known partially-fake video.** [ ] Tested with real / fully-fake /
partially-fake inputs. [ ] Errors shown as readable messages.
**Gate:** a non-expert can upload a video and correctly read the result unaided. **Record a short
screen capture here — it is the single most effective portfolio artifact this project produces.**

---

### PHASE 15 — Testing

**Objective:** the suite that makes the results trustworthy.
**Tasks:** unit tests (manifest, audio alignment, **GT alignment**, **AP metric**, post-processing,
model shapes, losses); integration tests (tiny end-to-end, API); property tests (segments sorted,
non-overlapping, in-bounds); regression test on a fixed fixture; coverage report; CI via GitHub
Actions (test-only — no deployment, per §0.2).
**Success:** [ ] All tests pass. [ ] Coverage > 70% on `src/`. [ ] **The AP metric and GT-alignment
tests exist and pass — these two protect every number in the project.** [ ] CI green.
**Gate:** suite green, coverage met.

---

### PHASE 16 — Documentation *(formerly Phase 17; deployment phase removed)*

**Objective:** make the work legible and defensible to someone who was not there.
**Tasks:** `README.md` (what/why/results/how to run, with the demo capture); `reports/results.md`
with all filled tables; `reports/dataset_statistics.md`; `reports/report_audit.md`;
`reports/final_report.md` (methodology, ablations, limitations); architecture diagrams; docstrings;
**an honest limitations section**; the resume bullet from Part 14.
**Success:** [ ] A stranger can reproduce your headline result from the README alone. [ ] Every
number traces to a logged run. [ ] **Limitations stated honestly.** [ ] No fabricated numbers anywhere.
**Gate:** documentation complete and reproducible.

---

# PART 12 — COMPUTATIONAL REQUIREMENTS

## 12.1 Your measured machine

```
GPU  : GTX 1650, 4 GB VRAM      ← tight but workable for Stage B
CPU  : Ryzen 5 3550H, 4c/8t     ← the bottleneck for Stage A extraction
RAM  : 5.94 GB                  ← ⚠️ THE BINDING CONSTRAINT
Disk : D: 167 GB free           ← ample
       C: 27 GB free            ← ⚠️ keep everything off C:
```

**Verdict: feasible for the full plan on subsets up to ~10k videos, using the cached two-stage
architecture. Not feasible for end-to-end fine-tuning or full-dataset training — use cloud for those.**

## 12.2 The RAM constraint, concretely

With ~2.5–3 GB usable after Windows:

| Practice | Why |
|---|---|
| `num_workers = 2` (never 8) | Each worker holds decoded frame buffers; 8 workers will OOM |
| Stream video, never load fully | A 30 s 1080p clip fully decoded is > 2 GB raw |
| `float16` for cached features | Halves both disk and RAM |
| Memory-map feature files (`mmap_mode='r'`) | OS pages on demand instead of loading everything |
| `persistent_workers=False` | Frees worker memory between epochs |
| Close Chrome while training | Not a joke — Chrome routinely holds 1–2 GB |
| Set pagefile on **D:** | C: has only 27 GB; thrashing there will freeze the machine |

## 12.3 Configuration tiers

### Minimum (your machine — the plan is written for this)
```
Subset          : dev-2k for all decisions, dev-10k for headline numbers
Batch size      : 8–16 (Stage B, cached features)
Grad accumulation: 2–4  → effective batch 32
Sequence length : T = 250 (10 s @ 25 fps)
Frames          : 25 fps, no temporal subsampling
Resolution      : 112 × 112 face crops
Precision       : AMP fp16
DataLoader      : num_workers=2, pin_memory=False
Backbone        : FROZEN — non-negotiable at 4 GB
Feature extract : overnight batch job, resumable
Stage B train   : 10–45 min per run  ← this is what makes 40 experiments possible
```

### Recommended (if you can access it)
```
GPU : RTX 3060 12 GB / RTX 4060 Ti 16 GB
RAM : 32 GB
CPU : 8 cores
→ batch 32–64, T = 500, optional partial backbone unfreezing, num_workers=8
```

### Cloud alternative — **strongly recommended for Phases 9–11 at scale**

| Platform | GPU | Free tier | Best for |
|---|---|---|---|
| **Kaggle Notebooks** | P100 16 GB / T4×2 | **30 h/week free** | ⭐ **Best free option.** Enough VRAM for end-to-end fine-tuning; persistent datasets |
| Google Colab (free) | T4 16 GB | Limited, preemptible | Quick experiments; **checkpoint every epoch to Drive** |
| Colab Pro | T4/L4/A100 | ~$10/mo | Full-dataset runs |
| Vast.ai / RunPod | RTX 3090/4090 | ~$0.20–0.50/h | Final full-dataset training; ~$10–25 total |

**Recommended split of labour:**
- **Local (GTX 1650):** development, debugging, preprocessing, all of Phases 0–8, the full ablation
  matrix on cached features, API and frontend. This is the majority of the project.
- **Cloud (Kaggle free tier):** full-dataset feature extraction, end-to-end fine-tuning, the final
  Phase 11 headline runs.

Keep the code **device-agnostic** (`device = "cuda" if available else "cpu"`, all paths from config)
so the same code runs in both places without edits.

## 12.4 Disk budget

| Item | Estimate |
|---|---|
| LAV-DF raw | `TO VERIFY` — check before downloading |
| Face crops, dev-10k | ~15–25 GB (`uint8 [T,112,112,3]`) |
| Visual features, dev-10k | ~2.5 GB (fp16) |
| Audio features, dev-10k | ~0.8 GB |
| Checkpoints (40 runs) | ~2 GB |
| Experiment logs | < 1 GB |
| **Total working set** | **~25–35 GB on D:** — comfortable within 167 GB |

> **Optimization:** face crops are the largest intermediate. Once visual features are extracted,
> crops can be deleted (regenerable from raw). Keep crops only for the smoke subset, for debugging.

## 12.5 Time budget (rough, `TO VERIFY` by measurement in Phase 2)

| Task | Estimate on this machine |
|---|---|
| Preprocess + extract, 2k videos | 2–4 h (overnight) |
| Preprocess + extract, 10k videos | 10–20 h (weekend) |
| One Stage-B training run | 10–45 min |
| Full ablation matrix (~40 runs) | ~20 h spread over days |

---

# PART 13 — MODERNIZATION

Every change below is justified. Nothing is replaced merely for being older.

| # | Original | Recommendation | Verdict | Reasoning |
|---|---|---|---|---|
| 1 | TensorFlow/Keras | **PyTorch** | ✅ Change | Dominant in research; better dynamic-shape handling for variable-length video; better debugging; more pretrained AV models. **The report lists both — PyTorch is a clean resolution, not a rejection** |
| 2 | ResNet-18 **vs** MobileNetV2 | **Decide by Experiment J** | ⚖️ **Measure** | §0.4 Finding 2 / §5.2. My prior is ResNet-18@112² with stated reasons, but the data decides |
| 3 | MFCC | **log-Mel** (MFCC as ablation) | ✅ Change, **measured** | DCT discards the fine spectral structure where vocoder artifacts live. Proven via Experiment K, not asserted |
| 4 | LSTM | **Transformer encoder** (LSTM as Baseline 4) | ✅ Change, **measured** | Forged spans are anomalous *relative* to the rest of the video; attention compares all pairs, an LSTM's compressed state cannot. Experiment D vs E |
| 5 | Simple fusion | **Bidirectional cross-attention** | ✅ Change, **measured** | Directly models "does audio at *t* explain video at *t*?" — which is the task. Concat kept as Baseline 3 |
| 6 | GAN augmentation (unspecified) | **Feature-space GAN + manipulation splicing**, F0/F1/F2 arms | ⚠️ **Reframe** | Full AV deepfake synthesis is infeasible on 4 GB. §I. And the three-arm design is what makes the claim *valid* |
| 7 | Flask | **FastAPI** | ✅ Change | Async, pydantic validation, auto OpenAPI docs, better for long-running inference jobs. Flask's sync model blocks on a 10 s inference |
| 8 | Softmax (2-way) | **Sigmoid + BCE** | ✅ Change | Equivalent for binary; simpler to calibrate/threshold; matches the per-frame head |
| 9 | No experiment tracking | **MLflow (local)** | ➕ Add | Without it, 40 ablation runs are unmanageable and Part 7 is not executable |
| 10 | No versioning | **Content-hashed features + versioned manifests + git SHA per run** | ➕ Add | §3.8. Eliminates an entire class of reproducibility bugs |
| 11 | No tests | **pytest suite** | ➕ Add | The AP-metric and GT-alignment tests protect *every number in the project* |
| 12 | Classification metrics only | **+ AP/AR at IoU thresholds** | ➕ Add | §9.3 — the original never measured its own central claim |
| 13 | No leakage control | **Enforced identity-disjoint splits with build-failing assertions** | ➕ Add | §3.5. The highest-impact correctness addition in this plan |
| 14 | — | **Modality dropout + 4-class diagnostics** | ➕ Add | §5.6 — defends the multimodal premise against silent collapse |
| 15 | Docker/deployment | **Removed** | ➖ Removed | Per your instruction (§0.2) |
| 16 | Config in code | **YAML + pydantic validation** | ➕ Add | Makes every experiment a config diff |

**Kept from the original, deliberately:** the overall multimodal architecture; the 1D-CNN audio
encoder; the lip-sync verification concept; temporal localization as the central goal; LAV-DF;
OpenCV, librosa, NumPy/Pandas/Matplotlib. **The report's core design is sound.** These changes
modernize the *implementation* of a good idea, not the idea.

---

# PART 14 — RESUME-READY OUTCOME

## 14.1 Metrics to collect *(placeholders until measured)*

**Classification:** video ROC-AUC, F1, precision, recall (test split, 3 seeds, mean ± std) —
against your own reproduced baseline **and** noting the report's historical 80% / 0.79.

**Localization — the differentiator:** AP@0.5, AP@0.75, AR@100, **mean boundary error in ms**,
**false-positive segment rate on real videos**.

**Ablation:** ΔAUC and ΔAP contributed by each of fusion, temporal, attention, GAN augmentation.

**Efficiency:** end-to-end latency per 10 s clip, model size (MB), peak VRAM, preprocessing
throughput (videos/min).

**Engineering:** test coverage %, number of tracked experiments, reproducibility (identical results
from a committed config).

## 14.2 Resume templates *(numbers stay as `___` until experimentally verified)*

**Primary (one line):**
> Developed a multimodal audio-visual deepfake detection system in **PyTorch** combining a
> **[BACKBONE]** visual encoder, a dilated 1D-CNN log-Mel audio encoder, and **cross-modal
> attention with a Transformer temporal encoder**, achieving **___ ROC-AUC / ___ F1** on the
> LAV-DF test set versus **___** for the strongest unimodal baseline, and localizing forged
> segments at **___ AP@IoU 0.5** with **±___ ms** mean boundary error at **___ s** end-to-end
> inference on a 4 GB GPU.

**Detailed (bulleted):**
> - Built an end-to-end audio-visual deepfake **detection and temporal localization** pipeline
>   (PyTorch, FastAPI, React) on the LAV-DF dataset, reaching **___ ROC-AUC** and **___ AP@0.5**
>   for forged-segment localization.
> - Designed a **cached two-stage architecture** enabling **___** tracked ablation experiments on
>   a 4 GB consumer GPU, reducing per-experiment training time from hours to **___ minutes**.
> - Ran a **7-experiment ablation** isolating the contribution of each component; **self-attention
>   over LSTM contributed +___ AP**, and cross-modal fusion **+___ AUC** over the best unimodal baseline.
> - Enforced **identity-disjoint splits** with automated leakage assertions, and implemented
>   **modality-dropout regularization** to prevent unimodal collapse, verified by per-manipulation-type
>   diagnostic breakdown.
> - Localized forged spans to **±___ ms** with a **___%** false-positive segment rate on genuine
>   videos, served through a REST API with a timeline visualization frontend.

**Interview one-liner:**
> "The original project classified whole videos. I rebuilt it to answer *where* the forgery is —
> and, just as importantly, to prove which architectural components actually earned their place,
> by ablating all of them under a leak-free protocol."

## 14.3 Fill-in tracker

| Placeholder | Source | Status |
|---|---|---|
| `[BACKBONE]` | Experiment J | ⬜ |
| ROC-AUC / F1 | Table 8.2.1 (test) | ⬜ |
| AP@0.5 / AP@0.75 | Table 8.2.2 (test) | ⬜ |
| Boundary error (ms) | Table 8.2.2 | ⬜ |
| FP-segment rate | Table 8.2.2 | ⬜ |
| Latency, model size | Table 8.2.3 | ⬜ |
| Ablation deltas | Part 7 | ⬜ |
| Experiment count | MLflow | ⬜ |

---

# PART 15 — FINAL DELIVERABLES

**1. Architecture diagram** — §5.1 (two-stage ASCII diagram; render as a proper figure in Phase 16).

**2. Implementation roadmap** — Part 11, Phases 0–16, each with gate criteria.

**3. Repository structure** — Part 10.

**4. Dataset pipeline** — Part 3 + Phases 1–3: acquire → manifest → validate → **leakage assertions**
→ subset → face crops → log-mel → cached features.

**5. Model architecture** — Part 5, with D-1 and C-1 resolved by Experiments J and K.

**6. Training pipeline** — config → cached features → Stage-B model → composite loss → AMP → 3 seeds
→ MLflow → best checkpoint on dev.

**7. Evaluation pipeline** — decoupled: predictions → parquet → metrics → tables/plots. Recomputable
without retraining.

**8. Experiment matrix** — Part 7 §7.2, Experiments A–K.

**9. Ablation plan** — Part 7 §7.3, four-criteria significance test, negative results reported.

**10. API architecture** — Part 2 §M + Phase 13. FastAPI, 4 endpoints, background jobs,
concurrency limited to 1.

**11. Frontend architecture** — Part 2 §N + Phase 14. React + Vite; **timeline visualization is
the centrepiece**.

**12. Testing strategy** — Part 11 Phase 15. Unit / integration / property / regression, with the
**AP-metric and GT-alignment tests as the critical two**.

**13. Local run strategy** *(replaces "Deployment strategy" per §0.2)* — `make api` runs uvicorn on
`:8000`; `npm run dev` runs Vite on `:5173`; a `make demo` target runs both. Model loaded from
`checkpoints/`. No containers, no cloud hosting, no CI deployment.

**14. Documentation plan** — Part 11 Phase 16.

**15. Resume metrics** — Part 14.

**16. Interview questions to prepare for** *(with where the answer lives)*

| Question | Answer lives in |
|---|---|
| Why multimodal instead of visual-only? | §1.1, Experiment C |
| Why self-attention rather than LSTM? | §H rationale, Experiment D vs E |
| How do you know the model isn't just reading audio? | §5.6, Table 8.2.4, Experiment I |
| How did you prevent data leakage? | §3.5 — identity-disjoint splits + build-failing assertions |
| Why AP instead of accuracy for localization? | §3.6, §8.1 |
| How did you pick the threshold? | §6.5 — tuned on dev, frozen before test |
| The report says ResNet-18 in one place, MobileNetV2 in another — what did you do? | §5.2 — measured both, Experiment J |
| Did the GAN augmentation actually help? | §I three-arm design; F2 vs F1, answered honestly |
| How would you scale this to production? | Cached features, batched extraction, ONNX; note the deliberate 4 GB constraint |
| What's the biggest weakness? | Generalization to unseen generators — a genuine, honest limitation |
| Why 40 ms hops? | §C — exact 1:1 alignment with 25 fps video |
| How do you know your AP implementation is right? | §6.7 — unit-tested against hand-computed cases |

**17. Risks and mitigations**

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | **6 GB RAM** exhaustion during preprocessing | **High** | High | Cached two-stage design; stream decoding; `num_workers=2`; resumable jobs; pagefile on D: |
| R2 | LAV-DF access delayed or unavailable | Medium | **High** | Request access in Phase 1 *immediately*; develop Phases 2–3 on any small AV set meanwhile |
| R3 | **Data leakage** inflating results | Medium | **Critical** | Official splits + build-failing identity assertions (§3.5) |
| R4 | **Modality collapse** — model ignores video | **High** | High | Modality dropout, sync loss, 4-class diagnostics (§5.6) |
| R5 | GT temporal misalignment | Medium | **Critical** | Inverse-transform assertion + **manual overlay inspection** (§6.3) |
| R6 | Wrong AP implementation invalidating all results | Medium | **Critical** | Unit tests against hand-computed cases (§6.7) |
| R7 | GAN augmentation provides no benefit | **High** | Low | Three-arm design; reporting a negative result is a valid, honest outcome |
| R8 | Disk exhaustion | Medium | Medium | Check download size first; everything on D:; delete face crops after extraction |
| R9 | Cannot reproduce the report's 80% | Medium | Medium | §9.4 reframes success around protocol validity and localization, not a single number |
| R10 | Scope creep | **High** | Medium | Phase gates; Phase 5 is already a defensible milestone |
| R11 | The 53.35% baseline turns out invalid | Medium | Medium | §0.4 Finding 3 — replace with a baseline you measure yourself |
| R12 | Overfitting on small subsets | High | Low | Expected and understood; scale to 10k for headline numbers; report both |
| R13 | Training interrupted (6 GB machine, long jobs) | **High** | Medium | Checkpoint every epoch; resumable extraction; assume interruption by default |

---

## APPENDIX A — All open `TO VERIFY` items

**Resolve in Phase 1 (report audit + dataset statistics):**

| # | Item | Where | Blocking? |
|---|---|---|---|
| 1 | Provenance of the **53.35%** existing-system figure | §0.4, §9.3 | **Yes** — Part 9 validity |
| 2 | Evaluation protocol behind the **80.00%** figure | §0.4, §9.3 | **Yes** — Part 9 validity |
| 3 | Existing-system precision / recall / F1 | §9.2 | Yes — improvements not computable without them |
| 4 | Whether the report used any localization metrics | §1.11 | Yes — determines the rebuild's claim |
| 5 | Report's training hyperparameters | §1.9 | No |
| 6 | Whether the report used sliding windows or dense prediction | §1.10 | No |
| 7 | Report's stated inference-latency target | §1.5 NFR-02 | No |
| 8 | LAV-DF actual video counts and class balance | §3.2 | Yes — drives loss weighting |
| 9 | LAV-DF download size | §3.2, §12.4 | **Yes** — disk planning |
| 10 | **Mean forged-span duration** | §3.2, §6.2 | **Yes** — determines `T` |
| 11 | Fake-frame fraction | §3.2, §3.6 | Yes — drives focal/pos_weight |
| 12 | Exact `metadata.json` field names | §3.3 | Yes — parser correctness |
| 13 | Whether an identity key exists for leakage checks | §3.5 | **Yes** — R3 |
| 14 | Actual fps / sample-rate consistency across files | §3.2 | Yes — alignment |
| 15 | Full-dataset extraction wall-clock on this machine | §7.4, §12.5 | No — measure in Phase 2 |

## APPENDIX B — Decision log *(fill as decisions are made)*

| ID | Decision | Options | Resolved by | Date | Outcome |
|---|---|---|---|---|---|
| **D-1** | Visual backbone | ResNet-18 / MobileNetV2 | Experiment J, Phase 4 | ⬜ | ⬜ |
| **C-1** | Audio features | MFCC / log-Mel | Experiment K, Phase 5 | ⬜ | ⬜ |
| B-1 | Face detector | MediaPipe / MTCNN / RetinaFace | Phase 2 | ⬜ | MediaPipe (recommended) |
| F-1 | Sync approach | Learned / pretrained SyncNet | Phase 8 | ⬜ | Learned (recommended) |
| G-1 | Fusion | Concat / gated / cross-attention | Experiments C vs E | ⬜ | ⬜ |
| H-1 | Temporal | BiLSTM / Transformer | Experiments D vs E | ⬜ | ⬜ |
| I-1 | GAN tier | Feature-space / face-region / full synthesis | Phase 9 | ⬜ | Tier 1 (recommended) |
| L-1 | Localization | Dense per-frame / sliding window | Phase 10 | ⬜ | Dense (recommended) |

---

## NEXT STEP

**This plan is awaiting your approval. No code has been written.**

Three things I need from you before Phase 0 begins:

1. **The report** — drop it into this folder. Part 1 and the 15 `TO VERIFY` items in Appendix A
   need auditing against the real document, and items 1–4 directly affect whether Part 9's
   comparison is valid at all.
2. **Confirm the hardware strategy** — local-only (subsets up to ~10k), or local + Kaggle free
   tier for the full-dataset runs? This changes Phase 1's download strategy.
3. **Approve or amend the plan** — particularly the reframing of GAN augmentation (§I), which is
   the largest deviation from the original report, and the decision to leave D-1 and C-1 to be
   settled by experiment rather than by choice.

On approval, we implement phase by phase, with the reasoning explained before any code is generated.
