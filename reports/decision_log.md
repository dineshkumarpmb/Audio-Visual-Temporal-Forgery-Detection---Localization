# Decision Log

The living version of `PROJECT_PLAN.md` Appendix B, plus the Phase −1 pre-flight decisions.
`PROJECT_PLAN.md` itself is treated as a signed artifact and is not edited; deltas are recorded here.

---

## Phase −1 pre-flight decisions — resolved 2026-08-12

### PF-1 · Hardware strategy (task PRE-2)

**Decision: Local + Kaggle free tier.** ✅ Approved 2026-08-12.

| | |
|---|---|
| **Local (GTX 1650, 4 GB / 6 GB RAM)** | Phases 0–8 in full, all preprocessing and Stage-A extraction for subsets, the complete ablation matrix on cached features, Phases 12–16 (optimization, API, frontend, tests, docs) |
| **Kaggle (free P100 16 GB, 30 h/week)** | Full-dataset feature extraction, any end-to-end fine-tuning, the final Phase 11 headline runs |

**Consequences accepted:**
1. Task **X-2 (device-agnostic code) is promoted from good practice to a hard requirement.** No
   hardcoded paths, no `cuda:0` literals, every path from config. A Phase-8 shortcut here becomes a
   Phase-11 refactor.
2. The dev-10k ceiling in §3.9 is **no longer a hard cap** — the `Full` row becomes reachable, so
   Part 9's targets need not carry a permanent subset caveat.
3. New work is required for cloud sync: mirroring the dataset/features as a Kaggle Dataset,
   and getting checkpoints and MLflow runs back to D:. Added as tasks **CL-1 … CL-6**.
4. Kaggle sessions are time-boxed and preemptible → **R13 (interruption) now applies to cloud runs
   too**, not just local ones. Resumable extraction and per-epoch checkpointing become mandatory
   rather than merely prudent.

---

### PF-2 · GAN augmentation scope (task PRE-3a)

**Decision: Tier 1 as planned, three-arm design retained.** ✅ Approved 2026-08-12.

- **In scope:** manipulation splicing (§I(a)) and the conditional feature-space GAN
  `G: (noise, label) → [T,256]` with a sequence discriminator (§I(b)).
- **Out of scope:** Tier 2 (face-region StyleGAN2-ADA/pix2pix) and Tier 3 (full AV synthesis via
  Wav2Lip + TTS) — infeasible on 4 GB VRAM within the project timeline.
- **Binding constraint:** Experiment F runs **all three arms F0/F1/F2**. The claim "the GAN helped"
  may be made **only** on `F2 > F1`, never on `F2 > F0`.
- **Pre-committed:** if `F2 ≈ F1`, the reported conclusion is *"the GAN added no value beyond
  classical augmentation."* This is recorded **before** running the experiment so the negative
  result cannot be quietly reframed after the fact.

This is the largest deviation from the original report and must be stated plainly in
`reports/final_report.md` (task P16-3) with the 4 GB VRAM justification.

---

### PF-3 · D-1 and C-1 resolved by experiment, not by choice (task PRE-3b)

**Decision: both settled empirically.** ✅ Approved 2026-08-12.

- **D-1** (ResNet-18 vs MobileNetV2) → **Experiment J**, Phase 4. Both extracted and trained on
  identical splits, seeds and schedules; compared on frame-level AUC **and** extraction throughput
  and peak VRAM.
- **C-1** (MFCC vs log-Mel) → **Experiment K**, Phase 5.

**Cost accepted:** one extra visual extraction pass over dev-2k, plus ~6 additional short Stage-B
training runs (~2–3 h total).

**Tie-break rules, fixed in advance** (so the decision cannot be rationalized after seeing results):
- **D-1:** if the AUC gap is < 1 point *and* the seed-variance intervals overlap, **take
  MobileNetV2** if its extraction is ≥2× faster — faster extraction buys more experiments, which is
  worth more than a fractional AUC point (§5.2).
- **C-1:** if the gap is within seed variance, **take log-Mel** — it is the simpler pipeline
  (no DCT stage) and preserves the fine spectral structure the plan argues for in §C-1.

---

## Phase 0 decisions — resolved 2026-08-12

### PF-4 · Numeric precision: fp32 locally, AMP fp16 on cloud

**Decided by measurement, and it overturns the plan.** `PROJECT_PLAN.md` §12.3 specifies
*"Precision: AMP fp16"* for the minimum (local) configuration. Benchmarking says otherwise.

Measured on the GTX 1650, 2048³ GEMM, reproducible across 6 trials in two independent scripts:

| Precision | Throughput | vs fp32 |
|---|---|---|
| fp32 | **2.38–2.56 TFLOP/s** | 1.00× |
| bf16 | 1.52 TFLOP/s | 0.60× |
| fp16 | **0.31–0.32 TFLOP/s** | **0.13×** |

A 3×3 convolution at 112² — much closer to real Stage-A work than a raw GEMM — shows the same
direction: **1601 img/s fp32 vs 303 img/s fp16 (0.19×)**.

**Cause:** the GTX 1650 is TU117, the one Turing die shipped **without tensor cores**. There is no
fast fp16 GEMM path, so cuBLAS falls back to a slow kernel. AMP's usual "free 2× speedup" simply
does not exist on this card; fp16 here buys memory and costs 5–8× compute.

**Decision:**

| Tier | Precision | Reasoning |
|---|---|---|
| Local (GTX 1650) | **fp32** | The memory saving is obtainable more cheaply by lowering batch size; the compute cost is not recoverable |
| Cloud (Kaggle P100 / T4) | **AMP fp16** | T4 has tensor cores, P100 has a fast fp16 path. **Re-run `make check-bench` there to confirm rather than assume** |

**Consequences:**
1. Precision becomes a **config field**, never a constant — the same code must be correct in both
   tiers (task X-2, decision PF-1).
2. Cached features stay `float16` **on disk** (Plan §D). That is storage, not compute, and is
   unaffected. Cast to fp32 on load.
3. §12.3's "Minimum" configuration block is superseded on this one line. Recorded here rather than
   by editing the plan.
4. Any future latency claim (Part 8 §8.1, Table 8.2.3) must state which precision it was measured
   under.

> This is the plan's own methodology working as intended: §0.3 forbids inheriting an unverified
> number, and the first thing Phase 0 measured contradicted a design assumption. Worth reporting in
> `reports/final_report.md` — it is a concrete example of measurement beating convention.

### PF-5 · Full JupyterLab excluded from dev dependencies

**Forced by a real failure, not preference.** `pip install -e ".[dev]"` aborted with
`OSError: [Errno 2] No such file or directory` on a JupyterLab webpack asset whose path exceeded
Windows' 260-character `MAX_PATH`. The project directory name
(`Audio-Visual Temporal Forgery Detection & Localization`, 54 chars) plus
`.venv/share/jupyter/labextensions/@jupyter-widgets/...` plus ~120-char bundle filenames overflows
the limit, and pip aborts the **entire transaction** — so ~20 unrelated packages failed to install
as collateral.

**Decision:** dev deps carry `ipykernel` + `nbformat` only. That is sufficient to run
`notebooks/*.ipynb` from VS Code or an external Jupyter. Anyone wanting the standalone Lab UI
installs it in a separate venv outside this path.

**⚠️ Open item — this will recur in Phase 14.** `node_modules` for React + Vite nests far deeper
than JupyterLab does. Enabling Windows long-path support is recommended **before** Phase 14, and
requires an elevated shell:

```powershell
# Run as Administrator, then reboot
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
```

Currently `LongPathsEnabled = 0`. Tracked as task **P14-0**.

---

## Phase 1 decisions — resolved 2026-08-27 / 2026-08-29

### PF-6 · Dataset access: attach on Kaggle, never download the raw 25.6 GB locally

**Decision: cloud-first data path.** ✅ Approved 2026-08-27. Supersedes task P1-2 as originally
written ("download LAV-DF to `data/raw/`") and the §12.4 disk budget's assumption that raw video
lives on D:.

**What changes.** A public Kaggle mirror of LAV-DF exists
(`elin75/localized-audio-visual-deepfake-dataset-lav-df`). Kaggle mounts an attached dataset
read-only at `/kaggle/input` — no download, no transfer wait, and it does **not** consume the
20 GB `/kaggle/working` budget. So Phases 1–4 run against the mounted copy, and only the derived
feature cache ever comes down to D:.

| | Before | After |
|---|---|---|
| Local raw video | 25.6 GB | **0 GB** (plus ~200 MB smoke-100 for the ✋ checks) |
| Local download | 25.6 GB | **~3.3 GB** — the dev-10k feature cache |
| Where Phases 1–4 run | local | **Kaggle** |
| Where Phases 4–11 run | local | local, unchanged |

**Why this works and is not a shortcut.** The plan already touches raw video exactly once: Phases
2–3 turn video into face crops + log-mel, Phase 4 turns those into a frozen-feature cache, and
every run from Phase 4 to Phase 11 trains on that cache. Moving the one-pass stage to the machine
that already holds the data is the natural placement, not a compromise.

**Size figure, stated precisely.** The ~3.3 GB is §12.4's **dev-10k** cache (2.5 GB visual fp16 +
0.8 GB audio). It is *not* the full-dataset figure — 136,304 videos would be ≈45 GB of features.
Full-dataset features stay on Kaggle and are never synced to D:; only metrics and checkpoints come
back (CL-5). This is consistent with PF-1, which already scoped the local track to dev-10k and the
`Full` row to Kaggle.

**Consequences accepted:**

1. **CL-1, CL-2, CL-3 move from "alongside Phases 9–11" to *before* P1-2.** The Kaggle account,
   the attached dataset and the thin notebook entrypoint are now Phase 1 prerequisites, not
   Phase 9 conveniences.
2. **X-2 (device-agnostic code) becomes load-bearing on day one.** Under PF-1 it was a hard
   requirement that would first be exercised around Phase 9. It is now exercised by the very first
   script that runs. This is strictly healthier — the requirement gets tested while the codebase is
   small enough to fix cheaply.
3. **New verification burden — the mirror is a community re-upload, not the authors' bucket.**
   It must be proven equivalent before anything is built on it. Added as **CL-7** with hard checks.
   The specific hazard: if the mirror was re-encoded, fps and duration shift and every
   `fake_periods` target silently moves. P1-5's `assert max(end) <= duration` catches the gross
   case; CL-7 covers the rest.
4. **Full-dataset feature extraction will not fit in one session.** ≈45 GB of features against a
   20 GB `/kaggle/working` cap and a 12 h session limit means sharded, resumable output across
   several sessions. Added as **CL-8**. R13 already demanded resumability; this makes it structural.
5. **The ✋ manual checks (P2-8, P2-10) still need real video locally.** Either pull the smoke-100
   subset (~200 MB) or render the 20 overlay clips on Kaggle and download those. Recorded against
   P2-10.
6. **PF-4's cloud half becomes testable immediately.** The P100/T4 have tensor cores; the local
   TU117 does not. AMP fp16 can now be benchmarked on the cloud side during Phase 1 rather than
   remaining an assumption until Phase 9.

**Options rejected:**

- **HuggingFace streaming.** The repo is a single 25.6 GB `LAV-DF.tar`, not sharded
  parquet/webdataset. `load_dataset(streaming=True)` gives sequential access only — adequate for a
  one-pass extraction, useless for shuffled training or stratified subsetting (P1-15). Also gated.
- **Colab + Google Drive shortcut.** The authors' Drive folder is open, so this avoids a local
  download too. Rejected on three counts: Colab's ~78 GB disk is ephemeral so the data is re-copied
  every session; sustained reads off mounted Drive are slow and hit quota errors; free GPU
  allocation is unguaranteed. Kaggle's persistent read-only mount dominates it for this workload.
- **Download the 25.6 GB to D: anyway.** Not wrong — 156.7 GB is free — but it buys nothing the
  mount does not, and costs a long interruptible transfer on a machine R13 assumes will be
  interrupted.

---

### PF-7 · The video timeline is `video_frames`, not `duration` — P2-1 is wrong as written

**Decision: `T == video_frames`.** ✅ Resolved 2026-08-29 by measurement during P1-5/P1-13.
Corrects task **P2-1** in PROJECT_PLAN Phase 2 and the semantics of the `duration` column in the
§2A manifest schema.

**What was measured.** For **all 136,304 entries**, metadata's `duration` satisfies

    duration == audio_frames / 16000 + 0.128        (exactly, 136,304/136,304)

It is a padded audio-derived figure, not a media duration. `ffprobe` on 28 stratified videos
confirms it exceeds *both* stream durations on every file:

| | metadata | ffprobe video | ffprobe audio |
|---|---|---|---|
| `dev/101081.mp4` | 4.224 s | 4.120 s | 4.160 s |
| `train/026282.mp4` | 18.944 s | 18.880 s | 18.880 s |

**Why P2-1 fails.** P2-1 says `assert T == round(duration × 25) ± 1`. Measured over the whole
dataset, `video_frames − round(duration × 25)` ranges from **−5 to −1**, and only **239 of
136,304** entries (0.18%) fall within ±1. That assertion would fail on 99.8% of the dataset — and
worse, anyone "fixing" it by *trusting* `duration` would allocate 2–5 phantom frames per video and
silently misalign every localization target against the real frames.

**The correct invariant.** `video_frames` matched `ffprobe`'s `nb_frames` on **28/28** spot-checked
files, and `r_frame_rate` is exactly 25.00 on all of them. So:

- the manifest's `duration` column is **`video_frames / 25`** — the timeline frames actually live on;
- metadata's own value is preserved as `duration_meta`, not discarded, so the derivation stays auditable;
- `bad_label` is judged against the **video** timeline, which is why 4 entries quarantine rather
  than the 0 that checking against the padded `duration` alone would give.

**Consequences accepted:**

1. **P2-1's assertion text must be replaced**, not merely relaxed. Recorded in TASKS.md.
2. **Phase 6 localization targets rasterise against `video_frames`.** A span end is clamped to
   `n_frames`; the ~0.1 s of padded tail has no frames and must not be allocated any.
3. **The 4 `bad_label` entries are real dirty data, not our bug** — checked by hand, all four have
   a span starting *after* the video ends (e.g. `train/092507.mp4`, span `[16.4, 17.454]` against
   16.12 s of video). They are quarantined, not repaired.

**Options rejected:**

- **Trust `duration` and pad the frame tensor.** Would invent 2–5 frames per video with no pixels
  behind them, and shift every target that Part 6 depends on.
- **Relax P2-1 to ±5.** Hides the finding instead of recording it, and ±5 is not a tolerance — it
  is a systematic offset with an exact closed form.

---

## Phase 2 decisions — resolved 2026-08-29

### PF-8 · Decision B-1 amended: MediaPipe Tasks API, with pinned model weights

**Decision: keep MediaPipe, change the API.** ✅ Resolved 2026-08-29. Amends **B-1**, which named
MediaPipe as the face detector, and the Phase 2 spec's "468 landmarks" via `mp.solutions.face_mesh`.

**What broke.** `mediapipe 1.0.0` (the version Phase 0 installed and validated) **removed
`mp.solutions` entirely**. `import mediapipe as mp; mp.solutions` raises `AttributeError`. The
wheel also ships **no model weights** — neither `.task` bundles nor `.tflite` — and the installed
`opencv-python` 5.0 no longer bundles Haar cascades either, so there was no in-package fallback.

**Resolution.** Use `mediapipe.tasks.python.vision.FaceLandmarker` with a `face_landmarker.task`
bundle fetched by `scripts/08_fetch_models.py` into `models/`. Underneath this is the same
BlazeFace + FaceMesh pipeline B-1 chose; only the Python surface and the weight distribution
changed, so B-1's comparison against MTCNN/RetinaFace still stands.

**Consequences accepted:**

1. **A network fetch is now a Phase 2 prerequisite.** Sizes and sha256 prefixes are pinned in the
   script: `face_landmarker.task` 3,758,596 B / `64184e229b263107`. A changed bundle changes every
   crop, so it must be a loud event, not silent drift in the cache.
2. **`models/` is gitignored** — 3.8 MB of weights are re-fetchable, not source.
3. **Landmark count is 478, not 468** (the Tasks bundle adds 10 iris points). Immaterial here: the
   5 alignment points are all in the original 468.

### PF-9 · `face_margin` defaults to 0.0, not the plan's 0.25

**Decision: margin 0.0 for LAV-DF.** ✅ Resolved 2026-08-29 by measurement. Amends the
`margin=0.25` in PROJECT_PLAN section B.

**What was measured.** Every LAV-DF video is **224×224** — the frames are already tight VoxCeleb2
face crops, not full scenes. There is no surrounding context for a margin to include, so widening
the crop only manufactures replicated border. Over 12 subjects, the fraction of crop pixels falling
outside the source frame:

| margin | out-of-frame pixels |
|---|---|
| **0.00** | **11.0%** mean, 17.5% max |
| 0.10 | 14.9% mean, 21.7% max |
| 0.25 | 21.4% mean, 29.1% max |

**Why it matters.** `BORDER_REPLICATE` smears the edge row outward. At 0.25 more than a fifth of
every crop would be fabricated pixels carrying no facial information — and a backbone will happily
learn the smear as a feature. The parameter stays configurable because a full-frame dataset would
genuinely want 0.25.

**Consequence:** even at 0.0 about 11% of each crop is replicated border. That is inherent to
aligning an already-cropped source and is visible in the contact sheets; it is not a bug.

### PF-10 · Face crops are ~7.5 MB/video — they must never be materialised in bulk

**Finding, recorded 2026-08-29.** Measured over smoke-100: **788,763,014 B for 100 videos = 7.52 MB
each**, at 112×112×3 uint8 over a mean ~200 frames.

| Subset | Crops on disk |
|---|---|
| smoke-100 | 0.75 GB |
| dev-2k | **14.7 GB** |
| dev-10k | **73.5 GB** |
| full (136,304) | **~1.0 TB** |

**Why this is not a crisis but must be designed around.** Section 3.9's "feature size" estimates
(25 MB smoke, 2.5 GB dev-10k) are for the *frozen backbone features*, not for these intermediate
crops, and X-6 already says to delete crops once features are extracted. The magnitude was simply
never quantified.

**Consequences accepted:**

1. **Phase 4 must extract features and delete crops per video**, never crop-all-then-feature-all.
   dev-10k's 73.5 GB does not fit D:'s free space comfortably and cannot fit Kaggle's 20 GB
   `/kaggle/working` at all.
2. **CL-8's sharding applies to crops as well as features.**
3. Only the smoke-100 crops are kept on disk (0.75 GB), per X-6.

**Extraction wall-clock (P2-13):** 1.49 s/video single-machine at `--workers 2`. Extrapolated:
dev-2k ≈ 50 min, dev-10k ≈ **4.1 h**, full ≈ **56 h** — which exceeds Kaggle's ~30 h/week GPU
quota in one pass and confirms CL-8's multi-session sharding is required, not optional. Detection
is CPU-bound here, so this figure is not improved by the GPU.

---

## Phase 3 decisions — resolved 2026-08-29

### PF-11 · The waveform is fitted to `video_frames × hop` before the STFT

**Decision: explicit pad/trim, so P3-4 holds by construction.** ✅ Resolved 2026-08-29.
Implements the fix PROJECT_PLAN Phase 3 asks for ("handle `center=True` padding here, not
downstream") — but neither option the plan offers is sufficient on its own.

**What was measured.** P3-4/P3-9 demand `audio_frames == video_frames ± 1` on **100%** of files,
"no rounding fudge". Over all 136,304 metadata entries, standard STFT framing gives:

| framing | frames within ±1 of `video_frames` |
|---|---|
| `center=True` (`1 + A//hop`) | 119,823 / 136,304 — **87.9%** |
| `center=False` (`1 + (A−n_fft)//hop`) | 86,440 / 136,304 — **63.4%** |

Neither passes. The residue is **not** a rounding artefact: the decoded track genuinely is not
`video_frames × 640` samples long. Measured on real files it drifts from exactly 0 up to about
+2.4 frames, and `audio_frames/640 − video_frames` spans −0.2 to −0.8 in metadata.

**Resolution.** `fit_to_video()` pads with zeros or trims the decoded waveform to exactly
`video_frames × hop_length` samples *before* the STFT, then `center=True` yields `T+1` frames of
which the trailing one is dropped. The frame count is then correct **by construction on 100% of
files**, not by luck. Measured over smoke-100 the adjustment is mean −1.32 frames, range
[−3.40, 0.00] — under 0.14 s, always at the tail.

**Why the size of the adjustment is reported.** Padding a waveform is a small fiction, so
`AudioResult.pad_frames` records it per file and the extraction script prints the distribution.
A silent fudge is exactly what the gate exists to prevent; a measured, bounded one is fine.

**Verified independently.** `scripts/11_verify_labels.py --modality audio` compares log-mel
between an `audio_only` fake and its original: divergence onset matches the labelled span start on
**5/5 pairs, median error 0.000 s**. Since the onset is counted in audio frames and compared
against a labelled time in seconds, that confirms the grid as well as the labels.

### PF-12 · Video and audio preprocessing configs are separate, hashed separately

**Decision: split `PreprocessConfig` into `VideoPreprocessConfig` + `AudioPreprocessConfig`.**
✅ Resolved 2026-08-29.

**Why.** Section 3.8's content-hashed cache invalidates on any config change. With one combined
config, retuning `n_mels` would invalidate **7.5 MB/video of face crops** (PF-10) — 73.5 GB at
dev-10k — to rebuild a 64 KB/video log-mel. Experiment K (MFCC vs log-mel) would have triggered
exactly that. Two configs, two cache roots, so each side invalidates only itself.

`AudioPreprocessConfig` also enforces the section C grid rather than trusting it: a validator
rejects any `hop_length` for which `sample_rate / hop_length != 25`, naming the correct value in
the error. The conventional 10 ms hop is the obvious thing for someone to reach for later, and it
would silently misalign every localization target.

**Default is log-mel, not MFCC** — the plan's C-1 recommendation. This does **not** resolve C-1:
that is settled by Experiment K in Phase 5. The MFCC arm is built and produces an identical time
axis (`--feature mfcc`), so the experiment is a one-flag change.

**Measured cost of the audio side:** 0.094 s/video, **64 KB/video** — 117× smaller than the face
crops, so PF-10's streaming constraint does not apply here. dev-10k log-mels would be ~0.6 GB.

---

## Appendix B — architectural decisions

| ID | Decision | Options | Resolved by | Date | Outcome |
|---|---|---|---|---|---|
| **PF-1** | Hardware strategy | Local-only / Local+Kaggle / defer | User, PRE-2 | 2026-08-12 | ✅ **Local + Kaggle free tier** |
| **PF-2** | GAN tier | Tier 1 / Tier 1 minus GAN / Tier 2 | User, PRE-3 | 2026-08-12 | ✅ **Tier 1, three-arm F0/F1/F2** |
| **PF-3** | D-1 & C-1 method | By experiment / by prior | User, PRE-3 | 2026-08-12 | ✅ **By experiment (J and K)** |
| **PF-4** | Numeric precision | AMP fp16 / fp32 | Measurement, Phase 0 | 2026-08-12 | ✅ **fp32 local, AMP fp16 cloud** — overturns §12.3 |
| **PF-5** | Jupyter in dev deps | full `jupyter` / `ipykernel` only | Phase 0 install failure | 2026-08-12 | ✅ **`ipykernel` + `nbformat` only** (MAX_PATH) |
| **PF-6** | Raw dataset location | download 25.6 GB to D: / attach on Kaggle / HF streaming / Colab+Drive | User, Phase 1 | 2026-08-27 | ✅ **Attach on Kaggle; only the ~3.3 GB dev-10k feature cache comes local** |
| **D-1** | Visual backbone | ResNet-18 / MobileNetV2 | Experiment J, Phase 4 | ⬜ | ⬜ Prior: ResNet-18 @112² |
| **C-1** | Audio features | MFCC / log-Mel | Experiment K, Phase 5 | ⬜ | ⬜ Prior: log-Mel |
| **B-1** | Face detector | MediaPipe / MTCNN / RetinaFace | Phase 2 | ⬜ | ⬜ Recommended: MediaPipe (CPU-only, keeps VRAM free, gives landmarks + mouth ROI in one pass) |
| **F-1** | Sync approach | Learned InfoNCE / pretrained SyncNet | Phase 8 | ⬜ | ⬜ Recommended: learned |
| **G-1** | Fusion | Concat / gated / cross-attention | Experiments C vs E | ⬜ | ⬜ |
| **H-1** | Temporal | BiLSTM / Transformer | Experiments D vs E | ⬜ | ⬜ |
| **I-1** | GAN tier detail | Feature-space / face-region / full synthesis | Phase 9 | 2026-08-12 | ✅ Feature-space (see PF-2) |
| **L-1** | Localization | Dense per-frame / sliding window | Phase 10 | ⬜ | ⬜ Recommended: dense |

---

## Data-derived decisions still pending measurement

| Parameter | Currently | Decided by | Task |
|---|---|---|---|
| Training crop `T` | provisionally 250 frames (10 s) | measured forged-span duration distribution | P1-11 |
| Focal `pos_weight` / class weighting | provisionally α=0.25, γ=2.0 | measured fake-frame fraction | P1-12 |
| Post-processing τ, median kernel, min-duration, merge-gap | plan defaults | dev grid search maximizing AP@0.5 | P10-10 |
| Loss weights λ_cls, λ_loc, λ_sync, λ_bnd | (1.0, 2.0, 0.3, 0.5) | dev tuning | P10-5 |
