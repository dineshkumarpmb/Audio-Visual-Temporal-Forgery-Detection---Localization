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

## Phase 4 decisions — resolved 2026-08-29

### PF-13 · Kaggle's single-file download endpoint has a hard quota — scale requires the mount

> ⚠️ **SUPERSEDED 2026-08-30 by PF-16.** The conclusion below — a per-session *volume
> quota* of ~300 files making dev-2k unobtainable locally — is wrong. The endpoint is
> rate-limited: excess concurrency returns 404, sustained volume returns 429 with
> `Retry-After`. At `--workers 5` with the 429 handler, dev-2k downloads unattended.
> The reasoning is kept here because Phase 4's first-pass numbers were produced under it.

**Finding, recorded 2026-08-29.** Phase 4 needs dev-2k (2,000 videos). It could not be
downloaded, and the reason is a hard external limit rather than anything fixable in code.

**What happened.** `scripts/05_fetch_metadata.py --subset dev-2k` began returning **HTTP 404**
for every file after roughly 300 successful downloads in a session. The 404 is misleading: the
files exist, they are in the mirror's own listing, and the *same URLs* succeed again after a
cooling-off period. Verified three ways:

- files that had downloaded successfully an hour earlier — including `README.md` and
  `metadata.min.json` — began 404ing too, so it is not per-file;
- the block cleared on its own after roughly an hour, then re-triggered after ~16 more files;
- the metadata API (`dataset_list`) kept working throughout, so the account is not blocked —
  only the file-download endpoint is throttled.

So Kaggle rate-limits this endpoint and signals it as 404 rather than 429. Concurrency made it
arrive sooner (8 workers) but 3 workers hit it too: it is a **volume quota**, not a concurrency
limit. Roughly 300 files per window.

**Consequences accepted:**

1. **Phase 4 ran at the scale actually obtainable: 262 clips** (train 58 / dev 185 / test 19)
   rather than dev-2k's 2,000. Every number in `reports/phase4_*.json` and `decision_d1.md`
   carries that caveat. The machinery is scale-independent and re-running is one command.
2. **This is what CL-2/CL-3 are for, and PF-6 was righter than it knew.** On Kaggle the dataset
   is *mounted* read-only at `/kaggle/input` — no downloads, no quota, no transfer. PF-6 chose
   that for disk reasons; it turns out to be the *only* way to reach dev-2k and above at all.
   Downloading was always a local convenience for Phases 1-3, where a 100-video smoke subset
   sufficed.
3. **The download path is now explicitly capped.** `--subset` fetching stays for smoke-scale work
   and the manual checks; anything larger goes through the Kaggle notebook. A comment in
   `05_fetch_metadata.py` says so, so nobody rediscovers this the slow way.
4. **The download order matters and was unlucky.** Ids are fetched sorted, and `dev_*` sorts
   before `test_*` and `train_*`, so the quota was spent almost entirely on dev clips. That is
   why the local train split is 58 and the dev split 185. Not a design choice — an artefact worth
   naming so the odd split sizes in Phase 4's report are not mistaken for stratification.

**What was *not* done:** nothing was re-split to compensate. Section 3.5 RULE 1 forbids
re-deriving splits, and a train set that borrowed from dev would make every number here
meaningless in exactly the way R3 warns about.

---

### PF-14 · MLflow tracking uses SQLite, not the bare file store

**Decision: `sqlite:///experiments/mlflow.db`.** ✅ Resolved 2026-08-29. Implements P4-8's
"MLflow (local, file-backed)" against a library that no longer supports it as written.

**What broke.** `mlflow 3.15.1` **raises** on a `file:` tracking URI:

> The filesystem tracking backend (e.g., './mlruns') is in maintenance mode and will not
> receive further updates. Please migrate to a database backend.

It can be forced with `MLFLOW_ALLOW_FILE_STORE=true`, but opting out of a deprecation to keep
using a store the vendor has frozen is borrowing trouble for a project that runs for many more
phases.

**Resolution.** SQLite — the migration path MLflow itself recommends. It preserves the part of
P4-8's intent that matters: a **single local file, no server, nothing to host or configure**. The
run directory layout under `experiments/` is unchanged, and checkpoints are still logged as
artefacts.

**Consequence:** `experiments/mlflow.db` is gitignored along with the rest of `experiments/`;
`reports/phase4_*.json` remains the committed, human-readable record, so no result depends on
having MLflow installed to read it.

---

### PF-15 · Mean pooling dilutes the forgery signal ~45x — measured, not assumed

**Finding, recorded 2026-08-29.** PROJECT_PLAN section J deviates from the source report by
specifying **attention pooling instead of mean pooling**, reasoning that "mean-pooling a
250-frame sequence where 50 frames are fake dilutes the signal 5:1". Phase 4's cached features
let that be measured rather than argued, and the real dilution is far worse than 5:1.

**Method.** Every fake names its `original`, so the two clips differ *only* in the forged span.
Comparing a fake against its original in the frozen ResNet-18 embedding space:

| | cosine distance |
|---|---|
| Mean-pooled embeddings (fake vs its original) | **0.0045** |
| Peak per-frame embedding distance | **0.2026** |
| **Ratio** | **45x** |

For context, the distance between two *different* identities' mean embeddings is **0.193** —
essentially the same magnitude as the peak per-frame forgery signal, and **43x larger** than the
mean-pooled forgery signal.

**What this means.** In mean-pooled space the forgery is two orders of magnitude weaker than
identity. A mean-pooling classifier is therefore being asked to find a 0.0045 perturbation
against a 0.193 nuisance axis. Attention pooling can concentrate on the frames where the signal
is 45x stronger; mean pooling structurally cannot.

This also predicts the shape of the failure: mean pooling should not merely score lower, it
should latch onto *identity* rather than manipulation. `MeanPoolBaseline` exists as the ablation
arm to demonstrate exactly that, with matched parameter count so the comparison isolates the
pooling operator.

**Corroborates P1-11.** The mean forged span is 0.650 s = 16 frames against clips averaging
~200 frames — roughly 8% of the timeline. A ~1/12 dilution of a localised signal, compounded by
averaging over a high-variance embedding, lands in the right order of magnitude for the 45x
measured here.

---

### PF-16 · The download endpoint is rate-limited, not quota-capped — PF-13 was wrong

**Finding, recorded 2026-08-30. This supersedes PF-13's central claim and reopens the
local path to dev-2k.**

PF-13 concluded that Kaggle's single-file download endpoint enforces a per-session
**volume quota** of roughly 300 files, signalled as 404, that it clears after about an
hour, and that dev-2k was therefore *unobtainable locally*. That conclusion is wrong, and
it is the sole reason Phase 4 ran on 262 clips with 58 in train and left its gate open.

**What re-testing showed.** The endpoint is rate-limited in two distinct ways, and both
are workable from this machine.

**1. Excess concurrency returns 404 for files that plainly exist.**

| Workers | Result | Throughput |
|---|---|---|
| 1 | **60/60 ok** | 31 files/min |
| 3 | **45/45 ok** | 67 files/min |
| 5 | **45/45 ok** | 108 files/min |
| 8 | 64 ok, then a solid run of 404s | — |

The decisive observation: **seconds after the 8-worker run began 404ing, a serial probe
got HTTP 200 on the very next ids.** A volume quota cannot behave that way. PF-13 tested
8 workers and 3 workers, saw 404s in both, and read that as proof the cause was not
concurrency — but its 3-worker run had already spent the burst allowance described below.

**2. Sustained volume returns 429 with `Retry-After`, not 404.**

After roughly 500 files in a burst every request returns:

    HTTP 429  {"code":429,"message":"TooManyRequests"}
    Retry-After: 180

`Retry-After` then counts down (180 → 130 → 5 s) as the window refills. This is a
scheduling instruction, not an error. PF-13 never saw it because it treated any failure
as the quota and stopped.

**Why PF-13 saw only 404s.** Its fetcher called `raise_for_status()` and counted every
exception toward one "quota hit" verdict, so a 404 from over-concurrency and a 429 from
volume were indistinguishable in its logs — and it stopped at the first run of either.

**Fix, in `scripts/05_fetch_metadata.py`:**

1. `fetch()` honours **429 + `Retry-After`** through a process-wide `THROTTLE`, so all
   workers back off together rather than each earning their own 429.
2. `--workers` defaults to **5**, the measured ceiling before 404s appear.
3. Only a *persistent* 404 — one that survives 3 retries — counts toward stopping. An
   earlier cut counted transient connection errors and halted a healthy run at 66 files.
4. Subset ids are fetched **train split first** (`SPLIT_PRIORITY`). PF-13 note 4 observed
   that ids sort `dev_* < test_* < train_*`, so a truncated run starves exactly the split
   that limits learning. This reorders arrival only — split membership still comes from
   the committed subset, so §3.5 RULE 1 is untouched.

**What PF-13 got right and keeps.** PF-6 still holds: the *full* 25.5 GB corpus stays on
Kaggle, and CL-2/CL-3 remain the right route for dev-10k and above. What changes is that
**subset-scale work no longer blocks on the Kaggle mount** — dev-2k is ~460 MB and comes
down unattended.

**Consequence for Phase 4.** The gate's failing condition was "both backbones clearly beat
majority-class", and the stated cause was 58 training clips. That constraint was an
artefact of the fetcher, not of the environment. Phase 4 was re-run with **train at 592
clips** (698 of dev-2k fetched before the run was stopped by choice, plus smoke-100) on an
**unchanged dev split**, and both arms then cleared the floor decisively -- see PF-17,
which also records what that number does *not* mean. Completing the remaining ~1,300 clips
needs only wall-clock, not a new capability.

---

### PF-17 · Phase 4's dev AUC contains a clip-length shortcut — the gate passes, the number does not mean what it looks like

**Finding, recorded 2026-08-30**, from `scripts/18_check_confound.py` /
`reports/confound_check.md`. Raised by P4-12's own re-run, not by a later phase.

**The gate's stated conditions are met.** With train at 592 clips instead of 58, on an
**unchanged 175-clip dev split**:

| Arm | 58-clip run | 592-clip run | vs majority floor |
|---|---|---|---|
| ResNet-18 + attention | 0.5667 ± 0.0121 | **0.6319 ± 0.0110** | **+0.132** |
| MobileNetV2 + attention | 0.5746 ± 0.0156 | **0.7189 ± 0.0086** | **+0.219** |
| ResNet-18 + mean-pool | 0.5764 ± 0.0131 | 0.6140 ± 0.0072 | +0.114 |

Both arms clear 0.5 by many seed-sd. That is a clear beat by any reading.

**But two results say it is not all forgery evidence.**

**1. `audio_only` fakes score 0.7551 — the *highest* of the three classes.** Phase 3
established that `audio_only` forgeries are **pixel-identical to their originals**; the
audio arm exists precisely because they are invisible to the visual method. A visual-only
model has nothing to detect there and should sit at chance. It is **+0.2551 above chance**
on clips it cannot see the manipulation in.

**2. Real clips are systematically shorter than fakes**, so clip length alone is a
classifier on this dev split:

| class | median frames | AUC from length alone | model |
|---|---|---|---|
| real | 171 | — | — |
| visual_only | 225 | 0.5916 | 0.7474 |
| audio_only | 226 | 0.6381 | 0.7551 |
| both | 207 | 0.6132 | 0.6580 |

Length alone reaches **AUC 0.6157** overall — most of the way to what the models score.

**This is R4 (shortcut learning) / §5.6's modality-collapse warning**, which X-7 requires
re-checking at every scale-up. The scale-up is exactly what surfaced it: at 58 training
clips `audio_only` sat at 0.52–0.56, i.e. at chance, because the model could not yet
exploit the shortcut. More data made the model *better at the artefact*.

**What is clean, and worth stating.**

- **R3 (identity leakage): clean.** 0 `source_id` values span train and dev.
- **The comparison is clean.** The dev split is unchanged between the two runs; only
  train grew. So the +0.065 / +0.144 improvements are real effects of training volume.
- **Every Phase 4 correctness diagnostic still passes**: overfit-a-batch 0.65 → 1e-6 on
  all three arms, determinism 4/4 re-extraction on both backbones, identity separation
  0.0015 within vs 0.205 between.

**Consequences accepted:**

1. **P4-12 is marked passed on its stated criteria, with this caveat attached to the
   number.** Marking it failed would be inventing a condition the gate does not contain;
   reporting the AUC bare would be worse. It travels with the caveat.
2. **The bare AUC must not enter Part 11's evidence table.** It is an upper bound
   containing a dataset artefact, not forgery-detection skill.
3. **Phase 5 is the control.** `audio_only` is precisely what the audio arm should catch
   and vision should not. If the audio model does not clearly beat 0.7551 on
   `audio_only`, the shortcut is doing the work in both arms and fusion will inherit it.
4. **Fix before Phase 6.** A length-matched dev evaluation, or a length-normalised view,
   should be settled before fusion — a shortcut this strong is otherwise inherited by
   every later phase, and Phase 10's localization cannot use it at all, since a per-frame
   task has no clip-length feature to lean on.

**Not done, deliberately:** the dev split was not re-drawn to balance length. §3.5 RULE 1
forbids re-deriving splits, and a length-matched *subset* of dev is an evaluation view,
not a new split — that is the form the fix should take.

---

## Phase 5 decisions — resolved 2026-08-30

### PF-18 · The audio encoder's padding must equal its dilation — section E's literal spec shrinks the sequence

**Finding, recorded 2026-08-30.** Section E specifies the encoder as
`4 × [Conv1d(k=3, pad=1) → BN → ReLU]` with dilations `[1, 2, 4, 8]`. Those two clauses are
inconsistent. For `k=3` a Conv1d outputs `L + 2·padding − 2·dilation`, so `pad=1` preserves
length **only at dilation 1**:

| layer | dilation | pad=1 output | pad=dilation output |
|---|---|---|---|
| 1 | 1 | `L` | `L` |
| 2 | 2 | `L − 2` | `L` |
| 3 | 4 | `L − 6` | `L` |
| 4 | 8 | `L − 14` | `L` |

Taken literally the stack loses **22 frames = 0.88 s**. Implemented as `padding=dilation`.

**Why this matters more than a shape bug.** It would not crash. It would silently break the
frame-for-frame correspondence Phase 3's alignment gate spent its whole existence
establishing — `hop_length=640` makes audio frame *t* equal video frame *t* by array index —
and Phase 10's per-frame targets would then be offset by 22 frames against the features they
label. The first symptom would be an inexplicable localization score, many phases later.

**P5-2 is exactly the check that catches it**, which is presumably why the plan asks for the
assertion in the same breath as the architecture. The assertion is in `forward()` and in
`tests/unit/test_audio_encoder.py`, including a test that reads the built layers so the
specific mistake cannot reappear.

**Two related masking decisions, same module:**

1. **Padded positions are re-zeroed after every block.** Dilation reaches 8 frames, so a
   padded neighbour would otherwise bleed into the last valid frames — fabricated evidence at
   the clip boundary, the audio analogue of what `face_found` masking prevents on the visual
   side. A test asserts a clip's embeddings are unchanged by what shares its batch.
2. **BatchNorm statistics still see padding.** Keeping BatchNorm is report-faithful, but its
   statistics shift with a batch's length mix. `--norm group` exists as the ablation arm that
   removes the effect. Not resolved by measurement yet; recorded so it is not forgotten.

---

### PF-19 · C-1 resolved → log-Mel — and the audio arm reads a dataset artefact, not only forgery

**Decision: `logmel`.** ✅ Resolved 2026-08-30 by Experiment K, per PF-3.

| | MFCC (B2a) | log-Mel (B2b) |
|---|---|---|
| dev AUC | 0.7677 ± 0.0488 | **0.9838 ± 0.0018** |
| dev accuracy | 0.5562 | 0.9067 |
| `audio_only` | 0.8683 | 0.9912 |
| `visual_only` | **0.5365** | **0.9728** |

The gap is **+0.2161** with disjoint seed intervals — 21× the 0.01 noise threshold — so the
pre-committed rule takes the higher arm and section C's stated prior is *confirmed by
measurement* rather than merely standing unrebutted. MFCC's seed spread (± 0.0488) is 27×
log-Mel's, which is its own argument: the DCT throws away most of the usable signal and what
is left is unstable.

**⭐ The PF-17 control passes.** Phase 4's visual arm scored 0.7551 on `audio_only` forgeries
it is pixel-blind to. Audio scores **0.9912** on exactly that class — the modality that can
genuinely see them is far ahead, which is the evidence that the audio arm reads acoustic
content rather than Phase 4's clip-length shortcut.

**⛔ But the mirror-image control fails, and it is a bigger problem.** `visual_only` fakes
have, by the dataset's own labels, **unmodified audio**. An audio model must be at chance.

- MFCC is: **0.5365** — the physically correct answer, and *below* the 0.5916 that clip
  length alone achieves.
- log-Mel is **0.9728**.

Clip length cannot explain it: both arms see identical lengths and MFCC does not exploit it.

**Measured, not inferred** (`scripts/21_check_audio_provenance.py`,
`reports/audio_provenance.json`). Comparing 8 `visual_only` fakes against the real video each
one names in `original`, using Phase 2's divergence-onset method:

- **8/8 diverge from t = 0**, median identical prefix **0 samples**;
- the labelled edits sit at a median of **3.85 s**.

So the audio stream of a `visual_only` fake is not a faithful copy of its original *anywhere*,
long before any manipulation. The test does not distinguish *why* — re-encode, resample, or a
global offset — only that the difference is **global rather than local**. That is enough: a
global, perfectly label-correlated difference exists in the audio of clips whose audio content
was never manipulated, and log-Mel preserves exactly the fine spectral structure needed to read
it while MFCC's DCT discards it.

**Consequences accepted:**

1. **C-1 stands as `logmel`.** It is the better arm on the task as measured, and the decision
   rule was fixed before the numbers. But *part of why* it wins is that it is better at reading
   the artefact — so the choice is right and the margin is not trustworthy.
2. **Phase 5's headline 0.9838 must never travel bare.** It is an upper bound containing a
   dataset-construction artefact. Same treatment as PF-17's number.
3. **⛔ Section 5.6's modality-collapse risk is now concrete.** Audio at 0.9838 against vision
   at 0.7189 gives a fusion model every incentive to ignore video entirely — the failure the
   plan warns "quietly defeats the project's premise". Phase 6 needs the modality-dropout
   diagnostic from its first run, not as a later ablation.
4. **Phase 10 is the natural corrective, and should be brought forward in importance.** A
   per-frame task cannot use clip length, and a *global* processing fingerprint is spread over
   the whole clip rather than concentrated in the forged span. Localization AP is therefore the
   metric this project can actually defend.

**Limits of this finding.** 8 pairs, all from dev, all `visual_only`. It establishes the
artefact exists and is global; it does not quantify how much of the 0.9838 it accounts for.
The clean way to measure that is a re-encode control — pass real clips through the same
mux/encode path and re-score — which needs the authors' pipeline and is out of scope here.
Recorded as the open question it is.

---

## Phase 6 decisions — resolved 2026-08-30

### PF-20 · Experiment C fails its prediction, the model collapses onto audio, and the collapse defences backfired

**Finding, recorded 2026-08-30.** All three results are negative, all three are measured,
and P6-8 passes on exactly that basis — it asks for the premise to be supported *or the
failure understood and documented*.

**1. Fusion does not beat its best single modality.**

| arm | dev AUC |
|---|---|
| A visual only (MobileNetV2) | 0.7189 ± 0.0086 |
| **B audio only (log-Mel)** | **0.9838 ± 0.0018** |
| C fusion, concat + defences | 0.9038 ± 0.0275 |
| C control, defences off | 0.9098 ± 0.0249 |

**C − max(A, B) = −0.0800.** The experiment matrix predicted `C > max(A,B)`; fusion is
*worse* than audio alone, and by more than its own seed spread. Adding a weak, partly
artefactual visual stream to a very strong audio one costs accuracy rather than adding to it.

**2. The fused model has collapsed onto audio (section 5.6).**

| stream zeroed at inference | dev AUC | drop |
|---|---|---|
| none | 0.9038 | — |
| **visual** | 0.9017 | **+0.0021 ± 0.0066** |
| audio | 0.6568 | +0.2471 ± 0.0239 |

Removing video costs nothing distinguishable from zero — one seed's AUC actually *improved*
without it. Removing audio costs 0.25. This is precisely the failure section 5.6 describes:
"a decent aggregate number and a model whose multimodal claim is hollow".

**3. ⚠️ The collapse defences made reliance on video *worse*, not better.**

| | drop when visual removed | uses video? |
|---|---|---|
| defences on (dropout 0.2, aux 0.3) | +0.0021 ± 0.0066 | **no** |
| **control, defences off** | **+0.0128 ± 0.0042** | **yes** |

The control's drop is positive on **all three seeds** and clears 2 sd; the defended arm's
straddles zero. So this is a real effect, not the arbitrary 0.01 threshold deciding it —
which is why the verdict now tests against seed spread rather than a fixed line.

Section 5.6 calls modality dropout "the single most effective intervention". On this data it
was counter-productive, and the plausible mechanism is specific rather than mysterious:
dropout zeroes the visual stream on 20% of clips, and since that stream is *already* weak
(0.7189, much of it PF-17's clip-length shortcut), the model learns it is unreliable and
downweights it further. An auxiliary head on a weak-and-artefactual modality pushes the same
way. The intervention assumes both modalities carry comparable signal; here one does not.

**Consequences accepted:**

1. **Baseline 3 is recorded as a negative result, not tuned until it wins.** Tuning against
   dev until fusion beat audio would be fitting the dev split, which §3.5 RULE 4 exists to
   prevent.
2. **This is not yet a verdict on multimodal fusion — only on *early concat* at this scale.**
   Section G's recommended bidirectional cross-attention is Phase 8, and it models "does
   audio at *t* explain video at *t*" rather than concatenating two opinions. It has a
   mechanism for using a weak visual stream that concat lacks. The comparison is now set up.
3. **Do not carry C's 0.9038 forward as the headline.** Audio alone is better and simpler,
   and both numbers sit on top of PF-17/PF-19's artefacts.
4. **The real fix is upstream, not architectural.** The visual arm is weak partly because it
   is reading clip length (PF-17). A length-matched evaluation and a stronger visual signal
   would change this experiment's premise; more fusion capacity will not.

**Limits.** 592 training clips, one fusion architecture, three seeds. This says early concat
fails here; it does not establish that no fusion can help. Phase 8 is the test that matters,
and it now has a documented baseline to beat.

---

### PF-27 · Default-mode training is not bit-reproducible; strict mode is (P11-10)

**Finding, recorded 2026-10-03**, from `scripts/39_check_reproducibility.py`.

Re-training Experiment G's seed 0 from its exact logged config (`train_config` identical)
did **not** reproduce it: dev AUC 0.9843 → 0.9783, frame AP 0.9253 → 0.9278, best epoch
12 → 23. The runs part at the first epoch (train loss 4.98351 vs 4.98380).
`seed_everything` pinned every RNG and made cuDNN deterministic, but never called
`torch.use_deterministic_algorithms(True)`, so other CUDA kernels (attention / scatter
backward, cuBLAS) summed in a run-dependent order and the difference compounded.

**Fix:** `seed_everything(strict=True)` / `scripts/34 --strict-determinism` adds
`torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Every op
in the section 5.1 model has a deterministic kernel (no error raised), at roughly 5-30%
more time per epoch. **Two strict runs are bit-identical**: every dev metric |Δ| = 0,
max per-frame score |Δ| = 0, same best epoch (`reports/reproducibility_strict.json`).

**Consequences accepted:** strict mode is opt-in, so every Phase 4-11 run keeps the code
path it was trained under; those runs are reproducible to about one seed std
(`reports/reproducibility.json`), not to 1e-4. The 3-seed, paired-by-seed protocol
already absorbs noise of that size -- no section 7.3 verdict rests on a smaller
difference. New training should pass `--strict-determinism`.

---

### PF-28 · Phase 11's ablations: neither the sync loss nor modality dropout earns its place at AP@0.5

**Finding, recorded 2026-10-03**, from `reports/evidence_table.md` (dev, 3 seeds, paired).

**H, the sync loss (G vs G with λ_sync = 0):** AP@0.5 +0.0023, spread 0.0287 -- **not
significant**. But on boundary quality it **hurts significantly**: AP@0.75 is 0.0962
lower *with* the sync loss (all 3 seeds, spread 0.0384), AP@0.95 0.038 vs 0.299, boundary
error 78 vs 56 ms. Phase 8 credited E's +0.0045 clip-AUC gain to the sync loss; for
localization it is a cost, not a gain.

**I, modality dropout (G vs G with p = 0):** AP@0.5 +0.0175, spread 0.0187 -- **not
significant** (seeds +0.039, -0.018, +0.031). Without it the model is the only arm in the
project whose localization crosses PF-21's 0.01 threshold: zeroing video costs
**0.060 AP@0.5** (0.022-0.105 per seed) against -0.000 for G. But the video it uses does not
help: visual-only fakes localize *worse* (AP@0.5 0.574 vs 0.682) and 10.2% of real clips
get a false segment (G: 0.0%). Modality dropout does not hold the model off video so much
as stop it leaning on video in an unhelpful way. PF-21's conclusion stands.

**Test (P11-8, the single run, 392 clips):** G scores AUC 0.9842, AP@0.5 0.8005, AP@0.75
0.5392 -- 0.05 / 0.09 below dev, the optimism expected from tuning on dev. Zeroing video
costs AUC 0.0016 and *raises* AP@0.5 by 0.0096: PF-21 holds on held-out data.

---

### PF-29 · Phase 12: no ONNX, no fp16 — the time is in face detection, not the model

**Decision, recorded 2026-10-04**, from `reports/benchmark.md` (`scripts/41_benchmark.py`,
20 dev clips, Chrome closed, burn-in pass first) and `reports/phase12_gate.md`.

**ONNX (P12-2) dropped, recommended and accepted before any Phase 12 code was written.** The
plan lists it as a speed lever; measured, it cannot be one here. The two GPU stages together
are ~8% of end-to-end time (MobileNetV2 ~0.31 s + Stage-B ~0.05 s per 10 s clip); decode +
MediaPipe landmarks are ~86%, and MediaPipe is already its own CPU runtime. The size gate is met
in plain PyTorch, and Phases 13–14 run locally in the same Python environment, so ONNX would
only add a second copy of the model to keep in parity. P12-2 is instead a **weights-only
bundle** (`scripts/40_export_model.py` → `models/avtfd_g_seed0.pt`, 31.1 MB: Stage-B +
backbone + frozen post-processing; `best.pt` is 66 MB only because it carries Adam's moments).

**Which seed ships:** the median dev AP@0.5 of G's three (seed0, 0.8634 between 0.8258 and
0.8674) — fixed as a rule before any Phase 12 number existed, so the demo neither flatters nor
undersells the results table.

**Three clean benchmark runs, same code, same clips** (s per 10 s clip, median / p90):

| variant | run 1 | run 2 | run 3 (committed) | verdict |
|---|---|---|---|---|
| `two_pass` (decode twice, as Phase 2–4) | 5.95 / 6.21 | 6.61 / 8.90 | 8.88 / 9.59 | — |
| **`optimized`** (decode once, fp32) | **3.65 / 3.88** | **5.03 / 5.30** | **4.45 / 5.26** | **shipped** |
| backbone fp16 | 3.76 / 4.00 | 5.75 / 6.52 | 5.75 / 6.16 | slower, drift 2.0e-4 |
| model fp16 (autocast) | 3.67 / 3.94 | 5.66 / 6.31 | 5.59 / 6.06 | slower, drift 5.3e-4 |

Absolute times move ~40% between runs on this laptop (2.1 GHz cap, thermals after a 12-minute
gate run), almost entirely in the CPU detection stage. **The orderings never move:** decoding
once wins every run, fp16 loses every run (confirming PF-4 on the real pipeline), and the worst
p90 of the shipped variant (5.30 s) clears the 10 s target by ~1.9x.

**fp16 is rejected** on both counts: slower, and not numerically identical (the adoption rule
admits a variant only with drift <= 1e-5, so P12-5's "accuracy unchanged" holds by
construction).

**What did help: decoding once.** Pass 1's frames are kept when they fit a 256 MB budget (a 10 s
LAV-DF clip is 37 MB) and the file is re-decoded only above it — bit-identical output
(unit-tested). The other P12-3 item, the model held in memory, is `VideoAnalyzer` + `warmup()`:
the first request after construction measured ~40 s (CUDA/cuDNN init + librosa/numba JIT) and
now costs ~0.3 s at startup instead (construction ~6 s).

**Accuracy (P12-5), all 175 dev clips end to end from raw `.mp4`:** max |Δ frame score| 5.4e-7
against the cached-feature predictions; AUC 0.9843, frame AP 0.9253, AP@0.5 0.8634, AP@0.75
0.5826 — all unchanged; segment count identical on every clip.

**Memory for Phase 13:** a fresh process holds 1,057 MB loaded + warm, 1,083 MB after the
longest clip (peak working set 1,182 MB); peak VRAM 242 MB of 4 GB.

**The lever left is `detect_every_n`** (landmarks every 5th frame). Raising it would cut the
dominant stage, but it changes the preprocessing the model was trained on, so it needs
re-extraction and re-validation — out of Phase 12's scope and not needed.

**Benchmark hygiene, learned the hard way:** with Chrome open the same work measured up to 2x
slower, and run order alone moved a median by 2x (cold disk + first-use process costs). The
benchmark therefore pre-reads every clip, runs an unrecorded burn-in pass, records ambient CPU /
RAM / Chrome with every run, measures memory in a fresh process, and gates on the p90.

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
| **D-1** | Visual backbone | ResNet-18 / MobileNetV2 | Experiment J, Phase 4 | 2026-08-30 | ✅ **MobileNetV2** — dev AUC 0.7189 vs 0.6319, a 0.087 gap far outside seed noise, so §5.2's speed tie-break never applies. Reverses the 2026-08-29 call made at 58 clips |
| **C-1** | Audio features | MFCC / log-Mel | Experiment K, Phase 5 | 2026-08-30 | ✅ **log-Mel** — dev AUC 0.9838 vs 0.7677, a 0.2161 gap with disjoint seed intervals. ⛔ Read with PF-19: part of the margin is a dataset artefact |
| **B-1** | Face detector | MediaPipe / MTCNN / RetinaFace | Phase 2 | ⬜ | ⬜ Recommended: MediaPipe (CPU-only, keeps VRAM free, gives landmarks + mouth ROI in one pass) |
| **F-1** | Sync approach | Learned InfoNCE / pretrained SyncNet | Phase 8, Exp H | 2026-10-03 | ✅ **Learned InfoNCE** built; Exp H shows it does not help AP@0.5 (not significant) and costs 0.096 AP@0.75 (PF-28) |
| **G-1** | Fusion | Concat / gated / cross-attention | Phase 8 decomposition (concat+Transformer vs cross+Transformer), Phase 11 | 2026-10-03 | ✅ **Cross-attention kept, but a tie**: frame AP +0.0387, spread 0.0561 — not significant |
| **H-1** | Temporal | BiLSTM / Transformer | Phase 8 decomposition (cross+BiLSTM vs cross+Transformer), Phase 11 | 2026-10-03 | ✅ **Transformer kept, but a tie**: frame AP +0.0153, spread 0.0196 — not significant |
| **I-1** | GAN tier detail | Feature-space / face-region / full synthesis | Phase 9 | 2026-08-12 | ✅ Feature-space (see PF-2) |
| **L-1** | Localization | Dense per-frame / sliding window | Phase 10 | 2026-10-02 | ✅ **Dense per-frame** + section 6.4 post-processing, frozen on dev |

---

## Data-derived decisions still pending measurement

| Parameter | Currently | Decided by | Task |
|---|---|---|---|
| Training crop `T` | provisionally 250 frames (10 s) | measured forged-span duration distribution | P1-11 |
| Focal `pos_weight` / class weighting | provisionally α=0.25, γ=2.0 | measured fake-frame fraction | P1-12 |
| Post-processing τ, median kernel, min-duration, merge-gap | plan defaults | dev grid search maximizing AP@0.5 | P10-10 |
| Loss weights λ_cls, λ_loc, λ_sync, λ_bnd | (1.0, 2.0, 0.3, 0.5) | dev tuning | P10-5 |
