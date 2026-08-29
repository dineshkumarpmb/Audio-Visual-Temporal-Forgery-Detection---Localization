# Implementation Task List — Audio-Visual Temporal Forgery Detection & Localization

Derived from `PROJECT_PLAN.md` (Parts 0–15, Phases 0–16).
IDs are stable — use them when referring to work.

Legend: ⛔ = blocking gate item · ⭐ = headline capability · 🔬 = experiment · ✋ = manual human check

---

## 📍 CURRENT STATUS — 2026-08-29, Phase 3 done (9/9) · next is Phase 4

| | |
|---|---|
| **Done** | Phase −1 (pre-flight) · Phase 0 (environment) — **gate green, 33/33** |
| **Done** | Phase −1 · Phase 0 · **Phase 1 (18/20)** · **Phase 2 (13/15)** · **Phase 3 (9/9)** · CL-1 ✅ · CL-7 ✅ |
| **Next** | **Phase 4 — Visual Baseline** (12 tasks) 🔬 Experiment A + Decision D-1. First phase with a model in it |
| **Progress** | ~61 of 203 tasks (≈30%) · 5 of 18 phases complete |
| **Tests** | **190 passing** — 174 unit + 16 integration against the real 136,304-entry dataset |
| **✋ Awaiting you** | P2-8 (contact sheets) · ⛔ P2-10 (watch 20 overlays) — artefacts rendered in `reports/figures/` |
| **Kaggle** | `dinesh1234567` · phone verified · **30 GPU h/week** · P100 16 GB or T4×2 · 12 h/session · 20 GB `/kaggle/working` |
| **Branch** | `master`, working tree clean |
| **Commits** | `ab2b620` baseline · `c53922b` decisions PF-1/2/3 · `ebf3196` Phase 0 · `badb58b` PF-6 · `3c0d0dc` CL-1 gate · `c872158` CL-1 credentials + API-drift fix |

**Phases 1–3 built the data and preprocessing layers.** `src/` holds `config.py`, `seed.py`,
`utils/console.py`, `data/{metadata,manifest,leakage,validate,subset}.py` and
`preprocessing/{video,face,align,audio,cache,pipeline}.py`, driven by `scripts/02`–`12`, with
**190 passing tests**. Both modalities are cached on one 40 ms grid and ready for a model.
`api/` and the model packages are still empty stubs — Phases 4–16 untouched.

### 🔴 The one thing still blocked

**PRE-1 — the original report is not on this machine.** It is the sole blocker for P1-3 (report
audit) and for Appendix-A items 1–7. Drop it at `docs/original_report.pdf`.

Impact, precisely: it blocks **Part 9's improvement comparison and Phase 16 (documentation)**.
It does **not** block Phases 2–15. If it never surfaces, the defensible move is to drop Part 9's
comparison rather than quote 53.35% / 80.00% without provenance — see
`reports/appendix_a_resolution.md`.

### Resume checklist

1. `make check` — Phase 0 gate, should print **GATE PASSED**, exit 0.
2. `make phase1` — metadata → CL-7 → manifest → subsets → statistics → tests.
3. `make phase2` — models → smoke-100 video → face crops → gate → label check → sheets → overlays.
   Pulls ~19 MB of video and 3.8 MB of weights; the 25.5 GB stays on Kaggle (PF-6).
4. `make phase3` — log-mel + MFCC on the video grid → ⛔ strict alignment gate → label check.
5. ✋ **Do P2-8 and P2-10**: open `reports/figures/contact_sheets/` and watch
   `reports/figures/overlays/`. These are the two things nothing automated can sign off.
6. Start **Phase 4** (visual baseline) at P4-1 — the first phase with a model in it.

### Mirror verified — CL-7 PASSED 10/10 (2026-08-29)

`reports/mirror_verification.md` · `reports/mirror_facts.json` · full listing in
`reports/mirror_listing.tsv.gz`.

| | |
|---|---|
| Ref | `elin75/localized-audio-visual-deepfake-dataset-lav-df` (public, v1) |
| Videos | **136,304** — mp4 set matches metadata **exactly in both directions** |
| Real / fake | **36,431 / 99,873** — matches the publication |
| Splits | train 78,703 · dev 31,501 · test 26,100 — all match |
| Size | **25,502,577,573 B = 25.50 GB** uncompressed |
| `metadata.min.json` | 33,837,990 B · sha256 `96d5f79bef1aa92f…` |
| Re-encode test | fps 25.00 **28/28** · `video_frames` byte-exact **28/28** · 16 kHz **28/28** |

Both CL-1 flags are now closed: the 24.84 GB `total_bytes` was Kaggle's *compressed* figure, and
`metadata.min.json` is present at the size the listing reports. Residual limit: it was not
byte-compared against the authors' HuggingFace copy, since PF-6 keeps that copy off this machine —
its sha256 is recorded so any future divergence is detectable.

### Risk status

**Phase 1 — all three resolved.** `fake_periods` are float **seconds** (asserted every load);
leakage assertions are **build-failing** with zero overlap on all three pairs; RAM was never a
factor (Phase 1 ran off a 33.8 MB JSON).

**Phase 2 — ground truth is now measured, not assumed.** §6.3 calls label alignment the
highest-risk step in the project. Comparing each visual fake against the real video it names in
`original`, divergence onset matches the labelled span start on **7/7 pairs, median error
0.000 s** (one frame = 0.04 s). That independently confirms the units, the timeline origin **and**
the `round(t×25)` rasterisation.

⚠️ **Note for anyone using `original` as a reference:** a fake is generally a *different length*
from its original (142 vs 136 frames, 331 vs 310) — LAV-DF replaces a word with a different word.
The clips desynchronise from the manipulation point onward, so only the **leading edge** of
divergence is informative; peak difference is not.

**Phase 3 — the alignment gate holds at 100%.** One audio frame is one video frame by
construction (`hop_length=640` = 40 ms = one frame at 25 fps), verified on 100/100 files. Label
verification now covers **all three fake classes, 12/12, median error 0.000 s** — the audio arm
reaches `audio_only` fakes, which are pixel-identical to their originals and so invisible to the
visual method.

⚠️ **RAM returns in Phase 4.** Crops are 7.52 MB/video (PF-10), so feature extraction must stream
and delete per video. Audio is only 64 KB/video and is not a constraint.

### Carried-forward open items

| Item | Blocks | Notes |
|---|---|---|
| 🔴 PRE-1 — original report missing | **P1-3, Part 9, Phase 16** | Not on this machine; drop at `docs/original_report.pdf`. Phases 2–15 do not need it. The only thing keeping the P1-20 gate at 3/4 |
| ⛔ PF-10 — crops are 7.52 MB/video | Phase 4 | dev-10k = 73.5 GB, full ≈ 1 TB. Phase 4 must featurise and delete **per video**; will not fit Kaggle's 20 GB `/kaggle/working` |
| ✋ P2-8 / ⛔ P2-10 manual checks | Phase 2 sign-off | Contact sheets and overlays rendered in `reports/figures/`. P2-10 is backed by 7/7 objective label verification but still needs a human watch |
| ✋ Attach the mirror in a Kaggle notebook | Phase 4+ on Kaggle | Manual browser step (*Add Data*). Phases 1–2 ran locally off metadata plus smoke-100 |
| P14-0 — Windows long paths disabled | Phase 14 | `LongPathsEnabled = 0`. Needs an elevated shell + reboot before `npm install` |
| PF-4 — precision re-benchmark on Kaggle | Phase 9–11 | fp32 locally is settled; the cloud figure is assumed until measured |

---

### Phase completion at a glance

| Phase | Tasks | Status |
|---|---|---|
| −1 Pre-flight | 5 | ✅ Complete (4/5 — PRE-1 blocked, Phase 16 only) |
| 0 Environment | 13 | ✅ Complete — gate 33/33 |
| 1 Dataset | 20 | ✅ **18/20** — gate 3/4; P1-3 blocked on PRE-1 |
| 2 Video preproc | 15 | ✅ **13/15** — gate 8/8; P2-8 & P2-10 await ✋ |
| 3 Audio preproc | 9 | ✅ **9/9** — gate 100/100, no manual checks |
| 4 Visual baseline | 12 | ⬜ **Next** — 🔬 Experiment A + Decision D-1 |
| 5 Audio baseline | 8 | ⬜ Not started |
| 6 Fusion | 8 | ⬜ Not started |
| 7 Temporal | 6 | ⬜ Not started |
| 8 Self-attention | 10 | ⬜ Not started |
| 9 Augmentation | 7 | ⬜ Not started |
| 10 Localization ⭐ | 14 | ⬜ Not started |
| 11 Ablations | 11 | ⬜ Not started |
| 12 Optimization | 6 | ⬜ Not started |
| 13 API | 12 | ⬜ Not started |
| 14 Frontend | 12 | ⬜ Not started (also gated by P14-0) |
| 15 Testing | 9 | ⬜ Not started |
| 16 Documentation | 11 | ⬜ Not started (needs PRE-1) |
| Cloud (Kaggle) | 8 | 🟡 3/8 — CL-1 ✅, CL-7 ✅ 10/10; CL-2/CL-3 when Phase 2 runs on Kaggle |
| Cross-cutting | 7 | 🔄 Continuous |

---

---

## PHASE −1 — Pre-flight (must clear before Phase 0) — ✅ **COMPLETE 4/5 (2026-08-12)**

PRE-2…PRE-5 resolved. PRE-1 (original report) remains 🔴 blocked, but gates Phase 16 only —
it is not on the critical path for Phases 0–15.

| ID | Task | Status |
|---|---|---|
| PRE-1 | ⛔ Locate & supply the original project report (PDF/DOCX) — Part 1 is a reconstruction until audited | 🔴 **BLOCKED — not on this machine.** Searched D:\ + Downloads/Desktop/Documents/OneDrive. See `reports/report_audit.md`. Drop it at `docs/original_report.pdf`. **Not on the critical path for Phases 0–15** |
| PRE-2 | ⛔ Decide hardware strategy | ✅ **Local + Kaggle free tier.** Local = Phases 0–8, ablations, API/frontend. Kaggle P100 = full-dataset extraction + final Phase 11 runs. Adds tasks CL-1…CL-6; makes X-2 mandatory |
| PRE-3 | ⛔ Approve/amend the plan (GAN reframing; D-1 / C-1 by experiment) | ✅ **Approved as written.** Tier 1 GAN with three arms F0/F1/F2; D-1 and C-1 settled by Experiments J and K. Tie-break rules pre-committed in `reports/decision_log.md` |
| PRE-4 | Request LAV-DF dataset access **immediately** (lead time is the #1 schedule risk, R2) | ✅ **DONE — no approval gate exists.** Open Google Drive/OneDrive links; HuggingFace needs only a click-through. **R2 downgraded to Low likelihood.** See `reports/dataset_access.md` |
| PRE-5 | Create the git repository; commit `PROJECT_PLAN.md` and this task list as the baseline | ✅ Done — repo initialised, `.gitignore` + `docs/` + `reports/` scaffolded |

**Bonus — two blocking Appendix-A items resolved early during PRE-4** (details in `reports/dataset_access.md`):

| Appendix A # | Item | Resolution |
|---|---|---|
| **#9** | LAV-DF download size | **25.6 GB** → download to D: (167 GB free); **never C:** (27 GB free) |
| **#12** | Exact metadata field names | File is **`metadata.min.json`** (not `metadata.json`); adds `n_fakes` and `audio_channels` beyond the §3.3 guess; `original: Optional[str]` confirms the `source_id` derivation path |

**New risk surfaced:** `fake_periods` is type-annotated `List[List[int]]` in the authors' loader, but
Part 6 assumes **float seconds**. If they are frame indices, every localization target is wrong by
~25×. Added as a hard check in P1-5.

---

## PHASE 0 — Environment & Hardware Validation — ✅ **COMPLETE (2026-08-12)**

Gate result: **33/33 checks passed, exit 0.** `make check` is now a reusable CI gate.

| ID | Task | Status |
|---|---|---|
| P0-1 | Install Python 3.10 from python.org (NOT the Store alias) | ✅ 3.10.11 via `winget Python.Python.3.10` (uses the python.org installer). Note 3.14 was already present — too new for mediapipe and outside LAV-DF's `<3.11` pin |
| P0-2 | Install ffmpeg, verify `ffmpeg -version` | ✅ ffmpeg + ffprobe 9.0 via `winget Gyan.FFmpeg` |
| P0-3 | venv on **D:**; `PIP_CACHE_DIR` to D: | ✅ `.venv` in-project on D:; `PIP_CACHE_DIR=D:\pip-cache` set as a user env var |
| P0-4 | PyTorch CUDA 11.8; verify `torch.cuda.is_available()` | ✅ torch **2.7.1+cu118** (the last cu118 release). Driver 526.47 supports CUDA 11.8 (≥522.06) but **not** 12.x (≥527.41) — so cu118 is forced, not just preferred. Verified with a real kernel launch, not just the flag |
| P0-5 | Install remaining deps | ✅ All 23 imports pass. mediapipe 1.0.0 works with numpy 2.2.6, so the planned `numpy<2` pin was dropped |
| P0-6 | Move pagefile to D: | ✅ **Already on D:** (18 GB) — no change needed |
| P0-7 | `pyproject.toml` + pinned `requirements.txt` | ✅ torch deliberately excluded from `dependencies` so `pip install -e .` cannot swap the CUDA build for a CPU one; `requirements.txt` carries the cu118 `--extra-index-url` |
| P0-8 | `Makefile` + `.gitignore` | ✅ make 4.4.1 installed via winget. All targets `.PHONY` — make is a command runner only, never dependency resolution, because the project path contains spaces and `&` |
| P0-9 | `.pre-commit-config.yaml` | ✅ Installed to `.git/hooks/pre-commit`. Includes a 5 MB large-file guard and notebook-output stripping |
| P0-10 | `scripts/00_check_env.py` | ✅ 33 checks; writes `reports/hardware_report.md` + `hardware_facts.json`; exits non-zero on failure |
| P0-11 | Benchmark honestly | ✅ GEMM per precision, VRAM ceiling, decode throughput — see finding below |
| P0-12 | `reports/hardware_report.md` | ✅ Generated from measurements only |
| P0-13 | ⛔ **Gate** | ✅ **PASSED** |

**Measured (`make check-bench`):**

| Metric | Value |
|---|---|
| fp32 GEMM | 2.38–2.56 TFLOP/s |
| bf16 GEMM | 1.52 TFLOP/s |
| fp16 GEMM | **0.31 TFLOP/s (0.13×)** |
| Peak allocatable VRAM | **3456 MB of 4096** (84%) |
| Decode, synthetic 640×480 | 894–1509 fps (36–60× realtime) |
| RAM free during run | **0.16–0.29 GB** ⚠️ |

**⛔ Finding — AMP fp16 must be OFF locally (decision PF-4).** fp16 is **5–8× slower** than fp32 on
this GPU, reproducibly (conv3×3 @112²: 1601 img/s fp32 vs 303 fp16). The GTX 1650 is TU117, the one
Turing die with **no tensor cores**. This contradicts `PROJECT_PLAN.md` §12.3's *"Precision: AMP
fp16"*. → **fp32 locally, AMP fp16 on Kaggle** (re-benchmark there to confirm). Precision becomes a
config field, not a constant. Disk-cached features stay fp16 — that's storage, not compute.

**⚠️ Open — RAM.** Only 0.16–0.29 GB free with the machine at rest. §12.2's "close Chrome before
training" is not a joke on this box; it is the operating procedure.

**⚠️ Open — Windows MAX_PATH (P14-0).** Installing full JupyterLab aborted the whole pip
transaction on a >260-char path (decision PF-5). Dev deps now carry `ipykernel` + `nbformat` only.
**This will recur with `node_modules` in Phase 14** — enable long paths first, elevated:
`New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`

---

## PHASE 1 — Dataset Acquisition & Validation — ✅ **18/20 (2026-08-29)**

**Gate P1-20 is 3 of 4 green.** Manifest validated, leakage assertions green, statistics
written. Only the report audit (P1-3) is outstanding, and it is blocked on PRE-1 — the
original report — which gates Part 9's comparison and Phase 16, not Phases 2–15.

Ran locally against `metadata.min.json` alone: per decision PF-6 no video was downloaded
except 28 stratified clips (5.6 MB) for the ffprobe checks metadata cannot answer.

| ID | Task | Status |
|---|---|---|
| P1-1 | Check LAV-DF download size | ✅ **25.50 GB** measured (25,502,577,573 B over 136,307 files). PRE-4's 25.6 GB confirmed |
| P1-2 | Confirm real folder structure vs §3.3 (decision **PF-6** — no local download) | ✅ **`LAV-DF/{train,dev,test}/*.mp4` + `metadata.json` + `metadata.min.json` + `README.md`** — matches §3.3, which guessed `metadata.json` only. Verified from the mirror's full file listing (`reports/mirror_listing.tsv.gz`). ✋ *Attaching* it in a Kaggle notebook stays a manual browser step, needed only when Phase 2 runs there |
| P1-3 | ⛔ **Report audit** — §0.4 Finding 3: provenance of 53.35% and the protocol behind 80.00% | 🔴 **BLOCKED on PRE-1.** The report is not on this machine. See `reports/report_audit.md`. Blocks **Part 9's comparison and Phase 16 only** — not Phases 2–15 |
| P1-4 | ⛔ Resolve all 15 Appendix-A `TO VERIFY` items, or escalate | ✅ **`reports/appendix_a_resolution.md`** — **8 resolved · 1 deferred by design (#15, Phase 2) · 6 escalated**, all six blocked on PRE-1 alone. Every blocking item not requiring the report is closed |
| P1-5 | ⛔ Parse `metadata.min.json`; assert `fake_periods` units and `n_fakes == len(fake_periods)` | ✅ **`fake_periods` are float SECONDS.** Ends are non-integral, max end 19.314 s vs max duration 19.968 s, and only 4/136,304 exceed `duration`. The ~25× catastrophe is ruled out. Enforced on every load by `assert_fake_periods_are_seconds()` |
| P1-6 | Implement `src/data/manifest.py`; write `manifest_v1.parquet` to the §A schema | ✅ **136,304 rows, 3.2 MB.** Full §2A schema plus `duration_meta`, `class_name`, `n_fakes`, `original`. `--probe` runs ffprobe where the video is mounted |
| P1-7 | Derive `source_id` from the `original` chain; document the derivation | ✅ **36,431 roots — exactly the real-video count**, and the root set equals the real-video set. Every fake resolves to its original. Cycles raise |
| P1-8 | ⛔ Three leakage assertions as **build-failing** tests | ✅ **Zero overlap on all three pairs** (train 21,254 / dev 8,271 / test 6,906 source_ids). `src/data/leakage.py`, enforced in `validate_manifest()` and in pytest |
| P1-9 | Implement `src/data/validate.py` — §3.7 quarantine + exclusion counts | ✅ **4 excluded of 136,304 (0.0029%)**, all `bad_label`. All 8 status codes implemented; metadata-only mode for PF-6 |
| P1-10 | Real dataset statistics: counts, 4-class breakdown, duration distribution | ✅ 136,304 total · 36,431/99,873 real/fake · 36,431/33,543/33,170/33,160 by class · 320.9 h |
| P1-11 | ⛔ Mean & distribution of forged-span durations — determines `T` | ✅ **mean 0.650 s (16.2 frames), median 0.664 s, max 1.600 s**, over 114,253 spans. Spans are *short* — a hard constraint on receptive field and median-filter width |
| P1-12 | ⛔ Fake-frame fraction — drives focal-loss / `pos_weight` | ✅ **6.4447%** (1,861,304 / 28,880,994) → **`pos_weight` 14.52**. Opposite direction from the 73.3% video-level fake share |
| P1-13 | Verify fps and sample-rate consistency across all files | ✅ **fps 25.00 on 28/28 probed, `video_frames` byte-exact on 28/28, 16 kHz mono on 136,304/136,304.** Surfaced ⛔ **PF-7** — see below |
| P1-14 | Write `reports/dataset_statistics.md` | ✅ Written from `manifest_v1.parquet`; every §3.2 provisional figure replaced with a measured one |
| P1-15 | `src/data/subset.py` + `scripts/03_make_subset.py`; smoke-100 / dev-2k / dev-10k | ✅ Stratified by split **and** 4-class, largest-remainder apportionment, quarantined rows excluded |
| P1-16 | **Commit the subset ID lists to git** | ✅ `data/manifests/subsets/` tracked (`.gitignore` fixed — `/data/` blocked descent, so the old negation could never match). `make verify-subsets` fails on drift |
| P1-17 | Write `tests/unit/test_manifest.py` | ✅ **90 tests pass** — 74 unit + 16 integration against the real 136,304-entry file |
| P1-18 | Content-hashed config caching (§3.8) + `seed_everything()` | ✅ `cache_dir()` = `features/{sha256(config)[:8]}`; seeds `random`/`numpy`/`torch`/cuDNN/`PYTHONHASHSEED` |
| P1-19 | Implement `src/config.py` — pydantic schemas | ✅ Frozen, `extra="forbid"`, cross-field validators. A typo is a startup error |
| P1-20 | ⛔ **Gate:** manifest validated, leakage green, statistics written, report audit complete | 🟡 **3 of 4.** Manifest ✅ · leakage ✅ · statistics ✅ · **report audit 🔴 blocked on PRE-1**. Phases 2–15 are unblocked; Part 9 and Phase 16 are not |

**⛔ Finding — decision PF-7: task P2-1 is wrong as written.** For all 136,304 entries
`duration == audio_frames/16000 + 0.128` exactly — a padded audio figure, not a media duration,
exceeding *both* stream durations on every file probed. P2-1's `assert T == round(duration × 25) ± 1`
holds for only **239 of 136,304** entries (offsets run −5..−1); it would fail on 99.8% of the
dataset. The correct invariant is **`T == video_frames`**, which matched ffprobe 28/28. The manifest's
`duration` column is `video_frames / 25`; metadata's value is kept as `duration_meta`.

**✅ CL-7 PASSED 10/10 — the mirror is equivalent** (`reports/mirror_verification.md`). File set
matches metadata exactly in both directions; real/fake and per-split counts match the publication;
fps/frame-counts byte-exact. The CL-1 size flag is resolved: Kaggle's 24.84 GB `total_bytes` is the
*compressed* figure, and the per-file sizes sum to **25.50 GB**.

---

## PHASE 2 — Video Preprocessing — ✅ **13/15 automated (2026-08-29)**

**Automated gate 8/8 green**: determinism (byte-identical), resumability (rebuilds match,
survivors untouched), detection rate (mean `face_found` 0.969). Two ✋ manual checks remain
yours: P2-8 (contact sheets) and ⛔ P2-10 (watch 20 overlays) — both have artefacts rendered
and waiting in `reports/figures/`.

Ran on smoke-100 locally (100 videos, 19 MB) plus 10 `original` counterparts, which is
exactly what PF-6 permits.

| ID | Task | Status |
|---|---|---|
| P2-1 | PyAV decoder with exact fps resampling to 25; assert frame count | ✅ `src/preprocessing/video.py`. **Assertion is `T == video_frames` per PF-7**, not `round(duration × 25) ± 1` — the latter fails on 99.8% of LAV-DF. Resampling implemented by nearest source index (a no-op here: every file is exactly 25.00) |
| P2-2 | MediaPipe detection + 5-point similarity alignment → 112×112 | ✅ `face.py` + `align.py`. ⚠️ **B-1 amended by PF-8**: mediapipe 1.0.0 removed `mp.solutions` and ships no weights, so this uses the Tasks API `FaceLandmarker` with a pinned bundle. Umeyama closed form, **not** `estimateAffinePartial2D` — its RANSAC default would break P2-11 |
| P2-3 | Detect every 5th frame, track/interpolate; handle gaps; emit `face_found` | ✅ Linear interpolation within `max_gap=15` frames (0.6 s ≈ the mean forged span). Wider gaps are **held, not interpolated**, and marked not-found — inventing landmarks across a 0.6 s gap could paper over an entire manipulated segment |
| P2-4 | Multiple-face policy — **document the choice** | ✅ **Most temporally consistent track, tie-broken by area.** Largest-per-frame is the obvious choice and is wrong: it flips identity mid-clip whenever a background face is briefly nearer. First detection takes the largest; thereafter nearest centre to the previous accepted face wins |
| P2-5 | Content-hashed, **resumable** caching (skip if output exists) | ✅ `data/interim/faces/{sha256(cfg)[:8]}/{video_id}.npz`. Writes are temp-file + atomic rename, so an interrupt cannot leave a truncated file that `is_cached()` would call done |
| P2-6 | Stream frames — never hold a full video in RAM; cap `num_workers=2` | ✅ Decoding is a generator throughout; two decode passes rather than buffering frames. Default `--workers 2` per §12.2 |
| P2-7 | Contact-sheet visualizer for 20 random videos | ✅ `scripts/viz_contact_sheet.py` — 8×4 grid, evenly sampled, **frames with `face_found=False` tinted red** so dropouts are unmissable |
| P2-8 | ✋ Inspect contact sheets: faces cropped, aligned, upright | 🟡 **20 sheets written and reviewed here** — faces upright, eyes level, one identity per clip, dropouts correctly flagged. Worst case `test_001689` (0.41) is the subject turning away, honestly marked rather than silently mis-cropped. ✋ **Yours to confirm**: `reports/figures/contact_sheets/` |
| P2-9 | `scripts/viz_overlay.py` — burn `fake_periods` on as a red overlay | ✅ 20 rendered. Red border + FORGED banner, timeline strip with playhead, frame/timestamp/class caption. Spans rasterised **identically to the training target** (`round(t×25)`, clamped) so what you watch is what the model is taught |
| P2-10 | ⛔✋ **Personally watch 20 overlay videos** | 🟡 **Backed by objective evidence, still needs your eyes.** `scripts/11_verify_labels.py` compares each visual fake against the real video it names in `original`: divergence must begin at the labelled span start. **7/7 pairs, median error 0.000 s, max 0.020 s** — half a frame. ✋ **Yours**: `reports/figures/overlays/` |
| P2-11 | Determinism test: byte-identical output across two runs | ✅ **6/6 byte-identical**, comparing raw array bytes across two separate cache roots (`scripts/10_verify_preprocessing.py`) |
| P2-12 | Verify interrupt-and-resume actually resumes | ✅ Half the cache deleted and re-run: rebuilt entries **byte-identical** to the deleted ones, survivors' mtime **unchanged** (skipped, not silently rewritten), resume 4.3 s vs 16.4 s full |
| P2-13 | Assert `face_found.mean() > 0.9`; measure extraction wall-clock | ✅ **mean 0.969** over smoke-100 (median 0.988, min 0.406; 6 of 100 below 0.9, all genuine head-turns). **1.49 s/video** at `--workers 2` → dev-10k ≈ 4.1 h, full ≈ **56 h** (see PF-10) |
| P2-14 | Write `tests/unit/test_video_preprocessing.py` | ✅ 40 tests; suite now **130 passing** |
| P2-15 | ⛔ **Gate:** determinism verified, visual inspection done, resumability confirmed | 🟡 **Automated 8/8 green** (`reports/preprocessing_verification.md`). Determinism ✅ · resumability ✅ · detection rate ✅. ✋ Awaiting your sign-off on P2-8 and ⛔ P2-10 |

**⛔ Finding — decision PF-10: crops are 7.52 MB/video.** Measured over smoke-100 (788,763,014 B /
100). Extrapolated: dev-2k **14.7 GB**, dev-10k **73.5 GB**, full **~1.0 TB**. §3.9's 25 MB/2.5 GB
figures are for *frozen features*, not these intermediates. So **Phase 4 must extract features and
delete crops per video**, never crop-all-then-feature-all: dev-10k cannot fit Kaggle's 20 GB
`/kaggle/working` and is uncomfortable on D:. Only smoke-100's 0.75 GB is kept (X-6).

**⚠️ Finding — decision PF-9: `face_margin` is now 0.0, not 0.25.** Every LAV-DF frame is
**224×224** — already a tight VoxCeleb2 face crop, not a full scene. A margin has no context to
add and only manufactures replicated border: measured over 12 subjects, 11.0% of crop pixels fall
outside the source frame at margin 0.0 versus **21.4% at 0.25**.

**⚠️ Finding — decision PF-8: Decision B-1's API is gone.** `mediapipe 1.0.0` removed
`mp.solutions` entirely and ships no model weights; `opencv-python` 5.0 no longer bundles Haar
cascades either. Phase 2 now depends on `scripts/08_fetch_models.py` fetching a pinned
`face_landmarker.task` (3,758,596 B, sha256 `64184e229b263107`) into a gitignored `models/`.

---

## PHASE 3 — Audio Preprocessing — ✅ **COMPLETE 9/9 (2026-08-29)**

⛔ **Gate P3-9 PASSED: 100/100 files aligned exactly**, re-verified with `--force` so every
file was re-derived rather than read from cache. No manual checks outstanding — this is the
first phase since Phase 0 to close completely.

| ID | Task | Status |
|---|---|---|
| P3-1 | ffmpeg demux → forced mono → resample to 16 kHz | ✅ `src/preprocessing/audio.py::decode_audio`. ffmpeg with explicit `-ac 1 -ar 16000` rather than librosa/audioread, so the downmix and resample are reproducible instead of depending on whichever backend gets selected |
| P3-2 | ⛔ Log-mel `n_fft=1024`, **`hop_length=640`**, `n_mels=80`, `fmin=20`, `fmax=7600` — lock and never change | ✅ §C verbatim, and **the lock is enforced, not trusted**: `AudioPreprocessConfig` rejects any `hop_length` where `sample_rate/hop_length != 25`, naming the correct value. The conventional 10 ms hop is what someone reaches for later, and it would silently misalign every target |
| P3-3 | Pre-emphasis + per-utterance CMVN; epsilon guard before log | ✅ `preemphasis(0.97)`, `cmvn()` per coefficient, `eps=1e-10` applied **before** the log. CMVN matters beyond convention here: without it a model can separate real from fake by recognising recording conditions and score well while learning nothing |
| P3-4 | ⛔ Assert `audio_frames == video_frames ± 1` on **100%** of files | ✅ **100/100 on smoke-100, exactly.** Achieved by `fit_to_video()` — see PF-11: neither `center=True` (87.9%) nor `center=False` (63.4%) reaches 100% on this dataset |
| P3-5 | Build the parallel MFCC path for Experiment K | ✅ `--feature mfcc`, DCT-II on the same log-mel so the arms differ by exactly one transform. Identical time axis (142×80 vs 142×40), separate cache root, 100/100 aligned |
| P3-6 | Synthetic 440 Hz tone round-trip → peak in the correct mel bin | ✅ Exact bin at 220 / 440 / 1000 / 4000 Hz (440 Hz → bin 11, centre 456.5 Hz), plus a monotonicity check |
| P3-7 | Flag all-zero waveforms as `no_audio` | ✅ `is_silent()` at 1e-6; silent clips are flagged and **skip CMVN** — normalising zero variance is dividing noise by noise. **0 silent clips in smoke-100** |
| P3-8 | Write `tests/unit/test_audio_align.py` | ✅ 48 tests; suite now **190 passing** |
| P3-9 | ⛔ **Gate:** alignment green on the whole subset — strict, no rounding fudge | ✅ **PASS — 100/100**, verified with `--force` so every file was re-derived, not read from cache. Exits non-zero on a single violation |

**⛔ Finding — decision PF-11: the plan's two options both fall short.** P3-4 demands 100%, and
over all 136,304 metadata entries standard framing gives `center=True` **87.9%** and
`center=False` **63.4%**. The residue is not rounding — the decoded track genuinely is not
`video_frames × 640` samples long, drifting up to +2.4 frames. `fit_to_video()` pads or trims the
waveform to exactly `video_frames × hop_length` *before* the STFT, so the count is right **by
construction**. Measured adjustment on smoke-100: mean **−1.32 frames**, range [−3.40, 0.00] —
under 0.14 s, always at the tail, and recorded per file so the size of the fudge stays visible.

**⚠️ Decision PF-12: configs split.** `PreprocessConfig` became `VideoPreprocessConfig` +
`AudioPreprocessConfig` with independent cache roots. Under one combined config, retuning `n_mels`
would have invalidated **7.5 MB/video of face crops** (73.5 GB at dev-10k) to rebuild a 64 KB
log-mel — and Experiment K would have triggered exactly that.

**✅ Audio verification closed the `audio_only` label gap.** `scripts/11_verify_labels.py` now runs
both modalities. Comparing log-mel between an `audio_only` fake and its original, divergence onset
matches the labelled span start on **5/5 pairs, median error 0.000 s**. Combined with the visual
side that is **12/12 across all three fake classes** — and because the onset is counted in *audio
frames* and compared against a labelled time in *seconds*, it confirms the §C grid as well as the
labels.

**Measured cost:** 0.094 s/video, **64 KB/video** — 117× smaller than face crops, so PF-10's
streaming constraint does not apply to audio. dev-10k log-mels would be ~0.6 GB.

---

## PHASE 4 — Visual Baseline (🔬 Experiment A + Decision D-1)

| ID | Task |
|---|---|
| P4-1 | Frozen-backbone feature extractor for **both** ResNet-18 (512-d) and MobileNetV2 (1280-d → learned linear → 512-d for parity) |
| P4-2 | `scripts/04_extract_visual.py` — batched, `torch.no_grad()` + autocast, fp16 output, resumable |
| P4-3 | Extract over dev-2k with both backbones; record throughput and peak VRAM for each |
| P4-4 | Determinism check (`model.eval()`, no dropout); embedding-norm sanity; t-SNE identity-separation plot |
| P4-5 | Attention-pooling head + classifier (`Linear 256→64 → GELU → Dropout → Linear 64→1`, sigmoid + BCE) |
| P4-6 | Build the training loop: `src/training/{trainer,loop,optim,callbacks}.py`, AMP fp16, grad accumulation, checkpoint every epoch |
| P4-7 | ⛔ **Overfit-a-batch test** — 10 samples to ~zero loss before any real training |
| P4-8 | Set up MLflow (local, file-backed): log git SHA, resolved config, all seeds, per-epoch metrics, checkpoint path, hardware, wall-clock |
| P4-9 | Train B1a (ResNet-18) and B1b (MobileNetV2), 3 seeds each |
| P4-10 | Implement Baseline 0 (majority-class + random) as the sanity floor |
| P4-11 | 🔬 Record Experiment J; ⛔ **resolve Decision D-1** in `Appendix B` with the measurements that decided it |
| P4-12 | ⛔ **Gate:** a working trained model, both backbones clearly beat majority-class, D-1 resolved by data |

---

## PHASE 5 — Audio Baseline (🔬 Experiment B + Decision C-1)

| ID | Task |
|---|---|
| P5-1 | Dilated stride-1 1D-CNN encoder: 4 × [Conv1d(k=3,pad=1) → BN → ReLU], 80→128→256→256→256, dilation [1,2,4,8] |
| P5-2 | ⛔ Assert output length **exactly** equals input length (any downsample destroys Phase 10) |
| P5-3 | `scripts/05_extract_audio.py` |
| P5-4 | Train B2a (MFCC) and B2b (log-Mel), 3 seeds each |
| P5-5 | Masked loss so padding never leaks into the objective; normalization stats computed on **train split only** |
| P5-6 | Overfit-a-batch test |
| P5-7 | 🔬 Record Experiment K; ⛔ **resolve Decision C-1** |
| P5-8 | ⛔ **Gate:** audio pipeline validated, C-1 resolved. ⭐ *Milestone — the project is already defensible here* |

---

## PHASE 6 — Multimodal Fusion (🔬 Experiment C)

| ID | Task |
|---|---|
| P6-1 | Early-concat fusion → MLP (`src/models/fusion/concat.py`) = Baseline 3 |
| P6-2 | Temporal alignment of the two cached feature streams on the shared index grid |
| P6-3 | Per-modality feature normalization — rule out scale mismatch letting one modality dominate |
| P6-4 | Implement **modality dropout** (p=0.2 per stream) now, not later (§5.6) |
| P6-5 | Implement per-modality auxiliary heads (collapse defence #3) |
| P6-6 | Build the 4-class diagnostic breakdown reporter (real / visual-only / audio-only / both) |
| P6-7 | 🔬 Run Experiment C; verify **C > max(A, B)** beyond seed variance — or investigate honestly why not |
| P6-8 | ⛔ **Gate:** multimodal premise empirically supported, or the failure understood and documented |

---

## PHASE 7 — Temporal Modeling (🔬 Experiment D)

| ID | Task |
|---|---|
| P7-1 | 2-layer BiLSTM, 256 hidden (`src/models/temporal/lstm.py`) = Baseline 4 |
| P7-2 | `pack_padded_sequence` for variable length; unit-test that **no gradient flows through padding** |
| P7-3 | Add the **per-frame head** (`[T,256] → [T,1]`) — first localization-relevant output |
| P7-4 | Add frame-level AP as a tracked metric |
| P7-5 | 🔬 Run Experiment D; verify D > C |
| P7-6 | ⛔ **Gate:** temporal contribution measured independently of attention |

---

## PHASE 8 — Self-Attention (🔬 Experiment E)

| ID | Task |
|---|---|
| P8-1 | 4-layer pre-norm Transformer encoder, 4 heads, d_model=256, d_ff=512, dropout 0.1, max_T=750 |
| P8-2 | Learned positional encoding + positional-encoding ablation |
| P8-3 | **Upgrade fusion to bidirectional cross-attention** (2 layers, A→V and V→A, concat + project) |
| P8-4 | Implement the sync head (component F): two 256-d projections → per-timestep cosine over ±5-frame window |
| P8-5 | Implement auxiliary **InfoNCE sync loss** (τ=0.07), positives = aligned pairs from **real** videos, negatives = shifted ≥10 frames |
| P8-6 | ⛔ **Desync test** — shift audio +400 ms on a real video and assert the sync score drops significantly. If not, the component is not working regardless of the loss curve |
| P8-7 | Training stability: warmup, pre-norm, gradient clipping at 1.0; attention head-entropy check for collapse |
| P8-8 | Attention-map visualization over the timeline (`notebooks/03_attention_viz.ipynb`) |
| P8-9 | 🔬 Run Experiment E; verify E > D beyond seed variance; confirm attention elevates on forged spans |
| P8-10 | ⛔ **Gate:** attention contribution measured, attention maps produced |

---

## PHASE 9 — Augmentation (🔬 Experiments F0 / F1 / F2)

| ID | Task |
|---|---|
| P9-1 | Implement **manipulation splicing** first — splice a real span from video X into video Y with exactly-known boundaries (likely the biggest single win) |
| P9-2 | Implement classical augmentation: spec-augment, temporal jitter, crop/colour jitter |
| P9-3 | Implement the conditional feature-space GAN: `G: (noise, label) → [T,256]` + sequence discriminator, spectral norm |
| P9-4 | Monitor GAN output diversity; inspect for mode collapse **before** using any samples |
| P9-5 | 🔬 Run **all three arms**: F0 (none) / F1 (classical) / F2 (classical + GAN), 3 seeds each |
| P9-6 | ⛔ Report **F2 vs F1** honestly — the only valid GAN claim. F2 ≈ F1 → "the GAN added nothing beyond classical augmentation" is a legitimate published result |
| P9-7 | ⛔ **Gate:** three arms run, conclusion recorded honestly |

---

## PHASE 10 — Temporal Localization ⭐ (🔬 Experiment G)

| ID | Task |
|---|---|
| P10-1 | ⛔ GT alignment: `fake_periods` → per-frame `[T]` target, with the **inverse-transform assertion** (target → intervals matches original within 1 frame) |
| P10-2 | Assert `target.sum() == 0` for every real video, exactly |
| P10-3 | Per-frame head + **focal loss** (α=0.25, γ=2.0) |
| P10-4 | Optional boundary head `[T,2]` (start-ness / end-ness) vs Gaussian-smoothed targets + BoundaryLoss |
| P10-5 | Combined loss `λ_cls·BCE + λ_loc·Focal + λ_sync·InfoNCE + λ_bnd·Boundary` starting at (1.0, 2.0, 0.3, 0.5); tune λ on dev |
| P10-6 | Post-processing chain (§6.4): median smooth(5) → threshold τ → binary closing(3) → extract runs → min-duration 0.4 s → merge gaps < 0.3 s → per-segment confidence → seconds |
| P10-7 | ⛔ Implement AP/AR for 1-D intervals and **unit-test against hand-computed toy cases** (perfect → 1.0; IoU 0.4 @ thr 0.5 → 0.0; one correct + one spurious → exact hand value). Cross-check against an established implementation |
| P10-8 | Synthetic known-span test → recovered IoU > 0.9 |
| P10-9 | Property tests: segments sorted, non-overlapping, within `[0, duration]` |
| P10-10 | Grid-search τ / median kernel / min-duration / merge-gap on **dev** maximizing AP@0.5, then **freeze** |
| P10-11 | Implement long-video chunking: 750-frame chunks, 125-frame overlap, average scores in overlap, post-process **once** on the stitched sequence |
| P10-12 | Measure mean boundary error (ms) and **false-positive segment rate on real videos** |
| P10-13 | ✋ Visually compare predicted timelines against ground truth for 20 videos |
| P10-14 | ⛔ **Gate:** AP@{0.5,0.75,0.95} + AR@{100,50,20,10} computed with a *verified* AP implementation, post-processing frozen before test |

---

## PHASE 11 — Ablation Experiments

| ID | Task |
|---|---|
| P11-1 | Build `src/evaluation/*` as **pure functions** — predictions.parquet + manifest → metrics.json, decoupled from training |
| P11-2 | `scripts/08_run_ablations.py` — run the full matrix A–K × 3 seeds (0,1,2) |
| P11-3 | 🔬 Experiment H — ablate the sync loss (λ_sync=0); verify G > H |
| P11-4 | 🔬 Experiment I — ablate modality dropout; use the 4-class breakdown to detect collapse |
| P11-5 | Compute mean ± std for every headline claim |
| P11-6 | Apply the **four §7.3 criteria** (magnitude > seed std, consistency on all 3 seeds, right metric, cost stated) to every claimed improvement |
| P11-7 | Fill Table 8.2.1 (classification), 8.2.2 (localization), 8.2.4 (4-class breakdown), 8.2.5 (decision resolutions) |
| P11-8 | ⛔ **Run the test split exactly once**, at the very end, after all hyperparameters and thresholds are frozen |
| P11-9 | ⛔ Include **negative results**; mark any sub-seed-variance difference as "not significant" |
| P11-10 | Reproducibility check: re-run one experiment from its logged config, confirm metrics match to ~1e-4 |
| P11-11 | ⛔ **Gate:** complete evidence table exists, every number traceable to a logged run |

---

## PHASE 12 — Model Optimization

| ID | Task |
|---|---|
| P12-1 | `scripts/09_benchmark.py` — profile end-to-end, broken into decode/align, visual extract, audio extract, Stage-B forward, post-process |
| P12-2 | ONNX export (`scripts/10_export_model.py`) |
| P12-3 | fp16 inference; batched extraction; model cached in memory |
| P12-4 | Fill Table 8.2.3 (latency, peak VRAM, peak RAM per component) |
| P12-5 | ⛔ **Re-verify accuracy after optimization** — quantization can degrade it; a demo that contradicts the results table is a failure |
| P12-6 | ⛔ **Gate:** < 10 s per 10 s clip, model < 50 MB, accuracy confirmed unchanged |

---

## PHASE 13 — FastAPI Backend

| ID | Task |
|---|---|
| P13-1 | `POST /api/v1/analyze` → returns job id |
| P13-2 | `GET /api/v1/jobs/{id}` → status / result |
| P13-3 | `GET /health` and `GET /api/v1/model-info` |
| P13-4 | Pydantic request/response schemas; auto-generated OpenAPI at `/docs` |
| P13-5 | In-process background job queue + job store |
| P13-6 | Strict input validation: max 200 MB, allowed [mp4, avi, mov, mkv], MIME + codec check, duration bounds, 300 s timeout |
| P13-7 | **Stream uploads to disk** — never read a large file into memory on a 6 GB machine |
| P13-8 | Load the model **once at startup**; verify by timing the second request |
| P13-9 | ⛔ Enforce a concurrency **semaphore of 1** — 6 GB RAM cannot support more |
| P13-10 | Structured error responses — oversized / corrupt / no-audio / no-face must never return a 500 stack trace |
| P13-11 | `tests/integration/test_api.py` with pytest + httpx covering every error case |
| P13-12 | ⛔ **Gate:** API tested and stable under all error cases |

---

## PHASE 14 — Frontend

| ID | Task |
|---|---|
| P14-0 | ⛔ **Enable Windows long-path support before `npm install`** (elevated + reboot). `node_modules` nests deeper than JupyterLab, which already broke a pip transaction in Phase 0 — see decision PF-5 |
| P14-1 | Scaffold React + Vite + TailwindCSS (`frontend/`) |
| P14-2 | Drag-drop upload component with progress |
| P14-3 | Job polling (1 s interval) with graceful error states |
| P14-4 | ⭐ **Timeline strip with forged spans highlighted** (green/red) — the centrepiece |
| P14-5 | Video player that seeks to a span on click |
| P14-6 | Verdict badge + confidence display |
| P14-7 | Per-frame score line chart |
| P14-8 | JSON download of the full result |
| P14-9 | Component tests (Vitest) |
| P14-10 | ✋ Test with a **real**, a **fully fake**, and a **partially fake** video; verify the timeline matches the returned segments |
| P14-11 | ⛔ **Record a screen capture of the demo** — the single most effective portfolio artifact this project produces |
| P14-12 | ⛔ **Gate:** a non-expert can upload a video and correctly read the result unaided |

---

## PHASE 15 — Testing

| ID | Task |
|---|---|
| P15-1 | Unit: `test_manifest`, `test_audio_align`, `test_model_shapes`, `test_losses`, `test_postprocess` |
| P15-2 | ⛔ Unit: `test_gt_alignment` — protects every localization number |
| P15-3 | ⛔ Unit: `test_ap_metric` — protects every experiment in Part 7 |
| P15-4 | Integration: `test_pipeline_e2e` on tiny fixtures (`tiny_real.mp4`, `tiny_fake.mp4`) |
| P15-5 | Property tests: segments sorted, non-overlapping, in-bounds |
| P15-6 | Regression test on a fixed fixture with committed expected outputs |
| P15-7 | Coverage report — target > 70% on `src/` |
| P15-8 | GitHub Actions CI (test-only; **no deployment**, per §0.2) |
| P15-9 | ⛔ **Gate:** suite green, coverage met, CI green |

---

## PHASE 16 — Documentation

| ID | Task |
|---|---|
| P16-1 | `README.md` — what / why / results / how to run, with the demo capture embedded |
| P16-2 | `reports/results.md` — all filled tables (8.2.1–8.2.5) |
| P16-3 | `reports/final_report.md` — methodology, ablations, limitations |
| P16-4 | Render the §5.1 two-stage architecture ASCII diagram as a proper figure |
| P16-5 | Complete Appendix B decision log (D-1, C-1, B-1, F-1, G-1, H-1, I-1, L-1) |
| P16-6 | Docstrings across `src/`; typed interfaces |
| P16-7 | ⛔ **Honest limitations section** — generalization to unseen generators, subset-scale caveats, any negative results |
| P16-8 | Fill the Part 14 resume templates and the §14.3 fill-in tracker from real measured numbers |
| P16-9 | Prepare answers to the 12 interview questions in §15.16 |
| P16-10 | `make demo` target (uvicorn :8000 + Vite :5173) documented as the local run strategy |
| P16-11 | ⛔ **Gate:** a stranger can reproduce the headline result from the README alone; every number traces to a logged run; no fabricated numbers anywhere |

---

## CLOUD — Kaggle track (new, from decision PF-1)

Not a separate phase — these are the tasks the local+Kaggle strategy adds.

**⚠️ CL-1, CL-2, CL-3 and CL-7 are now Phase 1 prerequisites, not Phase 9 work** (decision **PF-6**,
2026-08-27). They gate P1-2. CL-4…CL-6 and CL-8 still run alongside Phases 9–11.

| ID | Task |
|---|---|
| CL-1 | ✅ **COMPLETE (2026-08-29)** — account `dinesh1234567`, **10/10 checks passed** (`reports/kaggle_report.md`). Credentials at `~/.kaggle/kaggle.json`, ACL-restricted. ✋ Phone verification **attested yes**; ✋ GPU quota **attested 30 h remaining**. Free tier confirmed: ~30 h/week, P100 16 GB or T4×2, 12 h/session, 20 GB writable `/kaggle/working`. Re-verify quota before each long run — it resets weekly |
| CL-2 | Attach the public LAV-DF mirror `elin75/localized-audio-visual-deepfake-dataset-lav-df` read-only at `/kaggle/input` (no download, does not count against `/kaggle/working`). Fallback if CL-7 fails: upload the authors' copy as a private Kaggle Dataset |
| CL-3 | Write a thin Kaggle notebook entrypoint that clones the repo and calls the **same** `scripts/` — no logic duplicated in notebook cells (drift here silently invalidates cross-environment comparisons) |
| CL-4 | ⛔ Verify device-agnosticism: run one Phase-4 experiment on both local and Kaggle from the same config and confirm metrics match within seed variance |
| CL-5 | Checkpoint/artifact sync back to D: — MLflow runs and `best.pt` must land in the same `experiments/` tree so Part 11's evidence table stays single-source |
| CL-6 | Handle Kaggle preemption: per-epoch checkpointing + resume-from-checkpoint verified **before** launching any long run (R13 now applies to cloud too) |
| CL-7 | ⛔ **Prove the Kaggle mirror is equivalent to the authors' release** before building on it — it is a community re-upload. Check: file count = 136,304; real/fake = 36,431/99,873; `metadata.min.json` present and byte-identical to the HF/GitHub copy; ⛔ **spot-check 20 videos with `ffprobe` for fps = 25 and unchanged `duration`/`video_frames` vs metadata** — a silent re-encode shifts every `fake_periods` target. Escalate to the fallback in CL-2 if any check fails |
| CL-8 | Shard full-dataset feature extraction across sessions — ≈45 GB of features against a 20 GB `/kaggle/working` cap and a 12 h session limit means resumable, sharded output written to a Kaggle Dataset. Only metrics and checkpoints sync back to D: (CL-5); the 45 GB never does |

---

## Cross-cutting / continuous

| ID | Task |
|---|---|
| X-1 | Every phase: tests pass + reproducible from a committed config + descriptive commit + results in `reports/` (universal exit gate) |
| X-2 | ⛔ Keep code device-agnostic so the same code runs locally and on Kaggle without edits — **mandatory, not aspirational, per decision PF-1** |
| X-3 | `data/raw/` is immutable; `experiments/` is append-only; `src/` never imports from `api/` or `scripts/` |
| X-4 | Assume interruption: checkpoint every epoch, all long jobs resumable (R13) |
| X-5 | Watch RAM continuously — `num_workers=2`, mmap feature files, `persistent_workers=False`, close Chrome while training (R1) |
| X-6 | Delete face crops after visual features are extracted (keep only smoke-subset crops) to reclaim disk (R8) |
| X-7 | Track the 13 risks R1–R13; re-check R3 (leakage) and R4 (modality collapse) at every scale-up |

---

## Task count by phase

| Phase | Tasks |
|---|---|
| Pre-flight | 5 |
| P0 Environment | 13 |
| P1 Dataset | 20 |
| P2 Video preproc | 15 |
| P3 Audio preproc | 9 |
| P4 Visual baseline | 12 |
| P5 Audio baseline | 8 |
| P6 Fusion | 8 |
| P7 Temporal | 6 |
| P8 Self-attention | 10 |
| P9 Augmentation | 7 |
| P10 Localization ⭐ | 14 |
| P11 Ablations | 11 |
| P12 Optimization | 6 |
| P13 API | 12 |
| P14 Frontend | 12 |
| P15 Testing | 9 |
| P16 Documentation | 11 |
| Cloud (Kaggle track) | 8 |
| Cross-cutting | 7 |
| **Total** | **203** |

*Phase −1 complete except PRE-1 (report retrieval), which does not block Phases 0–15.*

*Completed as of 2026-08-29: ~61 of 203 tasks (≈30%). Phases −1, 0, 1, 2 and 3 done (Phase 1 at 18/20 with gate 3/4; Phase 2 at 13/15 with the automated gate 8/8; Phase 3 complete 9/9). CL-1 and CL-7 complete. 190 passing tests. Outstanding: P1-3 blocked on PRE-1, and the two ✋ Phase 2 manual checks.*
