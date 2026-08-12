# Implementation Task List — Audio-Visual Temporal Forgery Detection & Localization

Derived from `PROJECT_PLAN.md` (Parts 0–15, Phases 0–16). Nothing here is started yet.
IDs are stable — use them when referring to work.

Legend: ⛔ = blocking gate item · ⭐ = headline capability · 🔬 = experiment · ✋ = manual human check

---

## PHASE −1 — Pre-flight (must clear before Phase 0) — **IN PROGRESS (2026-08-12)**

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

## PHASE 0 — Environment & Hardware Validation

| ID | Task |
|---|---|
| P0-1 | Install Python 3.10 from python.org (NOT the Microsoft Store alias) |
| P0-2 | Install ffmpeg, add to PATH, verify `ffmpeg -version` |
| P0-3 | Create the venv on **D:** (C: has only 27 GB free); set `PIP_CACHE_DIR` to D: |
| P0-4 | Install PyTorch with CUDA 11.8 wheels; verify `torch.cuda.is_available() == True` on the GTX 1650 |
| P0-5 | Install remaining deps: torchvision, torchaudio, librosa, opencv-python, mediapipe, pandas, pyarrow, scikit-learn, matplotlib, mlflow, fastapi, uvicorn, pytest, ruff, pydantic, tqdm, av |
| P0-6 | Move the Windows pagefile to D: (prevents C: thrash under the 6 GB RAM constraint) |
| P0-7 | Write `pyproject.toml` (deps + ruff/mypy/pytest config) and pinned `requirements.txt` |
| P0-8 | Write `Makefile` (setup / features / train / test / api / demo) and `.gitignore` (data/, checkpoints/, logs/, *.npy) |
| P0-9 | Add `.pre-commit-config.yaml` |
| P0-10 | Write `scripts/00_check_env.py` — assert every dependency, print the hardware table, exit non-zero on failure |
| P0-11 | Benchmark honestly: matmul throughput, peak allocatable VRAM before OOM, single-video decode speed, free RAM under load |
| P0-12 | Write `reports/hardware_report.md` with measured numbers |
| P0-13 | ⛔ **Gate:** CUDA works, ffmpeg works, hardware report written, `00_check_env.py` exits 0 |

---

## PHASE 1 — Dataset Acquisition & Validation

| ID | Task |
|---|---|
| P1-1 | ✅ ~~Check LAV-DF download size~~ — **resolved in PRE-4: 25.6 GB, download to D:** |
| P1-2 | Download LAV-DF to `data/raw/` via HuggingFace CLI (resumable — matters at 25.6 GB, see R13); verify checksums; confirm real folder structure vs §3.3 |
| P1-3 | ⛔ **Report audit** — resolve §0.4 Finding 3: provenance of the 53.35% figure and the evaluation protocol behind 80.00%; write `reports/report_audit.md` |
| P1-4 | ⛔ Resolve all 15 Appendix-A `TO VERIFY` items, or explicitly escalate as unresolvable |
| P1-5 | Parse **`metadata.min.json`** (schema confirmed in PRE-4); ⛔ **print 10 raw `fake_periods` and assert `max(end) <= duration`** — determines whether values are seconds or frame indices (~25× error if misread); assert `n_fakes == len(fake_periods)` |
| P1-6 | Implement `src/data/manifest.py` — `ffprobe` every file, join with metadata, write `manifest_v1.parquet` to the §A schema |
| P1-7 | Derive `source_id` (identity key) — from `original` chain to root if no explicit key exists; document the derivation |
| P1-8 | ⛔ Implement the three leakage assertions (train∩dev, train∩test, dev∩test on `source_id`) as **build-failing** tests |
| P1-9 | Implement `src/data/validate.py` — quarantine per §3.7 (missing / corrupt / no_audio / no_video / too_short / too_long / bad_label); report exclusion counts + reasons |
| P1-10 | Compute real dataset statistics: total/real/fake counts, 4-class breakdown, duration distribution |
| P1-11 | ⛔ Compute **mean & distribution of forged-span durations** — this determines `T` in Part 6 |
| P1-12 | ⛔ Compute the **fake-frame fraction** — this drives focal-loss / `pos_weight` |
| P1-13 | Verify fps and sample-rate consistency across all files |
| P1-14 | Write `reports/dataset_statistics.md` — every provisional §3.2 figure replaced with a measured one |
| P1-15 | Implement `src/data/subset.py` + `scripts/03_make_subset.py`; build smoke-100 / dev-2k / dev-10k, stratified by split and 4-class taxonomy, fixed seed |
| P1-16 | **Commit the subset ID lists to git** so every experiment runs on literally the same videos |
| P1-17 | Write `tests/unit/test_manifest.py` |
| P1-18 | Implement content-hashed config caching (§3.8) and `src/seed.py::seed_everything()` |
| P1-19 | Implement `src/config.py` — pydantic schemas so configs are validated, not raw dicts |
| P1-20 | ⛔ **Gate:** manifest validated, leakage assertions green, real statistics written, report audit complete |

---

## PHASE 2 — Video Preprocessing

| ID | Task |
|---|---|
| P2-1 | PyAV decoder wrapper with exact fps resampling to 25; assert `T == round(duration × 25) ± 1` |
| P2-2 | MediaPipe face detection (Decision B-1) + 5-point similarity-transform alignment → 112×112 crops |
| P2-3 | Detect every 5th frame, track/interpolate between; handle detection gaps; emit `face_found` mask |
| P2-4 | Multiple-face policy: pick largest or most temporally consistent track — **document the choice** |
| P2-5 | Content-hashed, **resumable** caching to `data/interim/faces/{video_id}.npy` (skip if output exists) |
| P2-6 | Stream frames — never hold a full video in RAM; cap `num_workers=2` |
| P2-7 | Contact-sheet visualizer for 20 random videos |
| P2-8 | ✋ Inspect contact sheets: faces correctly cropped, aligned, upright |
| P2-9 | Write `scripts/viz_overlay.py` — burn `fake_periods` onto video as a red overlay |
| P2-10 | ⛔✋ **Personally watch 20 overlay videos** and confirm the highlighted spans genuinely look manipulated (§6.3 — catches label bugs nothing else will) |
| P2-11 | Determinism test: byte-identical output across two runs |
| P2-12 | Verify interrupt-and-resume actually resumes |
| P2-13 | Assert `face_found.mean() > 0.9` on a clean sample; measure and record extraction wall-clock |
| P2-14 | Write `tests/unit/test_video_preprocessing.py` |
| P2-15 | ⛔ **Gate:** determinism verified, visual inspection done, resumability confirmed |

---

## PHASE 3 — Audio Preprocessing

| ID | Task |
|---|---|
| P3-1 | ffmpeg demux → forced mono downmix → resample to 16 kHz |
| P3-2 | ⛔ Log-mel with `n_fft=1024`, **`hop_length=640`** (= 40 ms = 1 frame @ 25 fps), `n_mels=80`, `fmin=20`, `fmax=7600` — lock this and never change it |
| P3-3 | Pre-emphasis + per-utterance CMVN; epsilon guard (`1e-10`) before log |
| P3-4 | ⛔ Assert `audio_frames == video_frames ± 1` on **100% of files** (handle `center=True` padding here, not downstream) |
| P3-5 | Build the parallel MFCC path for Experiment K |
| P3-6 | Synthetic 440 Hz tone round-trip test → peak lands in the correct mel bin |
| P3-7 | Flag all-zero waveforms as `no_audio` |
| P3-8 | Write `tests/unit/test_audio_align.py` |
| P3-9 | ⛔ **Gate:** alignment assertion green on the whole subset — strict, no rounding fudge |

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

Runs alongside Phases 9–11. Not a separate phase — these are the tasks the local+Kaggle strategy adds.

| ID | Task |
|---|---|
| CL-1 | Create/verify the Kaggle account; confirm GPU quota (free tier: ~30 h/week, P100 16 GB or T4×2) |
| CL-2 | Mirror the dataset to Kaggle — attach the existing public LAV-DF Kaggle dataset if usable, else upload the cached **features** (~3.3 GB, far smaller than 25.6 GB of raw video) as a private Kaggle Dataset |
| CL-3 | Write a thin Kaggle notebook entrypoint that clones the repo and calls the **same** `scripts/` — no logic duplicated in notebook cells (drift here silently invalidates cross-environment comparisons) |
| CL-4 | ⛔ Verify device-agnosticism: run one Phase-4 experiment on both local and Kaggle from the same config and confirm metrics match within seed variance |
| CL-5 | Checkpoint/artifact sync back to D: — MLflow runs and `best.pt` must land in the same `experiments/` tree so Part 11's evidence table stays single-source |
| CL-6 | Handle Kaggle preemption: per-epoch checkpointing + resume-from-checkpoint verified **before** launching any long run (R13 now applies to cloud too) |

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
| Cloud (Kaggle track) | 6 |
| Cross-cutting | 7 |
| **Total** | **201** |

*Phase −1 complete except PRE-1 (report retrieval), which does not block Phases 0–15.*
