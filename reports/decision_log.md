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

## Phase 1 decisions — resolved 2026-08-27

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
