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

## Appendix B — architectural decisions

| ID | Decision | Options | Resolved by | Date | Outcome |
|---|---|---|---|---|---|
| **PF-1** | Hardware strategy | Local-only / Local+Kaggle / defer | User, PRE-2 | 2026-08-12 | ✅ **Local + Kaggle free tier** |
| **PF-2** | GAN tier | Tier 1 / Tier 1 minus GAN / Tier 2 | User, PRE-3 | 2026-08-12 | ✅ **Tier 1, three-arm F0/F1/F2** |
| **PF-3** | D-1 & C-1 method | By experiment / by prior | User, PRE-3 | 2026-08-12 | ✅ **By experiment (J and K)** |
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
