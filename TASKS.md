# Implementation Task List — Audio-Visual Temporal Forgery Detection & Localization

Derived from `PROJECT_PLAN.md` (Parts 0–15, Phases 0–16).
IDs are stable — use them when referring to work.

Legend: ⛔ = blocking gate item · ⭐ = headline capability · 🔬 = experiment · ✋ = manual human check

---

## 📍 CURRENT STATUS — 2026-10-03, Phase 11 complete · Phase 12 next

| | |
|---|---|
| **Done** | Phase −1 · Phase 0 · **P1 (18/20)** · **P2 (13/15)** · **P3 (9/9)** · **P4 (12/12)** · **P5 (8/8)** · **P6 (8/8)** · **P7 (6/6)** · **P8 (10/10)** · **P9 (7/7)** · **P10 (13/14)** · **P11 (11/11)** · CL-1 ✅ · CL-7 ✅ |
| **Resolved** | ⛔ **D-1 → MobileNetV2** (0.7189 vs 0.6319). ⛔ **C-1 → log-Mel** (0.9838 vs 0.7677). Both open decisions settled by experiment |
| **Next** | **Phase 12 — optimization** (ONNX, fp16, Table 8.2.3). ⛔ The test split is spent (`reports/test_once.lock`); P12-5 re-verifies accuracy on **dev**, never test |
| **Progress** | ~131 of 203 tasks (≈65%) · **13 of 18 phases complete** |
| **Tests** | **409 passing** (unit + integration against the real 136,304-entry dataset) |
| **⚠️ Caveat** | ⛔ **PF-17 / PF-19 / PF-20** — the artefacts are inherited by every arm since. Absolute numbers (test included) are **upper bounds**; the trustworthy quantities are the paired deltas in `reports/evidence_table.md`. ⛔ **PF-27**: Phase 4–11 runs reproduce to ~1 seed-sd, not 1e-4 — new training uses `--strict-determinism` |
| **✋ Awaiting you** | P2-8 (contact sheets) · ⛔ P2-10 (watch 20 overlays) — artefacts in `reports/figures/` · P10-13 (20 timelines, `reports/figures/localization/timelines_dev20.png`) |
| **Kaggle** | `dinesh1234567` · phone verified · **30 GPU h/week** · P100 16 GB or T4×2 · 12 h/session · 20 GB `/kaggle/working` |
| **Branch** | `master` |

**Phases 1–7 built the data, preprocessing, all three baselines and the temporal model.**
`src/` now holds `models/backbones/{visual,audio}.py`, `models/fusion/{concat,cross_attention}.py`,
`models/temporal/{lstm,transformer}.py`, `models/heads/{classification,frame,sync}.py`,
`losses/sync.py`, `localization/targets.py`, `training/{trainer,reporting}.py`,
`evaluation/metrics.py` and `data/dataset.py`, driven by `scripts/02`–`26`. Phase 9 added
`training/{augment,gan}.py`, `models/gan/{generator,discriminator}.py` and `scripts/30`–`31`.
`api/` and `inference/` are still stubs — Phases 11–16 are untouched. Phase 10 added
`evaluation/localization.py`, `localization/{postprocess,chunking}.py`, `losses/localization.py`,
`models/heads/boundary.py` and `scripts/32`–`35`.

### 🔬 Phase 11 result — evidence table + the single test run (2026-10-03)

**Gate P11-11 passed.** `reports/evidence_table.md` (Tables 8.2.1, 8.2.2, 8.2.4, 8.2.5 and
section 7.3 on 13 claims) and `reports/test_once.md`.

⛔ **P11-8 — test, scored exactly once** (392 clips, 291 fake / 101 real, 3 seeds, frozen
post-processing; a second run is refused by `scripts/38_test_once.py`):

| G (headline) | dev | **test** |
|---|---|---|
| clip ROC-AUC | 0.9889 | **0.9842 ± 0.0013** |
| frame AP | 0.9257 | **0.8903 ± 0.0037** |
| AP@0.5 | 0.8522 | **0.8005 ± 0.0167** |
| AP@0.75 | 0.6340 | **0.5392 ± 0.0299** |
| boundary error | 78 ms | **97 ± 14 ms** |
| FP rate on real | 0.000 | **0.026** |

Visual-only fakes are the weak class on test (AP@0.5 0.71 vs ~0.85). Zeroing video costs
AUC 0.0016 and *raises* AP@0.5 by 0.0096 — **PF-21 holds on held-out data.**

**Section 7.3 verdicts (dev, paired by seed).** Helps: **J** MobileNetV2 (+0.087 AUC),
**K** log-Mel (+0.216), **D > C** BiLSTM (+0.084). Hurts: **C vs B** fusion (−0.080 AUC),
**E vs D** on frame AP (−0.018). Not significant: G-1 cross-attention, H-1 Transformer,
F1, F2, focal, boundary head, **H** sync loss (+0.0023 AP@0.5, spread 0.0287) and **I**
modality dropout (+0.0175, spread 0.0187).

**PF-28** — the two new ablations' secondary metrics are real findings: the sync loss
**costs 0.096 AP@0.75** (all 3 seeds, beyond spread) and 22 ms of boundary error; without
modality dropout the model is the only arm whose localization uses video (0.060 AP@0.5
lost without it) — but unhelpfully: visual-only AP@0.5 falls 0.68 → 0.57 and 10% of real
clips get a false segment.

**PF-27 / P11-10** — re-training G seed 0 from its logged config drifted (AUC Δ 0.006,
best epoch 12 → 23): cuDNN determinism alone is not enough. With
`torch.use_deterministic_algorithms(True)` (`--strict-determinism`) two re-runs are
**bit-identical**. P11-10 passes in strict mode; earlier runs reproduce to ~1 seed-sd.

Test-split features: only 19 test clips existed locally, so the dev-2k test split was
fetched alone (`05_fetch_metadata.py --splits test`, 383 clips, 0 failed) and extracted —
train (592) and dev (175) are exactly Experiment G's. Experiment I seed 2 was killed by
host RAM pressure at epoch 23 (PF-26 again) and resumed from `last.pt`.

### 🔬 Phase 10 result — Experiment G (2026-10-02)

**Localization works on dev, and no objective change beats plain BCE at AP@0.5.**
`reports/experiment_g.md`, gate P10-14 passed. Dev numbers — λ, model selection and
post-processing were all chosen on dev; Phase 11's single test run is the held-out figure.

| arm | AP@0.5 | AP@0.75 | AP@0.95 | AR@10 | boundary err | FP on real |
|---|---|---|---|---|---|---|
| BCE control (F0 objective) | 0.8461 ± 0.0109 | 0.6425 | 0.0730 | 0.6542 | 69 ms | 0.020 |
| focal | 0.8441 ± 0.0114 | **0.6793** | **0.1870** | **0.6977** | **61 ms** | 0.048 |
| **full** (focal + boundary, headline) | **0.8522** ± 0.0187 | 0.6340 | 0.0383 | 0.6202 | 78 ms | **0.000** |

⛔ **Every AP@0.5 difference is inside seed spread — not significant.** The headline arm is
`full` only because the pre-committed rule picks the highest mean AP@0.5. The one significant
result is **focal − BCE: +0.0369 AP@0.75 and −7.3 ms boundary error** (all 3 seeds) — focal
loss sharpens boundaries. The boundary head (§5.4) did **not** help: −0.0454 AP@0.75, +17 ms,
both not significant, and AP@0.95 collapses to 0.038. λ tuning (P10-5) moved nothing beyond
spread, so the plan's defaults stand.

Post-processing tuning was the largest single lever: **+0.24 AP@0.5** for `full` over §6.4's
defaults (0.6112 → 0.8522), with a flat top — mostly from dropping τ to 0.2.

⛔ **PF-21 holds for localization too.** Zeroing video costs +0.0009 (BCE), −0.0000 (full) and
+0.0119 (focal) AP@0.5 — only focal crosses 0.01, barely, so localization is audio-driven.

**PF-26** — Phase 10 training is RAM-bound on this machine: the first `make phase10` run was
killed by host memory pressure mid-`full_l1` (≈1.5 GB free with Chrome open). Resume from
`last.pt` worked exactly (epoch 35). Close Chrome before long runs (X-5).

### 🔬 Phase 9 result — Experiment F (2026-09-19)

**The GAN added nothing beyond classical augmentation — and augmentation itself hurt.**
`reports/experiment_f.md`, gate P9-7 passed.

| arm | dev AUC | sd | frame AP | −visual | train clips |
|---|---|---|---|---|---|
| E full (Phase 8) | 0.9928 | 0.0011 | 0.8998 | +0.0010 | 592 |
| F0 no augmentation | 0.9919 | 0.0030 | 0.8946 | −0.0003 | 592 |
| F1 splice + classical | 0.9874 | 0.0021 | 0.8412 | +0.0009 | 888 |
| F2 F1 + GAN samples | 0.9883 | 0.0005 | 0.8508 | +0.0071 | 888 |

⛔ **P9-6, F2 − F1 = +0.0009** AUC (per seed −0.0008 / +0.0044 / −0.0008), inside the 0.0021
seed spread and the seeds disagree on the sign — **not significant**. Frame AP +0.0096, also
inside spread. The pre-committed headline applies: *the GAN added nothing beyond classical
augmentation*. P9-4 admitted all three seeds (pairwise ratio 1.01, std ratio 0.77, nn ratio 8.84).

⛔ **F1 − F0 = −0.0045 AUC and −0.0534 frame AP, significant on both** (beyond spread,
negative on all 3 seeds). Splicing + classical augmentation made the model *worse* on LAV-DF
dev, and hurt localization most — the opposite of the plan's "likely the biggest single win"
for P9-1. F0 reproduces Phase 8's E full (|Δ| 0.0009), so this is the augmentation, not the
harness. **Carry into Phase 10: train Experiment G without P9 augmentation unless a
localization-specific re-test says otherwise.**

⛔ **PF-21 still stands.** No arm uses video (threshold 0.01). F2's +0.0071 mean hides one seed
at +0.0204 and two near zero — seed-dependent, not a break. Caveat: only the `visual` splices
(a third of the spliced examples) have untouched audio, and the reliance metric runs on LAV-DF
dev where audio suffices, so it cannot see any visual skill learned for splices. A held-out
`visual`-splice dev view would measure it — logged as **PF-24**.

F2 seed 2 was interrupted on 2026-09-06 at epoch 16 and re-run from scratch on 2026-09-19 (a
resume would have lost epochs 0–15 of history). F0/F1 were trained at `fbec6c6`, F2 at `1e167fd`
— the difference is F2-only GAN code, recorded per arm in the report.

### 🔬 Phase 7 result — Experiment D (2026-09-05)

**D > C, decisively, and the gate passed.** `reports/experiment_d.md`.

| arm | dev AUC | sd |
|---|---|---|
| C fusion (no temporal) | 0.9038 | 0.0275 |
| **D fusion + BiLSTM** | **0.9883** | 0.0022 |

`D − C = +0.0844`, well beyond the larger seed spread (0.0275). Section 5.3 predicted
sequential memory would *lose* to global comparison; at this scale it comfortably beat early
concat instead. Phase 8's Transformer now has a much higher bar than the plan anticipated.

**P7-4 frame AP = 0.9183 ± 0.0180**, the first localization-relevant number in the project.
⛔ It survived its validity check: a positional prior (frame index alone, fitted on train)
scores 0.0744 against a 0.0606 chance floor and explains only **8.1%** of it
(`scripts/26_check_frame_confound.py`, `reports/frame_confound.json`). The frame head is
locating forgeries by content, not by clock position.

**⛔ But PF-20's collapse survived the BiLSTM, and that is the finding that matters most.**
Reliance on video went 0.0021 → **0.0020**: removing the entire visual stream costs nothing.
Frame AP without video is 0.8986 vs 0.9012 — the *localization* is audio-driven too. Recurrence
over an already-collapsed representation cannot un-collapse it, which is now measured rather
than argued. `visual_only` fakes score AUC 0.9702 with the visual pathway contributing nothing;
their audio is unmodified, so this is **PF-19's fingerprint, stronger than at Phase 6**.
Cross-attention (P8-3) is the last untried defence — see PF-21.

### 🔬 Phase 8 result — Experiment E (2026-09-05)

**E > D (+0.0045), but attention did not earn it.** `reports/experiment_e.md`, gate P8-10 passed.

| arm | dev AUC | sd | frame AP | lift | −visual |
|---|---|---|---|---|---|
| D (Phase 7, no sync) | 0.9883 | 0.0022 | 0.9183 | — | +0.0020 |
| D + sync loss | 0.9922 | 0.0018 | 0.9022 | 56.46× | +0.0028 |
| + cross-attention | 0.9920 | 0.0048 | 0.8847 | 53.17× | +0.0010 |
| + self-attention | 0.9917 | 0.0038 | 0.8613 | 25.84× | +0.0056 |
| **E full (§5.1 model)** | **0.9928** | 0.0011 | 0.8998 | 0.06× | +0.0010 |

⛔ **The decomposition is the point.** Measured against `D + sync` so the sync loss is held
constant: sync loss **+0.0039**, cross-attention **−0.0002**, self-attention **−0.0005**, both
together +0.0005. Seed spread is 0.0022, so every attention term is **below seed variance and
not significant** (§7.3 criterion 1). `E − D` is real; it is P8-5's InfoNCE sync loss, not
P8-1's Transformer or P8-3's cross-attention. Section 5.3 predicted global comparison would
beat sequential memory — at this scale it did not.

⛔ **P8-9's "attention elevates on forged spans" is seed-dependent.** Lift per seed is
0.06, 0.06, **59.17** — runs identical but for the seed, within 0.003 dev AUC, split between
attention that sits almost entirely on the forged span and attention that almost entirely
avoids it. Whether attention finds the forgery is **decoupled from classification
performance**, so no single attention figure represents the architecture. Median quoted; the
mean describes none of the runs. Frame AP also *fell* at every attention arm (0.9183 → 0.8613–0.8998).

⛔ **PF-21 is answered, negatively.** No arm broke the §5.6 collapse — every configuration
still loses ≈nothing when the visual stream is zeroed (+0.0010 to +0.0056, all under the 0.01
threshold). The project has now tried modality dropout, auxiliary heads, recurrence,
cross-attention and self-attention. **This is a dataset finding, not an architecture one**: on
LAV-DF at this scale the audio pathway is so much easier that no architectural encouragement
makes the visual pathway worth using. Belongs in limitations; PF-19's processing fingerprint
is the leading explanation.

The sync head passes its desync unit test (P8-6) but **does not separate real from fake at
clip level** (−0.0016) — it helps as a training signal, not as a detector.

### 🟢 What unblocked, and how

**⛔ PF-13 was wrong, and it was the only thing holding Phase 4's gate.** It recorded a hard
Kaggle *volume quota* (~300 files/session, signalled as 404) and concluded dev-2k was
"unobtainable locally" — so Phase 4 trained on **58 clips** and the gate stayed at 2/3.

Re-testing showed the endpoint is **rate**-limited, two separate ways:

| Behaviour | Signal | Fix |
|---|---|---|
| Too much concurrency | **404** on files that plainly exist — while a *serial* probe got 200 seconds later | `--workers ≤ 5` |
| Sustained volume | **429** + `Retry-After: 180` | shared `THROTTLE` sleeps on the header |

PF-13's fetcher called `raise_for_status()` and counted every exception as one "quota hit",
so it could not tell those apart and stopped at the first run of either. With the fix,
train went **58 → 592 clips** and both arms cleared the floor decisively. Full write-up in
**PF-16**; PF-13 keeps a superseded banner because Phase 4's first-pass numbers were
produced under it.

### 🔴 What is still blocked

**🔴 PRE-1 — the original report is not on this machine.** Sole blocker for P1-3 and
Appendix-A items 1–7. Blocks **Part 9's comparison and Phase 16**, not Phases 5–15. Drop it
at `docs/original_report.pdf`.

### Resume checklist

1. `make check` — Phase 0 gate, should print **GATE PASSED**, exit 0.
2. `make phase1` — metadata → CL-7 → manifest → subsets → statistics → tests.
3. `make phase2` — models → smoke-100 video → face crops → gate → label check → sheets → overlays.
4. `make phase3` — log-mel + MFCC on the video grid → ⛔ strict alignment gate → label check.
5. `make phase4` — extract both backbones → verify → train 3 arms → D-1 → ⛔ `make confound`.
6. `make phase5` — audio both arms → C-1 → ⛔ `make confound` + `make provenance`.
7. ✋ Watch `reports/figures/overlays/` and the contact sheets — the two things nothing
   automated can sign off (P2-8, P2-10).
8. `make phase6` — fusion + its ⛔ **control arm** → Experiment C → P6-8.
9. `make phase7` — Baseline 4 + BiLSTM → ⛔ `make frame-confound` → Experiment D → P7-6 gate.
10. **Phase 8** — `scripts/27_train_attention.py` (4 arms x 3 seeds) → `scripts/29_attention_maps.py`
    → `scripts/28_experiment_e.py` for Experiment E and the ⛔ P8-10 gate.
11. `make phase9` — F0 → F1 → F2 (F2 needs F1's checkpoint per seed; resumable per seed)
    → `scripts/31_experiment_f.py` for Experiment F and the ⛔ P9-7 gate.

### Risk status

**Phase 1 — all three resolved.** `fake_periods` are float **seconds** (asserted every
load); leakage assertions are **build-failing** with zero overlap on all three pairs; RAM
was never a factor (Phase 1 ran off a 33.8 MB JSON).

**Phase 2 — ground truth is measured, not assumed.** Divergence onset matches the labelled
span start on **7/7 pairs, median error 0.000 s** (one frame = 0.04 s).

⚠️ **Note for anyone using `original` as a reference:** a fake is generally a *different
length* from its original (142 vs 136 frames, 331 vs 310) — LAV-DF replaces a word with a
different word. Only the **leading edge** of divergence is informative.

**Phase 3 — the alignment gate holds at 100%.** One audio frame is one video frame by
construction (`hop_length=640` = 40 ms = one frame at 25 fps), verified on 100/100 files.
Label verification covers **all three fake classes, 12/12, median error 0.000 s**.

**⛔ Phases 4-5 — R4 is live, in two distinct forms.** Each modality has a class it is
physically blind to, and `scripts/18_check_confound.py` scores every arm on its own:

| arm | blind class | length alone | model | verdict |
|---|---|---|---|---|
| visual / resnet18 | `audio_only` | 0.6381 | 0.6296 | length fully explains it |
| visual / mobilenet_v2 | `audio_only` | 0.6381 | **0.7551** | ⛔ exceeds length |
| audio / logmel | `visual_only` | 0.5916 | **0.9728** | ⛔ far exceeds length |
| audio / mfcc | `visual_only` | 0.5916 | 0.5365 | ✅ at chance |

Two separate artefacts, with different fixes: **clip length** (real clips are shorter,
worth AUC 0.6157 on its own) and a **global audio processing fingerprint** (PF-19, measured
8/8). R3 (identity leakage) is clean: 0 `source_id` overlap between train and dev. X-7
requires re-checking at every scale-up — these scale-ups are what exposed both.

### Carried-forward open items

| Item | Blocks | Notes |
|---|---|---|
| ⛔ PF-17 — clip-length shortcut | Part 11, Phase 6 | Settle a length-matched dev view before fusion |
| ⛔ PF-19 — audio reads a global processing fingerprint | Part 11, Phase 6 | log-Mel scores 0.9728 on `visual_only`, whose audio is unmodified; 8/8 pairs diverge from t=0. A re-encode control would quantify it |
| ⛔ **PF-21** — §5.6 collapse survived the BiLSTM, **and Phase 8** | Part 11 limitations | **ANSWERED 2026-09-05, negatively.** Cross-attention was the last untried defence and it failed too: zeroing video costs +0.0028 (sync), **+0.0010 (cross)**, +0.0056 (self), +0.0010 (E full) — all under the 0.01 threshold. Modality dropout, auxiliary heads, recurrence, cross-attention and self-attention have now all been tried. **A dataset finding, not an architecture one** — on LAV-DF at this scale the audio pathway is too easy for the visual one to be worth using. Goes in limitations; PF-19 is the leading explanation |
| ⛔ **PF-22** — PF-19's fingerprint is stronger at D | Phase 8, Part 11 | `visual_only` fakes score AUC **0.9702** with the visual pathway contributing nothing, and their audio is unmodified by definition. Phase 7's 0.9883 is an upper bound; the `D − C` delta is the trustworthy part |
| **PF-23** — `average_precision` broke ties by row order | reports on disk | Fixed 2026-09-05 to resolve tie groups like `roc_auc` always has. Trained-model AP is unaffected (sigmoid outputs are never exactly tied), but the **`majority_class` Baseline-0 floor is a constant by construction**: Phases 4-6 recorded `ap 0.7383`, the correct value is **0.7200** (= the positive rate). Stale in three report JSONs; no headline moves. Mattered because a saturated frame head would have scored near-perfect frame AP |
| **PF-24** — reliance on video is measured only on LAV-DF dev | Phase 10–11 | Phase 9's audio-untouched `visual` splices are the one training signal that could force video use, but `dev_drop_visual` is scored on LAV-DF fakes where audio suffices. A held-out spliced dev view (by modality) would show whether F1/F2 learned any visual skill at all |
| **PF-26** — host RAM kills long runs | Phase 12+ | `make phase10` was reaped mid-run at ≈1.5 GB free, and Phase 11's Experiment I seed 2 again (2026-10-03); resume from `last.pt` exact both times. Close Chrome before long runs |
| ⛔ **PF-27** — default training is not bit-reproducible | all new training | Re-run of G seed 0 drifted ~1 seed-sd. `--strict-determinism` (`torch.use_deterministic_algorithms`) makes re-runs bit-identical; use it from now on |
| **PF-28** — sync loss costs boundary precision; modality dropout hides unhelpful video use | Part 11 limitations, Phase 16 | G − H = −0.096 AP@0.75 (significant). I relies on video for localization (0.060 AP@0.5) but does worse on visual-only fakes |
| ⛔ **Test split is spent** | Phases 12–16 | `reports/test_once.lock`. Accuracy re-checks after optimization (P12-5) run on dev |
| **PF-25** — P9 augmentation hurts on dev | Phase 10 | F1 − F0 = −0.0045 AUC / **−0.0534 frame AP**, significant on 3/3 seeds. Default Experiment G to no augmentation |
| PF-18 — BatchNorm sees padding | Phase 6+ | `--norm group` ablation exists, not yet measured |
| 🔴 PRE-1 — original report missing | **P1-3, Part 9, Phase 16** | Not on this machine; drop at `docs/original_report.pdf`. The only thing keeping the P1-20 gate at 3/4 |
| dev-2k is 698/2000 local | tighter Phase 4 numbers | Wall-clock only, ~500 files/hour. Does not change any Phase 4 conclusion |
| ⛔ PF-10 — crops are 7.52 MB/video | Phase 9+ | Streaming `--from-video` solves it; features are 228–569 KB/video |
| ✋ P2-8 / ⛔ P2-10 manual checks | Phase 2 sign-off | Rendered in `reports/figures/`. Backed by 7/7 objective label verification but still needs a human watch |
| ✋ Attach the mirror in a Kaggle notebook | dev-10k and above | Manual browser step (*Add Data*). No longer blocks subset-scale work (PF-16) |
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
| 4 Visual baseline | 12 | ✅ **12/12** — gate 3/3; D-1 ✅ MobileNetV2; ⛔ PF-17 caveat |
| 5 Audio baseline | 8 | ✅ **8/8** — gate passed; C-1 ✅ log-Mel; ⛔ PF-19 caveat |
| 6 Fusion | 8 | ✅ **8/8** — gate passed on a **negative** result: C < B, collapsed onto audio (PF-20) |
| 7 Temporal | 6 | ✅ **6/6** — gate P7-6 passed; D − C = **+0.0844**; frame AP **0.9183** |
| 8 Self-attention | 10 | ✅ **10/10** — gate P8-10 passed; E − D = +0.0045, all of it the sync loss |
| 9 Augmentation | 7 | ✅ **7/7** — gate P9-7 passed; F2 − F1 = +0.0009 (**not significant**); F1 − F0 = −0.0045 (augmentation hurt) |
| 10 Localization ⭐ | 14 | ✅ **13/14** — gate P10-14 passed; AP@0.5 **0.8522** (dev); arms tie at AP@0.5, focal significantly sharper at AP@0.75; P10-13 awaits ✋ |
| 11 Ablations | 11 | ✅ **11/11** — gate P11-11 passed; **test AUC 0.9842, AP@0.5 0.8005**; H and I not significant at AP@0.5; P11-10 bit-identical in strict mode (PF-27) |
| 12 Optimization | 6 | ⬜ Not started |
| 13 API | 12 | ⬜ Not started |
| 14 Frontend | 12 | ⬜ Not started (also gated by P14-0) |
| 15 Testing | 9 | ⬜ Not started |
| 16 Documentation | 11 | ⬜ Not started (needs PRE-1) |
| Cloud (Kaggle) | 8 | 🟡 2/8 — CL-1 ✅, CL-7 ✅. No longer on Phase 4's critical path (PF-16) |
| Cross-cutting | 7 | 🔄 Continuous |

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

## PHASE 4 — Visual Baseline — ✅ **COMPLETE 12/12, gate 3/3 (2026-08-30)**

⛔ **Gate P4-12 PASSED** — working model ✅ · both backbones clearly beat majority-class ✅ ·
D-1 resolved ✅. Read it together with **PF-17**: the AUC is a clear beat *and* it contains
a clip-length shortcut, so the bare number must not travel alone.

⛔ **D-1 resolved → MobileNetV2**, reversing the 2026-08-29 call. At 58 training clips the
two arms were 0.008 apart — inside noise — so §5.2's speed tie-break decided it and the
prior (ResNet-18) stood. At 592 clips the gap is **0.087**, far outside seed noise, so the
experiment separates them and the higher-scoring arm wins outright.

**What unblocked this: PF-16.** PF-13 recorded a hard Kaggle *volume quota* making dev-2k
"unobtainable locally", and that alone kept the gate at 2/3. It was a misdiagnosis — the
endpoint is **rate**-limited (excess concurrency → 404; sustained volume → 429 with
`Retry-After`). No new capability was needed, just a fetcher that honours the header.

| ID | Task | Status |
|---|---|---|
| P4-1 | Frozen-backbone extractor for **both** ResNet-18 (512-d) and MobileNetV2 (1280-d → 512 for parity) | ✅ `src/models/backbones/visual.py`. Both frozen, `train()` **overridden** so a `Trainer.train()` call cannot flip BatchNorm into batch-statistics mode. Parity comes from the head's `Linear(D→256)`, applied to **both** arms |
| P4-2 | `scripts/04_extract_visual.py` — batched, `no_grad`, fp16 output, resumable | ✅ `scripts/13_extract_visual.py` (04 was taken by Phase 1 stats). ⛔ **`--from-video` streams decode→align→featurise→discard**, so peak disk is one clip, not PF-10's 73.5 GB. **Compute fp32, store fp16** per PF-4 |
| P4-3 | Extract over dev-2k with both backbones; record throughput and peak VRAM | ✅ **786 clips** (698 of dev-2k + smoke-100), both backbones, 5/5 checks. Peak VRAM 179 / 180 MB. Feature size **228 KB/video (ResNet-18) vs 569 KB (MobileNetV2)** — ResNet-18 is 2.5× smaller, as 512-d vs 1280-d implies. ⚠️ End-to-end frames/s is **not comparable across arms this run** (21 vs 533 videos on a resumed cache); `reports/backbone_bench.json` carries the isolated like-for-like figure |
| P4-4 | Determinism check; embedding-norm sanity; t-SNE identity separation | ✅ **5/5 both backbones** (`scripts/16_verify_features.py`). Re-extraction **byte-identical 4/4**; no dead clips; norms sane (median 25.5 / 21.6). Identity separation **0.0015 within vs 0.205 between** |
| P4-5 | Attention-pooling head + classifier | ✅ `Linear(D→256)` → attention-pool → LayerNorm → `Linear(256→64)` → GELU → Dropout → `Linear(64→1)`, single logit + `BCEWithLogitsLoss` per §J. Padding **and** `face_found=False` frames are masked out of pooling |
| P4-6 | Training loop: AMP, grad accumulation, checkpoint every epoch | ✅ `src/training/trainer.py`. Early stopping on **dev** only; atomic per-epoch checkpoints; `resume()` restores model+optimiser+best. AMP present but **off by default** (PF-4) |
| P4-7 | ⛔ **Overfit-a-batch** — 10 samples to ~zero loss | ✅ **PASS on all three arms**: 0.6460 / 0.5403 / 0.6461 → **0.000000**. Runs automatically before every training run and **aborts it on failure** |
| P4-8 | MLflow: git SHA, config, seeds, metrics, checkpoint, hardware, wall-clock | ✅ All logged. ⚠️ **PF-14**: MLflow 3.15 *raises* on the file store the plan specifies; switched to SQLite (`experiments/mlflow.db`) |
| P4-9 | Train B1a (ResNet-18) and B1b (MobileNetV2), 3 seeds each | ✅ 9 runs on an **unchanged 175-clip dev split**, train 58 → 592: **ResNet-18 0.6319 ± 0.0110** (was 0.5667), **MobileNetV2 0.7189 ± 0.0086** (was 0.5746), mean-pool ablation 0.6140 ± 0.0072 |
| P4-10 | Baseline 0 (majority-class + random) as the sanity floor | ✅ majority AUC **0.5000** / acc **0.7200**; random AUC 0.5198. That accuracy is the point: the dev split is 72% fake, so accuracy is worthless as a headline |
| P4-11 | 🔬 Record Experiment J; ⛔ **resolve D-1** with the measurements | ✅ **D-1 → MobileNetV2** (`reports/decision_d1.md`), by the §5.2 rule applied mechanically. See below |
| P4-12 | ⛔ **Gate:** working trained model, both backbones clearly beat majority-class, D-1 resolved | ✅ **3 of 3**, with the PF-17 caveat recorded rather than hidden |

**⛔ D-1 resolved → MobileNetV2.** §5.2's pre-committed tie-break is "take MobileNetV2 iff
|AUC gap| < 0.01 **and** it is ≥2× faster". Both conditions are now False — but the rule is
a tie-break for when the experiment *cannot* separate the arms, and it can:

| Condition | Threshold | 58 clips | 592 clips |
|---|---|---|---|
| AUC gap within noise | < 0.01 | 0.0079 ✅ (tie-break applied) | **0.0870 ❌ (gap is real)** |
| MobileNetV2 ≥2× faster | ≥ 2.0× | 1.10× ❌ | 1.15× ❌ |
| Seed sd intervals overlap | — | yes | **no** |

At 58 clips the arms were indistinguishable, so the prior stood. At 592 the gap is 8.7× the
noise threshold and the sd intervals are disjoint, so the measurement decides it directly.
**The earlier ResNet-18 call was not wrong — it was correct on the evidence then available**,
which is exactly why PF-3 required deciding by experiment.

⚠️ **The backbone is 1.9% of extraction wall-clock**, so §5.2's "the efficiency argument
mostly evaporates under the cached architecture" is confirmed again. ResNet-18 keeps one
real advantage the AUC does not capture: its features are **2.5× smaller on disk**, which
matters for CL-8's 45 GB sharding against a 20 GB `/kaggle/working`.

**✅ PF-15 is vindicated; the earlier contradiction was a small-data artefact.** At 58 clips
mean-pool *beat* attention (0.5764 vs 0.5667), contradicting PF-15's measurement that the
forgery signal is 45× stronger at the peak frame. At 592 clips attention wins as PF-15
predicts (**0.6319 vs 0.6140**). Whether a head can exploit the peak was a training
question, and 58 clips could not answer it.

**⛔ PF-17 — what this gate does *not* prove.** `scripts/18_check_confound.py` /
`reports/confound_check.md`:

- **`audio_only` fakes score 0.7551 — the highest of the three classes.** They are
  **pixel-identical to their originals** (Phase 3), so a visual-only model should be at
  chance. It is **+0.2551 above chance on clips it cannot see the manipulation in.**
- **Real clips are shorter than fakes** (median 171 vs 207–226 frames), so **clip length
  alone reaches AUC 0.6157** — most of the way to the models' scores.
- This is **R4 / §5.6 shortcut learning**, and the scale-up is what surfaced it: at 58
  clips `audio_only` sat at chance because the model could not yet exploit the artefact.
- **R3 is clean**: 0 `source_id` values span train and dev.

➡️ **Phase 5 is the control.** `audio_only` is exactly what audio should catch and vision
should not. If the audio arm does not clearly beat 0.7551 there, the shortcut is doing the
work in both arms. A length-matched dev view should be settled **before Phase 6 fusion**.

**Remaining data, not remaining capability.** 698 of dev-2k's 2,000 clips are local; the
rest is wall-clock at Kaggle's ~500 files/hour (`make fetch-meta ARGS="--subset dev-2k"`).
Re-running at the full 2,000 is one command and does not change any conclusion above — it
tightens them.

---

## PHASE 5 — Audio Baseline — ✅ **COMPLETE 8/8, gate passed (2026-08-30)**

⛔ **Gate P5-8 PASSED** — audio pipeline validated ✅ · C-1 resolved ✅.
⭐ *Milestone: §4.2 calls the project "already defensible here".* It is — but read **PF-19**
before quoting the number, because the margin contains a dataset artefact.

⛔ **C-1 resolved → log-Mel**, confirming section C's prior *by measurement* rather than
leaving it unrebutted: **0.9838 ± 0.0018 vs MFCC's 0.7677 ± 0.0488**, a +0.2161 gap with
disjoint seed intervals — 21× the noise threshold.

| ID | Task | Status |
|---|---|---|
| P5-1 | Dilated stride-1 1D-CNN encoder: 4 × [Conv1d(k=3,pad=1) → BN → ReLU], 80→128→256→256→256, dilation [1,2,4,8] | ✅ `src/models/backbones/audio.py`. Receptive field **31 frames = 1.24 s**, stride 1 throughout. ⛔ **`padding=dilation`, not the plan's literal `pad=1`** — see PF-18 |
| P5-2 | ⛔ Assert output length **exactly** equals input length | ✅ Asserted in `forward()` **and** in `tests/unit/test_audio_encoder.py` — 16 parametrised cases (T = 1…500, both feature dims), plus a test reading the built layers so PF-18's mistake cannot reappear |
| P5-3 | `scripts/05_extract_audio.py` | ✅ Already Phase 3's `scripts/12_extract_audio.py`; re-run over Phase 4's exact corpus. **786 clips × 2 arms, 0 alignment violations, 0 failures, 0 silent** |
| P5-4 | Train B2a (MFCC) and B2b (log-Mel), 3 seeds each | ✅ 6 runs, `scripts/19_train_audio.py`. **log-Mel 0.9838 ± 0.0018**, **MFCC 0.7677 ± 0.0488**. Same 786 clips and the same unchanged dev split as Phase 4, so the modalities are directly comparable |
| P5-5 | Masked loss so padding never leaks; normalization stats on **train split only** | ✅ Padding masked in the encoder (re-zeroed after every block, so dilation cannot bleed it into valid frames), in pooling, and in the loss. Normalisation is **per-utterance CMVN**, which is stronger than train-only stats: statistics are computed *within* each clip, so no cross-clip quantity exists to leak at all |
| P5-6 | Overfit-a-batch test | ✅ **PASS both arms**, 0.7157 → 0.000000. Runs before every training run and aborts it on failure. Load-bearing here in a way it was not in Phase 4: this encoder trains from scratch rather than sitting on frozen features |
| P5-7 | 🔬 Record Experiment K; ⛔ **resolve Decision C-1** | ✅ **C-1 → log-Mel** (`reports/decision_c1.md`), by a rule fixed before the numbers were seen |
| P5-8 | ⛔ **Gate:** audio pipeline validated, C-1 resolved | ✅ **Passed**, with PF-19 recorded against the number |

**⭐ The PF-17 control passes — Phase 4's shortcut is real but audio is not riding it.**
The visual arm scored **0.7551** on `audio_only` fakes it is pixel-blind to. Audio scores
**0.9912** on that same class. The modality that can genuinely see those forgeries is far
ahead, which is what Phase 4's number needed as a check.

**⛔ PF-19 — the mirror-image control fails, and it matters more.** `visual_only` fakes have
**unmodified audio** by the dataset's own labels, so an audio model must be at chance:

| arm | `visual_only` | reading |
|---|---|---|
| clip length alone | 0.5916 | the PF-17 shortcut |
| **MFCC** | **0.5365** | ✅ physically correct — *below* the length baseline |
| **log-Mel** | **0.9728** | ⛔ scores on clips whose audio was never manipulated |

Length cannot explain it — both arms see identical lengths and MFCC does not exploit it.
So this was measured rather than argued (`scripts/21_check_audio_provenance.py`): comparing
8 `visual_only` fakes against the real video each names in `original`, by Phase 2's
divergence-onset method, **8/8 diverge from t = 0** with a median identical prefix of **0
samples**, while the labelled edits sit at a median of **3.85 s**. The audio stream is not a
faithful copy of its original *anywhere*, long before any manipulation. A global,
perfectly label-correlated difference exists, and log-Mel preserves exactly the fine
spectral structure needed to read it while MFCC's DCT discards it.

➡️ **C-1 still stands** — log-Mel is the better arm on the task as measured, and the rule was
pre-committed. But part of *why* it wins is that it reads the artefact better, so the choice
is right and the margin is not trustworthy.

**⛔ Section 5.6's modality-collapse risk is now concrete, not hypothetical.** Audio 0.9838
against vision 0.7189 gives a fusion model every incentive to ignore video entirely — the
failure the plan says "quietly defeats the project's premise". **Phase 6 needs the
modality-dropout diagnostic in its first run**, not as a later ablation.

**➡️ Phase 10 rises in importance.** A per-frame task cannot use clip length, and a *global*
processing fingerprint is spread across the clip rather than concentrated in the forged span.
Localization AP is the metric this project can most defend.

---

## PHASE 6 — Multimodal Fusion — ✅ **COMPLETE 8/8, gate passed on a negative result (2026-08-30)**

⛔ **Gate P6-8 PASSED.** It asks for the multimodal premise to be *"empirically supported,
**or the failure understood and documented**"*. The premise is **not** supported at this
scale — and the failure is measured rather than guessed, which is what the gate requires.
See **PF-20**.

| ID | Task | Status |
|---|---|---|
| P6-1 | Early-concat fusion → MLP (`src/models/fusion/concat.py`) = Baseline 3 | ✅ visual→256 ‖ audio→256 → `Linear(512→256)` → GELU → attention-pool → logit. Same head as Phases 4/5 verbatim, so the comparison measures the *modality*, not head capacity |
| P6-2 | Temporal alignment of the two cached streams on the shared index grid | ✅ `MultimodalFeatureDataset`. The grids already agree by construction (`hop_length=640` = 40 ms = one frame at 25 fps), so this is a truncation to `min(T_v, T_a)` — **asserted per clip** against Phase 3's ±1, not assumed |
| P6-3 | Per-modality feature normalization | ✅ Each stream projected then LayerNorm'd. Frozen ImageNet activations and CMVN'd log-mel arrive on wildly different scales; a raw concat would let the larger-norm stream win the first `Linear` by arithmetic. Tested: a 100× louder visual stream changes the fused std by <10× |
| P6-4 | Implement **modality dropout** (p=0.2 per stream) now, not later | ✅ Per *clip*, not per frame — the failure defended against is a global preference. ⚠️ **It backfired; see PF-20** |
| P6-5 | Per-modality auxiliary heads | ✅ A `_Head` on each stream, `aux_weight=0.3`. Tested that the visual aux gradient reaches `visual_proj` **and does not leak** into the audio encoder |
| P6-6 | 4-class diagnostic breakdown reporter | ✅ On every run, plus the section 5.6 modality ablation (zero each stream at inference) |
| P6-7 | 🔬 Run Experiment C; verify **C > max(A,B)** — or investigate honestly why not | ✅ **C − max(A,B) = −0.0800.** Investigated, not buried |
| P6-8 | ⛔ **Gate:** multimodal premise supported, or the failure understood | ✅ Passed on the second clause |

**🔬 Experiment C — three negative results, all measured.**

| arm | dev AUC |
|---|---|
| A visual only (MobileNetV2) | 0.7189 ± 0.0086 |
| **B audio only (log-Mel)** | **0.9838 ± 0.0018** |
| C fusion, concat + defences | 0.9038 ± 0.0275 |
| C control, defences off | 0.9098 ± 0.0249 |

**1. Fusion is worse than audio alone** — by 0.0800, more than its own seed spread. Adding a
weak, partly artefactual visual stream to a strong audio one costs accuracy.

**2. ⛔ The model collapsed onto audio.**

| stream zeroed | dev AUC | drop |
|---|---|---|
| visual | 0.9017 | **+0.0021 ± 0.0066** |
| audio | 0.6568 | +0.2471 ± 0.0239 |

Removing video costs nothing distinguishable from zero — one seed *improved* without it.
Exactly §5.6's "decent aggregate number and a model whose multimodal claim is hollow".

**3. ⚠️ The collapse defences made it worse.** The control arm relies on video **more** than
the defended one (+0.0128 ± 0.0042, positive on all three seeds, vs +0.0021 ± 0.0066 which
straddles zero). §5.6 calls modality dropout "the single most effective intervention"; here it
was counter-productive. Plausible mechanism: dropout zeroes an *already weak* visual stream on
20% of clips, so the model learns it is unreliable and downweights it further. The
intervention assumes both modalities carry comparable signal — on LAV-DF one does not.

**Not tuned until it won.** Adjusting dropout or aux weight against dev until fusion beat
audio would be fitting the dev split, which §3.5 RULE 4 exists to prevent.

➡️ **This is a verdict on *early concat*, not on fusion.** §G's recommended bidirectional
cross-attention (Phase 8, Experiment E) models "does audio at *t* explain video at *t*"
rather than concatenating two opinions, so it has a mechanism for a weak visual stream that
concat lacks. Phase 6's job was to give it a documented baseline to beat, and it has.

➡️ **The real fix is upstream.** The visual arm is weak partly because it reads clip length
(PF-17). A length-matched evaluation and a stronger visual signal change this experiment's
premise; more fusion capacity will not.

---

## PHASE 7 — Temporal Modeling (🔬 Experiment D) — ✅ **COMPLETE 6/6, gate passed (2026-09-05)**

Baseline 4 = Phase 6's fusion + a 2-layer BiLSTM, reusing `ConcatFusionBaseline.encode_streams`
verbatim so `D − C` isolates the recurrence. 3.91 M params, 3 seeds, same 786 clips and dev split.

| ID | Task | Status |
|---|---|---|
| P7-1 | 2-layer BiLSTM, 256 hidden (`src/models/temporal/lstm.py`) = Baseline 4 | ✅ `src/models/temporal/lstm.py` — `PackedBiLSTM` 2×256 bidirectional |
| P7-2 | `pack_padded_sequence` for variable length; unit-test that **no gradient flows through padding** | ✅ packed; asserted **two** ways — zero gradient at padded inputs, and identical output batched vs alone |
| P7-3 | Add the **per-frame head** (`[T,256] → [T,1]`) — first localization-relevant output | ✅ `src/models/heads/frame.py` + `src/localization/targets.py` |
| P7-4 | Add frame-level AP as a tracked metric | ✅ **0.9183 ± 0.0180**; ⛔ validity: positional prior explains only 8.1% |
| P7-5 | 🔬 Run Experiment D; verify D > C | ✅ **D 0.9883 vs C 0.9038 → +0.0844**, beyond seed spread |
| P7-6 | ⛔ **Gate:** temporal contribution measured independently of attention | ✅ ⛔ **GATE PASSED** — `attention: false` declared, controls match, 3 seeds |

---

## PHASE 8 — Self-Attention (🔬 Experiment E) — 🟢 **10/10 complete (2026-09-05)**

⚠️ **E changes two components at once** (concat→cross-attention, BiLSTM→Transformer), so
`AttentionFusionModel` exposes them as independent flags and P8-10 reads all four arms.
Otherwise `E − D` would be unattributable — the failure P7-6 exists to prevent.

| ID | Task | Status |
|---|---|---|
| P8-1 | 4-layer pre-norm Transformer encoder, 4 heads, d_model=256, d_ff=512, dropout 0.1, max_T=750 | ✅ `models/temporal/transformer.py`, pre-norm, hand-rolled to expose attention weights |
| P8-2 | Learned positional encoding + positional-encoding ablation | ✅ learned PE + ablation, tested by permutation-equivariance |
| P8-3 | **Upgrade fusion to bidirectional cross-attention** (2 layers, A→V and V→A, concat + project) | ✅ `models/fusion/cross_attention.py` |
| P8-4 | Implement the sync head (component F): two 256-d projections → per-timestep cosine over ±5-frame window | ✅ `models/heads/sync.py`, max-cosine over ±5 frames |
| P8-5 | Implement auxiliary **InfoNCE sync loss** (τ=0.07), positives = aligned pairs from **real** videos, negatives = shifted ≥10 frames | ✅ `src/losses/sync.py`, real videos only, within-clip negatives |
| P8-6 | ⛔ **Desync test** — shift audio +400 ms on a real video and assert the sync score drops significantly. If not, the component is not working regardless of the loss curve | ✅ ⛔ **PASSES** — aligned >0.99, +400 ms desync <0.5, 3-frame nudge >0.9 |
| P8-7 | Training stability: warmup, pre-norm, gradient clipping at 1.0; attention head-entropy check for collapse | ✅ pre-norm + grad clip 1.0; head entropy `(B, layers, heads)` free every forward |
| P8-8 | Attention-map visualization over the timeline (`notebooks/03_attention_viz.ipynb`) | ✅ `scripts/29_attention_maps.py` → 14 figures in `reports/figures/attention/`. Written as a script, not a notebook: the pre-commit hook strips notebook outputs, so the figures would vanish in review |
| P8-9 | 🔬 Run Experiment E; verify E > D beyond seed variance; confirm attention elevates on forged spans | ✅ **E 0.9928 vs D 0.9883 → +0.0045**, beyond the 0.0022 seed spread. ⛔ But the gain is the **sync loss** (+0.0039); attention is −0.0002/−0.0005, below seed variance. Lift is **seed-dependent** (0.06, 0.06, 59.17) — not confirmed |
| P8-10 | ⛔ **Gate:** attention contribution measured, attention maps produced | ✅ ⛔ **GATE PASSED** — 4 arms × 3 seeds, controls match D, maps present |

---

## PHASE 9 — Augmentation (🔬 Experiments F0 / F1 / F2) — ✅ **COMPLETE 7/7, gate passed (2026-09-19)**

| ID | Task | Status |
|---|---|---|
| P9-1 | Implement **manipulation splicing** first — splice a real span from video X into video Y with exactly-known boundaries (likely the biggest single win) | ✅ `training/augment.py`, real donor + real recipient, quarter-frame inset fixes a 1338/15000 boundary round-trip bug. Measured: **not** a win — see P9-6 |
| P9-2 | Implement classical augmentation: spec-augment, temporal jitter, crop/colour jitter | ✅ in feature space (time masking, jitter, `feature_noise`, `frame_dropout`); pixel crop/colour jitter deliberately replaced — the cache holds embeddings. Per-epoch re-randomisation wired via `set_epoch` |
| P9-3 | Implement the conditional feature-space GAN: `G: (noise, label) → [T,256]` + sequence discriminator, spectral norm | ✅ `models/gan/`, spectral norm, per-dim whitening, seed fed to every GRU step, MSGAN mode-seeking (weight 1.0) |
| P9-4 | Monitor GAN output diversity; inspect for mode collapse **before** using any samples | ✅ ⛔ ratios vs real batch, raises `ModeCollapseError`. All 3 seeds admitted: pairwise 1.15 / 1.16 / 0.72 (floor 0.50) |
| P9-5 | 🔬 Run **all three arms**: F0 (none) / F1 (classical) / F2 (classical + GAN), 3 seeds each | ✅ 9 runs, `experiments/phase9_F{0,1,2}/seed{0,1,2}`, `reports/phase9_F*.json` |
| P9-6 | ⛔ Report **F2 vs F1** honestly — the only valid GAN claim. F2 ≈ F1 → "the GAN added nothing beyond classical augmentation" is a legitimate published result | ✅ ⛔ **F2 − F1 = +0.0009, not significant** (seeds −0.0008/+0.0044/−0.0008). Headline: *the GAN added nothing beyond classical augmentation*. F1 − F0 = −0.0045 — augmentation hurt |
| P9-7 | ⛔ **Gate:** three arms run, conclusion recorded honestly | ✅ ⛔ **GATE PASSED** — `reports/experiment_f.md`, 3 arms × 3 seeds, controls match, F0 reproduces E full |

---

## PHASE 10 — Temporal Localization ⭐ (🔬 Experiment G) — ✅ **13/14, gate passed (2026-10-02)**

| ID | Task | Status |
|---|---|---|
| P10-1 | ⛔ GT alignment: `fake_periods` → per-frame `[T]` target, with the **inverse-transform assertion** (target → intervals matches original within 1 frame) | ✅ ⛔ `scripts/32_check_gt_alignment.py` over the full manifest: **99,873/99,873** fakes within one frame, worst 0.900 frames (`reports/gt_alignment.json`) |
| P10-2 | Assert `target.sum() == 0` for every real video, exactly | ✅ **36,431/36,431** real targets exactly zero |
| P10-3 | Per-frame head + **focal loss** (α=0.25, γ=2.0) | ✅ `losses/localization.py`; arm `focal`, 3 seeds |
| P10-4 | Optional boundary head `[T,2]` (start-ness / end-ness) vs Gaussian-smoothed targets + BoundaryLoss | ✅ `models/heads/boundary.py`; arm `full`, 3 seeds |
| P10-5 | Combined loss `λ_cls·BCE + λ_loc·Focal + λ_sync·InfoNCE + λ_bnd·Boundary` starting at (1.0, 2.0, 0.3, 0.5); tune λ on dev | ✅ `full_l1` / `full_l4` / `full_b1`, seed 0: AP@0.5 0.8712 / 0.8522 / 0.8378 vs 0.8634 at the default — all inside `full`'s 3-seed spread (0.0187), so **defaults kept** |
| P10-6 | Post-processing chain (§6.4): median smooth(5) → threshold τ → binary closing(3) → extract runs → min-duration 0.4 s → merge gaps < 0.3 s → per-segment confidence → seconds | ✅ `localization/postprocess.py` |
| P10-7 | ⛔ Implement AP/AR for 1-D intervals and **unit-test against hand-computed toy cases** (perfect → 1.0; IoU 0.4 @ thr 0.5 → 0.0; one correct + one spurious → exact hand value). Cross-check against an established implementation | ✅ ⛔ `evaluation/localization.py` + `tests/unit/test_localization_metrics.py`; cross-checked against the LAV-DF authors' evaluator (`avdeepfake1m` 0.0.4): 465 cases, **0 unexplained** (`reports/ap_crosscheck.json`) |
| P10-8 | Synthetic known-span test → recovered IoU > 0.9 | ✅ `tests/unit/test_postprocess.py` |
| P10-9 | Property tests: segments sorted, non-overlapping, within `[0, duration]` | ✅ `tests/unit/test_postprocess.py`; `assert_segment_contract` also runs on every dev prediction |
| P10-10 | Grid-search τ / median kernel / min-duration / merge-gap on **dev** maximizing AP@0.5, then **freeze** | ✅ ⛔ 7,776 configs; frozen to `configs/postprocess_frozen.json` (arm `full`: τ 0.2, kernel 5, min dur 0.2 s, gap 0.0 s). Flat top. Tuning bought +0.24 AP@0.5 over §6.4's defaults |
| P10-11 | Implement long-video chunking: 750-frame chunks, 125-frame overlap, average scores in overlap, post-process **once** on the stitched sequence | ✅ `localization/chunking.py` |
| P10-12 | Measure mean boundary error (ms) and **false-positive segment rate on real videos** | ✅ headline arm: **78 ± 16 ms**, FP rate on real **0.000**. Focal: 61 ± 2 ms, 0.048 |
| P10-13 | ✋ Visually compare predicted timelines against ground truth for 20 videos | ✋ **Awaiting you** — `reports/figures/localization/timelines_dev20.png` |
| P10-14 | ⛔ **Gate:** AP@{0.5,0.75,0.95} + AR@{100,50,20,10} computed with a *verified* AP implementation, post-processing frozen before test | ✅ ⛔ **GATE PASSED** — `reports/experiment_g.md`, 6/6 checks, test split never scored |

---

## PHASE 11 — Ablation Experiments — ✅ **COMPLETE 11/11, gate passed (2026-10-03)**

| ID | Task | Status |
|---|---|---|
| P11-1 | Build `src/evaluation/*` as **pure functions** — predictions + manifest → metrics, decoupled from training | ✅ `src/evaluation/runs.py`; Experiment G re-run through it reproduces its committed numbers exactly. `src/inference/predict.py` + `scripts/36` saved dev predictions for E and F0–F2 (rebuilt models match saved predictions to ~1e-6) |
| P11-2 | Run the full matrix A–K × 3 seeds (0,1,2) | ✅ A–G and J/K reused from Phases 4–10 (3 seeds each); H and I trained here. `make phase11` runs the protocol in order. The script is `37_evidence_table.py` (`08_` was already taken) |
| P11-3 | 🔬 Experiment H — ablate the sync loss (λ_sync=0); verify G > H | ✅ `experiments/phase11_H_nosync`. G − H = **+0.0023 AP@0.5, not significant** (spread 0.0287). But −0.096 AP@0.75, significant: the sync loss costs boundary precision (PF-28) |
| P11-4 | 🔬 Experiment I — ablate modality dropout; use the 4-class breakdown to detect collapse | ✅ `experiments/phase11_I_nomd`. G − I = **+0.0175 AP@0.5, not significant**. I uses video for localization (0.060 AP@0.5 lost without it) yet visual-only AP@0.5 falls 0.68 → 0.57 (PF-28) |
| P11-5 | Compute mean ± std for every headline claim | ✅ Every arm, every metric, in `reports/evidence_table.json` |
| P11-6 | Apply the **four §7.3 criteria** to every claimed improvement | ✅ 13 claims, paired by seed; deciding metric per criterion 3; params + train time per criterion 4 |
| P11-7 | Fill Table 8.2.1, 8.2.2, 8.2.4, 8.2.5 | ✅ `reports/evidence_table.md` (8.2.3 is Phase 12's) |
| P11-8 | ⛔ **Run the test split exactly once**, at the very end | ✅ ⛔ `scripts/38_test_once.py` — 392 clips, lock written before predicting, re-run refused. **AUC 0.9842, AP@0.5 0.8005, AP@0.75 0.5392** (`reports/test_once.md`) |
| P11-9 | ⛔ Include **negative results**; mark sub-seed-variance differences "not significant" | ✅ 2 claims hurt, 8 not significant — all listed |
| P11-10 | Reproducibility: re-run one experiment from its logged config, match to ~1e-4 | ✅ in strict mode — **bit-identical** (`reports/reproducibility_strict.json`). ❌ default mode drifts ~1 seed-sd (`reports/reproducibility.json`) → **PF-27** |
| P11-11 | ⛔ **Gate:** complete evidence table, every number traceable to a logged run | ✅ ⛔ **GATE PASSED** — 6/6 checks; every arm traced to its run dirs + git SHAs |

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

*Completed as of 2026-09-05: ~100 of 203 tasks (≈49%). Phases −1, 0, 1, 2, 3, 4, 5, 6, 7, 8 done.
**D-1 → MobileNetV2, C-1 → log-Mel**, Experiment C is a documented negative result (PF-20),
**Experiment D a decisive positive one** (+0.0844, gate P7-6 passed) whose frame AP survived its own
confound check (position explains 8.1%), and **Experiment E a mixed one** (+0.0045, gate P8-10
passed) where the decomposition assigns the gain to the **sync loss** and shows both attention
components below seed variance. CL-1 and CL-7 complete. 319 passing tests.
Outstanding: ⛔ PF-17/PF-19/PF-22 — the audio fingerprint **survived the BiLSTM and the Transformer**,
so no Phase 4–8 absolute number may enter Part 11's evidence table; only deltas may. ⛔ PF-21 is now
answered negatively and is a **limitations-section finding**: nothing tried breaks the §5.6 collapse.
P1-3 is blocked on PRE-1; two ✋ Phase 2 manual checks remain. dev-2k is 698/2000 local (PF-16).*
