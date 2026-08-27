# LAV-DF — Dataset Access Report

**Task:** PRE-4 (Phase −1)
**Date:** 2026-08-12
**Status:** ✅ Research complete — **no manual approval gate exists**.
**Updated 2026-08-27:** §9 superseded by decision **PF-6** — attach on Kaggle, no local download.

---

## 1. Headline finding — Risk R2 is largely eliminated

`PROJECT_PLAN.md` §15.17 rates **R2 (LAV-DF access delayed or unavailable)** as *Medium likelihood / High impact*,
and Phase 1's contingency was "start Phases 2–3 on any small AV video set and swap LAV-DF in later."

**That contingency is not needed.** There is no application form, no email request, and no human
review step. Access is:

- **Google Drive / OneDrive** — open links, no gating at all.
- **HuggingFace** — requires a logged-in account and a single click-through acceptance of the terms.
  Auto-granted; no maintainer approval queue.

**Revised R2:** Likelihood *Low*, Impact *High*. The residual risk is download bandwidth/interruption,
not permission.

---

## 2. Official sources

| Source | URL |
|---|---|
| GitHub repo (code + baselines) | https://github.com/ControlNet/LAV-DF |
| HuggingFace dataset | https://huggingface.co/datasets/ControlNet/LAV-DF |
| Google Drive | https://drive.google.com/drive/folders/1U8asIMb0bpH6-zMR_5FaJmPnC53lomq7?usp=sharing |
| OneDrive (Monash) | https://monashuni-my.sharepoint.com/:f:/g/personal/zhixi_cai_monash_edu/EklD-8lD_GRNl0yyJJ-cF3kBWEiHRmH4U5Dtg7eJjAOUlg?e=wowDpd |
| Terms & Conditions | https://github.com/ControlNet/LAV-DF/blob/master/TERMS_AND_CONDITIONS.md |

**Papers:** DICTA 2022 (arXiv:2204.06228) · CVIU 2023 journal version (arXiv:2305.01979)
**License:** CC BY-NC 4.0

---

## 3. Terms you are agreeing to

Four conditions, all acceptable for this project:

1. **Non-commercial research and educational use only.** ← A portfolio/interview project qualifies.
   Note this rules out any commercial productization of models trained on it.
2. **You must also accept the VoxCeleb2 agreements** (LAV-DF is built on VoxCeleb2) and acknowledge
   the dataset's deepfake origin.
3. **No warranties** of any kind from the authors.
4. **You indemnify the authors** against claims arising from your use.

> **Action:** cite the DICTA 2022 and CVIU 2023 papers in `reports/final_report.md` and `README.md`,
> and state the CC BY-NC 4.0 / non-commercial restriction in the README. Recorded as a Phase 16 item.

---

## 4. ⛔ RESOLVED — Appendix A item #9: download size

| | |
|---|---|
| **Total download size** | **25.6 GB** (per the HuggingFace dataset card) |
| C: free space | 27 GB → **would be catastrophic; do not download to C:** |
| D: free space | 167 GB → **comfortable** |

**Decision:** download to `D:\...\data\raw\`. After extraction the working set is
raw (25.6 GB) + face crops (~15–25 GB at dev-10k) + features (~3.3 GB) ≈ **45–55 GB peak**,
still well within 167 GB. Per §12.4, face crops get deleted after visual extraction.

This resolves a **blocking** `TO VERIFY` item and supersedes the "check before downloading"
caveat in §3.2 and §12.4.

---

## 5. ⛔ RESOLVED — Appendix A item #12: exact metadata field names

The plan's §3.3 guess was close but **wrong in three ways**. Confirmed from the authors' own loader
(`dataset/lavdf.py` in the LAV-DF repo):

```python
@dataclass
class Metadata:
    file: str  # e.g. "train/000001.mp4"
    n_fakes: int  # ← NOT in the plan's §3.3 guess
    fake_periods: List[List[int]]  # list of [start, end]
    duration: float
    original: Optional[str]  # None for real videos
    modify_video: bool
    modify_audio: bool
    split: str  # "train" | "dev" | "test"
    video_frames: int
    audio_channels: int  # ← NOT in the plan's §3.3 guess
    audio_frames: int
```

### Differences from the plan's §3.3 assumption

| # | Plan assumed | Reality | Consequence |
|---|---|---|---|
| 1 | File named **`metadata.json`** | File is **`metadata.min.json`** | Parser path must be updated (P1-5) |
| 2 | No `n_fakes` field | `n_fakes: int` exists | Free cross-check: assert `n_fakes == len(fake_periods)` — a cheap label-integrity test to add in P1-6 |
| 3 | No `audio_channels` field | `audio_channels: int` exists | Use it to pre-flag multi-channel audio before Phase 3's forced mono downmix (P3-1) |
| 4 | `"split"` key assumed | ✅ confirmed correct | Official splits are in the metadata — §3.5 Rule 1 is directly enforceable |
| 5 | `"original"` key assumed | ✅ confirmed, and typed `Optional[str]` | **This is the `source_id` derivation path** — see §6 below |

### ⚠️ One discrepancy to verify on download

`fake_periods` is type-annotated `List[List[int]]` (**int**), but §3.3 and all of Part 6 assume
**float seconds**. Either the annotation is loose (most likely — durations are floats and the
published example shows `[[2.44, 3.88]]`) or the values are frame indices.

**This is not cosmetic.** If they are frame indices and the code treats them as seconds, every
localization target is wrong by a factor of ~25 and the project silently fails.

**Action (P1-5):** print 10 raw `fake_periods` entries immediately after parsing and assert
`max(end) <= duration` — if values exceed duration, they are frame indices. Added to the
Phase 1 task list as a hard check.

---

## 6. Partial progress — Appendix A item #13: identity key for leakage checks

`original: Optional[str]` is confirmed present, which means the §3.5 Rule 3 derivation path exists:
follow `original` to its root to obtain `source_id`.

**Still `TO VERIFY` (P1-7):** whether a *VoxCeleb2 speaker identity* is recoverable. `original`
links a fake to its real source video, which prevents **near-duplicate leakage** — but if two
*different* real videos share one speaker and land in different splits, **identity leakage remains
possible**. The official split is likely already identity-disjoint (the paper reports 153 unique
subjects), but §3.5 Rule 3 says *assert it, do not trust it*.

Resolve in Phase 1 by checking whether the VoxCeleb2 speaker id is embedded in filenames or
recoverable from the VoxCeleb2 metadata.

---

## 7. Bonus finding — a legitimate published baseline for Part 9

The LAV-DF repo publishes the authors' own results on the official test split:

| Method | AP@0.5 | AP@0.75 | AP@0.95 | AR@100 | AR@50 | AR@20 | AR@10 |
|---|---|---|---|---|---|---|---|
| BA-TFD | 79.15 | 38.57 | 0.24 | 67.03 | 64.18 | 60.89 | 58.51 |
| BA-TFD+ | 96.30 | 84.96 | 4.44 | 81.62 | 80.48 | 79.40 | 78.75 |

**Why this matters for Part 9.** §0.4 Finding 3 flags that the report's 53.35% "existing system"
figure may be an invalid comparison. These numbers are a *real, published, same-dataset,
same-protocol* reference for the **localization** metrics — which is exactly what the original
report never measured (§9.3).

**Use them as context, not as a target.** Both are fully-trained end-to-end models on the *full*
dataset with far more compute than a 4 GB GTX 1650 running a frozen-feature Stage-B head on a 10k
subset. Reporting "BA-TFD+ reaches 96.3 AP@0.5 on full data; my cached-feature model reaches X on a
10k subset at Y% of the compute" is an honest and strong framing. Silently comparing against them
is not.

Also note the **AP@0.95 collapse (0.24 and 4.44)** even for the authors' own models — this
independently confirms §6.6's instruction to "expect low values, and say so" at IoU 0.95.

**Recorded for Phase 1 (P1-3)** as external context for the report audit, and for Phase 11 / 16 as
a comparison row in `reports/results.md`.

---

## 8. Toolchain note affecting Phase 0

The LAV-DF repo pins **Python ≥3.7, <3.11** and `pytorch_lightning == 1.7.*`.

- Our **Python 3.10** choice (P0-1) is inside their supported range — no conflict. ✅
- We are **not** using PyTorch Lightning (the plan writes a custom trainer in `src/training/`),
  so the 1.7.* pin does not bind us. It only matters if we later reuse their reference code to
  reproduce BA-TFD, which is listed as a stretch goal (§5.3) — not a current task.

---

## 9. Access procedure — **superseded 2026-08-27 by decision PF-6**

> **The 25.6 GB raw dataset is no longer downloaded locally.** It is attached read-only on Kaggle.
> Only the ~3.3 GB dev-10k feature cache ever lands on D:. See `reports/decision_log.md` PF-6.

### 9.1 Current procedure (Kaggle-first)

1. **CL-1** — Kaggle account, confirm GPU quota (~30 h/week, P100 16 GB or T4×2, 12 h/session).
2. **CL-2** — *Add Data* → attach `elin75/localized-audio-visual-deepfake-dataset-lav-df`.
   It mounts read-only at `/kaggle/input`: no download, no transfer wait, and it does **not**
   consume the 20 GB `/kaggle/working` budget.
3. **CL-7 ⛔** — prove the mirror equals the authors' release *before* building on it. It is a
   community re-upload. Verify file count, real/fake split, `metadata.min.json` integrity, and
   `ffprobe` fps/duration on 20 spot-checks. **A silent re-encode shifts every `fake_periods`
   target** — the same class of failure as the seconds-vs-frames hazard in §5.
4. **CL-3** — thin notebook entrypoint that clones the repo and calls the same `scripts/`.
5. **P1-2** — confirm the real folder structure against §3.3 and note any deviation.

### 9.2 Fallback, if CL-7 fails

Accept the terms on HuggingFace (log in → click through) or use the open Google Drive link, and
upload the authors' copy as a **private Kaggle Dataset**. Still no local 25.6 GB download.

### 9.3 Rejected alternatives

| Option | Why rejected |
|---|---|
| **HuggingFace streaming** | The repo is a single 25.6 GB `LAV-DF.tar`, not sharded parquet/webdataset. `streaming=True` gives sequential access only — fine for a one-pass extraction, useless for shuffled training or stratified subsetting (P1-15) |
| **Colab + Drive shortcut** | ~78 GB disk is ephemeral (re-copy every session); sustained reads off mounted Drive are slow and quota-limited; free GPU allocation unguaranteed |
| **Download 25.6 GB to D:** | Not wrong (156.7 GB free) but buys nothing the mount does not, at the cost of a long interruptible transfer on a machine R13 assumes will be interrupted |

### 9.4 What still needs real video on D:

The ✋ manual checks **P2-8** and **P2-10** require watching actual overlay videos. Either pull the
smoke-100 subset (~200 MB) or render the 20 overlay clips on Kaggle and download those.

---

## Open items carried into Phase 1

| # | Item | Task |
|---|---|---|
| 1 | Are `fake_periods` values seconds or frame indices? | P1-5 |
| 2 | Is a VoxCeleb2 speaker identity recoverable for full leakage checks? | P1-7 |
| 3 | Confirm the actual on-disk folder structure vs §3.3 | P1-2 |
| 4 | Confirm real/fake counts (published: 36,431 / 99,873 of 136,304) | P1-10 |
