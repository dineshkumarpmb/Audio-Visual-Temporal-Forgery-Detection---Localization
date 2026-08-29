# Appendix A — resolution of the 15 `TO VERIFY` items (task P1-4)

PROJECT_PLAN Appendix A lists 15 items that had to be resolved in Phase 1 "or explicitly
escalated as unresolvable". This is that accounting. Nothing below is inferred: each row
either cites the measurement that settled it or names the blocker that prevents it.

**Status: 8 resolved · 1 deferred by design · 6 escalated (all blocked on PRE-1).**

| # | Item | Blocking? | Status |
|---|---|---|---|
| 1 | Provenance of the **53.35%** existing-system figure | Yes | 🔴 **ESCALATED** — PRE-1 |
| 2 | Evaluation protocol behind the **80.00%** figure | Yes | 🔴 **ESCALATED** — PRE-1 |
| 3 | Existing-system precision / recall / F1 | Yes | 🔴 **ESCALATED** — PRE-1 |
| 4 | Whether the report used any localization metrics | Yes | 🔴 **ESCALATED** — PRE-1 |
| 5 | Report's training hyperparameters | No | 🔴 **ESCALATED** — PRE-1 |
| 6 | Sliding windows or dense prediction | No | 🔴 **ESCALATED** — PRE-1 |
| 7 | Report's stated inference-latency target | No | 🔴 **ESCALATED** — PRE-1 |
| 8 | LAV-DF video counts and class balance | Yes | ✅ **RESOLVED** |
| 9 | LAV-DF download size | Yes | ✅ **RESOLVED** |
| 10 | **Mean forged-span duration** | Yes | ✅ **RESOLVED** |
| 11 | Fake-frame fraction | Yes | ✅ **RESOLVED** |
| 12 | Exact metadata field names | Yes | ✅ **RESOLVED** |
| 13 | Whether an identity key exists for leakage checks | Yes | ✅ **RESOLVED** (with a caveat) |
| 14 | fps / sample-rate consistency | Yes | ✅ **RESOLVED** |
| 15 | Full-dataset extraction wall-clock | No | ⏸️ **DEFERRED** — Phase 2, by the plan's own instruction |

**Every blocking item that does not require the original report is resolved.** All six
escalations trace to the same single cause, and none of them block Phases 2–15.

---

## Resolved

### #8 — Video counts and class balance

Measured from `metadata.min.json` (136,304 entries) and cross-checked against the mirror's
own file listing, which matched as an **exact set** — no file in metadata missing from the
mirror, none on the mirror absent from metadata.

| | Count |
|---|---|
| Total | 136,304 |
| Real / fake | 36,431 / 99,873 |
| train / dev / test | 78,703 / 31,501 / 26,100 |
| real / visual_only / audio_only / both | 36,431 / 33,543 / 33,170 / 33,160 |

Fake sub-types are near-uniform (33.2k–33.5k), so the four-class diagnostic of §8.2.4 is
well-powered. The real:fake ratio is roughly 1:2.7 at the video level — the *opposite*
direction from the frame-level imbalance in #11. See `dataset_statistics.md`.

### #9 — Download size

**25,502,577,573 bytes = 25.50 GB (23.75 GiB)**, summed over all 136,307 listed files.

This also resolves the discrepancy flagged at CL-1: the Kaggle API's `total_bytes` reports
**24,842,461,534 B (23.14 GiB)**, which is the *compressed* dataset. The uncompressed content
sums to 25.50 GB, matching the published ~25.6 GB. There is no missing data. Moot for disk
planning either way — decision PF-6 means none of it lands on D:.

### #10 — Mean forged-span duration (determines `T`, §6.2)

Over **114,253 spans**:

| min | p05 | p25 | median | mean | p75 | p95 | max |
|---|---|---|---|---|---|---|---|
| 0.164 s | 0.244 s | 0.424 s | 0.664 s | **0.650 s** | 0.864 s | 1.144 s | **1.600 s** |

**At 25 fps the mean span is 16.2 frames and the longest in the entire dataset is 40.**
This is the single most design-relevant number in Phase 1: any temporal receptive field or
post-processing median filter wider than ~1.6 s cannot resolve a single span, and smoothing
at that scale erases the thing being localized.

### #11 — Fake-frame fraction (drives focal loss / `pos_weight`, §3.6)

**1,861,304 forged frames of 28,880,994 = 6.4447%** → BCE `pos_weight` ≈ **14.52**.

Note this points the opposite way from #8: **73.3% of videos are fake, but only 6.4% of
frames are.** An unweighted localization head predicts all-real and scores 93.6% frame
accuracy while being useless. Frame accuracy must never be a headline metric.

### #12 — Exact metadata field names

The file is **`metadata.min.json`**, not `metadata.json`. Confirmed against the authors'
own `README.md`, shipped inside the mirror. Eleven fields:

```
file · n_fakes · fake_periods · duration · original · modify_video ·
modify_audio · split · video_frames · audio_frames · audio_channels
```

The full `metadata.json` (137.6 MB) adds only `timestamps` and `transcript`, neither of
which this project uses — so the 33.8 MB minified file is the correct artefact to parse.

`fake_periods` is confirmed as **float seconds** (P1-5), not the `List[List[int]]` frame
indices the authors' loader type-annotates. See `dataset_statistics.md` for the evidence.

### #13 — Identity key for leakage checks — **resolved, with a caveat that must be read**

**No explicit identity field exists.** `source_id` is therefore *derived* by walking the
`original` pointer to its root (P1-7). The derivation is exact and self-validating: it
yields **36,431 roots — precisely the number of real videos**, and the root set equals the
real-video set exactly. Every fake resolves to the real video it was built from.

The three R3 assertions pass with **zero overlap** on all three split pairs:

| | train ∩ dev | train ∩ test | dev ∩ test |
|---|---|---|---|
| Shared `source_id` | 0 | 0 | 0 |

⚠️ **What this does not prove.** It proves no fake is split away from its own original. It
does **not** prove that two *different* real videos of the same VoxCeleb2 speaker sit in the
same split — `metadata.min.json` carries no speaker key, so that cannot be tested from
metadata at all. §3.5 RULE 2 names speaker memorisation as a distinct hazard from
near-duplicate leakage; only the second is closed here. Recorded rather than glossed, and
re-checked at every scale-up per X-7.

### #14 — fps / sample-rate consistency

Uniform across all 136,300 usable entries, and verified against the actual media on 28
stratified videos:

| Check | Result |
|---|---|
| `r_frame_rate` == 25.00 | **28/28** |
| `video_frames` == ffprobe `nb_frames` | **28/28** |
| sample rate == 16,000 Hz | **28/28** |
| audio channels == 1 (mono) | 136,304/136,304 |

Byte-exact frame counts mean the Kaggle mirror is **not a re-encode** — the specific hazard
CL-7 exists to rule out.

⛔ **But `duration` is not a media duration.** For all 136,304 entries,
`duration == audio_frames/16000 + 0.128` exactly, and it exceeds both stream durations on
every file probed. The video timeline is `video_frames / 25`. This invalidates task P2-1 as
written; see **decision PF-7**.

---

## Deferred by design

### #15 — Full-dataset extraction wall-clock

Appendix A marks this "No — measure in Phase 2", and it cannot honestly be answered earlier:
it requires the decode-and-detect pipeline that Phase 2 builds. Phase 0 measured the input to
that estimate (decode throughput 894–1509 fps synthetic, 36–60× realtime); the extraction
figure itself waits for P2-15.

---

## Escalated — items 1–7, all blocked on PRE-1

**Single cause:** the original project report is not on this machine. Searched `D:\`,
Downloads, Desktop, Documents and OneDrive — see `report_audit.md`. Drop it at
`docs/original_report.pdf` and these become answerable in one sitting.

**Impact, stated precisely:**

- **Items 1–4 are blocking for Part 9 only** — the "target improvement" comparison against
  the existing system. Without provenance for the 53.35% and 80.00% figures, §0.4 Finding 3
  stands: those numbers cannot be used as baselines, and Part 9's arithmetic has nothing
  valid to rest on.
- **Items 5–7 are non-blocking** by Appendix A's own marking; they affect how faithfully the
  rebuild mirrors the original, not whether it is correct.
- **Phases 2–15 are unaffected.** Nothing in preprocessing, modelling, localization,
  evaluation or serving depends on the report. Only **Phase 16** (documentation) and Part 9's
  comparison table do.

**Recommended handling if the report never surfaces:** drop Part 9's comparison entirely
rather than compare against unsourced numbers, and state in the README that the rebuild
reports absolute performance on a documented protocol with no claim of improvement over the
original. That is a defensible position; quoting 53.35% without provenance is not.
