# Original Project Report — Audit

**Task:** PRE-1 (Phase −1), completed in Phase 1 task P1-3
**Date opened:** 2026-08-12
**Status:** 🔴 **BLOCKED — the report has not been located on this machine.**

---

## 1. Search performed (2026-08-12)

`PROJECT_PLAN.md` §0.1 records that the report was not readable when the plan was drafted. That
finding was **re-tested independently today** with a wider search, and it still holds.

| Location searched | Filter | Result |
|---|---|---|
| `D:\` root + all top-level folders | names matching `*deepfake*`, `*forgery*`, `*LAV-DF*`, `*lavdf*` | Only `D:\portfolio\...\deepfake-detection.jpg` (a portfolio thumbnail) and this project folder |
| `C:\Users\Admin\Downloads` (depth 3) | `*.pdf`, `*.docx`, `*.doc` > 20 KB | 43 files — payslips, resumes, certificates, course PDFs. **No project report.** |
| `C:\Users\Admin\Desktop` | same | 2 files, both resumes |
| `C:\Users\Admin\Documents` | same | none |
| `C:\Users\Admin\OneDrive` | same | none |
| `D:\College details` | `*.pdf`, `*.docx`, `*.doc`, `*.pptx` | **empty** |
| `D:\My learning`, `D:\Notes` | same | **empty** |
| `D:\Personel learnings` | same | 25 files — all published ML/maths textbooks |
| `D:\Resume`, `D:\New folder (2)` | same | resumes only |

**Conclusion:** the report is not on local storage. It must be retrieved from an external source —
email attachment, college portal / LMS, a cloud drive, a phone, or a printed copy.

---

## 2. Why this blocks real work

Part 1 of the plan is a **reconstruction** from two weak sources (the user's typed summary and a
portfolio marketing blurb). Four Appendix-A items are marked **blocking** and *all four* can only
be answered by the report:

| # | Question | Why it blocks |
|---|---|---|
| 1 | Where did the **53.35%** "existing system" figure come from? | It is 3.35 pts above chance. If it was cited from another paper on a different test set, the entire "80% vs 53.35%" comparison in Part 9 is **invalid** and must be replaced with a baseline we measure ourselves (§0.4 Finding 3) |
| 2 | What evaluation protocol produced the **80.00%**? | Video-level or frame-level? Which split? What threshold? An accuracy with no protocol is not reproducible by definition |
| 3 | Existing-system precision / recall / F1 | Without them, §9.2's improvement figures are **not computable** for 3 of 4 metrics |
| 4 | Did the report use **any** localization metric? | §9.3: if a localization system was scored only with classification metrics, none of the four headline numbers measure the project's central claim — which changes what "improvement" even means |

**Non-blocking but wanted:** training hyperparameters (§1.9), sliding-window vs dense prediction
(§1.10), the stated latency target (§1.5 NFR-02).

---

## 3. Impact assessment — what can proceed without it

**Good news: almost everything.** The report gates *interpretation*, not *construction*.

| Phase | Blocked by the missing report? |
|---|---|
| 0 — Environment | ❌ No |
| 1 — Dataset (manifest, leakage, statistics, subsets) | ❌ No — only the §P1-3 audit sub-task is blocked |
| 2–3 — Preprocessing | ❌ No |
| 4–8 — Baselines, fusion, temporal, attention | ❌ No |
| 9–11 — Augmentation, localization, ablations | ❌ No |
| 12–15 — Optimization, API, frontend, tests | ❌ No |
| **16 — Documentation (Part 9 comparison section)** | ✅ **Yes** |
| **Part 14 resume claims referencing "vs the original"** | ✅ **Yes** |

**Therefore:** the report is **not on the critical path** for Phases 0–15. It becomes required at
Phase 16, and it is needed *earlier* only if you want the Part 9 comparison framed before then.

### Fallback if the report is never recovered

`PROJECT_PLAN.md` §9.4 already anticipates this. Targets T0/T3/T4/T5 (protocol validity,
localization metrics, ablation evidence, efficiency) are **entirely independent** of the report.
Only T1/T2 ("match / exceed the reported 80%") depend on it.

The §9.4 framing — *"the original reported classification metrics only; I rebuilt it with a
validated leak-free protocol, added temporal localization with AP/AR metrics the original never
measured, and ablated every component"* — stands on its own with **no reference to the report at
all**. If the report stays lost, that is the framing, and the project loses nothing essential.

In that case, Baseline 0 (§4.2) plus the published BA-TFD numbers (see `dataset_access.md` §7)
supply the external reference points instead.

---

## 4. Audit checklist — to complete when the report is supplied

Every `TO VERIFY` in Part 1 of `PROJECT_PLAN.md`:

- [ ] §1.2 — Is the existing-system limitation table accurate? Is 53.35% actually stated there?
- [ ] §1.3 — Is the 10-step proposed-system pipeline as reconstructed?
- [ ] §1.5 NFR-02 — Was an inference-latency target stated?
- [ ] §1.7 — **The ResNet-18 / MobileNetV2 contradiction** — confirm both appear and note exactly where
- [ ] §1.9 — Training hyperparameters (optimizer, LR, batch size, epochs, schedule)
- [ ] §1.9 — Was a localization loss included, or classification-only?
- [ ] §1.10 — Sliding windows or dense per-frame prediction?
- [ ] §1.11 — Were ROC-AUC, a confusion matrix, or any AP/AR metric reported?
- [ ] §0.4 F3 — Provenance of **53.35%** (cited / measured / different task formulation)
- [ ] §0.4 F3 — Evaluation protocol behind **80.00%**
- [ ] §9.2 — Existing-system precision / recall / F1, if a comparison table exists

---

## 5. Requested action

Place the report at:

```
D:\Audio-Visual Temporal Forgery Detection & Localization\docs\original_report.pdf
```

(`docs/` is created and git-tracked; the file itself is gitignored as source material.)

Any format works — PDF, DOCX, or even phone photos of the printed pages. Once it lands, this audit
is completed as part of Phase 1 task **P1-3**.

**Until then, Phase 0 proceeds.** The report is not a prerequisite for it.
