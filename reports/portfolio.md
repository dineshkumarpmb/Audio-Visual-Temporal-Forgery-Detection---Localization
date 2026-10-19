# Resume and portfolio text (P16-8)

The plan's Part 14 templates, filled from measured numbers only. Every figure carries a source
tag checked by `scripts/46_check_docs.py`. **Three template lines could not be filled honestly**
— the measurement contradicted them — and are rewritten below with the reason; use the rewritten
versions, never the originals.

## §14.3 fill-in tracker

| Placeholder | Value | Source | Status |
|---|---|---|---|
| `[BACKBONE]` | MobileNetV2 (AUC +0.087<!--n:reports/evidence_table.json#claims/0/metrics/auc/delta|sign3--> over ResNet-18) | Experiment J | ✅ |
| ROC-AUC / F1 | 0.984<!--n:reports/test_once.json#summary/auc/mean|3--> / 0.967<!--n:reports/test_once.json#summary/f1/mean|3--> | test, 3 seeds | ✅ |
| AP@0.5 / AP@0.75 | 0.80<!--n:reports/test_once.json#summary/ap@0.5/mean|2--> / 0.54<!--n:reports/test_once.json#summary/ap@0.75/mean|2--> | test, 3 seeds | ✅ |
| Boundary error | 97<!--n:reports/test_once.json#summary/boundary_error_ms/mean|int--> ms | test | ✅ |
| FP-segment rate | 2.6<!--n:reports/test_once.json#summary/fp_video_rate/mean|pct1-->% of real clips | test | ✅ |
| Latency, model size | 4.45<!--n:reports/benchmark.json#variants/optimized/s_per_10s/median|2--> s per 10 s clip, 31.1<!--n:reports/model_export.json#size_mb|1--> MB | Table 8.2.3 | ✅ |
| Ablation deltas | see `results.md` §7.3 — 3 of 13 claims helped | Phase 11 | ✅ |
| Experiment count | 66 logged training runs, 26 arms | `experiments/*/seed*/result.json` | ✅ |

Strongest unimodal baseline (audio only, dev): AUC 0.984<!--n:reports/evidence_table.json#summary/B/auc/mean|3-->.

## Primary (one line)

> Built a multimodal audio-visual deepfake **detection and temporal localization** system in
> **PyTorch** — a frozen **MobileNetV2** face encoder, a dilated 1D-CNN log-Mel audio encoder,
> **cross-modal attention and a Transformer** — reaching **0.984**<!--n:reports/test_once.json#summary/auc/mean|3--> ROC-AUC and
> **0.80**<!--n:reports/test_once.json#summary/ap@0.5/mean|2--> AP@0.5 for forged-segment localization on a held-out LAV-DF test split,
> with **97**<!--n:reports/test_once.json#summary/boundary_error_ms/mean|int--> ms mean boundary error and **4.45**<!--n:reports/benchmark.json#variants/optimized/s_per_10s/median|2--> s
> end-to-end inference per 10 s clip on a 4 GB GPU.

*Changed from the template:* it asked for the final model's AUC "versus ___ for the strongest
unimodal baseline". On dev the audio-only baseline scores 0.984<!--n:reports/evidence_table.json#summary/B/auc/mean|3--> too — the comparison would have
been a non-difference presented as a result, so it was removed.

## Detailed (bulleted)

> - Built an end-to-end audio-visual deepfake **detection and temporal localization** pipeline
>   (PyTorch, FastAPI, React) on LAV-DF, reaching **0.984**<!--n:reports/test_once.json#summary/auc/mean|3--> ROC-AUC and
>   **0.80**<!--n:reports/test_once.json#summary/ap@0.5/mean|2--> AP@0.5 on a test split scored exactly once.
> - Designed a **cached two-stage architecture** — features extracted once, the model trained on
>   them — that made a **66-run, 3-seed ablation** feasible on a 6 GB RAM / 4 GB GPU laptop, at
>   minutes per run.
> - Ran a **13-claim ablation** under a pre-registered significance rule and **reported the
>   failures**: early fusion lost to audio alone (AUC −0.080<!--n:reports/evidence_table.json#claims/2/metrics/auc/delta|sign3-->), GAN augmentation
>   added nothing (+0.001<!--n:reports/evidence_table.json#claims/8/metrics/ap@0.5/delta|sign3--> AP@0.5), and only the backbone, the audio
>   features and temporal context helped.
> - **Diagnosed a modality collapse** with per-manipulation-type evaluation: removing the entire
>   video stream costs 0.0016<!--n:reports/test_once.json#summary/visual_reliance_auc/mean|4--> AUC, traced to a whole-soundtrack processing
>   fingerprint in the dataset — and documented it as the system's main limitation.
> - Protected every localization number with an **exact-arithmetic ground-truth test** (which
>   caught a float-rounding bug shifting 2,938 label boundaries by a frame) and an AP metric
>   cross-checked against scikit-learn and the dataset authors' evaluator; 82% test coverage, CI.
> - Served it as a **FastAPI** service with streamed uploads and a one-job-at-a-time queue, and a
>   **React** front end with a forged-span timeline and a downloadable PDF report.

*Changed from the template:* (1) "self-attention over LSTM contributed +___ AP" — measured
+0.015<!--n:reports/evidence_table.json#claims/6/metrics/frame_ap/delta|sign3--> frame AP, inside seed noise: not a contribution. (2) "cross-modal
fusion +___ AUC over the best unimodal baseline" — measured −0.080<!--n:reports/evidence_table.json#claims/2/metrics/auc/delta|sign3-->: a loss. (3) "modality
dropout to prevent unimodal collapse, verified by…" — it did not prevent it; the bullet now
claims what was actually done, which was to find and measure the collapse.

## Interview one-liner

> "The original project classified whole videos. I rebuilt it to say *where* the forgery is —
> and to prove which parts earned their place. Most didn't: the model turned out to read the
> soundtrack and ignore the face, and I can show you exactly how I measured that."

## Using these numbers

- Quote **test** numbers, with "LAV-DF, dev-2k subset" and "3 seeds". Dev numbers are higher and
  were used for tuning.
- Do not compare the 95.2<!--n:reports/test_once.json#summary/accuracy/mean|pct1-->% accuracy to the original report's 80%: the
  original protocol is unknown (`results.md`, Part 9).
- If asked "does it work on real-world deepfakes?" the answer is *not measured* — one dataset,
  one family of generators (`final_report.md` §5).
