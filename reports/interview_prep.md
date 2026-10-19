# Interview preparation (P16-9)

The twelve questions from plan §15.16, answered from what was measured. Numbers carry source
tags checked by `scripts/46_check_docs.py`; every answer ends with where the evidence lives.

---

**1. Why multimodal instead of visual-only?**

A forgery can change the face, the voice or both, and a visual-only model is physically blind to
a voice-only edit. Measured, the visual-only arm reached AUC 0.719<!--n:reports/evidence_table.json#summary/A/auc/mean|3-->
against 0.984<!--n:reports/evidence_table.json#summary/B/auc/mean|3--> for audio-only. The honest follow-up: *naive* fusion (early concat)
did worse than audio alone — AUC −0.080<!--n:reports/evidence_table.json#claims/2/metrics/auc/delta|sign3--> — so "multimodal" only paid off once
temporal context was added. *Evidence: `results.md` Table 8.2.1; Experiments A, B, C.*

**2. Why self-attention rather than an LSTM?**

The plan predicted global comparison would beat sequential memory. It did not, at this scale:
Transformer over BiLSTM was +0.015<!--n:reports/evidence_table.json#claims/6/metrics/frame_ap/delta|sign3--> frame AP, inside seed noise. The
Transformer stayed because the pre-committed selection rule picked the configuration containing
it, and it handles long clips by chunking cleanly — not because it measurably helped.
*Evidence: `results.md` §7.3, claim H-1; `reports/experiment_e.md`.*

**3. How do you know the model isn't just reading audio?**

I don't — I know that it *is*. Zeroing the whole video stream costs
0.0016<!--n:reports/test_once.json#summary/visual_reliance_auc/mean|4--> test AUC. The four-class breakdown shows face-only fakes still
scoring 0.966<!--n:reports/test_once.json#four_class/visual_only/auc/mean|3--> AUC — because LAV-DF re-synthesises the whole soundtrack of
every fake, leaving a fingerprint even where the voice wasn't edited. Six defences were tried;
none broke the collapse. It's the first limitation in the report. *Evidence:
`final_report.md` §5; Table 8.2.4; PF-19, PF-21.*

**4. How did you prevent data leakage?**

Official identity-disjoint splits, and assertions that fail the build if any speaker identity
appears in two splits — checked at every scale-up, zero overlap each time. Separately, a
positional-confound check: frame index alone explains only
8.1<!--n:reports/frame_confound.json#positional_share_of_model|pct1-->% of the frame head's AP, so localization reads content, not clock
position. *Evidence: `reports/confound_check.md`, `reports/frame_confound.json`, plan §3.5.*

**5. Why AP instead of accuracy for localization?**

Forged frames are a small minority — about 6<!--n:reports/frame_confound.json#forged_frame_rate|pct0-->% of dev frames — so "everything
is real" scores 94% frame accuracy while localizing nothing. Segment AP at an IoU threshold
rewards finding the span *and* placing it, and punishes false segments on real videos.
*Evidence: plan §3.6, §8.1; `src/evaluation/localization.py`.*

**6. How did you pick the threshold?**

A grid search on dev over threshold, median kernel, closing, minimum duration and merge gap,
maximizing mean AP@0.5 across seeds — then frozen to `configs/postprocess_frozen.json` before
the test split was ever scored. It was the biggest single localization lever: +0.241<!--n:reports/experiment_g.json#tuning_gain_ap50/full|3-->
AP@0.5 over the plan's defaults. *Evidence: `reports/experiment_g.md`, `reports/test_once.md`.*

**7. The report says ResNet-18 in one place and MobileNetV2 in another — what did you do?**

Measured both under the same protocol (Experiment J). MobileNetV2 won by AUC
+0.087<!--n:reports/evidence_table.json#claims/0/metrics/auc/delta|sign3-->, far outside seed noise, so the speed tie-break never applied.
Same for the other contradiction, MFCC vs log-Mel: log-Mel by +0.216<!--n:reports/evidence_table.json#claims/1/metrics/auc/delta|sign3-->.
*Evidence: `results.md` Table 8.2.5; decisions D-1, C-1.*

**8. Did the GAN augmentation actually help?**

No. A three-arm design (no augmentation / classical / classical + GAN) isolates it: the GAN added
+0.001<!--n:reports/evidence_table.json#claims/8/metrics/ap@0.5/delta|sign3--> AP@0.5, with the seeds disagreeing on the sign. Classical splicing
augmentation itself *lowered* frame AP (−0.053<!--n:reports/evidence_table.json#claims/7/metrics/frame_ap/delta|sign3-->). *Evidence:
`reports/experiment_f.md`, PF-25.*

**9. How would you scale this to production?**

First, the time is not in the model: end to end is 4.45<!--n:reports/benchmark.json#variants/optimized/s_per_10s/median|2--> s per 10 s clip,
of which 3.84<!--n:reports/benchmark.json#variants/optimized/median_stage_s_per_10s/detect|2--> s is decoding and CPU face landmarks and
0.05<!--n:reports/benchmark.json#variants/optimized/median_stage_s_per_10s/model|2--> s is the model — so ONNX would not help, and was measured not
to. Scaling means GPU decoding, batched landmarking, and horizontal workers behind a queue (the
API already serialises jobs). But I'd fix the audio shortcut before shipping anything.
*Evidence: `reports/benchmark.md`, PF-29.*

**10. What's the biggest weakness?**

Two, and they compound: it ignores the face, and everything was measured on one dataset with
one family of generators. Whether it detects a real-world deepfake is not measured.
*Evidence: `final_report.md` §5.*

**11. Why 40 ms hops?**

At 16 kHz, a hop of 640 samples is exactly 40 ms — exactly one video frame at 25 fps. Audio frame
*t* and video frame *t* describe the same instant, so cross-modal alignment is an array index,
verified 100/100 files, with no interpolation to get wrong. *Evidence: plan §C;
`reports/preprocessing_verification.md`.*

**12. How do you know your AP implementation is right?**

Three independent checks: hand-computed cases; agreement with the dataset authors' own
evaluator on random problems (`scripts/33_crosscheck_ap.py`); and, in CI, property tests matching
scikit-learn on tie-heavy inputs plus a from-the-definition reference for interval AP. One bug
was real: frame AP once broke ties by row order (PF-23), caught and fixed. *Evidence:
`tests/unit/test_ap_metric.py`, `reports/ap_crosscheck.json`.*
