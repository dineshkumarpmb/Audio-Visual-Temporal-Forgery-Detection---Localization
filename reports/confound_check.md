# Validity check — how much of each baseline is forgery evidence?

Dev clips: **175**. Majority-class floor: **0.5000**.

Each modality has a class it is **physically blind to**. A model scoring well above
chance there is reading an artefact, not a forgery.

| arm | blind class | length alone | model | above chance | verdict |
|---|---|---|---|---|---|
| visual / resnet18 | `audio_only` | 0.6381 | 0.6296 | +0.1296 | ⛔ **above chance** |
| visual / mobilenet_v2 | `audio_only` | 0.6381 | 0.7551 | +0.2551 | ⛔ **above chance** |
| audio / logmel | `visual_only` | 0.5916 | 0.9728 | +0.4728 | ⛔ **above chance** |
| audio / mfcc | `visual_only` | 0.5916 | 0.5365 | +0.0365 | ✅ at chance |

## Artefact 1 — clip length

Real clips are shorter than fakes (median **171** frames vs 225, 226, 207), so length alone reaches **AUC 0.6157**.

| class | median frames | AUC from length alone |
|---|---|---|
| real | 171 | — |
| visual_only | 225 | 0.5916 |
| audio_only | 226 | 0.6381 |
| both | 207 | 0.6132 |

## Artefact 2 — processing history

An arm that beats **both** chance and the length baseline on its blind class has found
something in the signal itself. The cleanest evidence that these are two different
artefacts is that the two audio arms disagree on the same clips:

| arm | `visual_only` (audio content untouched) |
|---|---|
| length alone | 0.5916 |
| audio / logmel | 0.9728 |
| audio / mfcc | 0.5365 |

MFCC sits at chance on `visual_only`, which is the physically correct answer — and it
sees exactly the same clip lengths as log-Mel. So log-Mel's score there is **not** the
length shortcut; it is fine spectral structure that MFCC's DCT discards. Section C
predicted log-Mel would preserve precisely that structure. The catch is what it is
reading it *for*: on `visual_only` clips the speech was never re-synthesised, so what
remains to detect is that the file was **rebuilt** — a processing fingerprint, which
correlates perfectly with the label in LAV-DF and would not survive a dataset where
real clips were re-encoded too.

## What is clean

- **R3 (identity leakage): clean.** 0 `source_id` values
  appear in both train and dev.
- **The dev split is unchanged** across every run compared here, so all arms are scored
  on identical clips.

## What to do about it

1. Do not carry any bare AUC from Phases 4–5 into Part 11's evidence table; each needs
   its blind-class figure beside it.
2. **Before Phase 6 fusion.** Fusing two arms that share the length artefact compounds
   it. Settle a length-matched dev evaluation first.
3. **Section 5.6's modality-collapse risk is now concrete, not hypothetical.** The audio
   arm is far stronger than the visual one, so a fusion model has every incentive to
   ignore video — the failure mode that would quietly defeat the project's premise.
   Phase 6 needs the modality-dropout diagnostic from the start.
4. Phase 10's localization is the natural corrective: a per-frame task cannot use clip
   length, and a processing fingerprint is spread over the whole clip rather than
   concentrated in the forged span.
