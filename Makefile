# Audio-Visual Temporal Forgery Detection & Localization
#
# All targets are .PHONY on purpose. This project lives at a path containing
# spaces and an "&", which GNU make cannot express as a file target — so make is
# used purely as a command runner, never for dependency resolution. Keep every
# path in a recipe relative and quoted.

PY  := .venv/Scripts/python.exe
PIP := $(PY) -m pip

TORCH_INDEX := https://download.pytorch.org/whl/cu118
TORCH_PINS  := torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118

.DEFAULT_GOAL := help
.PHONY: help venv torch install install-dev check check-bench check-kaggle lint fmt \
        typecheck test test-cov fetch-meta manifest subset verify-subsets verify-mirror \
        stats phase1 models faces verify-preproc verify-labels sheets overlays phase2 \
        audio audio-mfcc phase3 visual overfit train-visual decide-d1 confound phase4 \
        train-audio decide-c1 provenance phase5 \
        train-fusion experiment-c phase6 \
        train-temporal frame-confound experiment-d phase7 \
        train-augmented experiment-f phase9 \
        gt-alignment crosscheck-ap train-localization experiment-g phase10 \
        test-features train-ablations evidence test-once reproduce phase11 \
        features train evaluate ablations benchmark api frontend demo clean-bench

help:  ## Show this help
	@echo Audio-Visual Temporal Forgery Detection ^& Localization
	@echo.
	@echo   Setup
	@echo     venv          Create the Python 3.10 virtualenv
	@echo     torch         Install PyTorch from the CUDA 11.8 index (NOT PyPI)
	@echo     install       Install project dependencies
	@echo     install-dev   Install project + dev dependencies + pre-commit hooks
	@echo.
	@echo   Gates
	@echo     check         Phase 0 environment gate
	@echo     check-bench   Phase 0 gate + benchmarks, writes reports/hardware_report.md
	@echo     check-kaggle  CL-1 gate, writes reports/kaggle_report.md
	@echo.
	@echo   Quality
	@echo     lint fmt typecheck test test-cov
	@echo.
	@echo   Pipeline
	@echo     fetch-meta manifest subset verify-subsets verify-mirror stats phase1
	@echo     models faces verify-preproc verify-labels sheets overlays phase2
	@echo     audio audio-mfcc phase3
	@echo     visual overfit train-visual decide-d1 confound phase4
	@echo     train-audio decide-c1 provenance phase5
	@echo     train-fusion experiment-c phase6
	@echo     features train evaluate ablations benchmark
	@echo.
	@echo   Serving
	@echo     api           uvicorn on :8000
	@echo     frontend      Vite dev server on :5173
	@echo     demo          Both together

# ── Setup ────────────────────────────────────────────────────────────────────

venv:  ## Create the venv using Python 3.10
	py -3.10 -m venv .venv
	$(PIP) install --upgrade pip setuptools wheel

torch:  ## Install CUDA 11.8 PyTorch. Driver 526.47 cannot run CUDA 12.x.
	$(PIP) install $(TORCH_PINS) --index-url $(TORCH_INDEX)

install:  ## Install runtime dependencies
	$(PIP) install -e .

install-dev:  ## Install runtime + dev dependencies and register pre-commit hooks
	$(PIP) install -e ".[dev]"
	$(PY) -m pre_commit install

# ── Gates ────────────────────────────────────────────────────────────────────

check:  ## Phase 0 gate — fails non-zero if the toolchain is not ready
# --no-report on purpose: the plain gate has no benchmark numbers to write, and
# letting it regenerate reports/hardware_report.md silently deletes the measured
# fp16/fp32 finding behind decision PF-4. Use check-bench to refresh the report.
	$(PY) scripts/00_check_env.py --no-report

check-bench:  ## Phase 0 gate plus GPU/VRAM/decode benchmarks
	$(PY) scripts/00_check_env.py --bench

# ── Quality ──────────────────────────────────────────────────────────────────

check-kaggle:  ## CL-1 gate - verify Kaggle access. Add --gpu-hours N --phone-verified to attest.
	$(PY) scripts/01_check_kaggle.py $(ARGS)

lint:
	$(PY) -m ruff check .

fmt:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

typecheck:
	$(PY) -m mypy src api

test:
	$(PY) -m pytest

test-cov:
	$(PY) -m pytest --cov=src --cov-report=term-missing --cov-report=html

# ── Pipeline (scripts land in their respective phases) ───────────────────────

fetch-meta: ## Phase 1 - fetch metadata.min.json from the Kaggle mirror (PF-6: no video)
	$(PY) scripts/05_fetch_metadata.py $(ARGS)

manifest:   ## Phase 1 - build + validate manifest_v1.parquet (leakage assertions are fatal)
	$(PY) scripts/02_build_manifest.py $(ARGS)

subset:     ## Phase 1 - build the reproducible 100 / 2k / 10k subsets
	$(PY) scripts/03_make_subset.py $(ARGS)

verify-subsets: ## Phase 1 - fail if a subset drifted from its committed id list (P1-16)
	$(PY) scripts/03_make_subset.py --verify

verify-mirror: ## CL-7 - prove the Kaggle mirror equals the authors' release (gates P1-2)
	$(PY) scripts/06_verify_mirror.py $(ARGS)

stats:      ## Phase 1 - measure the dataset, write reports/dataset_statistics.md
	$(PY) scripts/04_dataset_stats.py --probe-dir data/raw/_spotcheck

phase1:     ## Phase 1 - the whole pipeline, in order, reproducibly
# --force on the manifest step is deliberate. write_manifest() refuses to overwrite so
# that a stray run cannot silently mutate an artefact other stages already read; an
# explicit full-pipeline rebuild is the one case where regeneration IS the intent, and
# it is deterministic, so the rebuilt file is byte-identical unless the data changed.
	$(MAKE) fetch-meta
	$(MAKE) verify-mirror
	$(MAKE) manifest ARGS=--force
	$(MAKE) subset
	$(MAKE) stats
	$(PY) -m pytest tests/ -q

models:     ## Phase 2 - fetch the pinned MediaPipe weights (decision PF-8)
	$(PY) scripts/08_fetch_models.py

faces:      ## Phase 2 - extract aligned face crops (resumable; safe to re-run)
	$(PY) scripts/09_extract_faces.py $(ARGS)

verify-preproc: ## Phase 2 gate - determinism, resumability, detection rate
	$(PY) scripts/10_verify_preprocessing.py $(ARGS)

verify-labels: ## Phase 2 - check fake_periods against pixel evidence (section 6.3)
	$(PY) scripts/11_verify_labels.py $(ARGS)

sheets:     ## Phase 2 - contact sheets for the manual P2-8 check
	$(PY) scripts/viz_contact_sheet.py $(ARGS)

overlays:   ## Phase 2 - burn fake_periods onto video for the manual P2-10 watch
	$(PY) scripts/viz_overlay.py $(ARGS)

phase2:     ## Phase 2 - the whole pipeline, in order
	$(MAKE) models
	$(PY) scripts/05_fetch_metadata.py --subset smoke-100 --with-originals
	$(MAKE) faces
	$(MAKE) verify-preproc
	$(MAKE) verify-labels
	$(MAKE) sheets
	$(MAKE) overlays
	$(PY) -m pytest tests/ -q

audio:      ## Phase 3 - extract log-mel on the video frame grid (resumable)
	$(PY) scripts/12_extract_audio.py $(ARGS)

audio-mfcc: ## Phase 3 - the MFCC arm for Experiment K (separate cache)
	$(PY) scripts/12_extract_audio.py --feature mfcc $(ARGS)

phase3:     ## Phase 3 - audio pipeline + the strict alignment gate
	$(MAKE) audio ARGS=--force
	$(MAKE) audio-mfcc
# --modality both, not audio: the report is one document covering all three fake
# classes, and an audio-only run would overwrite the visual half.
	$(PY) scripts/11_verify_labels.py --modality both
	$(PY) -m pytest tests/unit/test_audio_align.py -q

visual:     ## Phase 4 - extract frozen visual features (ARGS=--backbone mobilenet_v2)
	$(PY) scripts/13_extract_visual.py --from-video $(ARGS)

overfit:    ## Phase 4 - the P4-7 overfit-a-batch gate, on its own
	$(PY) scripts/14_train.py --overfit-only $(ARGS)

train-visual: ## Phase 4 - train the visual baseline, 3 seeds (ARGS=--backbone ...)
	$(PY) scripts/14_train.py $(ARGS)

decide-d1:  ## Phase 4 - resolve Decision D-1 from the measurements
	$(PY) scripts/15_decide_d1.py

confound:   ## Phase 4 - P4-12 validity check (R4 shortcut learning, PF-17)
	$(PY) scripts/18_check_confound.py

phase4:     ## Phase 4 - both backbones end to end, then D-1 and the validity check
# Extract over dev-2k as well as smoke-100: P4-3 specifies dev-2k, and training defaults
# to the union of both. Extraction is resumable and skips videos not yet fetched, so this
# is safe to run against a partially downloaded subset.
	$(PY) scripts/13_extract_visual.py --subset smoke-100 --backbone resnet18     --from-video
	$(PY) scripts/13_extract_visual.py --subset smoke-100 --backbone mobilenet_v2 --from-video
	$(PY) scripts/13_extract_visual.py --subset dev-2k    --backbone resnet18     --from-video
	$(PY) scripts/13_extract_visual.py --subset dev-2k    --backbone mobilenet_v2 --from-video
	$(PY) scripts/16_verify_features.py --backbone resnet18
	$(PY) scripts/16_verify_features.py --backbone mobilenet_v2
	$(PY) scripts/14_train.py --backbone resnet18     --seeds 3
	$(PY) scripts/14_train.py --backbone mobilenet_v2 --seeds 3
	$(PY) scripts/14_train.py --backbone resnet18 --pooling mean --seeds 3
	$(MAKE) decide-d1
# ⛔ PF-17: the gate number is not trustworthy without this. Runs last so it reads the
# freshly written reports/phase4_*.json.
	$(MAKE) confound
	$(PY) -m pytest tests/ -q

train-audio: ## Phase 5 - train one audio arm (ARGS=--feature mfcc)
	$(PY) scripts/19_train_audio.py $(ARGS)

decide-c1:  ## Phase 5 - resolve Decision C-1 from the measurements
	$(PY) scripts/20_decide_c1.py

provenance: ## Phase 5 - PF-19: is visual_only audio a faithful copy of its original?
	$(PY) scripts/21_check_audio_provenance.py

phase5:     ## Phase 5 - audio baseline, both arms, then C-1 and the validity check
# Audio extraction is Phase 3's; re-run it over the same corpus Phase 4 used so the two
# modalities are scored on identical clips. Cheap (~0.09 s/video) and resumable.
	$(PY) scripts/12_extract_audio.py --subset dev-2k    --feature logmel
	$(PY) scripts/12_extract_audio.py --subset smoke-100 --feature logmel
	$(PY) scripts/12_extract_audio.py --subset dev-2k    --feature mfcc
	$(PY) scripts/12_extract_audio.py --subset smoke-100 --feature mfcc
	$(PY) scripts/19_train_audio.py --feature logmel --seeds 3
	$(PY) scripts/19_train_audio.py --feature mfcc   --seeds 3
	$(MAKE) decide-c1
# ⛔ PF-19: log-Mel scores 0.97 on clips whose audio is unmodified. The gate number is not
# trustworthy without these two.
	$(MAKE) confound
	$(MAKE) provenance
	$(PY) -m pytest tests/ -q

train-fusion: ## Phase 6 - train Baseline 3 (ARGS=--no-defences for the control)
	$(PY) scripts/22_train_fusion.py $(ARGS)

experiment-c: ## Phase 6 - Experiment C and the P6-8 gate
	$(PY) scripts/23_experiment_c.py

phase6:     ## Phase 6 - fusion baseline, its control arm, then Experiment C
	$(PY) scripts/22_train_fusion.py --seeds 3
# ⛔ The control arm is not optional. Without it "we defended against modality collapse"
# is a claim about code rather than a measurement -- and it is what showed the defences
# backfiring here (PF-20).
	$(PY) scripts/22_train_fusion.py --seeds 3 --no-defences
	$(MAKE) experiment-c
	$(PY) -m pytest tests/ -q

train-temporal: ## Phase 7 - train Baseline 4, fusion + BiLSTM (ARGS=--frame-weight 0)
	$(PY) scripts/24_train_temporal.py $(ARGS)

frame-confound: ## Phase 7 - P7-4 validity: does frame index alone explain frame AP?
	$(PY) scripts/26_check_frame_confound.py

experiment-d: ## Phase 7 - Experiment D and the P7-6 gate
	$(PY) scripts/25_experiment_d.py

phase7:     ## Phase 7 - temporal baseline, then Experiment D
# ⛔ The gradient check first. A BiLSTM that cannot fit 10 samples is broken in a way no
# amount of real training will localise -- and packing bugs live exactly here.
	$(PY) scripts/24_train_temporal.py --overfit-only
	$(PY) scripts/24_train_temporal.py --seeds 3
# ⛔ Before the gate reads the frame AP, measure what frame position alone scores. PF-17 and
# PF-19 are what happens when a headline is quoted without its shortcut floor.
	$(MAKE) frame-confound
	$(MAKE) experiment-d
	$(PY) -m pytest tests/ -q

train-augmented: ## Phase 9 - train one augmentation arm (ARGS=--arms F1 --seeds 3)
	$(PY) scripts/30_train_augmented.py $(ARGS)

experiment-f: ## Phase 9 - Experiment F and the P9-7 gate
	$(PY) scripts/31_experiment_f.py

phase9:     ## Phase 9 - the three augmentation arms, then Experiment F
# The arm order is a dependency, not a preference. F2's generator is trained on the fused
# embeddings F1's *checkpoint* produces, so the script refuses to run F2 for a seed whose F1
# checkpoint is missing. Both are resumable: a killed sweep re-reads result.json per seed.
	$(PY) scripts/30_train_augmented.py --arms F0 F1 F2 --seeds 3
	$(MAKE) experiment-f
	$(PY) -m pytest tests/ -q

gt-alignment: ## Phase 10 - P10-1/P10-2 ground-truth round trip over the whole manifest
	$(PY) scripts/32_check_gt_alignment.py

crosscheck-ap: ## Phase 10 - P10-7 AP/AR vs the LAV-DF authors' evaluator (REF=<dir>)
# The reference is not a dependency (it pins numpy<2). Fetch it into a scratch dir:
#   pip download avdeepfake1m==0.0.4 --no-deps -d <tmp>
#   python -m zipfile -e <tmp>/avdeepfake1m-0.0.4-*.whl <REF>
	$(PY) scripts/33_crosscheck_ap.py --reference "$(REF)"

train-localization: ## Phase 10 - train localization arms (ARGS=--arms full --seeds 3)
	$(PY) scripts/34_train_localization.py $(ARGS)

experiment-g: ## Phase 10 - tune + freeze post-processing on dev, Experiment G, gate P10-14
	$(PY) scripts/35_experiment_g.py

phase10:    ## Phase 10 - round trip, the three arms, lambda tuning, then Experiment G
# Evaluation reads only the saved dev predictions, so experiment-g can be re-run (a metric
# fixed, a figure redrawn) without retraining. crosscheck-ap needs REF and runs separately.
	$(MAKE) gt-alignment
	$(PY) scripts/34_train_localization.py --arms bce full focal --seeds 3
	$(PY) scripts/34_train_localization.py --arms full_l1 full_l4 full_b1 --seeds 1 --skip-overfit
	$(MAKE) experiment-g
	$(PY) -m pytest tests/ -q

features:   ## Phases 2-5 — Stage-A extraction (resumable; safe to re-run)
	$(PY) scripts/04_extract_visual.py
	$(PY) scripts/05_extract_audio.py

train:      ## Stage-B training. Override: make train CONFIG=configs/experiment/exp_e.yaml
	$(PY) scripts/06_train.py --config $(or $(CONFIG),configs/default.yaml)

evaluate:
	$(PY) scripts/07_evaluate.py --config $(or $(CONFIG),configs/default.yaml)

test-features: ## Phase 11 - fetch + extract the dev-2k TEST split only (train/dev untouched)
# Training loaders use every clip that has features, so fetching more train/dev clips would
# silently change the ablations' training set relative to Experiment G's. Test only.
	$(PY) scripts/05_fetch_metadata.py --subset dev-2k --splits test --max-hours 2
	$(PY) scripts/13_extract_visual.py --subset dev-2k --backbone mobilenet_v2 --from-video
	$(PY) scripts/12_extract_audio.py --subset dev-2k --feature logmel

train-ablations: ## Phase 11 - Experiments H (-sync) and I (-modality dropout), 3 seeds
	$(PY) scripts/34_train_localization.py --arms H_nosync I_nomd --seeds 3 --skip-overfit

evidence:   ## Phase 11 - the A-K evidence table, section 7.3 on every claim, gate P11-11
	$(PY) scripts/37_evidence_table.py

test-once:  ## ⛔ P11-8 - score the test split exactly once (refuses to run twice)
	$(PY) scripts/38_test_once.py --final

reproduce:  ## P11-10 - re-train G seed 0 (default + twice strict), compare to ~1e-4
# PF-27: the default mode (cuDNN deterministic only) is shown to drift by ~1 seed-sd; the
# strict mode (torch.use_deterministic_algorithms) is shown to reproduce.
	$(PY) scripts/34_train_localization.py --arms full --seeds 1 --out-root experiments/_repro --skip-overfit
	-$(PY) scripts/39_check_reproducibility.py --rerun experiments/_repro/phase10_full/seed0
	$(PY) scripts/34_train_localization.py --arms full --seeds 1 --out-root experiments/_repro_strict_a --skip-overfit --strict-determinism
	$(PY) scripts/34_train_localization.py --arms full --seeds 1 --out-root experiments/_repro_strict_b --skip-overfit --strict-determinism
	$(PY) scripts/39_check_reproducibility.py --original experiments/_repro_strict_a/phase10_full/seed0 --rerun experiments/_repro_strict_b/phase10_full/seed0 --out reports/reproducibility_strict.json

ablations:  ## Phase 11 - ablations, dev evidence, reproducibility, then the single test run
# Order is the protocol: every dev decision is made before test-once, which refuses to run
# until H and I exist. E/F predictions are inference-only re-runs of saved checkpoints.
	$(PY) scripts/36_dump_dev_predictions.py --arms phase8_full phase9_F0 phase9_F1 phase9_F2
	$(MAKE) train-ablations
	$(MAKE) reproduce
	-$(MAKE) evidence
	$(MAKE) test-features
	$(MAKE) test-once
	$(MAKE) evidence
	$(PY) -m pytest tests/ -q

phase11: ablations

benchmark:  ## Phase 12 — latency / VRAM / model size, fills Table 8.2.3
	$(PY) scripts/09_benchmark.py

# ── Serving ──────────────────────────────────────────────────────────────────

api:        ## FastAPI on :8000. Concurrency is capped at 1 by design (6 GB RAM).
	$(PY) -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1

frontend:
	cd frontend && npm run dev

demo:       ## API + frontend together
	start "avtfd-api" cmd /c "$(PY) -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1"
	cd frontend && npm run dev

clean-bench:
	@if exist "data\interim\_bench" rmdir /s /q "data\interim\_bench"
