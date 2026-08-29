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
        typecheck test test-cov fetch-meta manifest subset verify-subsets verify-mirror \n        stats phase1 \n        features train evaluate ablations \
        benchmark api frontend demo clean-bench

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

features:   ## Phases 2-5 — Stage-A extraction (resumable; safe to re-run)
	$(PY) scripts/04_extract_visual.py
	$(PY) scripts/05_extract_audio.py

train:      ## Stage-B training. Override: make train CONFIG=configs/experiment/exp_e.yaml
	$(PY) scripts/06_train.py --config $(or $(CONFIG),configs/default.yaml)

evaluate:
	$(PY) scripts/07_evaluate.py --config $(or $(CONFIG),configs/default.yaml)

ablations:  ## Phase 11 — the full A-K matrix across 3 seeds
	$(PY) scripts/08_run_ablations.py

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
