# Phase 5 tooling — export, parity, Triton config validation.
#
# WHY EVERY TARGET `cd`s TO THE REPO ROOT FIRST:
# this repository contains `mlops-pipeline/triton/`. Running `python -m pytest`
# (or any `python -m ...`) with the working directory set to `mlops-pipeline/`
# puts that directory on `sys.path`, so `import triton` resolves to our config
# directory instead of the real PyTorch Triton package. `torch._dynamo` imports
# `triton.language` during `import torch`, so the shadowing surfaces as a
# baffling `ImportError: cannot import name 'GenerationMixin' from
# 'transformers.generation'`. Running from the repo root avoids it entirely.
# See tests/conftest.py for the full explanation.

REPO   := $(abspath ..)
PYTHON := $(REPO)/ai-engine/.venv/bin/python

.PHONY: help export export-minilm parity validate verify-manifest stage test verify-regression clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-18s\033[0m %s\n",$$1,$$2}'

export: ## Export/stage every model (ONNX where applicable, native otherwise)
	cd $(REPO) && $(PYTHON) mlops-pipeline/export/export_onnx.py --model all

export-minilm: ## Export just MiniLM (fast; skips the 1.6 GiB Laya graph)
	cd $(REPO) && $(PYTHON) mlops-pipeline/export/export_onnx.py --model minilm

parity: ## Run the numerical-parity tests and print the real max-abs-diff numbers
	cd $(REPO) && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
	  $(PYTHON) -m pytest -o addopts="" -p no:warnings -s mlops-pipeline/tests/test_export_parity.py

validate: ## Validate every config.pbtxt against Triton's ModelConfig protobuf
	cd $(REPO) && $(PYTHON) mlops-pipeline/triton/validate_configs.py

verify-manifest: ## Re-hash artifacts and compare against manifest.json
	cd $(REPO) && $(PYTHON) mlops-pipeline/export/verify_manifest.py

stage: ## Copy exported artifacts into the Triton repository version dirs
	cd $(REPO) && $(PYTHON) mlops-pipeline/export/populate_triton_repo.py

test: ## Run the mlops test suite (export parity + Triton config validation)
	cd $(REPO) && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
	  $(PYTHON) -m pytest -o addopts="" -p no:warnings mlops-pipeline/tests/

verify-regression: ## Run all three service suites (the CI regression gate)
	cd $(REPO)/backend-core && ./.venv/bin/python -m pytest -q -o addopts="" -p no:warnings
	cd $(REPO)/ai-engine    && ./.venv/bin/python -m pytest -q -o addopts="" -p no:warnings
	cd $(REPO)/frontend-dashboard && npx vitest run

clean: ## Remove exported artifacts (large; gitignored)
	rm -rf $(REPO)/mlops-pipeline/export/artifacts
