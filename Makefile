.PHONY: help sync build test clean data train infer mlflow-ui

# TinyGPT Makefile (uv-first workflow).
#
# Usage:
#   make sync
#   make data
#   make train
#   make infer PROMPT="First Citizen:\n"
#   make mlflow-ui

UV ?= uv
PY ?= $(UV) run python

DATASET ?= shakespeare
CHECKPOINT ?= examples/gpt_mlx_weights.npz
TOKENIZER_MODEL ?= examples/tokenizer.model

# Inference defaults (override at invocation time)
PROMPT ?= First Citizen:\n
MAX_NEW_TOKENS ?= 128
TEMPERATURE ?= 0.8

# Training/inference extra args passthrough
TRAIN_ARGS ?=
INFER_ARGS ?=

help:
	@echo "TinyGPT targets:"
	@echo "  make sync        - create .venv + install deps (uv sync)"
	@echo "  make build       - build wheel/sdist (uv build)"
	@echo "  make test        - run tests (pytest)"
	@echo "  make data        - prepare dataset (default: shakespeare)"
	@echo "  make train       - run MLX training (logs to MLflow by default)"
	@echo "  make infer       - run inference from checkpoint"
	@echo "  make mlflow-ui   - launch MLflow UI (local file store)"
	@echo ""
	@echo "Common overrides:"
	@echo "  CHECKPOINT=examples/my.npz"
	@echo "  PROMPT='To be, or not to be:\n'"
	@echo "  TRAIN_ARGS='--num-epochs 5' (if supported by the script)"

sync:
	$(UV) sync

build:
	$(UV) build

test:
	$(UV) run pytest

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ .mypy_cache dist build *.egg-info
	rm -rf mlruns mlflow.db artifacts

data:
ifeq ($(DATASET),shakespeare)
	$(PY) data/shakespeare/prepare.py
else
	@echo "Unknown DATASET=$(DATASET). Supported: shakespeare"
	@exit 2
endif

train:
	# MLflow is enabled by default; disable with MLFLOW_ENABLE=0
	MLFLOW_ENABLE=$${MLFLOW_ENABLE:-1} \
	$(PY) examples/gpt_mlx.py --checkpoint "$(CHECKPOINT)" $(TRAIN_ARGS)

infer:
	MLFLOW_ENABLE=$${MLFLOW_ENABLE:-1} \
	$(PY) examples/gpt_mlx.py --inference \
		--checkpoint "$(CHECKPOINT)" \
		--prompt "$(PROMPT)" \
		--max-new-tokens "$(MAX_NEW_TOKENS)" \
		--temperature "$(TEMPERATURE)" \
		$(INFER_ARGS)

mlflow-ui:
	# Uses local ./mlruns by default (MLflow file store).
	$(UV) run mlflow ui --host 127.0.0.1 --port 5000
