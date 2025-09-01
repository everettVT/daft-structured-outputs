PYPROJECT := pyproject.toml
UV := $(shell command -v uv)
VENV_DIR := .venv

# Load environment vars from .env if present
ifneq (,$(wildcard .env))
include .env
export $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' .env)
endif

# vLLM defaults (override as needed)
MODEL ?= google/gemma-3n-e4b-it
GUIDED_BACKEND ?= guidance
DTYPE ?= bfloat16
GPU_MEM_UTIL ?= 0.85
HOST ?= 0.0.0.0
PORT ?= 8000

.PHONY: setup sync activate clean uv-check hf-auth vllm-serve

uv-check:
	@if [ -z "$(UV)" ]; then \
		echo "Error: 'uv' is not installed. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	fi

# Create the venv (if needed) and sync dependencies from pyproject.toml
setup: uv-check
	@echo "[uv] Creating venv and syncing dependencies from $(PYPROJECT)"
	@uv venv --python 3.12
	@uv sync

# Re-sync dependencies with the current pyproject.lock/pyproject.toml
sync: uv-check
	@echo "[uv] Syncing dependencies"
	@uv sync

# Reminder: activation won't persist from within make; prefer 'uv run <cmd>'
activate:
	@echo "Run to activate in your shell: source $(VENV_DIR)/bin/activate"
	@echo "Recommended: use 'uv run <command>' without activating."

# Authenticate to Hugging Face. If HF_TOKEN is set, login non-interactively; otherwise, prompt.
hf-auth: uv-check
	@if [ -n "$(HF_TOKEN)" ]; then \
		echo "HF_TOKEN is set; performing non-interactive login via huggingface_hub"; \
		uv run python -c "from huggingface_hub import login; import os; login(token=os.environ.get('HF_TOKEN'), add_to_git_credential=True)"; \
	else \
		echo "HF_TOKEN not set; launching interactive 'hf auth login'"; \
		uv run hf auth login; \
	fi

# Run vLLM OpenAI-compatible API server
vllm-serve: uv-check
	@echo "Starting vLLM OpenAI server on $(HOST):$(PORT) with model $(MODEL)"
	@uv run python -m vllm.entrypoints.openai.api_server \
		--model $(MODEL) \
		--guided-decoding-backend $(GUIDED_BACKEND) \
		--dtype $(DTYPE) \
		--gpu-memory-utilization $(GPU_MEM_UTIL) \
		--host $(HOST) --port $(PORT)

clean:
	@echo "Removing $(VENV_DIR)"
	@rm -rf $(VENV_DIR)

