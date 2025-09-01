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

clean:
	@echo "Removing $(VENV_DIR)"
	@rm -rf $(VENV_DIR)

