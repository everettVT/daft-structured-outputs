## daft-structured-outputs

A focused sandbox for exploring structured outputs with Daft, vLLM, and OpenAI-compatible clients.

### Table of Contents
- **Getting Started**
  - [Prerequisites](#prerequisites)
  - [Install and Setup](#install-and-setup)
  - [Environment Variables](#environment-variables)
  - [Make Targets](#make-targets)
- **Serving & Running**
  - [Start the vLLM OpenAI server](#start-the-vllm-openai-server)
  - [Run the workload scripts/notebooks](#run-the-workload-scriptsnotebooks)
- **Project Content**
  - [Canon](#canon) contains reference examples from ray, vllm, and sglang on structured outputs, as well as a full suite of `llm_generate` inference calls across the most common structured output methods.
  - [Friction](#friction) contains the original (giant) "Scaling Multimodal Structured Outputs with Gemma-3, vLLM, and Daft", as well as notebooks focused on individal pain points seperated for easier review.
  - [Workload](#workload) contains both a full walkthrough notebook and atomic python script for evaluating multimodal model performance on image understanding.
- **Testing**
  - [Integration tests](#integration-tests)
- **Troubleshooting**
  - [Common issues](#common-issues)

---

### Prerequisites
- **Python**: 3.12+
- **uv**: Fast Python package/venv manager. Install:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install and Setup
```bash
cd daft-structured-outputs
make setup
```
This creates a local `.venv` and syncs dependencies from `pyproject.toml`. Prefer running commands with `uv run` without activating the venv.

### Environment Variables
These are read by tests and examples. The `makefile` also exports them.
- `OPENAI_API_KEY`: Any non-empty value when using a local vLLM server (e.g., `none`).
- `OPENAI_BASE_URL`: Defaults to `http://0.0.0.0:8000/v1`.
- `HF_TOKEN`: Hugging Face token for model pulls. If not set, use `make hf-auth`.

Example:
```bash
export OPENAI_API_KEY=none
export OPENAI_BASE_URL=http://0.0.0.0:8000/v1
export HF_TOKEN=hf_...
```

### Make Targets
```bash
make setup        # Create venv and uv sync
make sync         # Re-sync dependencies
make activate     # Echo activation instructions (prefer `uv run`)
make clean        # Remove .venv
make hf-auth      # Run 'hf auth login' if HF_TOKEN is not set
make vllm-serve   # Start vLLM OpenAI-compatible server
```

---

### Start the vLLM OpenAI server
Defaults are aligned with project notebooks:
```bash
make vllm-serve \
  MODEL=google/gemma-3n-e4b-it \
  GUIDED_BACKEND=guidance \
  DTYPE=bfloat16 \
  GPU_MEM_UTIL=0.85 \
  HOST=0.0.0.0 PORT=8000
```
If you need Hugging Face auth:
```bash
make hf-auth
```

### Run the workload scripts/notebooks
- Python scripts (example):
```bash
uv run python workload/daft_mm_so_gemma3.py
```
- Notebooks: open in your IDE or Jupyter and ensure the environment variables above are set in the session.

---

### Canon
Curated examples and references. Open directly or execute through your IDE.
- `canon/vllm/vllm_structured_outputs.py`
- `canon/vllm/OpenAI Chat Completion Client.py`
- `canon/vllm/OpenAI Chat Completion Client with Tools.py`
- `canon/vllm/OpenAI Chat Completion Tool Calls.py`
- `canon/llm_generate_so_vllm_online.ipynb`
- `canon/ray/VLLM with Structural Output.ipynb`
- `canon/sglang/sglang_structured_outputs.ipynb`
- `canon/sglang/sglang_function_calling.ipynb`
- `canon/sglang/sglang_structured_outputs_for_reasoning_models.ipynb`

### Friction
Notes and experiments documenting rough edges, performance, and gotchas.
- `friction/FULL_MESS.ipynb`
- `friction/udf_performance.ipynb`
- `friction/udf_pains.ipynb`
- `friction/trying_to_hack_llm_generate_for_images.ipynb`

### Workload
End-to-end flows and runnable scripts.
- `workload/daft_mm_so_gemma3.py`
- `workload/daft_mm_so_vllm_walkthough.ipynb`

---

### Integration tests
Run against a live vLLM server (skips if unreachable):
```bash
uv run pytest -q tests/test_openai_vllm_integration.py
```
Environment variables used by the tests:
- `OPENAI_BASE_URL` (default `http://0.0.0.0:8000/v1`)
- `OPENAI_API_KEY` (default `none`)
- `MODEL` (default `google/gemma-3n-e4b-it`)
- `TEST_IMAGE_URL` (optional; enables the vision test)

---

### Common issues
- **vLLM server not reachable**: Ensure `make vllm-serve` is running; confirm `OPENAI_BASE_URL` and `PORT`.
- **HF auth required**: Run `make hf-auth` to authenticate if `HF_TOKEN` is not set.
- **GPU memory**: Adjust `GPU_MEM_UTIL` in `make vllm-serve` for your hardware.
- **Dependencies**: Re-run `make sync` after modifying `pyproject.toml`.
