<div align="center">

# Daft Structured Outputs 

Canonical multimodal workloads Sandbox for Structured Ouputs with Daft

<i> Featuring HuggingFace, vLLM, Gemma 3n, and OpenAI </i>

</div>

---

**Core Deliverable**
- [Notebook](/workload/Daft_Canonical_Workload_Gemma3_vLLM.ipynb)
- [Python Script](/workload/structured_outputs_workload.py)

**Project Content**
  - [References](/references) contains reference examples from ray, vllm, and sglang on structured outputs, as well as a full suite of `llm_generate` inference calls across the most common structured output methods.
  - [Friction](/friction) contains the original (giant) "Scaling Multimodal Structured Outputs with Gemma-3, vLLM, and Daft", as well as notebooks focused on individal pain points seperated for easier review.
  - [Workload](/workload) contains both a full walkthrough notebook and atomic python script for evaluating multimodal model performance on image understanding.
  - [Integration tests](/tests) for openai and llm_generate structured outputs usage patterns

---

### Prerequisites
- **Python**: 3.12+
- **uv**: Fast Python package/venv manager. Install:
```bash
pip install uv
```

### Install and Setup
Clone this repository and then run 
```bash
cd daft-structured-outputs
uv venv && uv sync
```
- This creates a local `.venv` and syncs dependencies from `pyproject.toml`. 
- Prefer running commands with `uv run` without activating the venv.

### Environment Variables
These are read by tests and examples. A `.env.examples` has been provided as a template. 
- `OPENAI_API_KEY`: Any non-empty value when using a local vLLM server (e.g., `none`).
- `OPENAI_BASE_URL`: Defaults to None. vLLM examples default to localhost:8000 
- `HF_TOKEN`: Hugging Face token for model pulls. If not set, use `make hf-auth`.
- `MODEL_ID`: for integration tests and CI

---

### Start the vLLM OpenAI server
Defaults are aligned with project notebooks:

```bash
uv run vllm.entrypoints.openai.api_server \
  --model google/gemma-3n-e4b-it \
  --enable-chunked-prefill \
  --guided-decoding-backend guidance \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --port 8000
```

You will need authenticate with Hugging Face to access Gemma-3
```bash
hf auth login
```

### Run the workload scripts/notebooks
- Python scripts (example):
```bash
uv run python workload/daft_mm_so_gemma3.py
```
- Notebooks: open in your IDE or Jupyter and ensure the environment variables above are set in the session.

---

### Integration tests
Run against a live vLLM server (skips if unreachable):
```bash
uv run pytest -q tests/test_openai_vllm_integration.py
```
Environment variables used by the tests:
- `OPENAI_BASE_URL` (default `http://0.0.0.0:8000/v1`)
- `OPENAI_API_KEY` (default `none`)
- `MODEL_ID` (default `google/gemma-3n-e4b-it`)
- `TEST_IMAGE_URL` (optional; enables the vision test)
- 

---

### Common issues
- **vLLM server not reachable**: Ensure `make vllm-serve` is running; confirm `OPENAI_BASE_URL` and `PORT`.
- **HF auth required**: Run `hf auth login` to authenticate if `HF_TOKEN` is not set.
- **GPU memory**: Adjust `GPU_MEM_UTIL` in `make vllm-serve` for your hardware.
- **Dependencies**: Re-run `uv sync` after modifying `pyproject.toml`.
