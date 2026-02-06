# Session 31 Progress - Triton Benchmark Parity (Chat Template)

**Date**: 2026-02-06  
**Session Name**: triton-benchmark-parity

---

## Objective

Align Triton benchmarking with the **exact training prompt format** by always applying Qwen2‑VL’s chat template, and make vLLM vs Triton comparisons apples‑to‑apples. Remove prompt overrides to eliminate implementation‑driven variance.

---

## Baseline

- **Last commit before this session**: `32112ca` — “Add Triton benchmark Q&A documentation and fix benchmark_triton.py”

---

## Summary of Changes (Since Last Commit)

### 1) `scripts/benchmark_triton.py`

**Core behavior changes**
- **Always apply chat template** using `Qwen2VLProcessor.apply_chat_template()` (matches training format).
- **Removed prompt override flags** (`--prompt`, `--system-prompt`, `--no-system`) to avoid input drift.
- **Training prompts are now constants** and always used.

**Processor & prompt handling**
- Added **processor loading with LRU cache** (`_load_processor()`), keyed by:
  - `QWEN2VL_PROCESSOR_PATH` env var (for local merged model paths)
  - fallback: `"Qwen/Qwen2-VL-7B-Instruct"`
- Added **chat-template builder**:
  - system + user (image + text) → `apply_chat_template(..., add_generation_prompt=True)`
  - prompt generated once from the first sample image

**Dataset/image handling**
- `load_test_image()` now returns **`tuple[Image.Image, str]`** instead of just `str`:
  - `[0]` = PIL image (needed for `build_chat_template_prompt`)
  - `[1]` = base64 string (needed for Triton HTTP/gRPC payload)
  - This is why image access in benchmark functions changed from `images[i]` to `images[i][1]`
  ```python
  # Old: images was list[str]
  image_b64 = images[0]          # Direct base64 string

  # New: images is list[tuple[PIL, str]]
  image_b64 = images[0][1]       # [1] extracts base64 from the tuple
  pil_image = images[0][0]       # [0] extracts PIL for chat template
  ```
- Dataset loading remains cached with LRU (`_load_test_dataset()`), and processor is now cached too.
- **Verified**: `build_chat_template_prompt()` produces the same prompt regardless of which image is passed (template uses generic `<|image_pad|>` placeholder, not image-specific content), so generating the prompt once from `images[0]` is correct.

**Concurrency / async**
- HTTP benchmark is **true async** (`aiohttp`, `asyncio.gather`, semaphore)
  - Old: used synchronous `requests.Session.post()` inside `async def` (not truly concurrent)
  - New: uses `aiohttp.ClientSession` with real async I/O
- gRPC benchmark is **true async** (`tritonclient.grpc.aio`, `asyncio.gather`, semaphore)
  - Old: synchronous `for` loop with `grpcclient.InferenceServerClient` (sequential, one at a time)
  - New: uses `grpcclient_aio.InferenceServerClient` with `await client.infer()`, bounded by semaphore
  - Added defensive `client.close()` pattern (`grpc.aio` close may return a coroutine)
- This makes HTTP and gRPC benchmarks comparable under concurrency.

**Usage docs**
- Updated header examples to show:
  - higher concurrency usage
  - `QWEN2VL_PROCESSOR_PATH` env variable for local processor files

---

## Rationale & Previous State

### Prompt Parity (Triton vs Training)

**Previous state (timeline)**
- **Phase 1 — quick baseline:** we started with a **user‑only prompt** (no system prompt) to get something running against Triton quickly.
  - Example (single string sent as `text_input`):
    ```text
    Detect the nutrition facts table in this image and return the bounding box coordinates.
    ```
- **Phase 2 — closer to training:** after comparing with the training recipe, we **added the system prompt** to match dataset prompts. We still used **simple string concatenation** because it was the fastest way to align wording.
  - Example concatenation:
    ```python
    prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"
    ```
  - Resulting raw text (no special tokens):
    ```text
    You are a Vision Language Model specialized in interpreting visual data from product images...

    Detect the bounding box of the nutrition table.
    ```
- **Phase 3 — exact format parity:** we realized Triton’s vLLM backend does **not** apply the Qwen2‑VL chat template, so concatenation still **missed special tokens** the model saw during training. That gap could explain accuracy drift, so we switched to **`apply_chat_template()`** and removed prompt overrides to keep inputs consistent.
  - What training‑style input actually looks like (conceptually):
    ```text
    <|im_start|>system
    ...system prompt...<|im_end|>
    <|im_start|>user
    <image>Detect the bounding box of the nutrition table.<|im_end|>
    <|im_start|>assistant
    ```

**Why change**
- Training and HF inference **use Qwen2‑VL chat templates** with special tokens.
- The mismatch could **degrade accuracy** or stability and makes vLLM vs Triton comparisons noisy.

**What we did**
- Always apply `Qwen2VLProcessor.apply_chat_template(..., add_generation_prompt=True)` on the client side.
- Prompt now matches training format exactly (system + user + image).
- Removed `--prompt`, `--system-prompt`, `--no-system` flags to prevent accidental drift.
- Added **LRU cache** for the processor (`_load_processor`) so template setup is loaded once per run.
- Added `QWEN2VL_PROCESSOR_PATH` env override to force local processor files when HF cache is unavailable.

---

### Benchmark Fairness (vLLM vs Triton)

**Previous state**
- vLLM benchmark used a **different system/user prompt** than training.
- Triton benchmark used yet another prompt string.

**Why change**
- Different prompts produce different token lengths and behaviors, which distorts performance and accuracy comparisons.

**What we did**
- Aligned both benchmarks to the training prompts so the comparison is apples‑to‑apples.

---

### Async Concurrency (HTTP + gRPC)

**Previous state**
- HTTP benchmark used a **synchronous** `requests` call inside `async def`, so it wasn’t truly concurrent.
- gRPC benchmark was **fully sequential** (for loop).

**Why change**
- We wanted **real concurrency** to measure throughput under load.

**What we did**
- HTTP: switched to `aiohttp` + `asyncio.gather` + semaphore.
- gRPC: switched to `tritonclient.grpc.aio` + `asyncio.gather` + semaphore.

---

### 2) `scripts/benchmark_vllm.py`

- **Aligned prompts to training**:
  - System prompt → training system prompt
  - User prompt → `"Detect the bounding box of the nutrition table."`
- Added note in header that **older benchmarks used different prompts**, and should be rerun for strict comparison.

---

### 3) `scripts/validate_triton_accuracy.py`

- Updated default prompt to **training user prompt** for consistency:
  - From: `"Detect the nutrition facts table in this image and return the bounding box coordinates."`
  - To: `"Detect the bounding box of the nutrition table."`

---

## Why This Matters

- Triton vLLM backend **does not apply chat templates**; it consumes `text_input` directly.
- Training and HF inference pipelines **do** apply Qwen2‑VL’s chat template.
- Without template parity, Triton inputs are **not equivalent to training**, leading to avoidable accuracy drift.

This session removes that gap by **always** applying the chat template in `benchmark_triton.py`.

---

## Files Modified (Uncommitted)

| File | Change Summary |
|------|----------------|
| `scripts/benchmark_triton.py` | Always apply chat template, remove prompt CLI flags, add processor LRU cache + env override, async HTTP+gRPC, PIL+base64 image handling |
| `scripts/benchmark_vllm.py` | Align prompts to training; add rerun note |
| `scripts/validate_triton_accuracy.py` | Update default prompt to training user prompt |

---

## Notes / Caveats

- **Processor path must match the deployed model.**  
  If HF cache is unavailable or mismatched, set:  
  `QWEN2VL_PROCESSOR_PATH=/path/to/merged/model`

- The prompt is generated once (from the first image) for efficiency.  
  This is correct because the template structure depends on **image presence**, not image content.

---

## Cloud Deployment Checklist (Docker-capable machine)

1. **Provision VM** with NVIDIA GPU + Docker support.
2. **Verify GPU + Docker**:
   - `nvidia-smi`
   - `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`
3. **Create directories** for weights + model repo:
   - e.g., `/workspace/models/` and `/workspace/triton_model_repository/`
4. **Transfer model weights** (BF16 and/or GPTQ).
5. **Transfer Triton model repo** + scripts (`deploy_triton.sh`, `benchmark_triton.py`, `validate_triton_accuracy.py`).
6. **Pull Triton image**: `docker pull nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3`
7. **Start Triton** using `scripts/deploy_triton.sh` (update paths inside).
8. **Health check**: `curl http://localhost:8000/v2/health/ready`
9. **Run benchmarks** (Triton + vLLM) with aligned prompts.
10. **Save results** to a new progress doc.

### Rsync Draft (local → remote VM)

```bash
# Set your remote target
REMOTE_USER=ubuntu
REMOTE_HOST=1.2.3.4
REMOTE_DIR=/workspace

# Create folders on remote
ssh ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_DIR}/models ${REMOTE_DIR}/triton_model_repository"

# Transfer Triton repo + scripts
rsync -av --progress \
  triton_model_repository/ \
  scripts/deploy_triton.sh \
  scripts/benchmark_triton.py \
  scripts/validate_triton_accuracy.py \
  scripts/benchmark_vllm.py \
  refactor_documentation/PROGRESS_20260206_SESSION31.md \
  ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/

# Transfer model weights (large)
rsync -av --progress /path/to/qwen2vl-nutrition-merged/ \
  ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/models/qwen2vl-nutrition-merged/

# (Optional) GPTQ weights
rsync -av --progress /path/to/qwen2vl-nutrition-merged-gptq-int4/ \
  ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/models/qwen2vl-nutrition-merged-gptq-int4/
```

**Note:** If you rely on local processor files, set:
`QWEN2VL_PROCESSOR_PATH=${REMOTE_DIR}/models/qwen2vl-nutrition-merged`

---

## Next Steps

1. **Run Triton benchmarks** with the new chat-template prompt.
2. **Rerun vLLM benchmarks** so results are apples‑to‑apples.
3. **Review accuracy** using `validate_triton_accuracy.py` with updated prompt.
4. **Commit** these changes once validated.
