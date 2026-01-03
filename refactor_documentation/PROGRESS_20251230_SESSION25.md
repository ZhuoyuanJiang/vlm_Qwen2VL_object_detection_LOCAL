# Session 25: Docker-Based vLLM Deployment (2025-12-30)

## Session Overview

This session continued from Session 24b, focusing on reorganizing the deployment notebook to use Docker for vLLM instead of installing vLLM in the conda environment.

## Problem Context (from Session 24b)

Installing `vllm>=0.6.0` silently upgraded PyTorch from 2.4.1 to 2.9.0, breaking flash-attn compatibility. The solution was:
1. Restore PyTorch 2.4.1 (done in Session 24b)
2. Use Docker for vLLM serving instead of conda install (this session)

## Changes Made

### notebooks/07_deployment_vllm_triton.py (.ipynb synced)

#### 1. Added Warning Section (after intro)
New markdown cell explaining Docker-based deployment:
- Why Docker instead of `pip install vllm`
- Workflow diagram showing Terminal (Docker) → HTTP API → Notebook
- Clear warning: "Do NOT run `pip install vllm` in your conda environment!"

#### 2. Updated Section 1.3 (Install Dependencies)
**Before:**
```python
# !pip install vllm>=0.6.0
# !pip install tritonclient[all]
# !pip install openai
```

**After:**
```python
# !pip install tritonclient[all]  # For Triton API calls
# !pip install openai             # For vLLM API calls

# ⚠️ DO NOT install vllm here! Use Docker instead.
```

Also updated dependency check to remove vllm, add requests.

#### 3. Updated Section 3.1 (Launch vLLM)
**Before:** Python command requiring vllm installed
```python
vllm_cmd = ["python", "-m", "vllm.entrypoints.openai.api_server", ...]
```

**After:** Docker command
```python
docker_vllm_cmd = f"""docker run --gpus all -d --name vllm-server \\
  -p {VLLM_PORT}:8000 \\
  -v {MERGED_MODEL_PATH}:/model:ro \\
  --ipc=host \\
  vllm/vllm-openai:v0.6.4.post1 \\
  --model /model \\
  --served-model-name {VLLM_MODEL_NAME} \\
  ..."""
```

Added:
- Useful Docker commands (logs, stop, rm)
- `check_vllm_server()` function to verify server is running

#### 4. Updated Section 3.4 (Stop vLLM)
**Before:** Subprocess-based termination
```python
def stop_vllm():
    vllm_process.terminate()
```

**After:** Docker-based stop
```python
def stop_vllm_docker():
    subprocess.run(["docker", "stop", "vllm-server"])
    subprocess.run(["docker", "rm", "vllm-server"])
```

## Architecture: Dev vs Serving Environments

```
┌─────────────────────────────────────────────────────────────┐
│  Dev Environment (conda: vlm_Qwen2VL_object_detection)      │
│  - PyTorch 2.4.1 + flash-attn 2.6.3                        │
│  - Training, inference, evaluation notebooks                │
│  - DO NOT install vLLM here                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Serving Environment (Docker)                               │
│  - vLLM container: vllm/vllm-openai:v0.6.4.post1           │
│  - Triton container: nvcr.io/nvidia/tritonserver:24.08-... │
│  - Isolated dependencies, no conflict with dev env          │
└─────────────────────────────────────────────────────────────┘
```

## Files Modified
- `notebooks/07_deployment_vllm_triton.py`
- `notebooks/07_deployment_vllm_triton.ipynb` (synced via jupytext)

## Known Linter Warnings (can be ignored)
The notebook has some unused imports and undefined variables in certain cells:
- `grpcclient` imported but not used (optional Triton feature)
- `np` not defined in some cells (numpy not imported in those cells)
- `vllm_cmd` and `vllm_process` references in old code sections

These are in optional/example cells and don't affect the main workflow.

## Next Steps
1. **Test Docker vLLM deployment** - Run the Docker command and test API
2. **Test Triton deployment** - Already uses Docker
3. **Commit changes** - Add notebook updates to git
4. **Update environment.yml header** - Already has warning (from Session 24b)

## Environment Policy Reminder

**DO NOT modify `environment.yml` or run `conda env export`**

- `environment.yml` (Dec 16 snapshot) is the "golden" reproducible state
- For vLLM/Triton: Use Docker
- For new dev packages: Manually edit environment.yml

---

## Session 25b: vLLM Deployment via Separate Conda Environment (2026-01-03)

### Context

Since Docker is not available on the lab servers (vllab4-vllab15), we deployed vLLM using a **separate conda environment** on vllab8. This approach isolates vLLM's dependencies (PyTorch 2.9.0) from the training environment (PyTorch 2.4.1 + flash-attn 2.6.3).

### Why a Separate Environment?

| Component | Training Env | vLLM Serving Env |
|-----------|--------------|------------------|
| PyTorch | 2.4.1+cu121 | 2.9.0+cu129 |
| flash-attn | 2.6.3 | Not needed |
| vLLM | ❌ Cannot install | 0.13.0 |
| Server | vllab11 | vllab8 |

- **vLLM 0.6.1+** is required for Qwen2-VL support
- vLLM 0.6.1+ requires PyTorch 2.5+, which breaks flash-attn
- Solution: Separate serving environment on a different server

### Infrastructure

| Server | Role | GPU | Storage |
|--------|------|-----|---------|
| vllab11 | Training/Development | 8x RTX 6000 Ada (48GB) | /ssd1/zhuoyuan/vlm_outputs/ |
| vllab8 | vLLM Serving | 8x RTX 3090 (24GB) | /ssd1/zhuoyuan/ |

### Step-by-Step Deployment Guide

#### Step 1: Create Directory Structure (on vllab8)

```bash
ssh vllab8
mkdir -p /ssd1/zhuoyuan/envs
mkdir -p /ssd1/zhuoyuan/vlm_outputs
```

#### Step 2: Create Conda Environment (on vllab8)

Using `--prefix` to store the environment on local SSD (not home directory to avoid quota issues):

```bash
conda create --prefix /ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving python=3.12 -y
conda activate /ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving
```

#### Step 3: Set UV Cache Directory (on vllab8)

Redirect uv cache to local SSD to avoid home directory quota issues:

```bash
export UV_CACHE_DIR=/ssd1/zhuoyuan/.cache/uv
mkdir -p /ssd1/zhuoyuan/.cache/uv

# Optional: Add to .bashrc for persistence
echo 'export UV_CACHE_DIR=/ssd1/zhuoyuan/.cache/uv' >> ~/.bashrc
```

#### Step 4: Install vLLM (on vllab8)

Using `uv` with `--torch-backend=auto` as recommended by vLLM docs:

```bash
pip install --upgrade uv
uv pip install vllm --torch-backend=auto
```

This installs:
- vLLM 0.13.0
- PyTorch 2.9.0+cu129
- All dependencies (~161 packages)

#### Step 5: Install Client Dependencies (on vllab8)

```bash
pip install openai pillow requests datasets
```

#### Step 6: Copy Model from vllab11 (on vllab8)

```bash
rsync -avz --progress vllab11:/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
  /ssd1/zhuoyuan/vlm_outputs/
```

Model size: ~16GB (4 safetensors files)

#### Step 7: Launch vLLM Server (on vllab8)

```bash
conda activate /ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving

CUDA_VISIBLE_DEVICES=0 vllm serve \
  /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
  --served-model-name qwen2vl-nutrition \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 4096 \
  --limit-mm-per-prompt '{"image":1}' \
  --port 8000
```

**Flags explained:**
- `CUDA_VISIBLE_DEVICES=0`: Use GPU 0 only
- `--served-model-name`: Name to use in API calls
- `--dtype bfloat16`: Match training precision
- `--max-model-len 4096`: Max context length
- `--limit-mm-per-prompt '{"image":1}'`: Allow 1 image per request (required for VLMs)
- `--port 8000`: HTTP port

**VRAM Usage:** ~22.9GB / 24GB (93% of RTX 3090)

#### Step 8: Test the API

From any terminal (vllab8 or other servers):

```bash
# Health check
curl http://vllab8:8000/health

# List models
curl http://vllab8:8000/v1/models
```

### API Usage Example

```python
import base64
import requests
from datasets import load_dataset

# Load test image
ds = load_dataset("openfoodfacts/nutrition-table-detection", split="val")
ds[0]['image'].save('/tmp/test_nutrition.jpg')

# Encode to base64
with open('/tmp/test_nutrition.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

# Send request
response = requests.post(
    "http://vllab8:8000/v1/chat/completions",
    json={
        "model": "qwen2vl-nutrition",
        "messages": [
            {"role": "system", "content": "You are a nutrition label detector..."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": "Detect the bounding box coordinates for the nutrition facts table."}
            ]}
        ],
        "max_tokens": 64,
        "temperature": 0.0,
        "stop": ["<|box_end|>"],
        "skip_special_tokens": False  # Important: include special tokens in output!
    }
)

result = response.json()
print(result['choices'][0]['message']['content'])
```

**Expected output:**
```
<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(272,494),(732,621)
```

### Key Findings

1. **`skip_special_tokens: False`** is required to see the special tokens (`<|object_ref_start|>`, `<|box_start|>`, etc.) in the output. vLLM defaults to `True`.

2. **`stop: ["<|box_end|>"]`** prevents repetition by stopping generation after the bounding box.

3. **Version pinning is important:** Don't use `pip install vllm>=0.6.0` - it will install the latest version which may have different PyTorch requirements. Pin the version: `vllm==0.13.0`.

4. **UV cache location:** Set `UV_CACHE_DIR` to local SSD before installing to avoid home directory quota issues.

### Architecture Diagram (Updated)

```
┌─────────────────────────────────────────────────────────────┐
│  vllab11: Dev Environment (conda: vlm_Qwen2VL_object_detection)  │
│  - PyTorch 2.4.1 + flash-attn 2.6.3                              │
│  - Training, inference, evaluation notebooks                      │
│  - Model stored at: /ssd1/zhuoyuan/vlm_outputs/                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ rsync model
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  vllab8: Serving Environment                                     │
│  - Conda env: /ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving │
│  - vLLM 0.13.0 + PyTorch 2.9.0+cu129                            │
│  - GPU: RTX 3090 (24GB) - uses ~22.9GB                          │
│  - API: http://vllab8:8000/v1/chat/completions                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP API
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Client (any server, notebook, or application)                   │
│  - Uses OpenAI-compatible API                                    │
│  - curl, requests, openai Python client                          │
└─────────────────────────────────────────────────────────────┘
```

### Cleanup (Optional)

After installation, you can delete the uv cache to save disk space:

```bash
rm -rf /ssd1/zhuoyuan/.cache/uv
```

The installed packages remain in the conda environment.

### Known Issue: Repetition Bug

**Status:** vLLM deployment is functional but exhibits a repetition bug.

When serving the fine-tuned model with vLLM, the model repeats its output indefinitely until `max_tokens` is reached:

```
<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(272,494),(732,621)<|box_end|>
<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(272,494),(732,621)<|box_end|>
...repeats...
```

**Key observations:**
- The same model works correctly with transformers `model.generate()` - no repetition
- vLLM seems to ignore the model's `generation_config.json` EOS tokens
- Adding `stop: ["<|box_end|>"]` prevents repetition (current workaround)

**Current workaround:**
```python
response = requests.post(
    "http://vllab8:8000/v1/chat/completions",
    json={
        ...
        "stop": ["<|box_end|>"],  # Prevents repetition
        "skip_special_tokens": False
    }
)
```

See `refactor_documentation/VLLM_REPETITION_BUG.md` for detailed analysis.

### Next Steps

1. Investigate vLLM repetition bug further - compare transformers vs vLLM generation behavior

2. Update `notebooks/07_deployment_vllm_triton.ipynb` with:
   - Separate conda env approach as alternative to Docker
   - `--limit-mm-per-prompt` flag for VLMs

3. Commit changes to git

---

## Files Modified/Created (Session 25)

### Created
- `refactor_documentation/PROGRESS_20251230_SESSION25.md` - This session documentation
- `refactor_documentation/VLLM_REPETITION_BUG.md` - Detailed analysis of the repetition bug
- `scripts/test_vllm_api.py` - Minimal vLLM API test script
- `scripts/test_vllm_with_visualization.py` - Comprehensive test with bbox parsing and visualization
- `requirements_vllm_serving.txt` - vLLM serving environment pip freeze
- `environment_vllm_serving.yml` - vLLM serving conda environment (Python 3.12, vLLM 0.13.0, PyTorch 2.9.0+cu129)

### Modified
- `notebooks/07_deployment_vllm_triton.py` - Added Docker-based vLLM deployment
- `notebooks/07_deployment_vllm_triton.ipynb` - Synced notebook version
