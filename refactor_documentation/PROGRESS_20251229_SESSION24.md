# Session 24: vLLM and Triton Deployment (2025-12-29)

## Session Overview

This session focused on deploying the trained Qwen2-VL nutrition detection model to production inference servers (vLLM and Nvidia Triton).

## Goals Accomplished

### 1. Created Deployment Notebook
**File**: `notebooks/07_deployment_vllm_triton.ipynb` (with synced `.py` file)

A comprehensive Jupyter notebook covering:
- Section 1: Configuration & GPU Setup
- Section 2: LoRA Merge (optional, skipped if merged model exists)
- Section 3: vLLM Deployment with OpenAI-compatible API
- Section 4: Triton Deployment with vLLM backend
- Section 5: Summary & Cleanup

### 2. Created Standalone Deployment Scripts

Three production-ready scripts for deployment without needing the notebook:

#### `scripts/serve_vllm.py`
Launches vLLM server with configurable options.

```bash
# Basic usage
python scripts/serve_vllm.py

# Custom configuration
python scripts/serve_vllm.py \
    --model /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
    --port 8000 \
    --gpu 2,3 \
    --model-name qwen2vl-nutrition
```

**Features**:
- Auto-detects available GPUs
- Configurable port, model path, dtype
- Health check after startup
- Graceful shutdown handling (Ctrl+C)

#### `scripts/setup_triton.py`
Creates Triton model repository structure.

```bash
# Basic usage
python scripts/setup_triton.py

# Custom paths
python scripts/setup_triton.py \
    --model /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
    --repo /ssd1/zhuoyuan/triton_model_repository
```

**Creates**:
- `model.json` - vLLM engine configuration
- `config.pbtxt` - Triton server configuration
- Prints Docker launch command

#### `scripts/test_deployment_inference.py`
Tests deployed model endpoints.

```bash
# Test vLLM
python scripts/test_deployment_inference.py --endpoint vllm

# Test Triton
python scripts/test_deployment_inference.py --endpoint triton

# Test with dataset image
python scripts/test_deployment_inference.py --use-dataset

# Test with custom image
python scripts/test_deployment_inference.py --image /path/to/image.jpg
```

**Features**:
- Tests both vLLM and Triton endpoints
- Parses bounding box output
- Validates response format
- Can load test images from HuggingFace dataset

### 3. Infrastructure Setup

#### Transferred Checkpoints
Moved all model checkpoints from vllab13 to vllab14 via rsync:
```bash
rsync -avz --progress zhuoyuan@vllab13:/ssd1/zhuoyuan/vlm_outputs/ /ssd1/zhuoyuan/vlm_outputs/
```

**Models available on vllab14**:
- `qwen2vl-nutrition-detection-r4-joint/` - Best LoRA adapter (0.8636 IoU)
- `qwen2vl-nutrition-detection-r4-joint-merged/` - Merged model (~16GB, ready for deployment)
- All other recipe outputs (r1, r2, r3, demo models)

#### Server Environment
- **Server**: vllab14
- **GPUs**: 8× NVIDIA RTX 6000 Ada (48GB each)
- **Available GPUs**: 2-7 (GPUs 0-1 may be in use)

## Key Concepts Explained

### OpenAI-Compatible API
vLLM provides an API that matches OpenAI's format exactly:

```python
# Same code works for both OpenAI and vLLM
from openai import OpenAI

# For OpenAI
client = OpenAI(api_key="sk-xxx", base_url="https://api.openai.com/v1")

# For vLLM (just change base_url)
client = OpenAI(api_key="dummy", base_url="http://localhost:8000/v1")

# Same API call
response = client.chat.completions.create(
    model="qwen2vl-nutrition",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### QLoRA Training vs Serving Quantization
- **Training**: Used 4-bit NF4 quantization to reduce GPU memory during training
- **Merged Model**: Full bf16 precision (~16GB) after `merge_and_unload()`
- **Serving**: Using bf16 (no additional quantization) to preserve 0.8636 IoU accuracy

### vLLM vs Triton
| Feature | vLLM | Triton |
|---------|------|--------|
| API | OpenAI-compatible | Native REST/gRPC |
| Complexity | Simple | More config files |
| Metrics | Basic | Prometheus-compatible |
| Multi-model | Manual | Built-in |
| Use case | Development, single model | Production, enterprise |

## Files Created This Session

| File | Purpose |
|------|---------|
| `notebooks/07_deployment_vllm_triton.py` | Deployment notebook (Jupytext source) |
| `notebooks/07_deployment_vllm_triton.ipynb` | Deployment notebook (Jupyter) |
| `scripts/serve_vllm.py` | vLLM server launcher |
| `scripts/setup_triton.py` | Triton repository setup |
| `scripts/test_deployment_inference.py` | Deployment testing |
| `refactor_documentation/Deployment_Plan.md` | Deployment planning document |

## Quick Reference Commands

### Launch vLLM Server
```bash
cd /home/zhuoyuan/projects/vlm_Qwen2VL_object_detection
python scripts/serve_vllm.py --gpu 2,3
```

### Test vLLM Deployment
```bash
# Health check
curl http://localhost:8000/health

# Test inference
python scripts/test_deployment_inference.py --endpoint vllm --use-dataset
```

### Setup Triton
```bash
python scripts/setup_triton.py

# Launch Triton (Docker)
docker run --gpus all --rm -it \
    --shm-size=4G \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v /ssd1/zhuoyuan/vlm_outputs:/ssd1/zhuoyuan/vlm_outputs:ro \
    -v /ssd1/zhuoyuan/triton_model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3 \
    tritonserver --model-repository=/models
```

## Next Steps

1. **Test vLLM deployment end-to-end** with real images
2. **Run Triton deployment** if enterprise features needed
3. **Performance benchmarking** - measure throughput and latency
4. **Continue Q&A** on Sections 3 & 4 of the notebook

## Session 24b: vLLM Debugging & Environment Fix (2025-12-30)

### Problem Summary

Attempted to launch vLLM server but encountered flash-attn import error. Investigation revealed that installing `vllm>=0.6.0` had silently upgraded PyTorch from 2.4.1 to 2.9.0, breaking flash-attn compatibility.

### Root Cause Analysis

1. **Dec 16**: Committed `environment.yml` with working versions:
   - `torch==2.4.1+cu121`
   - `flash-attn==2.6.3`

2. **Dec 26**: Ran `pip install vllm>=0.6.0` (in deployment notebook)
   - vLLM 0.13.0 was installed
   - vLLM 0.13.0 requires `torch==2.9.0`
   - pip automatically upgraded PyTorch to 2.9.0
   - This broke flash-attn (compiled for PyTorch 2.4.1)

3. **Dec 30**: Discovered the issue when:
   - vLLM server failed to start
   - Inference notebooks also failed with same error

### Error Message

```
ImportError: flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationESs
```

### Why flash-attn Broke

| Component | Compiled Against | Currently Installed | Status |
|-----------|------------------|---------------------|--------|
| flash-attn 2.6.3 | PyTorch 2.4.1 | PyTorch 2.9.0 | ❌ ABI mismatch |

When PyTorch upgrades, its C++ symbols change. Pre-compiled extensions like flash-attn become incompatible.

### Solution Applied

**Restored PyTorch 2.4.1** to match the working environment:

```bash
# Killed Jupyter kernel holding old libraries
kill <pid>

# Cleaned orphan temp directories from failed pip operations
rm -rf /path/to/site-packages/~*

# Restored PyTorch 2.4.1
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
```

**Verified fix:**
```
PyTorch: 2.4.1+cu121
flash_attn: 2.6.3
✅ flash_attn works!
```

### Dev vs Serving Environment Strategy

| Environment | Purpose | Tools |
|-------------|---------|-------|
| **Dev Environment** | Training, evaluation, notebooks | `vlm_Qwen2VL_object_detection` conda env |
| **Serving Environment** | Production API server | Docker container with vLLM |

**Why separate environments?**
- Dev needs flexibility, many libraries, Jupyter support
- Serving needs stability, specific optimized versions
- vLLM 0.13.0 requires PyTorch 2.9.0, but our flash-attn needs 2.4.1
- Separating avoids dependency conflicts

### Recommended vLLM Deployment (Docker)

Use Docker for vLLM serving - industry standard approach:

```bash
docker run --gpus all -p 8000:8000 \
  -v /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged:/model:ro \
  vllm/vllm-openai:v0.6.4.post1 \
  --model /model \
  --served-model-name qwen2vl-nutrition \
  --dtype bfloat16 \
  --max-model-len 4096
```

**Benefits:**
- No impact on dev environment
- Docker images stored on system disk (not home quota)
- Reproducible across servers
- Industry standard for ML deployment

### Lessons Learned

1. **Always pin versions** when installing new packages:
   ```bash
   pip install vllm==0.6.4 --no-deps  # Then manually install compatible deps
   ```

2. **Check what pip will change** before installing:
   ```bash
   pip install vllm --dry-run
   ```

3. **Keep environment.yml updated** after major changes for rollback capability

4. **Separate dev and serving environments** to avoid dependency conflicts

### Current Working State

```
torch==2.4.1+cu121
flash-attn==2.6.3
vllm==0.13.0 (installed but incompatible - use Docker instead)
```

- ✅ Training notebooks work
- ✅ Inference notebooks work (with flash_attention_2)
- ⏳ vLLM deployment: Use Docker approach

### Environment Management Policy

**IMPORTANT: Do NOT modify `environment.yml` or run `conda env export`**

The `environment.yml` file (snapshot date: 2025-12-16) is the "golden" reproducible environment for:
- Training
- Inference
- Evaluation notebooks

**For vLLM/Triton deployment:**
- Use **Docker** (see `notebooks/07_deployment_vllm_triton.ipynb`)
- Do NOT install vLLM into the conda environment
- Installing vLLM upgrades PyTorch and breaks flash-attn compatibility

**Why this policy?**
1. vLLM 0.13.0 requires PyTorch 2.9.0
2. Our flash-attn 2.6.3 was compiled for PyTorch 2.4.1
3. These are incompatible - installing vLLM breaks inference notebooks
4. Docker isolates vLLM dependencies completely

**If you need to add new development packages:**
- Manually edit `environment.yml` to add specific packages
- Do NOT use `conda env export` (it captures unwanted dependencies)

**If the environment gets corrupted:**
```bash
conda env remove -n vlm_Qwen2VL_object_detection
conda env create -f environment.yml
```

## Notes

- The `cleanup()` function in the notebook only affects your own Python process - it cannot kill other users' GPU processes
- rsync uses SSH authentication under the hood - secure by default
- Merged model is ~16GB in bf16, fits comfortably on single 48GB GPU
- Server changed from vllab14 to vllab11 on 2025-12-30
