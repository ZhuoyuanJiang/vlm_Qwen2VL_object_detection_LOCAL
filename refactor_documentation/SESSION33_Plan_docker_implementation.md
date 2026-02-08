# Session 33 Plan: Dockerfile for Triton Deployment

**Date**: 2026-02-07

---

## Background: What a Dockerfile Does

Think of a Dockerfile as a **recipe for building a Docker image**. The image is a snapshot of a computer with everything pre-installed. When you `docker run` that image, it creates a container (a running instance).

In Session 32, you used a pre-built NVIDIA image (`tritonserver:26.01-vllm-python-py3`) and then manually:

1. Copied config files (`config.pbtxt`, `model.json`) to the right locations on the cloud server
2. Updated model paths in `model.json` with `sed`
3. Ran a long `docker run` command with multiple `-v` volume mounts

A custom Dockerfile **bakes our configs into the image** so deployment becomes simpler — fewer manual steps, fewer things to get wrong.

---

## What We'd Build

**The image contains:**
- NVIDIA's Triton + vLLM (from their base image)
- Our config files (`config.pbtxt` + `model.json`) — already baked in
- A small entrypoint script that lets you pick which model to serve (`gptq` or `bf16`)

**What stays outside the image (mounted at runtime):**
- Model weights (~6.5GB GPTQ, ~15.5GB BF16) — too large to bake in, and you want flexibility to swap them

**End result — deployment goes from this:**
```bash
# Session 32: 5+ manual steps
rsync configs... → sed update paths... → docker run --gpus all -v /this:/that -v /this2:/that2 ...
```

**To this:**
```bash
# With Dockerfile: 1 command
docker run --gpus all -v /path/to/weights:/models/weights qwen2vl-triton gptq
```

---

## Steps

### Step 1: Write the Dockerfile

Extends NVIDIA's Triton image, copies our configs in.

**New file: `Dockerfile`** (project root)
- Base: `nvcr.io/nvidia/tritonserver:26.01-vllm-python-py3` (the tested image from Session 32)
- COPY our 2 model configs (`config.pbtxt` + `model.json`) into `/opt/triton_configs/` inside the image
- COPY the entrypoint script
- EXPOSE ports 8000 (HTTP), 8001 (gRPC), 8002 (Metrics)

### Step 2: Write an Entrypoint Script

A small bash script that selects GPTQ or BF16 model and starts Triton.

**New file: `docker/entrypoint.sh`**
- Takes one argument: `gptq` (default), `bf16`, or `both`
- Copies the selected model config from `/opt/triton_configs/` to `/models/triton_repo/`
- Runs `tritonserver --model-repository=/models/triton_repo`
- Prints helpful messages about what to mount and where

**Usage after building:**
```bash
# Build once
docker build -t qwen2vl-triton .

# Run GPTQ model
docker run --gpus all --rm -d --shm-size=4G \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v /path/to/gptq-weights:/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4:ro \
    qwen2vl-triton gptq

# Run BF16 model
docker run --gpus all --rm -d --shm-size=4G \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v /path/to/bf16-weights:/models/qwen2vl-nutrition-detection-r4-joint-merged:ro \
    qwen2vl-triton bf16
```

### Step 3: Fix Stale References in Tracked Files

While exploring for the Dockerfile, we found several tracked files (visible on GitHub) with outdated references:

#### Image version: `24.08` → `26.01`

| File | What to fix |
|------|-------------|
| `scripts/deploy_triton.sh` | Line 36: `TRITON_IMAGE` variable |
| `triton_model_repository/README.md` | Line 72: docker run example |
| `triton_model_repository/PATH_MAPPING.md` | Line 105: docker run example |
| `scripts/setup_triton.py` | Line 23: docstring example |
| `notebooks/07_deployment_vllm_triton.py` | Lines 866, 876, 1024 |

#### Endpoint: `/infer` → `/generate`

| File | What to fix |
|------|-------------|
| `scripts/validate_triton_accuracy.py` | Line 154: endpoint URL |
| `scripts/test_deployment_inference.py` | Line 262: endpoint URL |
| `scripts/setup_triton.py` | docstring output |
| `triton_model_repository/README.md` | Lines 27-28, 112, 124-125, 142 |
| `notebooks/07_deployment_vllm_triton.py` | Line 975 |

**Not fixing** (untracked or historical records):
- `scripts/Commented_*.py` — untracked, won't appear on GitHub
- `temp_*.txt` — untracked
- `refactor_documentation/` session docs — historical records, keep as-is

### Step 4: Update deploy_triton.sh

Beyond fixing the image version, also update `deploy_triton.sh` to reference the custom Dockerfile image as an alternative, and align model path conventions with `model.json`.

### Step 5: Test

- `docker build -t qwen2vl-triton .` should succeed (doesn't need GPU)
- Grep the repo for remaining `24.08` or `/infer` references in tracked files to confirm all fixed
- (Optional) If Vast.ai server is available: run the built image with actual weights to verify end-to-end

### Step 6: Document in Session 33

Create `refactor_documentation/PROGRESS_20260207_SESSION33.md` documenting what was done.

---

## Files to Create

1. `Dockerfile`
2. `docker/entrypoint.sh`
3. `refactor_documentation/PROGRESS_20260207_SESSION33.md`

## Files to Modify

1. `scripts/deploy_triton.sh` — image version + path conventions
2. `triton_model_repository/README.md` — image version + endpoints
3. `triton_model_repository/PATH_MAPPING.md` — image version
4. `scripts/setup_triton.py` — image version + endpoint in docstring
5. `scripts/validate_triton_accuracy.py` — `/infer` → `/generate`
6. `scripts/test_deployment_inference.py` — `/infer` → `/generate`
7. `notebooks/07_deployment_vllm_triton.py` — image version + endpoint
