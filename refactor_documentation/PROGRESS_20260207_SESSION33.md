# Session 33 Progress - Dockerfile for Triton Deployment

**Date**: 2026-02-07
**Session Name**: docker-triton

---

## Objective

Create a Dockerfile that packages the Triton Inference Server with our model configs, so deployment becomes a single `docker run` command instead of the multi-step manual process used in Session 32.

Additionally, clean up stale references in tracked files (old Triton image version, wrong endpoint).

---

## Part 1: Dockerfile + Entrypoint Script

### Problem

In Session 32, deploying to Vast.ai required:
1. Transferring config files (`config.pbtxt`, `model.json`) to the cloud server
2. Updating model paths in `model.json` with `sed`
3. Running a long `docker run` command with multiple `-v` volume mounts for both configs and weights

### Solution

A custom Dockerfile that bakes our configs into the image. Model weights are still mounted at runtime (too large to include in the image).

**Distribution strategy**: The Dockerfile lives in the repo. Users clone the repo, run `docker build`, download weights from HuggingFace, and `docker run`. We do NOT push a pre-built image to Docker Hub — the base Triton image alone is ~20GB, and maintaining a registry adds overhead. For a research project, Dockerfile-in-repo is standard.

### Files Created

**`Dockerfile`** (project root)
- Base image: `nvcr.io/nvidia/tritonserver:26.01-vllm-python-py3` (tested in Session 32)
- Copies both model configs into `/opt/triton_configs/` inside the image
- Uses `docker/entrypoint.sh` as the entrypoint
- Exposes ports 8000 (HTTP), 8001 (gRPC), 8002 (Metrics)
- Defaults to GPTQ model via `CMD ["gptq"]`

**`docker/entrypoint.sh`**
- Takes one argument: `gptq` (default), `bf16`, or `both`
- Copies the selected model config from `/opt/triton_configs/` to `/models/triton_repo/`
- Prints which weight path to mount
- Runs `tritonserver --model-repository=/models/triton_repo`

### Usage

```bash
# Build once
docker build -t qwen2vl-triton .

# Run GPTQ INT4 model
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

### Before vs After

| Aspect | Session 32 (manual) | Session 33 (Dockerfile) |
|--------|---------------------|-------------------------|
| Config transfer | `rsync` configs to cloud | Baked into image |
| Path fixup | `sed` to update model.json | Not needed |
| Docker command | Long `-v` for configs + weights | Only `-v` for weights |
| Reproducibility | Must repeat manual steps | `docker build` + `docker run` |

### Tested on Vast.ai

Built and tested the Dockerfile on the same Vast.ai RTX 4090 server used in Session 32.

**Build:**
```bash
# On Vast.ai: transferred Dockerfile + configs to /workspace/repo/
cd /workspace/repo
docker build -t qwen2vl-triton .
# Build completed in ~1 second (base image already cached from Session 32)
```

**Run:**
```bash
docker run --gpus all --rm -d --shm-size=4G --name triton-test \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v /workspace/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4:/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4:ro \
    qwen2vl-triton gptq
```

**Startup logs confirmed:**
- Entrypoint printed: `=== Starting Triton with GPTQ INT4 model ===`
- vLLM V1 engine (v0.13.0) initialized with `gptq_marlin` backend
- Model loaded in ~26s, 6.46 GiB VRAM
- `enable_prefix_caching=False` confirmed (unbiased config)
- HTTP/gRPC/Metrics services all started

**Benchmark result (GPTQ INT4, c=1, 5 requests, --vary-images):**

| Metric | Value |
|--------|-------|
| Success | 5/5 |
| P50 Latency | 545.8 ms |
| Min Latency | 456.6 ms |
| Max Latency | 3142.2 ms (CUDA graph warmup on first request) |
| Throughput | 0.97 req/s |

Results are consistent with Session 32 (~310ms P50 at steady state after warmup). The higher P50 here is because only 5 requests were sent and the first request (3142ms warmup) skews the median.

---

## Part 2: Stale Reference Cleanup (Complete)

Fixed all tracked files with outdated references. `notebooks/07_deployment_vllm_triton.py` was already up to date (user had updated it).

### Image version: `24.08` → `26.01`

| File | Status |
|------|--------|
| `scripts/deploy_triton.sh` | Fixed |
| `triton_model_repository/README.md` | Fixed |
| `triton_model_repository/PATH_MAPPING.md` | Fixed |
| `scripts/setup_triton.py` | Fixed (2 places) |
| `notebooks/10_Triton.py` + `10_Triton.ipynb` | Fixed |

### Endpoint: `/infer` → `/generate`

| File | Status |
|------|--------|
| `scripts/validate_triton_accuracy.py` | Fixed |
| `scripts/setup_triton.py` | Fixed |
| `triton_model_repository/README.md` | Fixed (table, examples, Important Note) |

**Not fixed** (intentional):
- `scripts/Commented_*.py` — untracked, not visible on GitHub
- `temp_*.txt` — untracked
- `refactor_documentation/` session docs — historical records, kept as-is

---

## Part 3: README and Documentation Updates (Complete)

- Added Triton deployment section to main `README.md` (prerequisites, quick start, model options, inference examples, ports, benchmarks)
- Updated project structure tree in `README.md` (added Dockerfile, docker/, triton_model_repository/, deployment scripts)
- Updated `scripts/README.md` to include all scripts organized by category (Training, Model Preparation, Serving, Benchmarking, Evaluation, Quick Tests)

---

## For README (reference for later)

The following information is needed for the GitHub README's Triton deployment section:

### Prerequisites
- Docker with GPU support (NVIDIA Container Toolkit)
- GPU with 24GB+ VRAM (RTX 4090, A10, etc.)
- Model weights downloaded from HuggingFace

### Quick Start
```bash
# 1. Clone the repo
git clone <repo-url>
cd vlm_Qwen2VL_object_detection

# 2. Download model weights from HuggingFace
# (TODO: add HuggingFace repo URL when confirmed)

# 3. Build the Docker image
docker build -t qwen2vl-triton .

# 4. Run inference server (GPTQ INT4 — faster, recommended)
docker run --gpus all --rm -d --shm-size=4G \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v /path/to/gptq-weights:/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4:ro \
    qwen2vl-triton gptq

# 5. Verify server is running
curl http://localhost:8000/v2/health/live

# 6. Send an inference request
curl -X POST http://localhost:8000/v2/models/qwen2vl_nutrition_gptq_int4/generate \
    -H "Content-Type: application/json" \
    -d '{"text_input": "Detect nutrition facts...", "image": "<base64>", "parameters": {"max_tokens": 100}}'
```

### Options
| Argument | Model | VRAM | Use Case |
|----------|-------|------|----------|
| `gptq` (default) | GPTQ INT4 | ~6.5 GB | Production (faster, ~310ms P50) |
| `bf16` | BF16 full-precision | ~15.5 GB | Accuracy baseline (~538ms P50) |
| `both` | Both models | ~22 GB (needs 2 GPUs) | A/B testing |

### Ports
| Port | Protocol | Purpose |
|------|----------|---------|
| 8000 | HTTP | REST API (`/v2/models/{name}/generate`) |
| 8001 | gRPC | gRPC API |
| 8002 | HTTP | Prometheus metrics |

---

## Files Created This Session

| File | Description |
|------|-------------|
| `Dockerfile` | Triton deployment image definition |
| `docker/entrypoint.sh` | Model selection + Triton startup script |
| `refactor_documentation/SESSION33_Plan_docker_implementation.md` | Implementation plan |
| `refactor_documentation/PROGRESS_20260207_SESSION33.md` | This file |

## Files Modified This Session

| File | What changed |
|------|-------------|
| `README.md` | Added Triton deployment section, updated project structure tree |
| `scripts/README.md` | Added all non-training scripts organized by category |
| `scripts/deploy_triton.sh` | `24.08` → `26.01` |
| `scripts/setup_triton.py` | `24.08` → `26.01`, `/infer` → `/generate` |
| `scripts/validate_triton_accuracy.py` | `/infer` → `/generate` |
| `triton_model_repository/README.md` | `24.08` → `26.01`, `/infer` → `/generate`, fixed Important Note |
| `triton_model_repository/PATH_MAPPING.md` | `24.08` → `26.01` |
| `notebooks/10_Triton.py` + `.ipynb` | `24.08` → `26.01` |

## TODO (for next session)

- Add Model Preparation section to README (merge LoRA + GPTQ quantization)
- Add standalone vLLM Serving section to README
- Add inference performance results to README
