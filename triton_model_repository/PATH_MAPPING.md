# Path Mapping: Lab Server → Cloud Server → Docker Container

This document helps you correctly map paths when deploying to a cloud server with Docker.

## The Three Path Contexts

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PATH CONTEXTS                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. LAB SERVER (where files currently are)                                  │
│     /ssd1/zhuoyuan/vlm_outputs/qwen2vl-...-gptq-int4/                       │
│     /home/zhuoyuan/projects/vlm_Qwen2VL.../triton_model_repository/         │
│                                                                             │
│  2. CLOUD SERVER HOST (where you transfer files to)                         │
│     /models/qwen2vl-...-gptq-int4/                                          │
│     /workspace/triton_model_repository/                                     │
│                                                                             │
│  3. DOCKER CONTAINER (what config files reference)                          │
│     /models/qwen2vl-...-gptq-int4/    ← model.json "model" path             │
│     /models/triton_repo/               ← tritonserver --model-repository    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Path Mapping

### Model Weights

| Context | Path |
|---------|------|
| Lab Server | `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4` |
| Cloud Host | `/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4` |
| Container | `/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4` (same, via mount) |

### Triton Config Repository

| Context | Path |
|---------|------|
| Lab Server | `/home/zhuoyuan/projects/vlm_Qwen2VL_object_detection/triton_model_repository` |
| Cloud Host | `/workspace/triton_model_repository` |
| Container | `/models/triton_repo` (via mount) |

## Docker Volume Mount Mapping

```bash
docker run \
    # Mount model weights: CLOUD_HOST_PATH:CONTAINER_PATH:ro
    -v /models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4:/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4:ro \

    # Mount config repo: CLOUD_HOST_PATH:CONTAINER_PATH
    -v /workspace/triton_model_repository:/models/triton_repo \

    # Tell Triton where to find configs (CONTAINER_PATH)
    tritonserver --model-repository=/models/triton_repo
```

## What to Update Before Deployment

### 1. model.json

The `"model"` field must be the **CONTAINER PATH**:

```json
{
    "model": "/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4"
}
```

This path must match the **right side** of your `-v` mount for model weights.

### 2. deploy_triton.sh (if using)

Update paths to match your cloud server's actual locations.

## Example: Complete Deployment Flow

```bash
# === ON LAB SERVER ===

# 1. Transfer model weights to cloud
rsync -avz --progress \
  /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4/ \
  user@cloud-server:/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4/

# 2. Transfer Triton config
rsync -avz --progress \
  /home/zhuoyuan/projects/vlm_Qwen2VL_object_detection/triton_model_repository/ \
  user@cloud-server:/workspace/triton_model_repository/


# === ON CLOUD SERVER ===

# 3. Verify files exist
ls -la /models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4/
ls -la /workspace/triton_model_repository/

# 4. Start Triton
docker run --gpus all --rm -it \
    --shm-size=4G \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v /models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4:/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4:ro \
    -v /workspace/triton_model_repository:/models/triton_repo \
    nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3 \
    tritonserver --model-repository=/models/triton_repo

# 5. Test
curl http://localhost:8000/v2/health/live
```

## Common Mistakes

### ❌ Wrong: Using lab server path in model.json
```json
{"model": "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-..."}
```
This path doesn't exist inside the Docker container!

### ❌ Wrong: Mismatched mount paths
```bash
# Mount to /data/model
-v /models/qwen2vl:/data/model

# But model.json says /models/qwen2vl
{"model": "/models/qwen2vl"}  # WRONG - doesn't match mount!
```

### ✅ Correct: Consistent paths
```bash
# Mount to /models/qwen2vl
-v /models/qwen2vl:/models/qwen2vl:ro

# model.json matches
{"model": "/models/qwen2vl"}  # CORRECT - matches mount!
```

## Quick Reference Card

```
┌────────────────────────────────────────────────────────────┐
│  FILE              │  WHAT PATH TO USE                      │
├────────────────────────────────────────────────────────────┤
│  model.json        │  CONTAINER path (right side of -v)    │
│  rsync/scp         │  CLOUD HOST path (left side of -v)    │
│  --model-repository│  CONTAINER path                       │
│  curl/client       │  localhost or cloud server IP         │
└────────────────────────────────────────────────────────────┘
```
