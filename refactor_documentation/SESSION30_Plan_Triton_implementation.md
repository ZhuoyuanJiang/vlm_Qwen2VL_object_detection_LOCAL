# SESSION30 Plan: NVIDIA Triton Implementation

**Created**: 2026-01-27
**Related Documentation**:
- [notebooks/10_Triton.ipynb](../notebooks/10_Triton.ipynb) - Educational material about Triton
- [PROGRESS_20260118_SESSION29.md](./PROGRESS_20260118_SESSION29.md) - GPTQ quantization experiments

---

## Objective & Goals

### Primary Goals (In Order of Priority)

1. **Learning Goal**: Understand what NVIDIA Triton is
   - What is Triton and how does it relate to vLLM?
   - When does Triton provide value vs standalone vLLM?
   - Gain a holistic understanding of the model serving ecosystem

2. **Production Goal**: Make it production-ready
   - Deploy the fine-tuned Qwen2-VL model to Triton
   - Understand production deployment best practices
   - Learn what enterprise features Triton provides

3. **Comparison Goal**: Compare performance with standalone vLLM (if applicable)
   - Understand if Triton adds overhead or provides speedups
   - Benchmark Triton vs standalone vLLM

---

## Current State

### Models Available

| Model | Path | Size | Notes |
|-------|------|------|-------|
| **BF16 (baseline)** | `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged` | ~15.5 GB | Full precision merged model |
| **GPTQ INT4** | `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4` | 6.45 GB | Quantized, vision encoder in BF16 |

### Session 29 Quantization Results (Completed)

| Metric | BF16 | GPTQ INT4 | Change |
|--------|------|-----------|--------|
| Model Size | 15.53 GB | 6.46 GB | **2.4x smaller** |
| KV Cache Capacity | 80,144 tokens | 249,936 tokens | **3.1x larger** |
| Mean IoU (accuracy) | 0.8458 | 0.8395 | **-0.74%** (negligible) |
| Detection Rate | 100% | 100% | Same |
| Throughput (c=1) | 0.86 req/s | 1.15 req/s | **+34%** |
| E2E Latency (c=1) | 1118 ms | 819 ms | **-27%** |
| TPOT (c=1) | 20.3 ms | 7.4 ms | **-64%** |

**Recommendation from Session 29**: Use GPTQ INT4 for production - negligible accuracy loss with significant speed and memory benefits.

### Current vLLM Serving (Working)

```bash
# BF16 model
CUDA_VISIBLE_DEVICES=0 vllm serve \
   /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
   --served-model-name qwen2vl-nutrition \
   --dtype bfloat16 \
   --trust-remote-code \
   --max-model-len 4096 \
   --limit-mm-per-prompt '{"image":1}' \
   --gpu-memory-utilization 0.9 \
   --port 8000

# GPTQ INT4 model
CUDA_VISIBLE_DEVICES=0 vllm serve \
   /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4 \
   --served-model-name qwen2vl-nutrition \
   --quantization gptq_marlin \
   --dtype half \
   --trust-remote-code \
   --max-model-len 4096 \
   --limit-mm-per-prompt '{"image":1}' \
   --gpu-memory-utilization 0.9 \
   --port 8000
```

---

## What is NVIDIA Triton?

### Definition

- **vLLM** = The inference engine (does the actual computation)
- **Triton** = The orchestration layer (manages models, routes requests, monitors health)

When using the vLLM backend, Triton essentially wraps vLLM. **Inference performance is nearly identical**—Triton adds management features, not speed.

### Architecture Overview

```
                    ┌─────────────────────────────────────────┐
                    │         NVIDIA Triton Server            │
                    │                                         │
   HTTP :8000 ─────►│  ┌─────────────────────────────────┐   │
   gRPC :8001 ─────►│  │     Request Router/Scheduler     │   │
Metrics :8002 ─────►│  └──────────────┬──────────────────┘   │
                    │                 │                       │
                    │    ┌────────────┼────────────┐          │
                    │    ▼            ▼            ▼          │
                    │ ┌──────┐   ┌──────┐    ┌──────┐        │
                    │ │Model │   │Model │    │Model │        │
                    │ │  v1  │   │  v2  │    │ GPTQ │        │
                    │ │(BF16)│   │(AWQ) │    │(INT4)│        │
                    │ └──────┘   └──────┘    └──────┘        │
                    │                                         │
                    │  Backends: vLLM, TensorRT, PyTorch...   │
                    └─────────────────────────────────────────┘
```

### Key Features

| Feature | What It Does | Why It Matters |
|---------|-------------|----------------|
| **Multi-model serving** | Serve BF16 + GPTQ models simultaneously | A/B testing, gradual rollout |
| **Model versioning** | `/models/qwen2vl/1/`, `/models/qwen2vl/2/` | Easy rollback, canary deployments |
| **gRPC endpoint** | Binary protocol, lower latency than HTTP | ~10-20% faster for high-throughput |
| **Prometheus metrics** | Built-in `/metrics` endpoint | Grafana dashboards, alerting |
| **Health checks** | `/v2/health/live`, `/v2/health/ready` | Kubernetes integration |
| **Dynamic batching** | Triton can batch requests (or defer to vLLM) | Flexibility in batching strategy |
| **Model ensembles** | Chain models (e.g., preprocessor → LLM → postprocessor) | Complex pipelines |

### When to Use Triton vs Standalone vLLM

| Scenario | Recommendation |
|----------|---------------|
| Single model, simple deployment | **Standalone vLLM** (simpler) |
| Multiple model versions in production | **Triton** |
| Need gRPC endpoint | **Triton** |
| Kubernetes/cloud-native deployment | **Triton** (better integration) |
| Quick prototyping | **Standalone vLLM** |
| Enterprise monitoring requirements | **Triton** |

### Performance: Triton vs Standalone vLLM

When using the vLLM backend:
- **Inference speed**: Nearly identical (same vLLM engine)
- **Overhead**: Small (~1-5% for request routing)
- **Throughput**: Same continuous batching benefits
- **Latency**: Slight increase due to extra layer

**Bottom line**: You don't use Triton for speed—you use it for **operational features** (versioning, monitoring, multi-model).

---

## Constraints

### Infrastructure Limitations

| Constraint | Details |
|------------|---------|
| **No Docker on lab server** | No sudo permissions on vllab8 |
| **Models on local SSD** | `/ssd1/zhuoyuan/vlm_outputs/` |
| **GPU available** | RTX 3090 x8 on vllab8 (but no Docker) |

### Strategy

1. **Rent a cloud GPU** with Docker support for Triton deployment
2. **Minimize cloud costs** by preparing as much as possible locally
3. **Transfer model weights** to cloud server when ready

### Cloud GPU Options

| Provider | Est. Cost | Notes |
|----------|-----------|-------|
| **Lambda Labs** | ~$1.10/hr (A10) | Easy setup, good prices |
| **RunPod** | ~$0.50-2/hr | Pay-per-use, pre-configured containers |
| **Vast.ai** | ~$0.30-1/hr | Cheapest GPU rentals |
| **AWS/GCP/Azure** | $1-4/hr | Enterprise options |

---

## Implementation Plan

### Phase 1: Local Preparation (No Docker/GPU Required)

**Goal**: Prepare all configuration files and scripts locally before renting cloud GPU.

**Task 1.1: Create Triton model repository structure**

```
triton_model_repository/
└── qwen2vl_nutrition/
    ├── 1/
    │   └── model.json          # vLLM backend config
    └── config.pbtxt            # Triton model config
```

**Task 1.2: Create `model.json` (vLLM backend configuration)**

```json
{
  "model": "/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4",
  "dtype": "half",
  "quantization": "gptq_marlin",
  "tensor_parallel_size": 1,
  "gpu_memory_utilization": 0.9,
  "max_model_len": 4096,
  "trust_remote_code": true
}
```

**Task 1.3: Create `config.pbtxt` (Triton model configuration)**

```protobuf
name: "qwen2vl_nutrition"
backend: "vllm"
max_batch_size: 0

input [
  {
    name: "text_input"
    data_type: TYPE_STRING
    dims: [ 1 ]
  },
  {
    name: "image"
    data_type: TYPE_STRING
    dims: [ 1 ]
    optional: true
  },
  {
    name: "sampling_parameters"
    data_type: TYPE_STRING
    dims: [ 1 ]
    optional: true
  }
]

output [
  {
    name: "text_output"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_MODEL
  }
]
```

**Task 1.4: Create deployment script (`scripts/deploy_triton.sh`)**

```bash
#!/bin/bash
# Triton deployment script for cloud GPU

MODEL_REPO="/models/triton_model_repository"
MODEL_WEIGHTS="/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4"

docker run --gpus all --rm -it \
    --shm-size=4G \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v ${MODEL_WEIGHTS}:${MODEL_WEIGHTS}:ro \
    -v ${MODEL_REPO}:/models/triton_model_repository \
    nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3 \
    tritonserver --model-repository=/models/triton_model_repository
```

**Task 1.5: Create Triton benchmark client (`scripts/benchmark_triton.py`)**

- HTTP client using `requests` library
- gRPC client using `tritonclient` library
- Metrics collection from Triton `/metrics` endpoint
- Compare with existing vLLM benchmark results

**Task 1.6: Create accuracy validation script (`scripts/validate_triton_accuracy.py`)**

- Send same validation slice to Triton
- Compare outputs with vLLM baseline
- Verify no accuracy drift from serving layer change

### Phase 2: Cloud GPU Setup

**Task 2.1: Rent cloud GPU instance**

Requirements:
- GPU: A10 or better (24GB+ VRAM)
- Docker support with GPU access
- ~50GB storage for model weights

**Task 2.2: Transfer model weights**

```bash
# From lab server to cloud
rsync -avz --progress \
  /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4 \
  user@cloud-server:/models/
```

**Task 2.3: Transfer configuration files**

```bash
rsync -avz --progress \
  triton_model_repository/ \
  scripts/deploy_triton.sh \
  scripts/benchmark_triton.py \
  user@cloud-server:/workspace/
```

### Phase 3: Triton Deployment

**Task 3.1: Pull Triton Docker image**

```bash
docker pull nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3
```

**Task 3.2: Start Triton server**

```bash
./scripts/deploy_triton.sh
```

**Task 3.3: Verify deployment**

```bash
# Health check
curl http://localhost:8000/v2/health/live

# Model info
curl http://localhost:8000/v2/models/qwen2vl_nutrition

# Test inference
curl -X POST http://localhost:8000/v2/models/qwen2vl_nutrition/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "text_input", "shape": [1], "datatype": "BYTES", "data": ["test prompt"]}]}'
```

### Phase 4: Benchmarking & Validation

**Task 4.1: Run accuracy validation**

```bash
python scripts/validate_triton_accuracy.py \
  --triton-url http://localhost:8000 \
  --baseline-outputs /path/to/vllm_baseline_outputs.json
```

Expected: 100% output match with vLLM (same model, same weights)

**Task 4.2: Run performance benchmarks**

```bash
# HTTP endpoint
python scripts/benchmark_triton.py \
  --endpoint http \
  --url http://localhost:8000 \
  --concurrency 1 4 8 \
  --output triton_http_benchmark.json

# gRPC endpoint
python scripts/benchmark_triton.py \
  --endpoint grpc \
  --url localhost:8001 \
  --concurrency 1 4 8 \
  --output triton_grpc_benchmark.json
```

**Task 4.3: Compare with standalone vLLM**

Create comparison table:

| Metric | Standalone vLLM | Triton (HTTP) | Triton (gRPC) |
|--------|-----------------|---------------|---------------|
| Throughput (c=1) | 1.15 req/s | TBD | TBD |
| E2E Latency (c=1) | 819 ms | TBD | TBD |
| TTFT (c=1) | 651 ms | TBD | TBD |

### Phase 5: Documentation

**Task 5.1: Create progress documentation**

Create `refactor_documentation/PROGRESS_YYYYMMDD_SESSION30.md`:
- Configuration details
- Deployment steps
- Benchmark results
- Comparison with standalone vLLM
- Recommendations

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `triton_model_repository/qwen2vl_nutrition/config.pbtxt` | Create | Triton model configuration |
| `triton_model_repository/qwen2vl_nutrition/1/model.json` | Create | vLLM backend configuration |
| `scripts/deploy_triton.sh` | Create | Triton deployment script |
| `scripts/benchmark_triton.py` | Create | Triton benchmark client (HTTP + gRPC) |
| `scripts/validate_triton_accuracy.py` | Create | Accuracy validation against vLLM baseline |
| `refactor_documentation/PROGRESS_*_SESSION30.md` | Create | Results documentation |

---

## Verification Plan

1. **Configuration validity**: `config.pbtxt` parses without errors
2. **Server health**: `/v2/health/live` returns 200
3. **Model loaded**: `/v2/models/qwen2vl_nutrition` shows model info
4. **Inference works**: Test request returns valid bbox output
5. **Accuracy preserved**: 100% output match with vLLM baseline
6. **Performance measured**: Benchmark results collected for comparison

---

## Questions to Resolve

1. **Model to deploy**: GPTQ INT4 (recommended) or BF16?
2. **Single model or multi-model**: Just GPTQ, or both BF16 + GPTQ for A/B testing?
3. **Cloud provider**: Lambda Labs vs RunPod vs Vast.ai?
4. **Benchmark scope**: Just verify it works, or comprehensive performance comparison?

---

## Triton API Reference

### Health Endpoints

```bash
# Liveness (server running)
GET /v2/health/live

# Readiness (model loaded)
GET /v2/health/ready

# Model-specific readiness
GET /v2/models/{model_name}/ready
```

### Inference Endpoint

```bash
POST /v2/models/{model_name}/infer
Content-Type: application/json

{
  "inputs": [
    {
      "name": "text_input",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["Your prompt here"]
    },
    {
      "name": "image",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["base64_encoded_image"]
    }
  ]
}
```

### Metrics Endpoint

```bash
# Prometheus metrics
GET :8002/metrics
```

---

## Official Documentation

- [Triton Inference Server Docs](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/)
- [Triton vLLM Backend](https://github.com/triton-inference-server/vllm_backend)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Triton Quick Deploy with vLLM](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Quick_Deploy/vLLM/README.html)

---

## Appendix A: VLM-Specific Considerations (Critical for Qwen2-VL)

### How Triton vLLM Backend Handles Images

The vLLM backend has built-in VLM support:

1. **Image Input Tensor**: The backend defines an optional `image` input tensor
2. **Base64 Encoding**: Images must be sent as base64-encoded strings
3. **PIL Conversion**: Backend decodes base64 → PIL RGB image
4. **vLLM Integration**: Sends to vLLM as `multi_modal_data: {"image": [PIL_image]}`

```python
# How the backend processes images (from vLLM backend source)
# 1. Receive base64 string
# 2. Decode: base64.b64decode(image_data)
# 3. Convert: PIL.Image.open(BytesIO(decoded)).convert("RGB")
# 4. Pass to vLLM: {"prompt": text, "multi_modal_data": {"image": [pil_image]}}
```

### Two API Shapes for Triton

| API Endpoint | Use Case | VLM Image Support |
|--------------|----------|-------------------|
| `/v2/models/<name>/generate` | Text-only LLMs | ❌ No image tensor |
| `/v2/models/<name>/infer` | VLMs with images | ✅ Has `image` tensor |

**For our VLM, we MUST use the `/infer` endpoint, NOT `/generate`.**

### Decoupled Transaction Policy

The vLLM backend uses "decoupled mode" for streaming capability:
- Even for non-streaming requests, the backend uses decoupled transactions
- Sample clients use `stream_infer` for gRPC
- This is a known limitation of the reference setup

### Chat Template Consideration

**Important**: If sending raw prompts without chat template formatting:
- Behavior may differ from vLLM's OpenAI chat/completions endpoint
- Our current vLLM setup uses OpenAI API which auto-applies chat template

**Options**:
1. Apply chat template client-side before sending to Triton
2. Use Triton's OpenAI-compatible frontend (port 9000) - but VLM support needs validation
3. Format prompts exactly as the model expects

### Example VLM Request to Triton

```python
import base64
import requests

# Encode image to base64
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

# Triton inference request
payload = {
    "inputs": [
        {
            "name": "text_input",
            "shape": [1],
            "datatype": "BYTES",
            "data": ["Detect the nutrition facts table in this image."]
        },
        {
            "name": "image",
            "shape": [1],
            "datatype": "BYTES",
            "data": [image_b64]
        },
        {
            "name": "sampling_parameters",
            "shape": [1],
            "datatype": "BYTES",
            "data": ['{"temperature": 0, "max_tokens": 100}']
        },
        {
            "name": "stream",
            "shape": [1],
            "datatype": "BOOL",
            "data": [False]
        }
    ]
}

response = requests.post(
    "http://localhost:8000/v2/models/qwen2vl_nutrition/infer",
    json=payload
)
```

---

## Appendix B: Model Control Modes

Triton supports different model lifecycle management modes:

| Mode | Behavior | Use Case |
|------|----------|----------|
| `none` (default) | Load all models at startup | Simple, single model |
| `explicit` | Load/unload via API | Multi-model, memory management |
| `poll` | Watch repo for changes | Development only, not production |

**For multi-model (BF16 + GPTQ on different GPUs)**:
```bash
tritonserver \
  --model-repository=/models \
  --model-control-mode=explicit
```

Then load specific models via API:
```bash
# Load GPTQ model
curl -X POST http://localhost:8000/v2/repository/models/qwen2vl_nutrition_gptq/load

# Unload BF16 model (free memory)
curl -X POST http://localhost:8000/v2/repository/models/qwen2vl_nutrition_bf16/unload
```

---

## Appendix C: GenAI-Perf Benchmarking Tool

NVIDIA provides **GenAI-Perf** for standardized LLM/VLM benchmarking:

```bash
# Install
pip install genai-perf

# Run benchmark against Triton
genai-perf \
  -m qwen2vl_nutrition \
  --service-kind triton \
  --backend vllm \
  --streaming \
  --concurrency 1 \
  --measurement-interval 10000 \
  --url localhost:8001  # gRPC port
```

**Metrics provided**:
- Time to First Token (TTFT)
- Inter-token latency
- Request throughput
- Output token throughput

**Note**: GenAI-Perf supports VLMs - verify image input support for your specific version.

---

## Appendix D: Security Considerations

### trust_remote_code

Our Qwen2-VL model requires `trust_remote_code=true`:
- Treat the model directory as executable code
- Verify model provenance before deployment
- In production, consider building a custom image with pre-verified code

### Request Size Limits

Base64-encoded images can be large:
- A 1MB image becomes ~1.33MB in base64
- Set appropriate request timeouts
- Consider image compression/resizing client-side

### API Authentication

Triton's OpenAI-compatible frontend supports header-based auth:
```bash
# Restrict access via headers
--openai-api-key=your-secret-key
```

For production, add authentication layer (API gateway, reverse proxy).
