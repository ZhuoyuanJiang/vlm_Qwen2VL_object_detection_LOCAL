# Session 30 Progress - NVIDIA Triton Deployment

**Date**: 2026-02-02
**Session Name**: triton

---

## Objective

Deploy the fine-tuned Qwen2-VL nutrition detection model to NVIDIA Triton Inference Server with vLLM backend. This session focuses on:

1. **Learning**: Understanding what Triton is and how it relates to vLLM
2. **Production Deployment**: Creating configuration files and scripts for production-ready deployment
3. **A/B Testing**: Deploying both BF16 and GPTQ INT4 models for comparison

---

## Background: Why Triton?

### Triton vs Standalone vLLM

```
┌─────────────────────────────────────────────────────────┐
│                    DEPLOYMENT OPTIONS                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Option A: Standalone vLLM (current setup)              │
│  ┌─────────────────────────────────────────────┐        │
│  │  vLLM Server                                │        │
│  │  - OpenAI-compatible API                    │        │
│  │  - Continuous batching                      │        │
│  │  - Single model focus                       │        │
│  └─────────────────────────────────────────────┘        │
│                                                         │
│  Option B: Triton + vLLM Backend (this session)         │
│  ┌─────────────────────────────────────────────┐        │
│  │  Triton Inference Server                    │        │
│  │  ┌───────────────────────────────────────┐  │        │
│  │  │  vLLM Backend (same inference engine) │  │        │
│  │  └───────────────────────────────────────┘  │        │
│  │  + Model versioning & A/B testing          │        │
│  │  + Multi-model serving                     │        │
│  │  + gRPC + HTTP endpoints                   │        │
│  │  + Enterprise monitoring (Prometheus)      │        │
│  └─────────────────────────────────────────────┘        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Key Insight**: Triton wraps vLLM as a backend. Inference performance is nearly identical (~1-5% overhead). Triton adds **operational features**, not speed.

### When to Use Triton

| Scenario | Recommendation |
|----------|---------------|
| Single model, simple deployment | Standalone vLLM (simpler) |
| Multiple model versions in production | **Triton** |
| Need gRPC endpoint | **Triton** |
| Kubernetes/cloud-native deployment | **Triton** |
| Quick prototyping | Standalone vLLM |
| Enterprise monitoring requirements | **Triton** |

---

## Environment

- **Lab Server**: vllab8 (no Docker access - no sudo permissions)
- **Deployment Target**: Cloud GPU with Docker support
- **Triton Image**: `nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3`
- **Models**: BF16 baseline + GPTQ INT4 (from Session 29)

---

## Models

| Model | Path | Size | GPU Assignment |
|-------|------|------|----------------|
| BF16 Baseline | `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged` | ~15.5 GB | GPU 0 |
| GPTQ INT4 | `/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4` | 6.45 GB | GPU 1 |

---

## Files Created

### 1. Educational Materials

#### `notebooks/10_Triton.ipynb` and `notebooks/10_Triton.py`

Educational notebook explaining:
- What is Triton Inference Server
- Architecture overview with ASCII diagrams
- Key features (multi-model, versioning, gRPC, Prometheus)
- When to use Triton vs standalone vLLM
- Performance comparison

---

### 2. Planning Documents

#### `refactor_documentation/SESSION30_Plan_Triton_implementation.md`

Comprehensive implementation plan including:
- Objectives and goals
- Current state summary
- Triton architecture explanation
- Infrastructure constraints
- Phase-by-phase implementation plan
- VLM-specific considerations (Appendix A)
- Model control modes (Appendix B)
- GenAI-Perf benchmarking tool (Appendix C)
- Security considerations (Appendix D)

---

### 3. Triton Model Repository

#### Directory Structure

```
triton_model_repository/
├── README.md                              # Quick start guide
├── PATH_MAPPING.md                        # Path mapping for cloud deployment
│
├── qwen2vl_nutrition_bf16/                # BF16 full-precision model
│   ├── config.pbtxt                       # Triton configuration (GPU 0)
│   └── 1/                                 # Version 1
│       └── model.json                     # vLLM engine configuration
│
└── qwen2vl_nutrition_gptq_int4/           # GPTQ INT4 quantized model
    ├── config.pbtxt                       # Triton configuration (GPU 1)
    └── 1/                                 # Version 1
        └── model.json                     # vLLM engine configuration
```

#### Why the `1/` Directory?

The `1/` is a **version number**. Triton supports multiple versions of the same model:

```
qwen2vl_nutrition_gptq_int4/
├── config.pbtxt           # Shared across all versions
├── 1/                     # Version 1
│   └── model.json
├── 2/                     # Version 2 (future: retrained model)
│   └── model.json
└── 3/                     # Version 3
    └── model.json
```

**API endpoints with versioning**:
```bash
# Use latest version (default)
POST /v2/models/qwen2vl_nutrition_gptq_int4/infer

# Use specific version
POST /v2/models/qwen2vl_nutrition_gptq_int4/versions/1/infer
POST /v2/models/qwen2vl_nutrition_gptq_int4/versions/2/infer
```

---

### 4. Configuration Files

#### `config.pbtxt` - Triton Model Configuration

This file defines the Triton-level configuration for each model.

**Key sections explained**:

```protobuf
# Model identity
name: "qwen2vl_nutrition_gptq_int4"    # Used in API endpoints
backend: "vllm"                         # Use vLLM as inference backend

# Batching (0 = let vLLM handle it)
max_batch_size: 0

# Input tensors (API contract)
input [
  {
    name: "text_input"                  # Required: text prompt
    data_type: TYPE_STRING
    dims: [ 1 ]
  },
  {
    name: "image"                       # Optional: base64-encoded image
    data_type: TYPE_STRING
    dims: [ 1 ]
    optional: true
  },
  {
    name: "sampling_parameters"         # Optional: JSON with temp, max_tokens
    data_type: TYPE_STRING
    dims: [ 1 ]
    optional: true
  },
  {
    name: "stream"                      # Optional: enable streaming
    data_type: TYPE_BOOL
    dims: [ 1 ]
    optional: true
  }
]

# Output tensor
output [
  {
    name: "text_output"
    data_type: TYPE_STRING
    dims: [ -1 ]                        # Variable length
  }
]

# GPU placement
instance_group [
  {
    count: 1
    kind: KIND_MODEL
    gpus: [ 1 ]                         # Pin to specific GPU
  }
]

# Required for vLLM streaming support
model_transaction_policy {
  decoupled: true
}
```

#### `model.json` - vLLM Engine Configuration

This file maps directly to vLLM CLI flags:

| vLLM CLI Flag | model.json Key |
|---------------|----------------|
| `--model PATH` | `"model": "PATH"` |
| `--dtype bfloat16` | `"dtype": "bfloat16"` |
| `--quantization gptq_marlin` | `"quantization": "gptq_marlin"` |
| `--gpu-memory-utilization 0.9` | `"gpu_memory_utilization": 0.9` |
| `--max-model-len 4096` | `"max_model_len": 4096` |
| `--trust-remote-code` | `"trust_remote_code": true` |
| `--limit-mm-per-prompt '{"image":1}'` | `"limit_mm_per_prompt": {"image": 1}` |

**BF16 model.json**:
```json
{
    "model": "/models/qwen2vl-nutrition-detection-r4-joint-merged",
    "tokenizer_mode": "auto",
    "trust_remote_code": true,
    "dtype": "bfloat16",
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 4096,
    "limit_mm_per_prompt": {"image": 1}
}
```

**GPTQ INT4 model.json**:
```json
{
    "model": "/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4",
    "tokenizer_mode": "auto",
    "trust_remote_code": true,
    "dtype": "half",
    "quantization": "gptq_marlin",
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 4096,
    "limit_mm_per_prompt": {"image": 1}
}
```

---

### 5. Deployment Scripts

#### `scripts/deploy_triton.sh`

Shell script to start Triton server with correct Docker mounts.

**Features**:
- Pre-flight checks (Docker, GPU support, path existence)
- Single model or dual-model mode
- Detached mode support
- Configurable paths at top of script

**Usage**:
```bash
# Start with both models (default)
./deploy_triton.sh

# Single model mode - GPTQ only
./deploy_triton.sh --single gptq

# Single model mode - BF16 only
./deploy_triton.sh --single bf16

# Run in background (detached)
./deploy_triton.sh --detach

# Show help
./deploy_triton.sh --help
```

**Configuration** (edit at top of script):
```bash
# Model weights paths (on cloud server host)
BF16_MODEL_PATH="/models/qwen2vl-nutrition-detection-r4-joint-merged"
GPTQ_MODEL_PATH="/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4"

# Triton model repository path (on cloud server host)
TRITON_REPO_PATH="/workspace/triton_model_repository"

# Docker image
TRITON_IMAGE="nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3"
```

**What it runs**:
```bash
docker run --gpus all --rm -it \
    --shm-size=4G \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v ${BF16_MODEL_PATH}:${BF16_MODEL_PATH}:ro \
    -v ${GPTQ_MODEL_PATH}:${GPTQ_MODEL_PATH}:ro \
    -v ${TRITON_REPO_PATH}:/models/triton_repo \
    nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3 \
    tritonserver --model-repository=/models/triton_repo
```

---

#### `scripts/benchmark_triton.py`

Benchmark script for Triton HTTP and gRPC endpoints.

**Features**:
- HTTP and gRPC endpoint support
- Configurable concurrency
- Multiple model benchmarking
- Latency percentiles (p50, p90, p99)
- JSON output for analysis

**Usage**:
```bash
# Basic HTTP benchmark
python scripts/benchmark_triton.py --model qwen2vl_nutrition_gptq_int4

# Benchmark both models
python scripts/benchmark_triton.py \
    --model qwen2vl_nutrition_bf16 qwen2vl_nutrition_gptq_int4

# gRPC benchmark (requires tritonclient)
python scripts/benchmark_triton.py \
    --endpoint grpc \
    --model qwen2vl_nutrition_gptq_int4

# High concurrency test
python scripts/benchmark_triton.py \
    --num-requests 50 \
    --concurrency 8 \
    --vary-images \
    --output benchmark_results.json

# Custom Triton URL
python scripts/benchmark_triton.py \
    --http-url http://cloud-server:8000 \
    --grpc-url cloud-server:8001 \
    --model qwen2vl_nutrition_gptq_int4
```

**Arguments**:
| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `qwen2vl_nutrition_gptq_int4` | Model name(s) to benchmark |
| `--endpoint` | `http` | Endpoint type: `http`, `grpc`, or `both` |
| `--http-url` | `http://localhost:8000` | Triton HTTP URL |
| `--grpc-url` | `localhost:8001` | Triton gRPC URL |
| `--num-requests` | `20` | Number of requests to send |
| `--concurrency` | `1` | Number of concurrent requests |
| `--vary-images` | `False` | Use different images for each request |
| `--output` | `None` | Output JSON file path |

**Metrics Collected**:
- Throughput (req/s)
- Latency: avg, min, max, p50, p90, p99
- Success/failure counts
- Total time

**Sample Output**:
```
============================================================
BENCHMARK RESULTS: qwen2vl_nutrition_gptq_int4 (HTTP)
============================================================

[Configuration]
  Requests:    20
  Concurrency: 4

[Results]
  Successful:  20/20
  Failed:      0
  Total time:  8.45s
  Throughput:  2.37 req/s

[Latency (successful requests)]
  Avg:  1684.2 ms
  Min:  812.3 ms
  Max:  2156.8 ms
  P50:  1623.5 ms
  P90:  2089.4 ms
  P99:  2145.2 ms
============================================================
```

---

#### `scripts/validate_triton_accuracy.py`

Accuracy validation script comparing Triton outputs vs vLLM baseline.

**Features**:
- Compares against cached baseline outputs
- Calculates IoU metrics
- Exact match and bbox match rates
- Identifies mismatched samples

**Usage**:
```bash
# Basic validation
python scripts/validate_triton_accuracy.py \
    --triton-url http://localhost:8000 \
    --model qwen2vl_nutrition_gptq_int4 \
    --baseline /ssd1/zhuoyuan/vlm_outputs/quantization_experiments/bf16_baseline_outputs.json

# With output file
python scripts/validate_triton_accuracy.py \
    --triton-url http://localhost:8000 \
    --model qwen2vl_nutrition_gptq_int4 \
    --baseline /path/to/baseline.json \
    --output triton_validation_results.json

# Validate subset of samples
python scripts/validate_triton_accuracy.py \
    --triton-url http://localhost:8000 \
    --model qwen2vl_nutrition_bf16 \
    --baseline /path/to/baseline.json \
    --num-samples 50
```

**Arguments**:
| Argument | Default | Description |
|----------|---------|-------------|
| `--triton-url` | `http://localhost:8000` | Triton server URL |
| `--model` | `qwen2vl_nutrition_gptq_int4` | Model name on Triton |
| `--baseline` | Required | Path to baseline outputs JSON |
| `--num-samples` | All | Number of samples to validate |
| `--output` | `None` | Output JSON file path |

**Metrics Calculated**:
- Exact match rate (string comparison)
- Bbox match rate (parsed coordinates)
- Mean/Median IoU
- IoU > 0.5 rate
- IoU > 0.7 rate
- Average latency

**Sample Output**:
```
============================================================
VALIDATION RESULTS
============================================================

Model: qwen2vl_nutrition_gptq_int4
Samples: 100

[Match Rates]
  Exact match:  98.0% (98/100)
  Bbox match:   100.0% (100/100)

[IoU Metrics]
  Mean IoU:     0.9856
  Median IoU:   0.9912
  IoU > 0.5:    100.0%
  IoU > 0.7:    100.0%

[Latency]
  Avg latency:  824.5 ms

[Validation Status]
  ⚠ 2 outputs differ from baseline
============================================================
```

---

## VLM-Specific Considerations

### Image Handling in Triton vLLM Backend

The vLLM backend has built-in VLM support:

1. **Image Input Tensor**: Optional `image` input tensor
2. **Base64 Encoding**: Images must be sent as base64-encoded strings
3. **PIL Conversion**: Backend decodes base64 → PIL RGB image
4. **vLLM Integration**: Sends to vLLM as `multi_modal_data: {"image": [PIL_image]}`

### API Endpoint for VLMs

**Important**: Use `/infer` endpoint, NOT `/generate`:

| API Endpoint | VLM Support |
|--------------|-------------|
| `/v2/models/<name>/generate` | ❌ No image tensor |
| `/v2/models/<name>/infer` | ✅ Has `image` tensor |

### Example VLM Request

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
    "http://localhost:8000/v2/models/qwen2vl_nutrition_gptq_int4/infer",
    json=payload
)
print(response.json())
```

---

## Path Mapping for Cloud Deployment

### The Three Path Contexts

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PATH CONTEXTS                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. LAB SERVER (where files currently are)                                  │
│     /ssd1/zhuoyuan/vlm_outputs/qwen2vl-...-gptq-int4/                       │
│                                                                             │
│  2. CLOUD SERVER HOST (where you transfer files to)                         │
│     /models/qwen2vl-...-gptq-int4/                                          │
│                                                                             │
│  3. DOCKER CONTAINER (what config files reference)                          │
│     /models/qwen2vl-...-gptq-int4/    ← model.json "model" path             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Path Mapping Table

| File | What Path to Use |
|------|------------------|
| `model.json` | CONTAINER path (right side of `-v` mount) |
| `rsync`/`scp` commands | CLOUD HOST path (left side of `-v` mount) |
| `--model-repository` | CONTAINER path |

### Common Mistake

```bash
# WRONG - lab server path in model.json
{"model": "/ssd1/zhuoyuan/vlm_outputs/qwen2vl-..."}

# CORRECT - container path (matches Docker mount)
{"model": "/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4"}
```

---

## Deployment Workflow

### Step 1: Transfer Files to Cloud Server

```bash
# Transfer BF16 model weights (~15GB)
rsync -avz --progress \
  /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
  user@cloud:/models/

# Transfer GPTQ INT4 model weights (~7GB)
rsync -avz --progress \
  /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4 \
  user@cloud:/models/

# Transfer Triton config repository
rsync -avz --progress \
  triton_model_repository/ \
  user@cloud:/workspace/triton_model_repository/

# Transfer scripts
rsync -avz --progress \
  scripts/deploy_triton.sh \
  scripts/benchmark_triton.py \
  scripts/validate_triton_accuracy.py \
  user@cloud:/workspace/scripts/
```

### Step 2: Start Triton Server

```bash
# On cloud server
cd /workspace
./scripts/deploy_triton.sh
```

### Step 3: Verify Deployment

```bash
# Health check
curl http://localhost:8000/v2/health/live

# Check models
curl http://localhost:8000/v2/models/qwen2vl_nutrition_bf16
curl http://localhost:8000/v2/models/qwen2vl_nutrition_gptq_int4
```

### Step 4: Run Benchmarks

```bash
# Benchmark both models
python scripts/benchmark_triton.py \
    --model qwen2vl_nutrition_bf16 qwen2vl_nutrition_gptq_int4 \
    --num-requests 20 \
    --concurrency 4 \
    --output triton_benchmark_results.json
```

### Step 5: Validate Accuracy

```bash
python scripts/validate_triton_accuracy.py \
    --model qwen2vl_nutrition_gptq_int4 \
    --baseline /path/to/bf16_baseline_outputs.json \
    --output triton_validation_results.json
```

---

## Triton API Reference

### Health Endpoints

```bash
# Liveness (server running)
curl http://localhost:8000/v2/health/live

# Readiness (models loaded)
curl http://localhost:8000/v2/health/ready

# Model-specific readiness
curl http://localhost:8000/v2/models/qwen2vl_nutrition_gptq_int4/ready
```

### Model Info

```bash
# List all models
curl http://localhost:8000/v2/models

# Model details
curl http://localhost:8000/v2/models/qwen2vl_nutrition_gptq_int4
```

### Inference

```bash
# HTTP inference
POST http://localhost:8000/v2/models/{model_name}/infer
Content-Type: application/json

# gRPC inference
localhost:8001 (use tritonclient library)
```

### Metrics

```bash
# Prometheus metrics
curl http://localhost:8002/metrics
```

---

## Files Summary

| File | Purpose | Status |
|------|---------|--------|
| `notebooks/10_Triton.ipynb` | Educational notebook | ✅ Committed |
| `notebooks/10_Triton.py` | Jupytext sync | ✅ Committed |
| `refactor_documentation/SESSION30_Plan_Triton_implementation.md` | Implementation plan | ✅ Committed |
| `triton_model_repository/README.md` | Quick start guide | ✅ Created |
| `triton_model_repository/PATH_MAPPING.md` | Path documentation | ✅ Created |
| `triton_model_repository/qwen2vl_nutrition_bf16/config.pbtxt` | BF16 Triton config | ✅ Created |
| `triton_model_repository/qwen2vl_nutrition_bf16/1/model.json` | BF16 vLLM config | ✅ Created |
| `triton_model_repository/qwen2vl_nutrition_gptq_int4/config.pbtxt` | GPTQ Triton config | ✅ Created |
| `triton_model_repository/qwen2vl_nutrition_gptq_int4/1/model.json` | GPTQ vLLM config | ✅ Created |
| `scripts/deploy_triton.sh` | Deployment script | ✅ Created |
| `scripts/benchmark_triton.py` | Benchmark script | ✅ Created |
| `scripts/validate_triton_accuracy.py` | Validation script | ✅ Created |

---

## Next Steps

1. **Rent cloud GPU** with Docker support (Lambda Labs, RunPod, Vast.ai)
2. **Transfer files** using rsync commands above
3. **Deploy and test** Triton server
4. **Run benchmarks** comparing Triton vs standalone vLLM
5. **Validate accuracy** to ensure no drift from serving layer change
6. **Document results** in a follow-up progress file

---

## References

- [Triton Inference Server Docs](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/)
- [Triton vLLM Backend](https://github.com/triton-inference-server/vllm_backend)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Session 29: GPTQ Quantization](./PROGRESS_20260118_SESSION29.md)
- [Session 30 Plan](./SESSION30_Plan_Triton_implementation.md)
