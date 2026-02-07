# Session 32 Progress - Triton Cloud Deployment (Vast.ai)

**Date**: 2026-02-06
**Session Name**: triton-cloud-deploy

---

## Objective

Deploy the fine-tuned Qwen2-VL nutrition detection model to NVIDIA Triton Inference Server on a cloud GPU (Vast.ai), benchmark both GPTQ INT4 and BF16 models sequentially, and compare Triton performance with standalone vLLM.

---

## Cloud Server Specs

| Spec | Value |
|------|-------|
| **Provider** | Vast.ai |
| **GPU** | NVIDIA GeForce RTX 4090 (24GB VRAM) |
| **OS** | Ubuntu 22.04.5 LTS |
| **NVIDIA Driver** | 580.95.05 |
| **CUDA Version** | 13.0 |
| **Docker** | 28.1.1 |
| **Disk** | ~289 GB free |
| **SSH** | Port 10241 |

---

## Step 0: Provision and Access the Cloud Server

### 0.1 Rent a Vast.ai Instance

Requirements:
- GPU with 24GB+ VRAM (RTX 4090, A10, etc.)
- Docker support with GPU access
- ~50GB+ storage for model weights + Docker images

### 0.2 Verify GPU + Docker on Cloud Server

```bash
# Check GPU
nvidia-smi

# Verify Docker can access GPU
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If Docker GPU fails with `could not select device driver`, install the NVIDIA Container Toolkit:

```bash
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

### 0.3 Set Up SSH Access from Lab Server

The lab server needs SSH access to the cloud server for file transfers (rsync).

**On lab server:**
```bash
# Print your public key
cat ~/.ssh/id_rsa.pub
# If no key exists: ssh-keygen -t rsa -b 4096
```

**On cloud server** — add the lab server's public key:
```bash
echo "YOUR_LAB_SERVER_PUBLIC_KEY" >> ~/.ssh/authorized_keys
```

**Note**: Since `~/.ssh/` is on the NAS shared across all lab servers (vllab4-vllab15), one key works from any lab server.

**(Optional) For IDE remote access (VS Code / Cursor)** — also add your local machine's key:
```bash
echo "YOUR_LOCAL_MACHINE_PUBLIC_KEY" >> ~/.ssh/authorized_keys
```

**Test from lab server:**
```bash
# Example: ssh -p <PORT> <USER>@<HOST> "echo connected"
ssh -p 10241 root@82.141.118.42 "echo connected"
```

---

## Step 1: Transfer Files from Lab Server to Cloud

Run on the **lab server**. Adjust `REMOTE`, `PORT`, and paths for your setup.

```bash
# === Configuration (adjust for your setup) ===
REMOTE="root@82.141.118.42"
PORT=10241
REMOTE_DIR="/workspace"

# Local paths (lab server)
GPTQ_LOCAL="/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4"
BF16_LOCAL="/ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged"
TRITON_REPO_LOCAL="~/projects/vlm_Qwen2VL_object_detection/triton_model_repository"
SCRIPTS_LOCAL="~/projects/vlm_Qwen2VL_object_detection/scripts"

# === Create directories on cloud ===
ssh -p ${PORT} ${REMOTE} "mkdir -p ${REMOTE_DIR}/models ${REMOTE_DIR}/triton_model_repository ${REMOTE_DIR}/scripts ${REMOTE_DIR}/results"

# === Transfer 1: GPTQ INT4 weights (~6.5 GB) ===
rsync -avz --progress -e "ssh -p ${PORT}" ${GPTQ_LOCAL}/ ${REMOTE}:${REMOTE_DIR}/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4/

# === Transfer 2: BF16 weights (~16 GB) ===
rsync -avz --progress -e "ssh -p ${PORT}" ${BF16_LOCAL}/ ${REMOTE}:${REMOTE_DIR}/models/qwen2vl-nutrition-detection-r4-joint-merged/

# === Transfer 3: Triton configs + scripts (tiny) ===
rsync -avz --progress -e "ssh -p ${PORT}" ${TRITON_REPO_LOCAL}/ ${REMOTE}:${REMOTE_DIR}/triton_model_repository/
rsync -avz --progress -e "ssh -p ${PORT}" ${SCRIPTS_LOCAL}/{deploy_triton.sh,benchmark_triton.py,validate_triton_accuracy.py,benchmark_vllm.py} ${REMOTE}:${REMOTE_DIR}/scripts/
```

**Tip**: Run transfers 1, 2, and 3 in separate terminal tabs to parallelize.

---

## Step 2: Fix Configs for Cloud Server

Run on the **cloud server**:

### 2a. Remove `gpus` lines from `config.pbtxt`

The original configs used `gpus: [0]` / `gpus: [1]` for dual-GPU A/B testing. However, **`KIND_MODEL` does not allow `gpus` specification** — vLLM manages GPU placement internally. Specifying GPUs with `KIND_MODEL` causes Triton to fail with:

> `Invalid argument: instance group ... has kind KIND_MODEL but specifies one or more GPUs`

```bash
# Remove gpus lines from both configs
sed -i '/gpus:/d' /workspace/triton_model_repository/qwen2vl_nutrition_gptq_int4/config.pbtxt
sed -i '/gpus:/d' /workspace/triton_model_repository/qwen2vl_nutrition_bf16/config.pbtxt
```

### 2b. Remove `_comment` fields from `model.json`

Triton's vLLM backend passes **all** keys in `model.json` directly to `AsyncEngineArgs`. JSON doesn't support comments, and `_comment` fields cause:

> `AsyncEngineArgs.__init__() got an unexpected keyword argument '_comment'`

```bash
sed -i '/_comment/d' /workspace/triton_model_repository/qwen2vl_nutrition_gptq_int4/1/model.json
sed -i '/_comment/d' /workspace/triton_model_repository/qwen2vl_nutrition_bf16/1/model.json
```

### 2c. Remove `disable_log_stats` and `disable_log_requests` from `model.json`

These are **server-level flags**, not `AsyncEngineArgs` parameters. Including them causes:

> `AsyncEngineArgs.__init__() got an unexpected keyword argument 'disable_log_requests'`

```bash
# Use python to cleanly remove both keys from both files
python3 -c "
import json
for path in [
    '/workspace/triton_model_repository/qwen2vl_nutrition_gptq_int4/1/model.json',
    '/workspace/triton_model_repository/qwen2vl_nutrition_bf16/1/model.json',
]:
    with open(path) as f: d = json.load(f)
    d.pop('disable_log_stats', None)
    d.pop('disable_log_requests', None)
    with open(path, 'w') as f: json.dump(d, f, indent=4)
"
```

### 2d. Update `model.json` paths

Update model paths to match where weights were placed on the cloud server:

```bash
# GPTQ model.json
sed -i 's|/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4|/workspace/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4|' /workspace/triton_model_repository/qwen2vl_nutrition_gptq_int4/1/model.json

# BF16 model.json
sed -i 's|/models/qwen2vl-nutrition-detection-r4-joint-merged|/workspace/models/qwen2vl-nutrition-detection-r4-joint-merged|' /workspace/triton_model_repository/qwen2vl_nutrition_bf16/1/model.json
```

### Final `model.json` (GPTQ INT4)

After all fixes, the GPTQ `model.json` should look like:

```json
{
    "model": "/workspace/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4",
    "tokenizer_mode": "auto",
    "trust_remote_code": true,
    "dtype": "half",
    "quantization": "gptq_marlin",
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 4096,
    "limit_mm_per_prompt": {
        "image": 1
    }
}
```

**Rule of thumb**: Only include keys that are valid `AsyncEngineArgs` parameters. No comments, no server-level flags.

---

## Step 3: Pull Triton Docker Image

Run on **cloud server** (can run in parallel with file transfers):

```bash
# We tried 25.11 first (vLLM 0.11.0) but hit bugs. Now using 26.01 (vLLM 0.13.0).
docker pull nvcr.io/nvidia/tritonserver:26.01-vllm-python-py3
```

This image is ~20 GB.

### Choosing the Right Triton Image

The Triton image version determines the bundled vLLM version. Check your host driver with `nvidia-smi`.

| Triton Image | vLLM Version | Min Driver (matrix) | Notes |
|-------------|-------------|-------------------|-------|
| `24.08-vllm-python-py3` | 0.5.3 | ~535+ | Too old — missing `limit_mm_per_prompt` |
| `25.11-vllm-python-py3` | 0.11.0 | ~580+ | Has `cu_seqlens_q` bug with Qwen2-VL |
| `26.01-vllm-python-py3` | **0.13.0** | 590.48+ | Matches our standalone vLLM. **Best choice.** |

**Our choice**: `26.01` (vLLM 0.13.0) — matches the exact vLLM version we used for standalone benchmarks on the lab server.

**Driver note**: The compatibility matrix says 590.48+ but our lab server runs vLLM 0.13.0 on driver 550.144.03, so forward compatibility likely works. Our Vast.ai server has driver 580.95.05 which should be fine.

**Lessons learned**:

The `24.08` image (vLLM 0.5.3) failed because:
1. `KIND_MODEL` + `gpus` specification is not allowed
2. `_comment` fields in `model.json` get passed to `AsyncEngineArgs` (JSON has no comments)
3. `limit_mm_per_prompt` didn't exist in vLLM 0.5.x

The `25.11` image (vLLM 0.11.0) fixed config issues but hit a **vLLM bug with Qwen2-VL**:

4. **V1 engine bug**: `RuntimeError: cu_seqlens_q must be on CUDA`
   - Happens during startup memory profiling: `profile_run()` → `get_multimodal_embeddings()` → `_process_video_input()`
   - vLLM V1 engine creates a tensor on CPU that should be on CUDA
   - This is a vLLM bug, not a Triton or config issue
   - **The same bug would occur with standalone `vllm serve` using vLLM 0.11.0**

5. **V0 engine fallback fails too**: Setting `VLLM_USE_V1=0` gives `AssertionError: <EMPTY MESSAGE>` — V0 engine is likely deprecated/broken in vLLM 0.11.0.

6. **`enforce_eager=True` attempted**: Disables CUDA graph compilation to change profiling behavior. Status: TBD.

### Root Cause Summary

The issue is **not Triton** — it's a **vLLM version mismatch**:
- Our standalone vLLM on the lab server is **0.13.0** (works with Qwen2-VL)
- Triton 25.11 bundles vLLM **0.11.0** (has a bug with Qwen2-VL multimodal profiling)
- Triton 26.01 bundles vLLM **0.13.0** (would work) but the compatibility matrix says it requires driver **590.48+**
- Our Vast.ai server has driver **580.95.05**

### Driver Compatibility: Forward Compatibility May Work

Key insight: Our **lab server runs vLLM 0.13.0 on driver 550.144.03** (even older than the Vast.ai driver 580.95.05), and it works fine. The NVIDIA driver compatibility matrix is about the CUDA toolkit version in the container vs the driver's supported CUDA version. Docker containers use the host's driver but bundle their own CUDA runtime, so **forward compatibility** often works — as long as the host driver supports the CUDA API version the container needs.

**Decision**: Try Triton 26.01 anyway. If the driver is too old, Docker will fail at startup with a clear CUDA error, so there's no risk.

### Resolution Path

| Step | Action | Status |
|------|--------|--------|
| 1 | Try `enforce_eager=True` with Triton 25.11 (vLLM 0.11.0) | Failed — same `cu_seqlens_q` error |
| 2 | Pull Triton 26.01 (vLLM 0.13.0) despite driver matrix | **Success!** |
| 3 | Proceed with benchmarks | In progress |

**Triton 26.01 startup confirmed** with driver 580.95.05. The container logged:
```
WARNING: CUDA Minor Version Compatibility mode ENABLED.
Using driver version 580.95.05 which has support for CUDA 13.0.
This container was built with CUDA 13.1 and will be run in Minor Version Compatibility mode.
```
This is just an informational warning — the container runs fine in this mode. The vLLM V1 engine (v0.13.0) initialized successfully with `gptq_marlin` backend and the model loaded to READY state.

---

## Step 4: Install Python Dependencies

The benchmark scripts run on the **host machine** (not inside Docker), so dependencies must be installed on the host. The Vast.ai host has Python 3.10.

Run on **cloud server** (use the requirements file for reproducibility):

```bash
pip install -r requirements_triton_benchmark.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

The above commands were how we originally discovered the dependencies ad-hoc:
```bash
# What we ran initially (before creating requirements_triton_benchmark.txt):
pip install aiohttp numpy Pillow datasets qwen-vl-utils
pip install 'tritonclient[all]'
pip install 'transformers>=4.45,<5.0'    # 5.x breaks Qwen2VLProcessor
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install 'jinja2>=3.1.0'
```

### Dependency errors encountered

| Error | Cause | Fix |
|-------|-------|-----|
| `TypeError: argument of type 'NoneType' is not iterable` in `video_processing_auto.py` | `transformers==5.1.0` has a bug with Qwen2VL's video processor auto-detection | Pin `transformers>=4.45,<5.0` |
| `PyTorch was not found. Models won't be available` | Host doesn't have torch installed (only the Docker container does) | `pip install torch --index-url .../cpu` |
| `AutoVideoProcessor requires the Torchvision library` | `Qwen2VLProcessor.from_pretrained()` needs torchvision for image processing | `pip install torchvision --index-url .../cpu` |
| `apply_chat_template requires jinja2>=3.1.0. Your version is 3.0.3` | Host's jinja2 is too old for chat template rendering | `pip install 'jinja2>=3.1.0'` |

**Key insight**: The Triton Docker container has its own Python environment with all ML dependencies. But our benchmark scripts run on the host, which is a bare Ubuntu with minimal packages. The host needs its own compatible set of dependencies for the processor/tokenizer.

**Reproducible setup**: The project includes `requirements_triton_benchmark.txt` — a pip-installable subset of `environment_vllm_serving.yml` with CPU-only torch (sufficient for tokenizer/processor). On any new cloud machine:
```bash
pip install -r requirements_triton_benchmark.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

**How this environment was built**: Dependencies were initially installed ad-hoc on the Vast.ai host while debugging errors (see table above). Afterwards, all version pins in `requirements_triton_benchmark.txt` were aligned to `environment_vllm_serving.yml` to maintain a single source of truth. The only intentional difference is torch/torchvision using `+cpu` instead of `+cu129` — the benchmark client only needs PyTorch for the tokenizer/processor, not GPU inference (that runs inside the Triton container). The Vast.ai host environment was then downgraded to match these aligned versions, and all benchmarks were re-verified.

---

## Phase A: Deploy & Benchmark GPTQ INT4

### Start Triton

```bash
docker run --gpus all --rm -d \
    --shm-size=4G \
    --name triton-gptq \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v /workspace/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4:/workspace/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4:ro \
    -v /workspace/triton_model_repository/qwen2vl_nutrition_gptq_int4:/models/triton_repo/qwen2vl_nutrition_gptq_int4 \
    nvcr.io/nvidia/tritonserver:26.01-vllm-python-py3 \
    tritonserver --model-repository=/models/triton_repo

# Watch logs — wait for "Started GRPCInferenceService" + "Started HTTPService"
docker logs -f triton-gptq
# Ctrl+C to stop following (container keeps running)
```

### Verify

```bash
curl http://localhost:8000/v2/health/live
curl http://localhost:8000/v2/models/qwen2vl_nutrition_gptq_int4
```

### Critical: `/infer` vs `/generate` Endpoints

Triton exposes two HTTP endpoints for inference:

| Endpoint | Format | Works with `decoupled: true`? |
|----------|--------|-------------------------------|
| `/v2/models/{name}/infer` | Structured tensor format (`inputs: [{name, shape, datatype, data}]`) | **No** — returns HTTP 501 |
| `/v2/models/{name}/generate` | Flat key-value JSON (`{"text_input": "...", "image": "..."}`) | **Yes** |

Our `config.pbtxt` has `model_transaction_policy { decoupled: true }`. This is **required** by the vLLM backend because vLLM supports streaming (sending multiple partial responses per request). Even though we don't use streaming, the vLLM backend needs decoupled mode.

**Error we hit**: All 20 benchmark requests instantly failed (0.11s total) with:
```
HTTP 501: {"error": "HTTP end point doesn't support models with decoupled transaction policy"}
```

This was confusing because:
- The model showed as `READY` in Triton
- Health checks passed
- `curl .../generate` worked fine (we tested manually)
- But the benchmark script was using `/infer`, not `/generate`

**Fix**: Updated `benchmark_triton.py` to use the `/generate` endpoint:

```python
# Before (BROKEN with decoupled vLLM models):
url = f"{http_url}/v2/models/{model_name}/infer"
payload = {"inputs": [{"name": "text_input", "shape": [1], "datatype": "BYTES", "data": [...]}]}

# After (WORKS with decoupled vLLM models):
url = f"{http_url}/v2/models/{model_name}/generate"
payload = {"text_input": "...", "image": "base64...", "parameters": {"max_tokens": 100}}
```

The `/generate` response format is also simpler:
```json
{"model_name": "...", "model_version": "1", "text_output": "the generated text"}
```

**gRPC and decoupled models**: The gRPC *protocol* supports streaming RPCs, so decoupled models can work over gRPC. However, our benchmark script uses `client.infer()` (unary RPC), not `client.stream_infer()` (streaming RPC). Unary `infer()` fails with decoupled models just like HTTP `/infer` does. To benchmark gRPC, the script would need to be updated to use `stream_infer()`. This was not implemented; gRPC benchmarks were skipped.

### Benchmark

```bash
export QWEN2VL_PROCESSOR_PATH=/workspace/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4

# HTTP concurrency=1
python /workspace/scripts/benchmark_triton.py \
    --endpoint http --model qwen2vl_nutrition_gptq_int4 \
    --num-requests 20 --concurrency 1 \
    --output /workspace/results/gptq_http_c1.json

# HTTP concurrency=4
python /workspace/scripts/benchmark_triton.py \
    --endpoint http --model qwen2vl_nutrition_gptq_int4 \
    --num-requests 20 --concurrency 4 \
    --output /workspace/results/gptq_http_c4.json

# gRPC concurrency=1
python /workspace/scripts/benchmark_triton.py \
    --endpoint grpc --model qwen2vl_nutrition_gptq_int4 \
    --num-requests 20 --concurrency 1 \
    --output /workspace/results/gptq_grpc_c1.json
```

### Stop

```bash
docker stop triton-gptq
```

### Results (Unbiased — prefix caching OFF, varied images)

| Metric | HTTP c=1 | HTTP c=4 |
|--------|----------|----------|
| Successful | 20/20 | 20/20 |
| Total time (s) | 12.15 | 5.07 |
| Throughput (req/s) | **1.65** | **3.94** |
| Avg Latency (ms) | 607.4 | 1012.9 |
| Min Latency (ms) | 260.5 | 857.9 |
| Max Latency (ms) | 3282.0 | 1113.5 |
| P50 Latency (ms) | **469.7** | **1012.5** |
| P90 Latency (ms) | 559.1 | 1098.9 |
| P99 Latency (ms) | 2765.5 | 1111.4 |

**Note on prefix caching**: We set `enable_prefix_caching: false` in model.json, which is passed to vLLM's `AsyncEngineArgs`. No error was raised, so the flag was accepted. As a second layer of protection, `--vary-images` sends a different image per request, so even if the flag were somehow ignored, there would be minimal KV cache reuse across requests.

Commands used:
```bash
# Disable prefix caching in model.json (one-time, before starting container)
python3 -c "import json; d=json.load(open('/workspace/triton_model_repository/qwen2vl_nutrition_gptq_int4/1/model.json')); d['enable_prefix_caching']=False; json.dump(d,open('/workspace/triton_model_repository/qwen2vl_nutrition_gptq_int4/1/model.json','w'),indent=4)"

export QWEN2VL_PROCESSOR_PATH=/workspace/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4

# c=1 with --vary-images (20 unique images, no prefix cache reuse)
python3 /workspace/scripts/benchmark_triton.py --endpoint http --model qwen2vl_nutrition_gptq_int4 --num-requests 20 --concurrency 1 --vary-images --output /workspace/results/gptq_http_c1_unbiased.json

# c=4 with --vary-images
python3 /workspace/scripts/benchmark_triton.py --endpoint http --model qwen2vl_nutrition_gptq_int4 --num-requests 20 --concurrency 4 --vary-images --output /workspace/results/gptq_http_c4_unbiased.json
```

<details>
<summary>Earlier biased results (prefix caching ON, same image repeated — for reference only)</summary>

| Metric | HTTP c=1 | HTTP c=4 |
|--------|----------|----------|
| Throughput (req/s) | 2.16 | 10.20 |
| P50 Latency (ms) | 309.9 | 325.0 |

These were ~50% faster than real-world because prefix caching reused KV cache across identical requests.
</details>

**Notes**:
- Max latency (~3282ms) on c=1 is the first request triggering CUDA graph compilation.
- gRPC benchmark was skipped — our script uses `client.infer()` (unary RPC) which doesn't work with decoupled models. Would need `stream_infer()` (streaming RPC) instead.

---

## Phase B: Deploy & Benchmark BF16

### Start Triton

```bash
docker run --gpus all --rm -d \
    --shm-size=4G \
    --name triton-bf16 \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v /workspace/models/qwen2vl-nutrition-detection-r4-joint-merged:/workspace/models/qwen2vl-nutrition-detection-r4-joint-merged:ro \
    -v /workspace/triton_model_repository/qwen2vl_nutrition_bf16:/models/triton_repo/qwen2vl_nutrition_bf16 \
    nvcr.io/nvidia/tritonserver:26.01-vllm-python-py3 \
    tritonserver --model-repository=/models/triton_repo

docker logs -f triton-bf16
```

### Benchmark

```bash
export QWEN2VL_PROCESSOR_PATH=/workspace/models/qwen2vl-nutrition-detection-r4-joint-merged

python /workspace/scripts/benchmark_triton.py \
    --endpoint http --model qwen2vl_nutrition_bf16 \
    --num-requests 20 --concurrency 1 \
    --output /workspace/results/bf16_http_c1.json

python /workspace/scripts/benchmark_triton.py \
    --endpoint http --model qwen2vl_nutrition_bf16 \
    --num-requests 20 --concurrency 4 \
    --output /workspace/results/bf16_http_c4.json

docker stop triton-bf16
```

### Results (Unbiased — prefix caching OFF, varied images)

| Metric | HTTP c=1 | HTTP c=4 |
|--------|----------|----------|
| Successful | 20/20 | 20/20 |
| Total time (s) | 16.79 | 6.20 |
| Throughput (req/s) | **1.19** | **3.23** |
| Avg Latency (ms) | 839.6 | 1235.9 |
| Min Latency (ms) | 485.3 | 1074.8 |
| Max Latency (ms) | 3605.1 | 1331.1 |
| P50 Latency (ms) | **703.3** | **1236.2** |
| P90 Latency (ms) | 769.5 | 1325.0 |
| P99 Latency (ms) | 3075.1 | 1330.6 |

Commands used:
```bash
export QWEN2VL_PROCESSOR_PATH=/workspace/models/qwen2vl-nutrition-detection-r4-joint-merged

python3 /workspace/scripts/benchmark_triton.py --endpoint http --model qwen2vl_nutrition_bf16 --num-requests 20 --concurrency 1 --vary-images --output /workspace/results/bf16_http_c1_unbiased.json

python3 /workspace/scripts/benchmark_triton.py --endpoint http --model qwen2vl_nutrition_bf16 --num-requests 20 --concurrency 4 --vary-images --output /workspace/results/bf16_http_c4_unbiased.json
```

<details>
<summary>Earlier biased results (prefix caching ON, same image repeated — for reference only)</summary>

| Metric | HTTP c=1 | HTTP c=4 |
|--------|----------|----------|
| Throughput (req/s) | 1.45 | 6.41 |
| P50 Latency (ms) | 538.3 | 560.1 |
</details>

---

## Step 9: Copy Results Back to Lab Server

Run on **lab server**:

```bash
REMOTE="root@82.141.118.42"
PORT=10241
rsync -avz --progress -e "ssh -p ${PORT}" ${REMOTE}:/workspace/results/ ~/projects/vlm_Qwen2VL_object_detection/triton_results/
```

---

## Comparison: Triton vs Standalone vLLM (Session 29)

All Session 32 results below are **unbiased** (prefix caching OFF, varied images).

| Metric | vLLM Standalone GPTQ (Session 29) | Triton GPTQ HTTP c=1 | Triton BF16 HTTP c=1 |
|--------|-----------------------------------|----------------------|----------------------|
| GPU | RTX 3090 (vllab8) | RTX 4090 (Vast.ai) | RTX 4090 (Vast.ai) |
| vLLM Version | 0.13.0 | 0.13.0 (Triton 26.01) | 0.13.0 (Triton 26.01) |
| Throughput (req/s) | 1.15 | **1.65** | 1.19 |
| E2E / P50 Latency (ms) | 819 | **469.7** | 703.3 |
| TPOT (ms) | 7.4 | — | — |

**Scaling with concurrency**:

| Metric | Triton GPTQ c=1 | Triton GPTQ c=4 | Triton BF16 c=1 | Triton BF16 c=4 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Throughput (req/s) | 1.65 | **3.94** | 1.19 | **3.23** |
| P50 Latency (ms) | 469.7 | 1012.5 | 703.3 | 1236.2 |

**Key observations**:
- **GPTQ INT4 is ~1.5x faster** than BF16 at P50 latency (470ms vs 703ms), consistent with 4-bit quantization reducing memory bandwidth and computation.
- **Concurrency tradeoff**: c=4 gives ~2.4-2.7x throughput but P50 latency increases ~2x. This is realistic — with varied images, each request needs full prefill, so batching increases per-request latency.
- **RTX 4090 vs RTX 3090**: The GPTQ P50 dropped from 819ms → 470ms (~1.7x faster).
- **Prefix caching impact**: The earlier biased results (same image, prefix caching ON) showed P50=310ms — about 50% faster than the unbiased 470ms. This shows prefix caching has a large effect when input prompts share common prefixes.

**Caveat**: The Session 29 vs Session 32 comparison involves different hardware. For a strict Triton-overhead measurement, rerun standalone vLLM benchmarks on the same RTX 4090.

---

## Troubleshooting & Error Log

This section documents every error we encountered during deployment, in chronological order. These are common pitfalls when deploying VLMs on Triton — documenting them saves hours of debugging next time.

### Error 1: Docker GPU fails
```
docker: Error response from daemon: could not select device driver
```
**Cause**: NVIDIA Container Toolkit not installed.
**Fix**:
```bash
apt-get update && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

### Error 2: SSH Permission Denied (lab → cloud)
```
Permission denied (publickey)
```
**Cause**: Lab server's public key not in cloud server's `~/.ssh/authorized_keys`.
**Fix**: Copy `~/.ssh/id_rsa.pub` from lab server, add to cloud's `~/.ssh/authorized_keys`.
**Gotcha**: The key must be a **single unbroken line**. If pasted with line breaks (e.g., from a terminal with word wrap), SSH silently rejects it. Use `cat ~/.ssh/id_rsa.pub` and copy the entire output as one line.
```bash
chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
```

### Error 3: `KIND_MODEL but specifies one or more GPUs` (Triton 24.08)
```
Invalid argument: instance group ... has kind KIND_MODEL but specifies one or more GPUs
```
**Cause**: `config.pbtxt` had `gpus: [1]` with `kind: KIND_MODEL`. The vLLM backend manages GPU placement internally — you cannot specify GPUs in the instance group.
**Fix**: Remove all `gpus:` lines from `config.pbtxt`.

### Error 4: `unexpected keyword argument '_comment'` (Triton 24.08)
```
AsyncEngineArgs.__init__() got an unexpected keyword argument '_comment'
```
**Cause**: `model.json` had `_comment` fields for documentation. Triton's vLLM backend passes **every key** in `model.json` directly to `AsyncEngineArgs`. JSON has no native comment syntax, and `_comment` is not a valid engine argument.
**Fix**: Remove all `_comment` fields from `model.json`. Document in `config.pbtxt` comments instead.

### Error 5: `unexpected keyword argument 'limit_mm_per_prompt'` (Triton 24.08, vLLM 0.5.3)
```
AsyncEngineArgs.__init__() got an unexpected keyword argument 'limit_mm_per_prompt'
```
**Cause**: vLLM 0.5.3 (bundled in Triton 24.08) is too old — `limit_mm_per_prompt` was added in a later version.
**Fix**: Upgrade to a newer Triton image (25.11+ with vLLM 0.11.0+).

### Error 6: `unexpected keyword argument 'disable_log_requests'` (Triton 25.11, vLLM 0.11.0)
```
AsyncEngineArgs.__init__() got an unexpected keyword argument 'disable_log_requests'
```
**Cause**: `disable_log_stats` and `disable_log_requests` are **server-level flags** (passed to `vllm serve`), not `AsyncEngineArgs` parameters. They don't belong in `model.json`.
**Fix**: Remove both fields from `model.json`.

### Error 7: `RuntimeError: cu_seqlens_q must be on CUDA` (Triton 25.11, vLLM 0.11.0 V1 engine)
```
RuntimeError: cu_seqlens_q must be on CUDA
```
Full traceback: `profile_run()` → `get_multimodal_embeddings()` → `_process_video_input()`

**Cause**: A bug in vLLM 0.11.0's V1 engine where a tensor is created on CPU instead of CUDA during the startup memory profiling phase for multimodal (Qwen2-VL) models. This is a vLLM bug, not a Triton issue.
**Fix**: Use a Triton image with vLLM 0.13.0+ (Triton 26.01). Adding `enforce_eager=True` to model.json did NOT fix this.

### Error 8: `AssertionError: <EMPTY MESSAGE>` (Triton 25.11, vLLM 0.11.0 V0 engine)
```
AssertionError: <EMPTY MESSAGE>
```
**Cause**: Setting `VLLM_USE_V1=0` to fall back to the V0 engine doesn't work because V0 is deprecated/broken in vLLM 0.11.0.
**Fix**: Don't use V0. Use Triton 26.01 (vLLM 0.13.0) where V1 works correctly.

### Error 9: HTTP 501 — `/infer` with decoupled models
```json
{"error": "HTTP end point doesn't support models with decoupled transaction policy"}
```
**Cause**: Our `config.pbtxt` has `model_transaction_policy { decoupled: true }`, which is **required** by the vLLM backend for streaming support. The `/v2/models/{name}/infer` HTTP endpoint does not support decoupled models — it expects exactly one response per request, but decoupled mode allows multiple (for streaming).

**Symptom**: All benchmark requests fail instantly (20 requests in 0.11 seconds) with no useful error in the benchmark output — you only see "Failed: 20" with zero latency, because Triton rejects them before any inference happens.

**Fix**: Use the `/v2/models/{name}/generate` endpoint instead, which supports decoupled models and uses a simpler flat JSON format. See the "Critical: `/infer` vs `/generate` Endpoints" section above.

### Error 10: Host Python dependency issues
Multiple errors when running benchmark scripts on the host:
- `transformers==5.1.0` has a bug with Qwen2VLProcessor (video processing auto-detection)
- PyTorch not installed on host (only inside Docker container)
- torchvision not installed (required by processor)
- jinja2 too old for `apply_chat_template()`

**Fix**: See Step 4 above for the full dependency install commands.

### Summary: What Triton actually is

After debugging all these issues, here's the mental model:

**Triton is an orchestration layer**, not an inference engine. It wraps vLLM (which does the actual GPU inference) and adds:
- Model versioning and lifecycle management
- HTTP/gRPC API standardization
- Multi-model serving on one server
- Health checks, metrics, and monitoring

But this means:
1. **vLLM version matters more than Triton version** — if vLLM has a bug with your model, Triton can't fix it
2. **`model.json` = `AsyncEngineArgs`** — every key gets passed directly to vLLM, no extras allowed
3. **`config.pbtxt` = Triton orchestration** — input/output tensor definitions, instance groups, transaction policy
4. **Decoupled mode is mandatory for vLLM** — use `/generate` (HTTP) or `stream_infer()` (gRPC), not `/infer` (HTTP) or unary `infer()` (gRPC)
5. **Host vs Container** — benchmark scripts run on the host, model inference runs inside Docker. They need separate dependency management.

---

## Files on Cloud Server

```
/workspace/
├── models/
│   ├── qwen2vl-nutrition-detection-r4-joint-merged/          # BF16 (~16 GB)
│   └── qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4/ # GPTQ (~6.5 GB)
├── triton_model_repository/
│   ├── qwen2vl_nutrition_bf16/
│   │   ├── config.pbtxt
│   │   └── 1/model.json
│   └── qwen2vl_nutrition_gptq_int4/
│       ├── config.pbtxt
│       └── 1/model.json
├── scripts/
│   ├── deploy_triton.sh
│   ├── benchmark_triton.py
│   ├── validate_triton_accuracy.py
│   └── benchmark_vllm.py
└── results/
    ├── gptq_http_c1.json
    ├── gptq_http_c4.json
    ├── gptq_grpc_c1.json
    ├── bf16_http_c1.json
    └── bf16_http_c4.json
```

---

## Quick Reference: All Commands

Copy-paste ready commands for reproducing the full deployment and benchmarking. Run each block on the indicated machine.

### On Lab Server: Transfer files to cloud

```bash
REMOTE="root@82.141.118.42"
PORT=10241

ssh -p ${PORT} ${REMOTE} "mkdir -p /workspace/models /workspace/triton_model_repository /workspace/scripts /workspace/results"

# Transfer 1: GPTQ INT4 weights (~6.5 GB)
rsync -avz --progress -e "ssh -p ${PORT}" /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4/ ${REMOTE}:/workspace/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4/

# Transfer 2: BF16 weights (~16 GB)
rsync -avz --progress -e "ssh -p ${PORT}" /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged/ ${REMOTE}:/workspace/models/qwen2vl-nutrition-detection-r4-joint-merged/

# Transfer 3: Triton configs + scripts
rsync -avz --progress -e "ssh -p ${PORT}" ~/projects/vlm_Qwen2VL_object_detection/triton_model_repository/ ${REMOTE}:/workspace/triton_model_repository/
rsync -avz --progress -e "ssh -p ${PORT}" ~/projects/vlm_Qwen2VL_object_detection/scripts/{deploy_triton.sh,benchmark_triton.py,validate_triton_accuracy.py,benchmark_vllm.py} ${REMOTE}:/workspace/scripts/
rsync -avz --progress -e "ssh -p ${PORT}" ~/projects/vlm_Qwen2VL_object_detection/requirements_triton_benchmark.txt ${REMOTE}:/workspace/scripts/
```

### On Cloud Server: Fix configs

```bash
# Remove gpus lines (KIND_MODEL doesn't allow GPU specification)
sed -i '/gpus:/d' /workspace/triton_model_repository/qwen2vl_nutrition_gptq_int4/config.pbtxt
sed -i '/gpus:/d' /workspace/triton_model_repository/qwen2vl_nutrition_bf16/config.pbtxt

# Remove _comment fields (model.json passes ALL keys to AsyncEngineArgs)
sed -i '/_comment/d' /workspace/triton_model_repository/qwen2vl_nutrition_gptq_int4/1/model.json
sed -i '/_comment/d' /workspace/triton_model_repository/qwen2vl_nutrition_bf16/1/model.json

# Remove server-level flags and update paths
python3 -c "
import json
for path in [
    '/workspace/triton_model_repository/qwen2vl_nutrition_gptq_int4/1/model.json',
    '/workspace/triton_model_repository/qwen2vl_nutrition_bf16/1/model.json',
]:
    with open(path) as f: d = json.load(f)
    d.pop('disable_log_stats', None)
    d.pop('disable_log_requests', None)
    d.pop('enforce_eager', None)
    with open(path, 'w') as f: json.dump(d, f, indent=4)
"

# Update model paths to cloud locations
sed -i 's|/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4|/workspace/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4|' /workspace/triton_model_repository/qwen2vl_nutrition_gptq_int4/1/model.json
sed -i 's|/models/qwen2vl-nutrition-detection-r4-joint-merged|/workspace/models/qwen2vl-nutrition-detection-r4-joint-merged|' /workspace/triton_model_repository/qwen2vl_nutrition_bf16/1/model.json
```

### On Cloud Server: Pull image + install deps

```bash
docker pull nvcr.io/nvidia/tritonserver:26.01-vllm-python-py3

# Install benchmark client dependencies (CPU torch is sufficient)
pip install -r /workspace/scripts/requirements_triton_benchmark.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### On Cloud Server: Disable prefix caching for unbiased benchmarks

```bash
python3 -c "
import json
for path in ['/workspace/triton_model_repository/qwen2vl_nutrition_gptq_int4/1/model.json', '/workspace/triton_model_repository/qwen2vl_nutrition_bf16/1/model.json']:
    with open(path) as f: d = json.load(f)
    d['enable_prefix_caching'] = False
    with open(path, 'w') as f: json.dump(d, f, indent=4)
"
```

### On Cloud Server: GPTQ INT4 Benchmark

```bash
# Start Triton
docker run --gpus all --rm -d --shm-size=4G --name triton-gptq -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /workspace/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4:/workspace/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4:ro -v /workspace/triton_model_repository/qwen2vl_nutrition_gptq_int4:/models/triton_repo/qwen2vl_nutrition_gptq_int4 nvcr.io/nvidia/tritonserver:26.01-vllm-python-py3 tritonserver --model-repository=/models/triton_repo

# Wait for startup (look for "Started HTTPService")
docker logs -f triton-gptq

# Health check
curl http://localhost:8000/v2/health/live && echo "" && curl -s http://localhost:8000/v2/models/qwen2vl_nutrition_gptq_int4 | python3 -m json.tool | head -5

# Set processor path
export QWEN2VL_PROCESSOR_PATH=/workspace/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4

# HTTP c=1 (use --vary-images for unbiased results)
python3 /workspace/scripts/benchmark_triton.py --endpoint http --model qwen2vl_nutrition_gptq_int4 --num-requests 20 --concurrency 1 --vary-images --output /workspace/results/gptq_http_c1.json

# HTTP c=4
python3 /workspace/scripts/benchmark_triton.py --endpoint http --model qwen2vl_nutrition_gptq_int4 --num-requests 20 --concurrency 4 --vary-images --output /workspace/results/gptq_http_c4.json

# Stop
docker stop triton-gptq
```

### On Cloud Server: BF16 Benchmark

```bash
# Start Triton
docker run --gpus all --rm -d --shm-size=4G --name triton-bf16 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /workspace/models/qwen2vl-nutrition-detection-r4-joint-merged:/workspace/models/qwen2vl-nutrition-detection-r4-joint-merged:ro -v /workspace/triton_model_repository/qwen2vl_nutrition_bf16:/models/triton_repo/qwen2vl_nutrition_bf16 nvcr.io/nvidia/tritonserver:26.01-vllm-python-py3 tritonserver --model-repository=/models/triton_repo

# Wait for startup
docker logs -f triton-bf16

# Set processor path
export QWEN2VL_PROCESSOR_PATH=/workspace/models/qwen2vl-nutrition-detection-r4-joint-merged

# HTTP c=1
python3 /workspace/scripts/benchmark_triton.py --endpoint http --model qwen2vl_nutrition_bf16 --num-requests 20 --concurrency 1 --vary-images --output /workspace/results/bf16_http_c1.json

# HTTP c=4
python3 /workspace/scripts/benchmark_triton.py --endpoint http --model qwen2vl_nutrition_bf16 --num-requests 20 --concurrency 4 --vary-images --output /workspace/results/bf16_http_c4.json

# Stop
docker stop triton-bf16
```

### On Lab Server: Copy results back

```bash
REMOTE="root@82.141.118.42"
PORT=10241
rsync -avz --progress -e "ssh -p ${PORT}" ${REMOTE}:/workspace/results/ ~/projects/vlm_Qwen2VL_object_detection/triton_results/
```
